# dask_ddp_wrapper.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from dask.distributed import Client, wait



@dataclass(frozen=True)
class GangConfig:
    backend: str = "auto" # "auto" | "nccl" | "gloo"
    device: str = "auto" # "auto" | "cuda" | "cpu"
    init_method: str = "env://"
    required_resources: Optional[Dict[str, float]] = None  # e.g. {"GPU": 1}
    master_port: int = 23456 # free/open on rank-0 host, saturn cloud used 23456
    join_timeout_s: int = 600
    max_restarts: int = 50
    restart_backoff_s: float = 5.0


def _resolve_device_backend(cfg: GangConfig) -> Tuple[str, str]:
    # Choose device
    if cfg.device != "auto":
        device = cfg.device
    else:
        device = "cuda" if (cfg.required_resources or {}).get("GPU", 0) > 0 else "cpu"

    # Choose backend
    if cfg.backend != "auto":
        backend = cfg.backend
    else:
        backend = "nccl" if device == "cuda" else "gloo"

    return device, backend


def _eligible_workers(client: Client, required_resources: Optional[Dict[str, float]]) -> List[str]:
    info = client.scheduler_info()
    workers = info.get("workers", {}) or {}

    if not required_resources:
        return sorted(workers.keys())

    addrs: List[str] = []
    for addr, w in workers.items():
        wres = (w.get("resources", {}) or {})
        ok = all(float(wres.get(k, 0.0)) >= float(v) for k, v in required_resources.items())
        if ok:
            addrs.append(addr)

    return sorted(addrs)


def _pick_master_addr(worker_addr: str) -> str:
    # "tcp://10.0.0.12:12345" -> "10.0.0.12"
    hostport = worker_addr.split("://", 1)[-1]
    host = hostport.rsplit(":", 1)[0]
    return host


def _ddp_worker_entry(
    *,
    train_fn: Callable[..., Any],
    train_kwargs: Dict[str, Any],
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    backend: str,
    init_method: str,
    device: str,
) -> Any:
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' but torch.cuda.is_available() is False")

        # Assumes 1 GPU per Dask worker process/pod
        os.environ.setdefault("LOCAL_RANK", "0")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    try:
        dist.init_process_group(backend=backend, init_method=init_method)  # env:// uses MASTER_* 
        return train_fn(**train_kwargs)
    finally:
        # PyTorch recommends cleaning up process groups to avoid exit hangs
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


def ensure_workers(client, n_workers, timeout=120):
    try:
        client.wait_for_workers(n_workers, timeout=timeout)
    except TimeoutError:
        print("Workers not ready, restarting...")
        client.restart()
        client.wait_for_workers(n_workers, timeout=timeout)


def run_elasticish_ddp_on_dask(
    *,
    client: Client,
    train_fn: Callable[..., Any],
    train_kwargs: Dict[str, Any],
    cfg: GangConfig = GangConfig(),
    resolve_checkpoint: Optional[Callable[[], Optional[str]]] = None,
) -> Any:
    attempt = 0
    last_exc = None

    while attempt <= cfg.max_restarts:
        try:
            # Ensure workers are ready
            ensure_workers(client, len(cfg.required_resources or {"GPU": 1}), timeout=120)

            device, backend = _resolve_device_backend(cfg)
            workers = _eligible_workers(client, cfg.required_resources)
            if not workers:
                time.sleep(cfg.restart_backoff_s)
                continue

            world_size = len(workers)
            master_worker = workers[0]
            master_addr = _pick_master_addr(master_worker)
            master_port = int(cfg.master_port)

            run_kwargs = dict(train_kwargs)
            if resolve_checkpoint is not None:
                ckpt = resolve_checkpoint()
                if ckpt is not None:
                    run_kwargs["checkpoint_file"] = ckpt

            futures = []
            for rank, w in enumerate(workers):
                fut = client.submit(
                    _ddp_worker_entry,
                    train_fn=train_fn,
                    train_kwargs=run_kwargs,
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    backend=backend,
                    init_method=cfg.init_method,
                    device=device,
                    resources=cfg.required_resources,
                    workers=[w],
                    allow_other_workers=False,
                    pure=False,
                    retries=3,
                )
                futures.append(fut)
            
            # Wait for all futures to complete
            results = client.gather(futures)


            # Wait for first completion (success or error)
            done, not_done = wait(futures, return_when="FIRST_COMPLETED", timeout=cfg.join_timeout_s)
            failed = [f for f in done if f.status == "error"]
            if failed:
                last_exc = failed[0].exception()
                client.cancel(futures, force=True)
                attempt += 1
                time.sleep(cfg.restart_backoff_s)
                continue

            # Wait for completion or first error (no timeout)
            done, not_done = wait(futures, return_when="FIRST_COMPLETED")
            failed = [f for f in done if f.status == "error"]
            if failed:
                last_exc = failed[0].exception()
                client.cancel(futures, force=True)
                attempt += 1
                time.sleep(cfg.restart_backoff_s)
                continue

            results = client.gather(futures)
            return results[0] if results else None

        except Exception as e:
            last_exc = e
            attempt += 1
            time.sleep(cfg.restart_backoff_s)
            client.restart()  # Restart the cluster for spot market interruptions

    raise RuntimeError(f"DDP gang failed after {cfg.max_restarts} restarts") from last_exc





def submit_train_to_cluster(client, train_func, **train_kwargs):
    # Get list of workers
    workers = list(client.scheduler_info()["workers"].keys())
    futures = []

    for rank, worker in enumerate(workers):
        # Pass rank and world_size to your train function
        kwargs = train_kwargs.copy()
        kwargs["rank"] = rank
        kwargs["world_size"] = len(workers)
        # Submit one training job per worker
        future = client.submit(
            train_func,
            workers=[worker],
            **kwargs
        )
        futures.append(future)

    # Optionally, gather results
    results = client.gather(futures)
    return results
