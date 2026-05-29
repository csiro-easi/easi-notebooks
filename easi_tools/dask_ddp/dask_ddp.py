import os
from typing import List, Callable, Any, Dict
from datetime import timedelta

from dask.distributed import Client
import torch
import torch.distributed as dist


def _scheduler_info_on_scheduler(dask_scheduler):
    return dask_scheduler.identity()


def _get_worker_info(client: Client) -> List[Dict]:
    sched_identity = client.run_on_scheduler(_scheduler_info_on_scheduler)
    workers = sched_identity["workers"]
    worker_keys = sorted(workers.keys())

    workers_by_host: Dict[str, List[str]] = {}
    for key in worker_keys:
        host = workers[key]["host"]
        workers_by_host.setdefault(host, []).append(key)

    all_workers = []
    global_rank = 0
    for host in sorted(workers_by_host.keys()):
        local_rank = 0
        for worker in workers_by_host[host]:
            all_workers.append(
                dict(
                    worker=worker,
                    local_rank=local_rank,
                    global_rank=global_rank,
                    host=host,
                )
            )
            local_rank += 1
            global_rank += 1
    return all_workers


def run(
    client: Client,
    pytorch_function: Callable,
    *args,
    backend: str = "nccl",
    master_port: int = 23456,
    timeout_s: int = 120,
    **kwargs
):
    all_workers = _get_worker_info(client)
    world_size = len(all_workers)
    master_addr = all_workers[0]["host"]

    futures = []
    for w in all_workers:
        fut = client.submit(
            dispatch_with_ddp,
            pytorch_function=pytorch_function,
            master_addr=master_addr,
            master_port=master_port,
            rank=w["global_rank"],
            world_size=world_size,
            local_rank=w["local_rank"],
            backend=backend,
            timeout_s=timeout_s,
            workers=[w["worker"]],
            *args,
            **kwargs,
        )
        futures.append(fut)

    return futures


def dispatch_with_ddp(
    pytorch_function: Callable,
    master_addr: Any,
    master_port: Any,
    rank: int,
    world_size: int,
    local_rank: int,
    *args,
    backend: str = "nccl",
    timeout_s: int = 120,
    **kwargs
) -> Any:
    os.environ["MASTER_ADDR"] = str(master_addr)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # train() can rely on LOCAL_RANK
    os.environ["LOCAL_RANK"] = str(local_rank)

    # safety: pick the correct GPU before NCCL init
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    try:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=timeout_s),
        )
        return pytorch_function(*args, **kwargs)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
