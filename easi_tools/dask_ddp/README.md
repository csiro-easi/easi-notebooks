# easi_tools/dask_ddp

Short, experimental tooling to run PyTorch DDP on Dask workers and supervise/resume training when transient failures occur (spot-worker interruptions, scheduler restarts).

This module contains several attempts at making DDP work reliably on Dask: an initial simple dispatcher, an older wrapper-style attempt that proved unreliable, and a newer supervise/retry-based approach that is generally usable but still has edge cases. The code is experimental and some failure modes are untested — this README explains how to use what exists, what to expect, and recommended next steps for someone taking over.

---

## Contents

- `dask_ddp.py` — simple initial dispatcher-style implementation exposing `run(...)` and `dispatch_with_ddp(...)` to start a DDP job across Dask workers.
- `dask_ddp_wrapper.py` — older wrapper-style attempt (includes `GangConfig`, worker selection, and `run_elasticish_ddp_on_dask(...)`) that proved unreliable in practice and is not recommended for new runs.
- `supervise.py` — newer supervise-and-retry approach which is the recommended starting point (`run_training_with_retries(...)`); it handles checkpoint resolution and retry logic but does not fully handle mid-epoch worker dropouts.
- `scheduler_notify.py` — Scheduler plugin to emit worker removal events (used by the supervisor to detect mid-run worker drops).

---

## Current status & caveats

- The `supervise.py`-based workflow is the most recent and is the recommended starting point: it scales the cluster, resolves a resume checkpoint on S3, and retries training runs. However, **it does not reliably handle worker dropouts mid-training** — the job needs to be manually cancelled and must be restarted (resume from checkpoint if available).
- `dask_ddp_wrapper.py` is an earlier, wrapper-style attempt that proved unreliable in practice and is not recommended for new runs.
- `monitor_and_gather_futures` and `WorkerDropEvents` (scheduler plugin) are partially tested; ensure `scheduler_notify.dask_setup` is installed on your scheduler for drop-detection to work.
- If a worker dies mid-epoch, PyTorch DDP will error because the world size changed; Due to `Pub/Sub` no longer being avaliable, the driver will not know anything has failed and will run infefinately.
- `scheduler_notify.dask_setup` attempts to circumvent these issues but is untested

---

## Quick usage examples

### 1) Supervise a training run with retries and checkpoint resume (recommended)
```python
from easi_tools.dask_ddp.supervise import run_training_with_retries

run_training_with_retries(
    client=client,
    cluster=cluster,
    ddp_module=dask_ddp,            # or your wrapper exposing .run
    train_fn=train,
    num_workers=NUM_WORKERS,
    max_attempts=5,
    s3_bucket=bucket,
    output_dir=output_dir,
    checkpoint_file=None,
    # extra train kwargs...
)
```

### 2) Using the wrapper (older attempt — less reliable)
```python
from easi_tools.dask_ddp.dask_ddp_wrapper import run_elasticish_ddp_on_dask, GangConfig

cfg = GangConfig(required_resources={"GPU": 1}, max_restarts=10, restart_backoff_s=5.0)
result = run_elasticish_ddp_on_dask(
    client=client,
    train_fn=train,
    train_kwargs=train_kwargs,
    cfg=cfg,
    resolve_checkpoint=lambda: resolve_resume_checkpoint(...),  # optional
)
```
There is also an issue with larger clusters not fully utilising the number of workers avaliable so these also attempt to solve tat issue by:
```python
# Make sure the scheduler's view is used
sched_identity = client.run_on_scheduler(lambda dask_scheduler: dask_scheduler.identity())
_real_scheduler_info = client.scheduler_info
client.scheduler_info = lambda: sched_identity
```


Notes:
- Install the `scheduler_notify` plugin on the scheduler (`client.run_on_scheduler(dask_ddp.scheduler_notify.dask_setup)`) so `monitor_and_gather_futures` can detect worker drops.
- `resolve_resume_checkpoint` in `supervise.py` inspects S3 for `model_latest.pth`, `model_best.pth`, or a `model_latest.json` pointer.

---

## Why PyTorch Elastic is a better long-term approach

The current pattern restarts the job when worker membership changes. PyTorch Elastic (aka `torch.distributed.elastic` / `torchrun --rdzv`) provides a better option:

- Elastic allows dynamic membership: workers can join / leave and the training can continue (depending on algorithm and rendezvous settings) without a full restart.
- Combining Elastic with frequent checkpointing reduces wasted work and can avoid restarting the entire job when a small number of workers drop.
- Practical approach: use a rendezvous service (etcd, c10d rdzv endpoint, or a Dask-scheduler-backed rendezvous) and refactor the training entrypoint to use elastic launches.

Note: adopting Elastic requires careful integration and testing — it changes how process groups and rank assignment are handled.

---

## Recommended next steps for the person taking over

1. Add unit and integration tests that simulate worker drop mid-run and validate the restart/resume flow. (Start with small, local tests.)
2. Ensure `scheduler_notify` is deployed on the scheduler at cluster start in your environment; add helper or doc to do this automatically for your deployment.
3. Harden `monitor_and_gather_futures` by combining future exception checks with scheduler events and clear cancellation/retry logic.
4. Promote frequent, small checkpoints in training so resuming is fast and less error-prone.
5. Prototype PyTorch Elastic or similar for the project and compare failure handling vs the current restart/resume approach.

---

## Contact & further notes
If you want, I can add a minimal example notebook (or tests) that demonstrates the wrapper + supervise flow end-to-end and simulate worker drop behavior.

---

