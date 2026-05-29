import time
import traceback
from dask.distributed import wait, as_completed
import json
import boto3
from botocore.exceptions import ClientError
from typing import Callable, Sequence
from dask.distributed import Client


def _s3join(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name


def resolve_resume_checkpoint(
    s3_bucket: str,
    output_dir: str,
    checkpoint_file: str | None,
    resume_mode: str = "auto",   # "auto"|"latest"|"best"|"none"
) -> str | None:
    if resume_mode == "none":
        return None

    if checkpoint_file:
        return checkpoint_file

    s3 = boto3.client("s3")

    latest_meta_key = _s3join(output_dir, "model_latest.json")
    latest_key = _s3join(output_dir, "model_latest.pth")
    best_key = _s3join(output_dir, "model_best.pth")

    def exists(key: str) -> bool:
        try:
            s3.head_object(Bucket=s3_bucket, Key=key)
            return True
        except ClientError:
            return False

    if resume_mode in ("best",):
        return best_key if exists(best_key) else None

    # auto/latest:
    # 1) Prefer latest.json pointer
    try:
        obj = s3.get_object(Bucket=s3_bucket, Key=latest_meta_key)
        meta = json.loads(obj["Body"].read().decode("utf-8"))
        k = meta.get("latest")
        if k and exists(k):
            return k
    except ClientError:
        pass

    # 2) Fallback to model_latest.pth
    if exists(latest_key):
        return latest_key

    # 3) (optional) in auto mode, fallback to best if latest missing
    if resume_mode == "auto" and exists(best_key):
        return best_key

    return None

# Untested: Assumes schedular_notify.py is on the schedular already
def monitor_and_gather_futures(client, futs, check_interval=5):
    TOPIC = "worker-health"
    seen = 0

    while True:
        done, not_done = wait(futs, timeout=0.2)
        if not not_done:
            return client.gather(futs)

        # If any future errors, surface it
        for future in not_done:
            if future.status == "error":
                future.result()  # raises the worker exception

        # React only to actual worker removals
        events = client.get_events(TOPIC)  # list of (timestamp, message) [web:33]
        new = events[seen:]
        seen = len(events)
        if any(msg.get("event") == "worker_removed" for ts, msg in new):
            client.cancel(futs)
            raise RuntimeError("Worker dropped mid-run; retrying")

        time.sleep(check_interval)



def run_training_with_retries(
    client,
    cluster,
    ddp_module,
    train_fn,
    num_workers: int,
    max_attempts: int,
    *,
    s3_bucket: str,
    output_dir: str,
    checkpoint_file: str | None,
    **train_kwargs,
):
    last_err = None

    for attempt in range(1, max_attempts + 1):
        print(f"[attempt {attempt}/{max_attempts}] scaling to {num_workers} workers")
        cluster.scale(num_workers)
        client.wait_for_workers(num_workers, timeout=600)

        resume_key = resolve_resume_checkpoint(s3_bucket, output_dir, checkpoint_file)
        print(f"[attempt {attempt}] resume checkpoint: {resume_key}")

        try:
            futs = ddp_module.run(
                client,
                train_fn,
                s3_bucket=s3_bucket,
                output_dir=output_dir,
                checkpoint_file=resume_key,
                **train_kwargs,
            )

            # Monitor futures instead of blind gather
            monitor_and_gather_futures(client, futs, num_workers)

            print(f"[attempt {attempt}] training finished")
            return  # success

        except Exception as e:
            last_err = e
            print(f"[attempt {attempt}] training failed: {type(e).__name__}: {e}")
            traceback.print_exc()

            try:
                client.restart(timeout="5m", wait_for_workers=True)
            except Exception as restart_err:
                print(f"[attempt {attempt}] client.restart failed: {restart_err}")
                raise

            time.sleep(min(30, 2 * attempt))

    raise RuntimeError(f"Exceeded max_attempts={max_attempts}") from last_err