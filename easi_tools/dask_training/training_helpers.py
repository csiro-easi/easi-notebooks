# training_helpers.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP




class Task:
    def compute_loss(self, outputs, batch):
        raise NotImplementedError

    def compute_metrics(self, outputs, batch):
        # return a dict of scalar floats
        return {}

class SegmentationTask(Task):
    def __init__(self, num_classes, class_weights, ignore_index=255):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
        )
        self.ignore_index = ignore_index

    def compute_loss(self, outputs, batch):
        labels = batch["labels"]
        return self.loss_fn(outputs, labels)

    def compute_metrics(self, outputs, batch):
        labels = batch["labels"]
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            valid = labels != self.ignore_index
            correct = (preds[valid] == labels[valid]).sum().item()
            total = valid.sum().item()
            acc = correct / max(total, 1)
        return {"accuracy": acc}


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v  # keep meta as is
    return out


def wrap_model_with_ddp(model, lr, device_id: int, param_selector=None):
    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id], output_device=device_id)
    if param_selector is None:
        params = filter(lambda p: p.requires_grad, ddp_model.parameters())
    else:
        params = param_selector(ddp_model)
    optimizer = optim.Adam(params, lr=lr)
    return ddp_model, optimizer


def wrap_model_with_fsdp(model, lr, param_selector=None):
    fsdp_model = FSDP(model, use_orig_params=True)
    if param_selector is None:
        params = filter(lambda p: p.requires_grad, fsdp_model.parameters())
    else:
        params = param_selector(fsdp_model)
    optimizer = optim.Adam(params, lr=lr)
    return fsdp_model, optimizer




def train_one_epoch_log(
    model,
    optimizer,
    scheduler,
    task,
    dataloader,
    device,
    log_every: int = 50,
    sync_cuda: bool = True,
):
    import time
    import torch

    model.train()

    total_loss = 0.0
    total_metric = 0.0
    num_batches = 0

    # Averages over measured steps only
    t_fetch_total = 0.0          # CPU wall time spent in next(data_iter)
    t_to_total = 0.0             # CPU wall time spent in to_device (incl any sync)
    t_step_cpu_total = 0.0       # CPU wall time for step region
    t_step_gpu_ms_total = 0.0    # GPU elapsed time for step region (CUDA events)
    n_meas = 0

    data_iter = iter(dataloader)

    while True:
        # Ensure previous iteration's GPU work doesn't leak into "fetch" timing.
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        # ---- fetch timing (CPU) ----
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        t1 = time.perf_counter()

        if isinstance(batch, (list, tuple)):
            batch = batch[0]

        # ---- to_device timing (CPU) ----
        t2 = time.perf_counter()
        batch = to_device(batch, device)
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        # ---- step timing (CPU + GPU events) ----
        optimizer.zero_grad(set_to_none=True)

        use_events = torch.cuda.is_available()
        if use_events:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        if use_events:
            start_ev.record()

        outputs = model(batch["inputs"])
        loss = task.compute_loss(outputs, batch)
        loss.backward()
        optimizer.step()

        if use_events:
            end_ev.record()

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t5 = time.perf_counter()

        if use_events:
            step_gpu_ms = float(start_ev.elapsed_time(end_ev))
        else:
            step_gpu_ms = 0.0

        metrics = task.compute_metrics(outputs, batch)

        total_loss += float(loss.item())
        total_metric += float(metrics.get("accuracy", 0.0))
        num_batches += 1

        if (num_batches - 1) % log_every == 0:
            t_fetch_total += (t1 - t0)
            t_to_total += (t3 - t2)
            t_step_cpu_total += (t5 - t4)
            t_step_gpu_ms_total += step_gpu_ms
            n_meas += 1

    if num_batches > 0:
        total_loss /= num_batches
        total_metric /= num_batches

    if scheduler is not None:
        scheduler.step()

    denom = max(n_meas, 1)
    return {
        "loss": float(total_loss),
        "accuracy": float(total_metric),
        "t_fetch": float(t_fetch_total / denom),
        "t_to_device": float(t_to_total / denom),
        "t_step_cpu": float(t_step_cpu_total / denom),
        "t_step_gpu_ms": float(t_step_gpu_ms_total / denom),
        "timing_samples": int(n_meas),
        "num_batches": int(num_batches),
    }



def train_one_epoch(model, optimizer, scheduler, task, dataloader, device):
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    num_batches = 0
    print("DEBUG: entering train_one_epoch")
    for batch in dataloader:
        print("DEBUG: got batch from dataloader")
        # Unwrap DALI's list-of-dicts if needed
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        print("DEBUG epoch batches:", num_batches)
        batch = to_device(batch, device)
        inputs = batch["inputs"]

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = task.compute_loss(outputs, batch)
        loss.backward()
        optimizer.step()

        metrics = task.compute_metrics(outputs, batch)
        total_loss += loss.item()
        total_metric += metrics.get("accuracy", 0.0)
        num_batches += 1

    if num_batches > 0:
        total_loss /= num_batches
        total_metric /= num_batches

    if scheduler is not None:
        scheduler.step()

    return {"loss": float(total_loss), "accuracy": float(total_metric)}


def evaluate(model, task, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = to_device(batch, device)
            inputs = batch["inputs"]
            outputs = model(inputs)

            loss = task.compute_loss(outputs, batch)
            metrics = task.compute_metrics(outputs, batch)

            total_loss += loss.item()
            total_metric += metrics.get("accuracy", 0.0)
            num_batches += 1

    if num_batches > 0:
        total_loss /= num_batches
        total_metric /= num_batches

    return {"loss": float(total_loss), "accuracy": float(total_metric)}
