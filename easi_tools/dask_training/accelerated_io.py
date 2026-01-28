# accelerated_io.py

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import cupy as cp
#import torch
#from torch.utils import dlpack

from collections import OrderedDict

import zarr
import icechunk as ic

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator


@dataclass(frozen=True)
class GeoBatchSpec:
    x_key: str = "features"
    y_key: Optional[str] = "labels"      # None for inference/SSL
    # Optional: enforce patching at read time if the stored samples are larger
    patch_hw: Optional[Tuple[int, int]] = (224, 224)


def _is_cuda_array(a) -> bool:
    return hasattr(a, "__cuda_array_interface__")


def _xp(a):
    if cp is not None and _is_cuda_array(a):
        return cp
    return np


class BatchAdapter:
    def __init__(self, label_mapper=None, label_lut=None, ignore_index=255):
        self.label_mapper = label_mapper
        self.ignore_index = ignore_index

        self.label_lut_np = None
        self.label_lut_cp = None
        if label_lut is not None:
            self.label_lut_np = np.asarray(label_lut, dtype=np.int64)
            if cp is not None:
                self.label_lut_cp = cp.asarray(self.label_lut_np)

    def __call__(self, x_raw, y_raw, meta: dict):
        xp = _xp(x_raw)
        x = xp.asarray(x_raw, dtype=xp.float32)

        yp = xp

        if self.label_lut_np is not None:
            y = yp.asarray(y_raw)
            nan_mask = yp.isnan(y) if yp.issubdtype(y.dtype, yp.floating) else None

            y_int = y.astype(yp.int64, copy=False)
            lut = self.label_lut_cp if (yp is cp and self.label_lut_cp is not None) else self.label_lut_np
            y = lut[y_int]

            if nan_mask is not None:
                y = y.astype(yp.int64, copy=False).copy()
                y[nan_mask] = self.ignore_index

            y = y.astype(yp.int64, copy=False)
        else:
            # Note: if label_mapper is NumPy-only, keep this path CPU-only for now.
            y = yp.asarray(y_raw, dtype=yp.int64)
            if self.label_mapper is not None:
                y = self.label_mapper.map(y)

        return x, y




class IcechunkDaliIterator:
    def __init__(
        self,
        bucket: str,
        repo_prefix: str,
        snapshot_id: Optional[str],
        spec: GeoBatchSpec,
        adapter: BatchAdapter,
        batch_size: int,
        indices: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        region: str = "ap-southeast-2",
        branch: str = "main",
    ):
        if spec.y_key is None:
            raise ValueError("spec.y_key must be set for training with num_outputs=2")

        self.adapter = adapter
        self.batch_size = batch_size
        self.shuffle = shuffle

        storage = ic.s3_storage(bucket=bucket, prefix=repo_prefix, region=region, from_env=True)
        repo = ic.Repository.open(storage)
        session = repo.readonly_session(snapshot_id=snapshot_id) if snapshot_id else repo.readonly_session(branch=branch)

        with zarr.config.enable_gpu():
            root = zarr.open_group(store=session.store, mode="r")
            self.x_arr = root[spec.x_key]
            self.y_arr = root[spec.y_key]

        n = self.x_arr.shape[0]
        self.indices = np.arange(n) if indices is None else np.array(indices)
        self.n = len(self.indices)
        self._cache = OrderedDict()
        self.i = 0
        self._cache_bytes = 0
        self._cache_max_bytes = 2 * 1024**3  # tune as needed
        # default: do not cache CuPy samples, probably not needed with dali
        self._cache_gpu = False  


    def _nbytes(self, x, y):
        xb = int(getattr(x, "nbytes", 0))
        yb = int(getattr(y, "nbytes", 0))
        return xb + yb

    def _maybe_cache_put(self, idx, x, y):
        is_gpu = hasattr(x, "__cuda_array_interface__")
        if is_gpu and not self._cache_gpu:
            return

        b = self._nbytes(x, y)
        if b > self._cache_max_bytes:
            return

        while self._cache_bytes + b > self._cache_max_bytes and len(self._cache) > 0:
            _, (old_x, old_y, old_b) = self._cache.popitem(last=False)
            self._cache_bytes -= old_b

        self._cache[idx] = (x, y, b)
        self._cache_bytes += b

    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        if self._cache_max_bytes > 0:
            self._cache.clear()
            self._cache_bytes = 0
        return self

    def _get_sample(self, idx: int):
        if idx in self._cache:
            self._cache.move_to_end(idx)
            x, y, _ = self._cache[idx]
            return x, y

        x_raw = self.x_arr[idx]
        y_raw = self.y_arr[idx]
        x, y = self.adapter(x_raw, y_raw, meta={"index": idx})

        self._maybe_cache_put(idx, x, y)
        return x, y


    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
    
        start = self.i
        end = min(self.i + self.batch_size, self.n)
        self.i = end
        idxs = self.indices[start:end]
    
        # Fast path: contiguous indices -> one slice read
        if len(idxs) > 0:
            idx0 = int(idxs[0])
            contiguous = np.all(idxs == (idx0 + np.arange(len(idxs))))
        else:
            contiguous = False
    
        if contiguous:
            x_raw_b = self.x_arr[idx0:idx0 + len(idxs)]
            y_raw_b = self.y_arr[idx0:idx0 + len(idxs)]
    
            # Apply adapter per-sample (still Python), but Zarr access becomes a single slice
            x_list, y_list = [], []
            for j in range(len(idxs)):
                x_np, y_np = self.adapter(x_raw_b[j], y_raw_b[j], meta={"index": int(idxs[j])})
                x_list.append(x_np)
                y_list.append(y_np)
    
            xp = _xp(x_list[0])
            x_b = xp.stack(x_list, axis=0)
            y_b = xp.stack(y_list, axis=0)
            return (x_b, y_b)
    
        # Fallback: random indices
        x_list, y_list = [], []
        for idx in idxs:
            x_np, y_np = self._get_sample(int(idx))
            x_list.append(x_np)
            y_list.append(y_np)

    
        xp = _xp(x_list[0])
        x_b = xp.stack(x_list, axis=0)
        y_b = xp.stack(y_list, axis=0)
        return (x_b, y_b)


# def make_external_source_fn(
#     bucket, repo_prefix, snapshot_id,
#     batch_size, indices, shuffle,
#     region="ap-southeast-2", branch="main",
#     spec=None, adapter=None,
# ):
#     if spec is None:
#         spec = GeoBatchSpec(x_key="features", y_key="labels", patch_hw=None)
#     if adapter is None:
#         adapter = BatchAdapter(label_mapper=None)

#     state = {"it": None}

#     def get_iter():
#         return IcechunkDaliIterator(
#             bucket=bucket,
#             repo_prefix=repo_prefix,
#             snapshot_id=snapshot_id,
#             spec=spec,
#             adapter=adapter,
#             batch_size=batch_size,
#             indices=indices,
#             shuffle=shuffle,
#             region=region,
#             branch=branch,
#         )

#     def source():
#         if state["it"] is None:
#             state["it"] = iter(get_iter())
#         try:
#             batch = next(state["it"])
#         except StopIteration:
#             state["it"] = None 
#             raise
#         return batch["inputs"], batch["labels"]

#     return source


@pipeline_def()
def dali_pipeline(eii):
    images, labels = fn.external_source(
        source=eii,
        num_outputs=2,
        device="gpu",
        batch=True,
        parallel=False,
        cycle="raise",
    )
    return images, labels


def make_dali_iterator(
    base_indices,
    batch_size,
    device_id,
    bucket,
    repo_prefix,
    snapshot_id,
    shuffle,
    threads=8,
    prefetch_queue_depth=6,
    spec=None,
    adapter=None,
):
    # eii_fn = make_external_source_fn(
    #     bucket=bucket,
    #     repo_prefix=repo_prefix,
    #     snapshot_id=snapshot_id,
    #     batch_size=batch_size,
    #     indices=base_indices,
    #     shuffle=shuffle,
    #     spec=spec,
    #     adapter=adapter,
    # )

    eii_fn = IcechunkDaliIterator(
        bucket=bucket,
        repo_prefix=repo_prefix,
        snapshot_id=snapshot_id,
        spec=spec or GeoBatchSpec(x_key="features", y_key="labels", patch_hw=None),
        adapter=adapter or BatchAdapter(label_mapper=None),
        batch_size=batch_size,
        indices=base_indices,
        shuffle=shuffle,
        region="ap-southeast-2",
        branch="main",
    )

    pipe = dali_pipeline(
        eii=eii_fn,
        batch_size=batch_size,
        num_threads=threads,
        device_id=device_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    pipe.build()

    return DALIGenericIterator(
        [pipe],
        output_map=["inputs", "labels"],
        auto_reset=True,
    )



def get_num_samples(bucket, repo_prefix, snapshot_id=None, branch="main", region="ap-southeast-2",
                    x_key="features"):
    storage = ic.s3_storage(bucket=bucket, prefix=repo_prefix, region=region, from_env=True)
    repo = ic.Repository.open(storage)
    session = repo.readonly_session(snapshot_id=snapshot_id) if snapshot_id else repo.readonly_session(branch=branch)
    with zarr.config.enable_gpu():
        root = zarr.open_group(store=session.store, mode="r")
        return root[x_key].shape[0]
