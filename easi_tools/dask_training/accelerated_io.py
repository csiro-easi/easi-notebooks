# accelerated_io.py

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
#import cupy as cp
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

# Just in case
def _to_numpy(a):
    # cupy supports .get() to transfer to host; numpy arrays won't have it
    get = getattr(a, "get", None)
    if callable(get):
        return get()
    return np.asarray(a)



class BatchAdapter:
    """
    Fast, picklable adapter.
    If label_lut is provided, mapping is vectorized: y_np = label_lut[y_np].
    """
    def __init__(self, label_mapper=None, label_lut=None, ignore_index=255):
        self.label_mapper = label_mapper
        self.label_lut = label_lut
        self.ignore_index = ignore_index


    def __call__(self, x_raw: Any, y_raw: Any, meta: dict):
        x_np = np.asarray(_to_numpy(x_raw), dtype=np.float32)
    
        if self.label_lut is not None:
            y_np = _to_numpy(y_raw)
    
            nan_mask = np.isnan(y_np) if np.issubdtype(y_np.dtype, np.floating) else None
    
            y_int = y_np.astype(np.int64, copy=False)
            y_mapped = self.label_lut[y_int]
    
            if nan_mask is not None:
                y_mapped = np.asarray(y_mapped, dtype=np.int64).copy()
                y_mapped[nan_mask] = self.ignore_index
    
            y_np = np.asarray(y_mapped, dtype=np.int64)
    
        else:
            y_np = np.asarray(y_raw)
            if self.label_mapper is not None:
                y_np = self.label_mapper.map(y_np)
            y_np = np.asarray(y_np, dtype=np.int64)
    
        return x_np, y_np




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
        self.i = 0
        self._cache = OrderedDict()
        self._cache_max = 256  # tune; 0 disables


    def __iter__(self):
        self.i = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        if self._cache_max > 0:
            self._cache.clear()
        return self

    def _get_sample(self, idx: int):
        if self._cache_max > 0 and idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
    
        x_raw = self.x_arr[idx]
        y_raw = self.y_arr[idx]
        x_np, y_np = self.adapter(x_raw, y_raw, meta={"index": idx})
    
        if self._cache_max > 0:
            self._cache[idx] = (x_np, y_np)
            if len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
    
        return x_np, y_np


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
    
            x_b = np.stack(x_list, axis=0)
            y_b = np.stack(y_list, axis=0)
            return {"inputs": x_b, "labels": y_b, "meta": {"indices": idxs}}
    
        # Fallback: random indices
        x_list, y_list = [], []
        for idx in idxs:
            x_np, y_np = self._get_sample(int(idx))
            x_list.append(x_np)
            y_list.append(y_np)

    
        x_b = np.stack(x_list, axis=0)
        y_b = np.stack(y_list, axis=0)
        return {"inputs": x_b, "labels": y_b, "meta": {"indices": idxs}}


def make_external_source_fn(
    bucket, repo_prefix, snapshot_id,
    batch_size, indices, shuffle,
    region="ap-southeast-2", branch="main",
    spec=None, adapter=None,
):
    if spec is None:
        spec = GeoBatchSpec(x_key="features", y_key="labels", patch_hw=None)
    if adapter is None:
        adapter = BatchAdapter(label_mapper=None)

    state = {"it": None}

    def get_iter():
        return IcechunkDaliIterator(
            bucket=bucket,
            repo_prefix=repo_prefix,
            snapshot_id=snapshot_id,
            spec=spec,
            adapter=adapter,
            batch_size=batch_size,
            indices=indices,
            shuffle=shuffle,
            region=region,
            branch=branch,
        )

    def source():
        if state["it"] is None:
            state["it"] = iter(get_iter())
        try:
            batch = next(state["it"])
        except StopIteration:
            state["it"] = None 
            raise
        return batch["inputs"], batch["labels"]

    return source


@pipeline_def()
def dali_pipeline(eii):
    images, labels = fn.external_source(
        source=eii,
        num_outputs=2,
        device="gpu",
        batch=True,
        parallel=False,
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
    prefetch_queue_depth=8,
    spec=None,
    adapter=None,
):
    eii_fn = make_external_source_fn(
        bucket=bucket,
        repo_prefix=repo_prefix,
        snapshot_id=snapshot_id,
        batch_size=batch_size,
        indices=base_indices,
        shuffle=shuffle,
        spec=spec,
        adapter=adapter,
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
