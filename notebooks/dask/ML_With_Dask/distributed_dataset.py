# distributed_dataset.py
"""
Dataset and training utilities that run entirely on Dask workers.
Nothing runs on the local notebook except job submission.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import xarray as xr
import fsspec
import numpy as np
from typing import Dict, List, Tuple, Optional
import io


class RemoteZarrDataset(Dataset):
    """
    Dataset that initializes completely on remote workers.
    No S3 calls happen on the client/notebook.
    """
    
    def __init__(
        self,
        zarr_path: str,
        patch_size: int = 224,
        worker_rank: int = 0,
        world_size: int = 1,
        feature_var: str = 'features',
        label_var: str = 'labels',
        num_threads: int = 8,
    ):
        self.zarr_path = zarr_path
        self.patch_size = patch_size
        self.worker_rank = worker_rank
        self.world_size = world_size
        self.feature_var = feature_var
        self.label_var = label_var
        # Might not use but whatevre
        self.num_threads = num_threads
        
        # These are set lazily on first access
        self._ds = None
        self._indices = None
        self._metadata = None
    
    def _ensure_initialized(self):
        """Initialize on first access (runs on worker, not client)."""
        if self._ds is not None:
            return
        
        # Open connection
        mapper = fsspec.get_mapper(self.zarr_path)
        self._ds = xr.open_zarr(mapper, consolidated=True)
        
        # Get metadata
        features = self._ds[self.feature_var]
        chunks = features.chunks
        
        n_time = len(chunks[0])
        n_y = len(chunks[2])
        n_x = len(chunks[3])
        
        self._metadata = {
            'n_time': n_time,
            'n_y': n_y,
            'n_x': n_x,
            'total_patches': n_time * n_y * n_x,
            'dims': dict(features.sizes)
        }
        
        # Compute this worker's indices
        total = self._metadata['total_patches']
        per_worker = total // self.world_size
        start = self.worker_rank * per_worker
        end = start + per_worker if self.worker_rank < self.world_size - 1 else total
        self._indices = list(range(start, end))
    
    def __len__(self) -> int:
        self._ensure_initialized()
        return len(self._indices)
    
    def _global_to_coords(self, global_idx: int) -> Tuple[int, int, int]:
        """Convert global index to (time, y, x) chunk coordinates."""
        m = self._metadata
        patches_per_time = m['n_x'] * m['n_y']
        
        t = global_idx // patches_per_time
        remainder = global_idx % patches_per_time
        y = remainder // m['n_x']
        x = remainder % m['n_x']
        
        return t, y, x
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_initialized()
        
        # Map local index to global index
        global_idx = self._indices[idx]
        t, y_chunk, x_chunk = self._global_to_coords(global_idx)
        
        # Pixel bounds
        y_start = y_chunk * self.patch_size
        x_start = x_chunk * self.patch_size
        y_end = min(y_start + self.patch_size, self._metadata['dims']['y'])
        x_end = min(x_start + self.patch_size, self._metadata['dims']['x'])
        
        # Load
        features = self._ds[self.feature_var].isel(
            time=t, y=slice(y_start, y_end), x=slice(x_start, x_end)
        ).values
        
        labels = self._ds[self.label_var].isel(
            time=t, y=slice(y_start, y_end), x=slice(x_start, x_end)
        ).values
        
        # Pad if needed
        if features.shape[-2:] != (self.patch_size, self.patch_size):
            features = self._pad(features, 3)
            labels = self._pad(labels, 2)
        
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(labels).long()
        )
    
    def _pad(self, arr, ndim):
        h, w = arr.shape[-2], arr.shape[-1]
        pad_h, pad_w = self.patch_size - h, self.patch_size - w
        if ndim == 3:
            return np.pad(arr, ((0,0), (0,pad_h), (0,pad_w)))
        return np.pad(arr, ((0,pad_h), (0,pad_w)))
    
    def get_batch(self, indices: list) -> tuple:
        """
        Fetch multiple items concurrently using threads.
        Call this instead of iterating with DataLoader.
        """
        from concurrent.futures import ThreadPoolExecutor
        import torch
        
        with ThreadPoolExecutor(max_workers=min(len(indices), 8)) as executor:
            results = list(executor.map(self.__getitem__, indices))
        
        features = torch.stack([r[0] for r in results])
        labels = torch.stack([r[1] for r in results])
        
        return features, labels
    
    def iter_batches(self, batch_size: int, shuffle: bool = True):
        """
        Generator that yields batches with concurrent fetching.
        Use this instead of DataLoader inside Dask workers.
        """
        import random
        
        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            yield self.get_batch(batch_indices)