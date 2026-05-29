# easi_tools.dask_training

**Purpose:** Helpers for efficient I/O and training workflows for EO segmentation tasks. Includes DALI/Icechunk-backed iterators, label mappers, training loop helpers, and evaluation/visualisation utilities.

---

## Highlights

- `accelerated_io`: `GeoBatchSpec`, `BatchAdapter`, `IcechunkDaliIterator` and `make_dali_iterator` for high-throughput GPU training.
- `label_mappers`: `WorldCoverLabelMapper`, `DictLUTLabelMapper` for mapping raw label codes to training classes.
- `training_helpers`: `SegmentationTask`, training/evaluation helpers (`train_one_epoch`, `train_one_epoch_log`, `evaluate`), and DDP/FSDP wrappers.
- `acc_testing`: convenience functions for loading/plotting training logs and test harnesses.

---

## Quick start

Example: create a DALI iterator and run one epoch:

```python
from easi_tools.dask_training import GeoBatchSpec, BatchAdapter, make_dali_iterator
from easi_tools.dask_training.label_mappers import WorldCoverLabelMapper

spec = GeoBatchSpec(x_key="features", y_key="labels", patch_hw=(224,224))
mapper = WorldCoverLabelMapper(num_classes=11, ignore_index=255)
adapter = BatchAdapter(label_mapper=mapper)

it = make_dali_iterator(base_indices=np.arange(N), batch_size=8, device_id=0,
                        bucket="my-bucket", repo_prefix="proj/ds-icechunk", snapshot_id=None,
                        shuffle=True, spec=spec, adapter=adapter)

# Typical training loop helpers in `training_helpers` accept this iterator
```

For examples in testing and plotting of results, see `acc_testing.test_model()` and `plot_training_history()`.

---

## API summary (key items)

- `GeoBatchSpec`, `BatchAdapter`, `IcechunkDaliIterator`, `IcechunkExternalSource`
- `make_dali_iterator(...)` → `DALIGenericIterator`
- `WorldCoverLabelMapper`, `DictLUTLabelMapper`
- `SegmentationTask`, `train_one_epoch`, `train_one_epoch_log`, `evaluate`
- `acc_testing` helpers: `load_latest_training_log`, `plot_training_history`, `test_model`

---

## Dependencies & Notes

Typical dependencies:
- `numpy`, `cupy` (optional), `nvidia-dali`, `torch`, `torchmetrics`, `icechunk`, `zarr`, `s3fs`, `boto3`, `matplotlib`, `seaborn`, `pandas`.

GPU-specific features (DALI, CuPy) require appropriate CUDA drivers and the NVIDIA DALI packages installed in your environment.

---
## Future Work
- Improve data pipeline further
- Better customisation for `ddp` and `fsdp`
- Include more task options
- Replacement for `dask-pytorch-ddp` with retry logic (explore `PyTorch Elastic`)
- Include all dependancies + libraries in a resusable image
- 