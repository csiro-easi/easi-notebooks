# easi_tools.dask_helpers

**Purpose:** Utilities for building, verifying and writing regional EO datasets from STAC into Zarr or Icechunk-backed stores on S3. Designed for use with ODC/STAC pipelines and downstream training pipelines that expect per-sample `features` and `labels` arrays.

---

## Highlights 

- `make_region_geobox`, `load_region_labels`, `load_region_season`: load labels and seasonal composites for a geographic `bbox`.
- `RegionSampler` / `GridRegionSampler`: produce training patches from region-level arrays.
- `STACZarrBuilder`: build per-region Zarrs and combine them into a single S3 Zarr dataset.
- `STACIceChunkBuilder`: incrementally build Icechunk-backed datasets on S3 (with resume support).
- `verify_s3_zarr_dataset` / `verify_icechunk_dataset`: quick structural checks and visual inspections.

---

## Quick start

1. Create a config and spec using `load_config()` and `spec_from_config()`. See `JSON_Specs.md`
2. Open a STAC catalog with `make_catalog(cfg)`.
3. Build a dataset:

```python
from easi_tools.dask_helpers import STACZarrBuilder, STACIceChunkBuilder

builder = STACZarrBuilder(catalog, bucket="my-bucket", base_prefix="proj/", spec=spec)
s3_path = builder.build(final_dataset_name="training_dataset_v1.zarr")

# or Icechunk
ic_builder = STACIceChunkBuilder(catalog, bucket="my-bucket", base_prefix="proj/", dataset_name="ds1", spec=spec)
snapshot_id = ic_builder.build(branch="main", resume=True)
```

Use `verify_s3_zarr_dataset(...)` or `verify_icechunk_dataset(...)` to sanity-check and visualize samples.

---

## API summary (key items)

- `load_config(path_or_s3_uri)` → dict
- `spec_from_config(cfg)` → `EODataSpec` dataclass
- `make_catalog(cfg)` → `pystac_client.Client`
- `GridRegionSampler(tile_size)` → `RegionSampler` implementation
- `STACZarrBuilder` and `STACIceChunkBuilder` → dataset builders
- `verify_s3_zarr_dataset(...)` / `verify_icechunk_dataset(...)`

---

## Dependencies & Notes

Typical dependencies:
- `odc.stac`, `xarray`, `pystac-client`, `planetary_computer` (optional), `boto3`, `s3fs`, `zarr`, `icechunk`

This package expects AWS credentials available (env or IAM role) for S3 operations.


---
## Future Work
- Parallel writes to S3 storage
- Further dataset version control and branching support
