# EASI Notebooks - AI Agent Instructions

## Project Overview
Educational Jupyter notebooks for learning Earth observation (EO) analysis using the **Open Data Cube (ODC)** on CSIRO's EASI deployments. Notebooks demonstrate accessing satellite data (Landsat, Sentinel-2/1), cloud-based analytics with Dask, and geospatial analysis workflows.

**Core focus**: Training materials and examples adapted for multi-regional EASI deployments with examples for each geographic region/deployment environment.

## Repository Structure

### Key Directories
- **`notebooks/`** - Training and tutorial notebooks organized by topic
  - `01-welcome-to-easi.ipynb`, `02-data-storage.ipynb` - foundational concepts
  - `dask/` - distributed processing examples (01 intro through 06 tuning)
  - `data_products/` - product-specific access patterns (Sentinel-2, Sentinel-1, VIIRS, NOVASAR)
- **`easi_tools/`** - Reusable helper modules for notebooks (no external dependencies beyond ODC stack)
- **`html/`** - Generated HTML versions (auto-created by pre-commit hook)
- **`bin/`** - Repository management scripts (`pre-commit`, `apply_hooks.sh`)
- **`resources/`** - Images, supplementary files

### easi_tools Modules (Import in Notebooks)
```python
from easi_tools import EasiDefaults, initialize_dask, mostcommon_crs, xarray_object_size, heading
```

| Module | Purpose |
|--------|---------|
| `deployments.py` | `EasiDefaults` class: deployment-specific config lookup (region, database, products, CRS) |
| `notebook_utils.py` | `heading()`, `display_table()`, Dask initialization (`initialize_dask()`), dashboard helpers |
| `load_s2l2a.py` | Sentinel-2 L2A offset/scaling corrections for COG data from Element84 STAC |

## Critical Patterns

### Deployment Configuration Pattern
Use `EasiDefaults()` to access deployment-specific variables:

```python
from easi_tools import EasiDefaults

ed = EasiDefaults()  # Auto-detect from DB_DATABASE env var, OR
ed = EasiDefaults(deployment='asia')  # Explicit deployment name

# Access deployment-specific values:
ed.domain  # e.g., 'asia.easi-eo.solutions'
ed.location  # e.g., 'Lake Tempe, Indonesia'
ed.latitude, ed.longitude  # Recommended AOI bounds
ed.time  # (start, end) tuple for typical examples
ed.productmap['sentinel-2']  # Deployment-specific product name mapping
```

**Supported deployments** (from `deployment_map`):
- `adias`, `asia`, `cal`, `chile`, `csiro`, `dcceew`, `sub-apse2`
- Each has: `domain`, `db_database`, `productmap`, `target` (CRS/resolution), `aliases`, `qa_mask`

### Datacube Query Pattern with Deployment Config
```python
from datacube import Datacube
from easi_tools import EasiDefaults, mostcommon_crs

dc = Datacube()
ed = EasiDefaults()

# Query parameters use deployment-specific product names
query = {
    'product': ed.productmap['sentinel-2'],  # e.g., 's2_l2a' or 'sentinel_2_c1_l2a'
    'x': ed.longitude,
    'y': ed.latitude,
    'time': ed.time,
    'output_crs': 'EPSG:3577'  # or use mostcommon_crs(dc, query)
}
ds = dc.load_data('ga', **query)
```

### Band Aliases and QA Mask Convention
Different EASI deployments use different band names (GA ARD vs Element84 STAC):

```python
# Access deployment-specific aliases
aliases = ed.deployment.get('aliases', {}).get('landsat', {})
# {'qa_band': 'oa_fmask', 'nir': 'nbart_nir', 'red': 'nbart_red', ...}

# Apply QA masks using deployment config
qa_mask = ed.deployment.get('qa_mask', {}).get('landsat', {})
# {'fmask': 'valid'}  OR  {'nodata': False, 'cloud': 'not_high_confidence', ...}
```

### Sentinel-2 L2A Offset Handling
Element84 STAC collections (sentinel-2-l2a, sentinel-2-c1-l2a) require care due to ESA reprocessing:

```python
from easi_tools.load_s2l2a import highest_sequence_number

# When multiple versions of same scene exist, select highest processing version:
matches = dc.find_datasets(product='s2_l2a', **query)
filtered = highest_sequence_number(list(matches))  # Returns dict keyed by scene ID

# Handle offset-corrected vs non-corrected data:
# Pre-2022-01-25 (PB < 04.00): refl = dn / 10000
# Post-2022-01-25 (PB >= 04.00): refl = (dn - 1000) / 10000
```

### Dask Initialization for Large Analysis
Use `initialize_dask()` for memory-intensive workflows:

```python
from easi_tools import initialize_dask

# Local cluster (single machine)
cluster, client = initialize_dask(use_gateway=False, workers=(2, 4))

# OR Dask Gateway (on EASI hub - multi-machine)
cluster, client = initialize_dask(use_gateway=True, workers=(1, 10), wait=True)

# Use cluster for chunked operations
ds_chunked = ds.chunk({'time': 10})  # Dask will parallelize
result = ds_chunked.resample(time='M').mean().compute()
```

## Notebook Development Workflow

### Adding/Updating Notebooks
1. **Create/edit** `.ipynb` in `notebooks/` (or subdirectory)
2. **Apply pre-commit hook** to register the hook:
   ```bash
   sh bin/apply_hooks.sh  # One-time setup in local clone
   ```
3. **Run the notebook** end-to-end to generate outputs
4. **Commit** - pre-commit hook automatically:
   - Strips cell outputs (reduces repo size)
   - Generates HTML version in `html/`
5. **Update documentation**:
   - Add entry to `html/readme.md` for discoverability
   - Update `whats_new.md` with summary
6. **Push** and create PR

### Notebook Best Practices
- **Import easi_tools early**: `from easi_tools import EasiDefaults`
- **Parameterize for deployments**: Use `EasiDefaults()` not hardcoded values
- **Document band aliases**: Show which deployment uses which band names
- **Handle sparse data**: Some deployments may lack certain products in certain regions
- **Use Dask** for time-series or multi-region analysis (see `dask/` examples)

## Code Quality Standards

### Notebook Practices
- Minimal code cells - use markdown explanations between cells
- Cell outputs are cleared before commit (pre-commit hook)
- HTML versions sync to `html/` directory for browsing
- Compatible with Jupyter Lab (not limited to classic notebook)

### Python Modules (in easi_tools)
- No external dependencies beyond ODC/datacube stack
- Type hints where reasonable
- Docstrings for public functions
- Logging via standard `logging` module

## Troubleshooting

**Issue**: "Deployment could not be found automatically"
- **Cause**: `DB_DATABASE` environment variable not set
- **Fix**: Either set `DB_DATABASE` env var OR pass deployment name: `EasiDefaults(deployment='asia')`

**Issue**: Different product names across deployments
- **Cause**: GA ARD products (e.g., `ga_s2am_ard_3`) vs Element84 STAC (e.g., `sentinel_2_c1_l2a`)
- **Fix**: Always use `ed.productmap['sentinel-2']` - it resolves to deployment-specific name

**Issue**: Sentinel-2 time-series has gaps or inconsistent values
- **Cause**: Multiple processing baselines (PB) with different offset handling
- **Fix**: Use `highest_sequence_number()` filter to standardize, or check metadata for `PROCESSING_BASELINE`

**Issue**: Dask cluster hangs or out-of-memory
- **Cause**: Unbounded task graph or excessive dataset chunking
- **Fix**: Limit parallelism with `workers` tuple, rechunk data (`ds.chunk({'time': 30})`), use Dask Gateway for scale

## Quick Start

**Run a simple notebook example**:
```python
# In notebook cell
from easi_tools import EasiDefaults
from datacube import Datacube

ed = EasiDefaults()  # Detects deployment from environment
dc = Datacube()

# Query with deployment defaults
data = dc.load(
    product=ed.productmap['sentinel-2'],
    x=ed.longitude, y=ed.latitude,
    time=ed.time
)
data.nbands.plot()  # Quick viz
```

**View all notebooks**: Open `html/readme.md` or browse `notebooks/` directory

**Test local changes**: Run notebook end-to-end, commit (hook generates HTML), verify in `html/`
