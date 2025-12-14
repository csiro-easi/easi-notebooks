# EASI Notebooks - What's new <img align="right" src="resources/csiro_easi_logo.png">

### 2025-11-20

- Add NovaSAR-1 RTC Gamma0 notebook, `notebooks/data_products/novasar-1-gamma0.ipynb`

### 2024-12-04

- Add Sentinel-1 RTC Gamma0 notebook, `notebooks/data_products/sentinel-1-gamma0.ipynb`
- Consistency updates to `notebooks/01-welcome-to-easi.ipynb` and the `notebooks/data_products/sentinel-2-*l2a.ipynb` notebooks
- Consolidate and update `easi-tools` functions, including for use with python3.12

### 2024-06-05
- Add Sentinel-2 Collection 1 L2a notebook, `notebooks/data_products/sentinel-2-c1-l2a.ipynb`

### 2024-05-22
- Rename `tutorials` directory to `notebooks`
- Move `data-products` to `notebooks` directory
- Simplify names of the `Dask` notebooks
- Add `adding-python-libraries` notebook
- Simplify `html/readme.md`, and update for the above changes

### 2024-05-09
- Update `tutorial/dask/05_-_Go_Big_or_Go_Home_Part_2.ipynb` to use `hvplot` rather than `holoviews` - the syntax is simpler and it avoids the bug that exists in `holoviews` working with `xarray` that has crept into upstream libraries.

### 2023-06-09

Large update to ensure the notebooks can work across all EASI deployments
- Rename EasiNotebooks to EasiDefaults (`easi-tools/deployments.py`)
- Update all notebooks to use EasiDefaults
- Update EasiDefaults to work on all deployments (that are defined)
- Generalised some of the language used in the Dask notebooks (`tutorials/dask/*.ipynb`)
- Updated the rendered HTML versions of the dask notebooks
- Various related updates

### 2023-05-24

Add a tutorial notebook for using scratch and project buckets
- See `tutorial/easi-scratch-bucket.ipynb`

### 2023-03-28

Add a tutorial notebook for data storage
- See `tutorial/02-data-storage.ipynb`

### 2023-02-16

Updated Welcome to EASI notebook
- Added `easi_tools/deployments.py`

### 2022-12-02

Added dask notebooks
- See `tutorial/dask/*.ipynb`

### 2022-12-02

Initial commit to this repository
- Structure
- Pre-commit hook
- README.md, LICENSE, whats_new.md
- Welcome to EASI notebook (copy from https://github.com/csiro-easi/eocsi-hackathon-2022)
