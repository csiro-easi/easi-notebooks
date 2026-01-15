from core_helpers import *
import xarray as xr


# --- Per-region write to temporary Zarrs ---
class STACZarrBuilder:
    def __init__(self, catalog, bucket: str, base_prefix: str,
                 spec: EODataSpec, sampler: RegionSampler | None = None):
        self.catalog = catalog
        self.bucket = bucket
        self.base_prefix = base_prefix
        self.spec = spec
        self.sampler = sampler or GridRegionSampler(tile_size=spec.tile_size)
    
    @property
    def s3_base(self) -> str:
        return f"s3://{self.bucket}/{self.base_prefix}"

    @property
    def tmp_prefix(self) -> str:
        return f"{self.s3_base}/tmp"

    def safe_search_items(self, collections, bbox, datetime=None, query=None,
                          max_items=150, retries=3):
        return safe_search_items(
            catalog=self.catalog,
            collections=collections,
            bbox=bbox,
            datetime=datetime,
            query=query,
            max_items=max_items,
            retries=retries,
        )

    # thin wrappers that pass self.spec/self.catalog
    def make_region_geobox(self, bbox):
        return make_region_geobox(self.spec, bbox)

    def load_region_labels(self, bbox):
        return load_region_labels(self.catalog, self.spec, bbox)

    def load_region_season(self, bbox, time_range, geobox):
        return load_region_season(self.catalog, self.spec, bbox, time_range, geobox)

    def region_to_patches(self, region: RegionSpec):
        return region_to_patches(self.catalog, self.spec, region, self.sampler)

    def build(self, final_dataset_name: str) -> str:
        """
        Build per-region Zarrs, combine into one Zarr on S3, and
        return the final s3:// path.
        """
        s3_path_final = f"{self.s3_base}/{final_dataset_name}"
        print(f"Final Storage Path: {s3_path_final}")
        print(f"Temp region prefix: {self.tmp_prefix}")

        region_zarr_paths = []

        for region in self.spec.regions:
            print(f"\n=== Building patches for {region.name} ===")
            try:
                patches = self.region_to_patches(region)
            except APIError as e:
                print(f"  Skipping {region.name} due to STAC API error: {e}")
                continue

            if not patches:
                print(f"  No patches for {region.name}, skipping.")
                continue

            ds_region = xr.concat(patches, dim="sample")
            ds_region = ds_region.chunk({
                "sample": 8,
                "time": -1,
                "band": -1,
                "y": self.spec.tile_size,
                "x": self.spec.tile_size,
            })
            ds_region.attrs["n_samples"] = ds_region.sizes["sample"]
            ds_region.attrs["n_time_steps"] = len(self.spec.seasonal_windows)
            ds_region.attrs["bands"] = self.spec.band_names

            region_path = f"{self.tmp_prefix}/{region.name}.zarr"
            print(f"  Writing region dataset to: {region_path}")
            ds_region.to_zarr(region_path, mode="w", consolidated=True)

            region_zarr_paths.append(region_path)

        print("\nTemp region stores:")
        for p in region_zarr_paths:
            print(" ", p)

        print("Combining region Zarrs into final dataset...")

        parts = [xr.open_zarr(p, consolidated=True) for p in region_zarr_paths]
        ds_final = xr.concat(parts, dim="sample")

        ds_final = ds_final.chunk({
            "sample": 32,
            "time": -1,
            "band": -1,
            "y": self.spec.tile_size,
            "x": self.spec.tile_size,
        })

        ds_final.attrs["n_samples"] = ds_final.sizes["sample"]
        ds_final.attrs["n_time_steps"] = len(self.spec.seasonal_windows)
        ds_final.attrs["bands"] = self.spec.band_names

        print(ds_final)
        print(f"Writing combined dataset to: {s3_path_final}")
        ds_final.to_zarr(s3_path_final, mode="w", consolidated=True)
        print("[Done]")

        return s3_path_final


def verify_s3_zarr_dataset(
    *,
    bucket: str,
    userid: str,
    project_name: str,
    dataset_name: str = "training_dataset_v3.zarr",
    s3_zarr_path: str | None = None,
    samples_to_plot: list[int] | None = None,
    expected_time_steps: int = 3,
    season_names: list[str] | None = None,
    max_samples: int = 2,
    make_plots: bool = True,
):
    """
    Verify a standard Zarr dataset stored in S3 (via s3fs) and optionally plot a few samples.

    Parameters
    ----------
    s3_zarr_path:
        If provided, overrides bucket/userid/project_name/dataset_name and is used directly.
        Example: "s3://my-bucket/path/to/dataset.zarr"

    Returns
    -------
    ds : xarray.Dataset
        Opened dataset (lazy until compute()).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import s3fs

    if season_names is None:
        season_names = ["Jan-Feb", "Jun-Jul", "Sep-Oct"]

    if samples_to_plot is None:
        samples_to_plot = list(range(max_samples))

    if s3_zarr_path is None:
        s3_zarr_path = f"s3://{bucket}/{userid}/{project_name}/{dataset_name}"

    print(f"Target Storage Path: {s3_zarr_path}")

    # --- 1) Open Zarr from S3 ---
    try:
        fs = s3fs.S3FileSystem()
        store = s3fs.S3Map(root=s3_zarr_path, s3=fs, check=False)
        ds = xr.open_dataset(store, engine="zarr", chunks={})  # lazy dask-backed [web:59]
    except Exception as e:
        print(f"⚠️ Error loading Zarr: {e}")
        raise

    # --- 2) Structural checks ---
    if "features" not in ds or "labels" not in ds:
        raise KeyError("Expected variables 'features' and 'labels' in dataset.")

    features = ds["features"]
    labels = ds["labels"]

    num_samples = ds.sizes.get("sample")
    n_time_steps = ds.sizes.get("time")

    if num_samples is None or n_time_steps != expected_time_steps:
        raise KeyError(
            "Loaded Zarr file is structurally incorrect. "
            f"Expected dimension 'sample' and 'time' size {expected_time_steps}, "
            f"got sample={num_samples}, time={n_time_steps}."
        )

    print("✅ Dataset loaded. Dimensions:")
    print(f"  Samples (N): {num_samples}")
    print(f"  Time Steps (T): {n_time_steps}")
    print(f"  Spatial (Y, X): {features.sizes.get('y')}, {features.sizes.get('x')}")

    band_names = list(ds.attrs.get("bands", []))
    print(f"  Bands: {band_names}")

    def _get_rgb(data_array, bands_list):
        """
        Extract bands for RGB from a DataArray using coordinate selection:
        data_array.sel(band='red'), etc.
        Output: (T, Y, X, Color)
        """
        stack = []
        for b in bands_list:
            # This implementation assumes the dataset has a 'band' coordinate with names.
            band_data = data_array.sel(band=b)  # (time, y, x)
            stack.append(band_data)

        img = xr.concat(stack, dim="color").values.astype("float32")  # (Color, T, Y, X)

        img = img / 10000.0
        img = np.nan_to_num(img, nan=0.0)

        img = np.transpose(img, (1, 2, 3, 0))  # (T, Y, X, Color)

        if img.max() == 0:
            return None

        for t in range(img.shape[0]):
            time_slice = img[t, :, :, :]
            valid = time_slice[time_slice > 0]
            if valid.size > 0:
                p2, p98 = np.percentile(valid, (2, 98))
                if p98 > p2:
                    img[t, :, :, :] = (time_slice - p2) / (p98 - p2)

        return np.clip(img, 0, 1)

    # --- 3) Optional plotting ---
    if not make_plots:
        return ds

    plot_samples = [s for s in samples_to_plot if 0 <= s < num_samples]
    if len(plot_samples) == 0:
        print("⚠️ No valid samples requested for plotting; returning dataset.")
        return ds

    cols = 3
    rows = len(plot_samples) * n_time_steps
    plt.figure(figsize=(15, 4 * rows))

    plot_idx = 1
    for s in plot_samples:
        sample_features = features.isel(sample=s).compute()
        sample_label = labels.isel(sample=s).compute()
        label_np = sample_label.values

        unique_vals = np.unique(label_np)
        print(f"DEBUG: Sample {s} unique label values: {unique_vals}")

        if sample_features.values.max() == 0:
            print(f"⚠️ Sample {s} appears empty (all zeros). Skipping.")
            continue

        rgb_stack = _get_rgb(sample_features, ["red", "green", "blue"])
        fc_stack = _get_rgb(sample_features, ["swir1", "nir", "red"])

        if rgb_stack is None or fc_stack is None:
            print(f"⚠️ Skipping Sample {s}: RGB/false-color stack failed.")
            continue

        for t in range(n_time_steps):
            time_name = season_names[t] if t < len(season_names) else f"Time {t}"

            ax1 = plt.subplot(rows, cols, plot_idx)
            ax1.imshow(rgb_stack[t, :, :, :])
            ax1.set_title(f"S{s}-{time_name}: True Color")
            ax1.axis("off")
            plot_idx += 1

            ax2 = plt.subplot(rows, cols, plot_idx)
            ax2.imshow(fc_stack[t, :, :, :])
            ax2.set_title(f"S{s}-{time_name}: False Color")
            ax2.axis("off")
            plot_idx += 1

            ax3 = plt.subplot(rows, cols, plot_idx)
            masked_lbl = np.ma.masked_where(label_np == 255, label_np)
            im = ax3.imshow(masked_lbl, cmap="tab20", interpolation="nearest")

            if s == plot_samples[0] and t == 0:
                plt.colorbar(im, fraction=0.046, pad=0.04)

            ax3.set_title(f"S{s}: Static Labels")
            ax3.axis("off")
            plot_idx += 1

    plt.tight_layout()
    plt.show()
    return ds
