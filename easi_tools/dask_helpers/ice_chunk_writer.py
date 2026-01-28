from .core_helpers import *
import icechunk as ic
from icechunk.xarray import to_icechunk
import boto3
import s3fs
import zarr
import xarray as xr
from pystac_client.exceptions import APIError


class STACIceChunkBuilder:
    def __init__(
        self,
        catalog,
        bucket: str,
        base_prefix: str,
        dataset_name: str,
        spec: EODataSpec,
        sampler: RegionSampler | None = None,
        region: str | None = "ap-southeast-2",
    ):
        self.catalog = catalog
        self.spec = spec
        self.bucket = bucket
        self.base_prefix = base_prefix
        self.dataset_name = dataset_name
        self.sampler = sampler or GridRegionSampler(tile_size=spec.tile_size)

        # Each dataset gets its own clean S3 prefix for the Icechunk repo
        self.repo_prefix = f"{base_prefix}/{dataset_name}-icechunk"
        session = boto3.session.Session()
        # Configure Icechunk S3 storage and repo once in __init__
        self.storage = ic.s3_storage(
            bucket=self.bucket,
            prefix=self.repo_prefix,
            region=region,
            from_env=True,   # uses your usual AWS creds / role
        )
        # open_or_create is the recommended pattern for scripts rerun many times
        self.repo = ic.Repository.open_or_create(self.storage)  # [web:1][web:64]

    @property
    def s3_base(self) -> str:
        return f"s3://{self.bucket}/{self.base_prefix}"

    def cleanup_repo(self):
        fs = s3fs.S3FileSystem(anon=False)
        path = f"s3://{self.bucket}/{self.repo_prefix}"
        print("Deleting Icechunk repo prefix:", path)
        fs.rm(path, recursive=True)

    # thin wrapper around your existing function-based helper
    def region_to_patches(self, region: RegionSpec):
        return region_to_patches(self.catalog, self.spec, region, self.sampler)

    def build(self, branch: str = "main", resume: bool = True, strict_resume: bool = False) -> str:
            """
            Incrementally build a single Icechunk-backed dataset on S3.
    
            If resume=True:
              - Detect an existing dataset on this branch.
              - Skip regions already written (tracked via 'regions_done').
              - Append only missing regions along 'sample'.
    
            If resume=False:
              - Overwrite from scratch (first write uses mode='w').
    
            Returns the final snapshot_id string for this build.
            """
            total_samples = 0
            last_snapshot_id: str | None = None
            regions_done: set[str] = set()
    
            resume_effective = resume
            first_region = True
    
            if resume:
                try:
                    _ = self.repo.lookup_branch(branch)  # probe branch tip
                    ro_session = self.repo.readonly_session(branch)
                    ds_existing = xr.open_zarr(ro_session.store, consolidated=False)
    
                    total_samples = int(ds_existing.sizes.get("sample", 0) or 0)
                    regions_done = set(ds_existing.attrs.get("regions_done", []))
                    print(
                        f"[resume] Found existing dataset on '{branch}': "
                        f"{total_samples} samples, {len(regions_done)} regions_done."
                    )
                    first_region = False
                except Exception as e:
                    if strict_resume:
                        raise RuntimeError(
                            f"strict_resume=True and no readable dataset found on branch '{branch}'."
                        ) from e
    
                    print(
                        f"[resume] No readable dataset found on '{branch}' (will start fresh). "
                        f"Reason: {type(e).__name__}: {e}"
                    )
                    resume_effective = False
                    first_region = True
            else:
                resume_effective = False
                first_region = True
                print(
                    f"resume=False: will overwrite any existing dataset on branch '{branch}' "
                    f"(first write uses mode='w')."
                )
    
            for region in self.spec.regions:
                if resume_effective and region.name in regions_done:
                    print(f"  Region {region.name} already present, skipping.")
                    continue
    
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
    
                # ---- TRAINING-OPTIMIZED VIEW: drop 'band' variable but keep names ----
                # Will help later
                if "band" in ds_region.coords:
                    band_names = ds_region.coords["band"].values.tolist()  # ['blue','green',...]
                    ds_region = ds_region.drop_vars("band")                # remove object coord var
                    ds_region.attrs["bands"] = band_names                  # keep mapping in attrs
                else:
                    # Fallback: still set bands from spec if coord missing
                    ds_region.attrs["bands"] = self.spec.band_names
    
                ds_region = ds_region.chunk({
                    "sample": 8,
                    "time": -1,
                    "band": -1,
                    "y": self.spec.tile_size,
                    "x": self.spec.tile_size,
                })
                ds_region.attrs["n_time_steps"] = len(self.spec.seasonal_windows)
                ds_region.attrs["bands"] = self.spec.band_names
    
                print(f"  Writing {ds_region.sizes['sample']} samples from {region.name}")
    
                # New writable session for this write 
                session = self.repo.writable_session(branch)
    
                if first_region:
                    # Only ever used on a true fresh build (resume=False)
                    to_icechunk(ds_region, session, mode="w")
                    first_region = False
                else:
                    # Append along 'sample' with automatic chunk alignment
                    to_icechunk(
                        ds_region,
                        session,
                        append_dim="sample",
                        align_chunks=True,
                    )
    
                # Update regions_done within this transaction
                regions_done.add(region.name)
                root = zarr.open_group(session.store, mode="r+")
                root.attrs["regions_done"] = list(regions_done)
    
                last_snapshot_id = session.commit(
                    f"append {ds_region.sizes['sample']} samples from {region.name}"
                )
                total_samples += ds_region.sizes["sample"]
    
            if last_snapshot_id is None:
                if resume_effective:
                    # Nothing new to write; dataset already complete for this spec
                    print(
                        "[resume] No new regions were written; all regions are already present. "
                        "Returning current branch tip."
                    )
                    # Return the current branch tip snapshot
                    ro_session = self.repo.readonly_session(branch)
                    return ro_session.snapshot_id  # or whatever your caller expects
                else:
                    # Fresh build with no data is probably a real problem
                    raise RuntimeError("No regions produced data; nothing was written.")
            print(f"[Done] Icechunk repo s3://{self.bucket}/{self.repo_prefix}")
            print(f"Total samples (this branch view): {total_samples}")
            print(f"Final snapshot id: {last_snapshot_id}")
            return last_snapshot_id


def verify_icechunk_dataset(
    *,
    bucket: str,
    repo_prefix: str,
    branch: str = "main",
    region: str | None = None,
    from_env: bool = True,
    samples_to_plot: list[int] | None = None,
    expected_time_steps: int = 3,
    season_names: list[str] | None = None,
    max_samples: int = 2,
    make_plots: bool = True,
    print_ds: bool = False,
):
    """
    Verifies an Icechunk-backed dataset was saved correctly and optionally plots a few samples.

    Returns
    -------
    ds : xarray.Dataset
        Opened dataset (lazy until compute()).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if region is None:
        session_aws = boto3.session.Session()
        region = session_aws.region_name or "ap-southeast-2"

    if season_names is None:
        season_names = ["Jan-Feb", "Jun-Jul", "Sep-Oct"]

    if samples_to_plot is None:
        samples_to_plot = list(range(max_samples))

    # --- 1) Load dataset from Icechunk ---
    print("⏳ Verifying Icechunk-backed dataset...")
    storage = ic.s3_storage(
        bucket=bucket,
        prefix=repo_prefix,
        region=region,
        from_env=from_env,
    )
    repo = ic.Repository.open(storage)
    session = repo.readonly_session(branch=branch)

    # xarray.open_zarr expects a Zarr store; Icechunk provides session.store
    ds = xr.open_zarr(session.store, consolidated=False, chunks={})
    
    if print_ds:
        print(ds)
        
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

    # Band names are attributes in your pipeline
    band_names = list(ds.attrs.get("bands", []))
    print(f"  Bands: {band_names}")

    band_index = {name: i for i, name in enumerate(band_names)}

    def _get_rgb(data_array, bands_list):
        """
        Extract bands for RGB from a DataArray, assuming dims include 'band'
        and output needed is (T, Y, X, Color) for plotting.
        """
        stack = []
        for b in bands_list:
            if b not in band_index:
                raise KeyError(f"Band '{b}' not found in ds.attrs['bands']: {band_names}")
            idx = band_index[b]
            band_data = data_array.isel(band=idx)  # (time, y, x)
            stack.append(band_data)

        img = xr.concat(stack, dim="color").values.astype("float32")  # (Color, T, Y, X)

        # Scale Sentinel-2 reflectance (common 0..10000 scaling)
        img = img / 10000.0
        img = np.nan_to_num(img, nan=0.0)

        # (Color, T, Y, X) -> (T, Y, X, Color)
        img = np.transpose(img, (1, 2, 3, 0))

        if img.max() == 0:
            return None

        # Simple per-time-step contrast stretch
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

    cols = 3  # RGB, False Color, Labels
    # rows = (#samples plotted) * (time steps)
    plot_samples = [s for s in samples_to_plot if 0 <= s < num_samples]
    if len(plot_samples) == 0:
        print("⚠️ No valid samples requested for plotting; returning dataset.")
        return ds

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
        fc_stack  = _get_rgb(sample_features, ["swir1", "nir", "red"])

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

            # Add colorbar once
            if s == plot_samples[0] and t == 0:
                plt.colorbar(im, fraction=0.046, pad=0.04)

            ax3.set_title(f"S{s}: Static Labels")
            ax3.axis("off")
            plot_idx += 1

    plt.tight_layout()
    plt.show()
    return ds
