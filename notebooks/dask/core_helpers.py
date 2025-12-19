from dataclasses import dataclass
from typing import Dict, List, Tuple

import odc.stac
import boto3
from odc.geo.geobox import GeoBox
from odc.geo.geom import box as odc_box
import time as _time
from pystac_client.exceptions import APIError
import importlib
import numpy as np
import os
#from pathlib import Path
import xarray as xr
from pystac_client import Client as PClient
import planetary_computer as pc
import json


@dataclass
class RegionSpec:
    name: str
    bbox: Tuple[float, float, float, float]


@dataclass
class EODataSpec:
    regions: List[RegionSpec]
    seasonal_windows: List[str]
    bands_s2: List[str]          # raw band IDs loaded from STAC
    band_names: List[str]        # canonical names in same order as a subset
    band_map: Dict[str, str]     # mapping raw -> canonical
    resolution: int
    output_crs: str
    tile_size: int


def _read_text_from_s3(uri: str) -> str:
    parsed = urlparse(uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


def load_config(config_path_or_uri: str) -> dict:
    if config_path_or_uri.startswith("s3://"):
        return json.loads(_read_text_from_s3(config_path_or_uri))
    with open(config_path_or_uri, "r") as f:
        return json.load(f)


def spec_from_config(cfg: dict) -> EODataSpec:
    regions: list[RegionSpec] = []
    for r in cfg["regions"]:
        bbox = tuple(r["bbox"])
        if len(bbox) != 4:
            raise ValueError(f"Region {r.get('name')} bbox must have 4 numbers, got: {r['bbox']}")
        regions.append(RegionSpec(name=r["name"], bbox=bbox))  # type: ignore[arg-type]

    return EODataSpec(
        regions=regions,
        seasonal_windows=list(cfg["seasonal_windows"]),
        bands_s2=list(cfg["bands_s2"]),
        band_names=list(cfg["band_names"]),
        band_map=dict(cfg["band_map"]),
        resolution=int(cfg["resolution"]),
        output_crs=str(cfg["output_crs"]),
        tile_size=int(cfg["tile_size"]),
    )


ALLOWED_MODIFIERS = {
    "none": None,
    "planetary_computer.sign_inplace": "planetary_computer.sign_inplace",
    "planetary_computer": "planetary_computer.sign_inplace",
    "myproj.modifiers.noop": "myproj.modifiers.noop",
}


def resolve(path: str):
    mod, attr = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), attr)


def make_catalog(cfg: dict):
    cat = cfg["catalog"]
    key = cat.get("modifier", "none")
    target = ALLOWED_MODIFIERS.get(key, "__INVALID__")
    if target == "__INVALID__":
        raise ValueError(f"Unsupported modifier {key!r}. Allowed: {list(ALLOWED_MODIFIERS)}")
    modifier = None if target is None else resolve(target)
    return PClient.open(cat["url"], modifier=modifier)


# Refractor tested and working
class RegionSampler:
    def sample_patches(
        self,
        features_region: xr.DataArray,  # (band, time, y, x)
        labels_region: xr.DataArray,    # (y, x)
    ) -> list[xr.Dataset]:
        raise NotImplementedError


class GridRegionSampler(RegionSampler):
    def __init__(self, tile_size: int):
        self.tile_size = tile_size

    def sample_patches(self, features_region, labels_region):
        tile_size = self.tile_size
        ny, nx = features_region.sizes["y"], features_region.sizes["x"]
        ny_tiles = ny // tile_size
        nx_tiles = nx // tile_size

        patches = []
        sample_idx = 0
        for iy in range(ny_tiles):
            for ix in range(nx_tiles):
                ys = slice(iy * tile_size, (iy + 1) * tile_size)
                xs = slice(ix * tile_size, (ix + 1) * tile_size)

                feat_patch = features_region.isel(y=ys, x=xs)
                lab_patch = labels_region.isel(y=ys, x=xs)

                feat_patch = feat_patch.assign_coords(
                    y=np.arange(tile_size),
                    x=np.arange(tile_size),
                )
                lab_patch = lab_patch.assign_coords(
                    y=np.arange(tile_size),
                    x=np.arange(tile_size),
                )

                ds = xr.Dataset(
                    {"features": feat_patch, "labels": lab_patch}
                ).expand_dims(sample=[sample_idx])

                patches.append(ds)
                sample_idx += 1

        return patches


def safe_search_items(catalog, collections, bbox, datetime=None, query=None,
                      max_items=150, retries=3):
    for i in range(retries):
        t0 = _time.time()
        print(f"  STAC search start (try {i+1}/{retries}) "
              f"{collections} {datetime} bbox={bbox}")
        try:
            search = catalog.search(
                collections=collections,
                bbox=bbox,
                datetime=datetime,
                query=query,
                max_items=max_items,
            )
            items = search.item_collection()
            dt = _time.time() - t0
            return items
        except APIError as e:
            dt = _time.time() - t0
            print(f"  STAC search failed after {dt:.1f}s: {e}")
            msg = str(e)
            if "maximum allowed time" in msg and i < retries - 1:
                wait = 2 ** i
                print(f"  STAC timeout, retry {i+1}/{retries} after {wait}s...")
                time.sleep(wait)
                continue
            raise

        
# --- Region-level geobox and loaders ---
def make_region_geobox(spec: EODataSpec, bbox):
    """
    Full-region GeoBox; patches will be spec.tile_size later via slicing.
    """
    geom_4326 = odc_box(bbox[0], bbox[1], bbox[2], bbox[3], crs="EPSG:4326")
    geom_3577 = geom_4326.to_crs(spec.output_crs)
    return GeoBox.from_bbox(
        geom_3577.boundingbox,
        crs=spec.output_crs,
        resolution=spec.resolution,
    )


def load_region_labels(catalog, spec: EODataSpec, bbox):
    geobox = make_region_geobox(spec, bbox)
    items = safe_search_items(
        catalog=catalog,
        collections=["esa-worldcover"],
        bbox=bbox,
        max_items=25,
    )
    if not items:
        return None, geobox

    ds = odc.stac.load(
        items,
        bands=["map"],
        geobox=geobox,
        chunks={"y": 1024, "x": 1024},
        resampling="nearest",
        fail_on_error=False,
    )
    labels = ds["map"].isel(time=0, drop=True).fillna(255).astype("uint8")
    return labels, geobox


def load_region_season(catalog, spec: EODataSpec, bbox, time_range, geobox):
    try:
        items = safe_search_items(
            catalog=catalog,
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": 90}},
            max_items=100,
        )
    except APIError as e:
        print(f"  [season {time_range}] STAC search failed: {e}")
        return None

    if not items:
        return None

    ds = odc.stac.load(
        items,
        bands=spec.bands_s2,
        geobox=geobox,
        chunks={"y": 1024, "x": 1024},
        fail_on_error=False,
    )
    if ds.sizes.get("time", 0) == 0:
        return None

    if "SCL" in ds:
        qa = ds["SCL"]
        valid = ((qa == 4) | (qa == 5) | (qa == 6) |
                 (qa == 7) | (qa == 2) | (qa == 11))
        masked = ds.where(valid)
        comp = masked.median(dim="time")
        comp = comp.fillna(masked.min(dim="time")).fillna(0)
    else:
        comp = ds.median(dim="time").fillna(0)

    # Now use the spec-driven band selection/rename
    raw_band_ids = list(spec.band_map.keys())
    comp = comp[raw_band_ids]
    comp = comp.rename(spec.band_map).astype("uint16")
    return comp


def region_to_patches(
    catalog,
    spec: EODataSpec,
    region: RegionSpec,
    sampler: RegionSampler,
    ):
    bbox = region.bbox

    labels_region, geobox = load_region_labels(catalog, spec, bbox)
    if labels_region is None:
        print(f"  No labels for {region.name}, skipping.")
        return []

    comps = []
    for t in spec.seasonal_windows:
        comp = load_region_season(catalog, spec, bbox, t, geobox)
        if comp is None:
            print(f"  Skipping {region.name} for window {t} (no data / STAC error)")
            return []
        comps.append(comp)

    features_region = xr.concat(comps, dim="time")
    features_region = features_region.to_array("band").transpose("band", "time", "y", "x")

    return sampler.sample_patches(features_region, labels_region)