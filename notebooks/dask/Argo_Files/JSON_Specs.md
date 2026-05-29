## Purpose

JSON files define a geospatial sampling spec: where to sample (regions), when to sample (seasonal windows), what to load (bands), and how to format outputs (projection, resolution, tile size).

## File structure

### Top-level keys

- `regions` (required): List of named bounding boxes to sample.
- `seasonal_windows` (required): List of date ranges that define each “time step”.
- `bands_s2` (required): Sentinel-2 asset/band identifiers to request from STAC.
- `band_names` (required): Human-friendly names for the model input channels.
- `band_map` (required): Mapping from Sentinel-2 band IDs to `band_names`.
- `resolution` (required): Pixel size in meters for resampling.
- `output_crs` (required): CRS used for output tiles (string like `"EPSG:3577"`).
- `tile_size` (required): Tile width/height in pixels (square tiles).
- `catalog` (required): STAC endpoint configuration.

## Regions format

Each region entry must be:

```json
{ "name": "unique_region_name", "bbox": [min_lon, min_lat, max_lon, max_lat] }
```

Rules:
- `name` must be unique across the file (used for logging/IDs).
- `bbox` must be in **WGS84 lon/lat** order: `[west, south, east, north]`.
- Use decimal degrees.
- Ensure `min_lon < max_lon` and `min_lat < max_lat`.

Example:

```json
{ "name": "tasmania_forest_c", "bbox": [146.5, -42.0, 146.7, -41.8] }
```

## Seasonal windows format

`seasonal_windows` is a list of strings in the form:

```
"YYYY-MM-DD/YYYY-MM-DD"
```

Rules:
- Each entry defines one time slice (so the number of windows equals the dataset `time` dimension).
- Windows should not overlap unless you intentionally want repeated sampling periods.
- Keep consistent count across train/val/test if your model expects a fixed number of time steps.

Example:

```json
"2021-01-01/2021-02-28"
```

## Bands and mapping

### `bands_s2`
List of Sentinel-2 band identifiers used in the STAC items, plus optional auxiliary layers.

Example:

```json
["B02", "B03", "B04", "B8A", "B11", "B12", "SCL"]
```

Notes:
- If you include `"SCL"` (scene classification), decide whether it is used as:
  - an input channel, or
  - a mask/quality layer (common).
- If `"SCL"` is included in `bands_s2` but not in `band_map`, it usually means it’s not part of the model’s continuous reflectance channels.

### `band_names`
Human-friendly names for the model channels (and the order you expect them).

Example:

```json
["blue", "green", "red", "nir", "swir1", "swir2"]
```

### `band_map`
Maps each reflectance band ID to a friendly channel name.

Example:

```json
{
  "B02": "blue",
  "B03": "green",
  "B04": "red",
  "B8A": "nir",
  "B11": "swir1",
  "B12": "swir2"
}
```

Rules:
- Every key in `band_map` should appear in `bands_s2`.
- Every value in `band_map` should appear in `band_names`.
- `band_names` should not contain duplicates.

## Output formatting

- `resolution` (meters): e.g. `20` means 20 m pixels.
- `output_crs`: e.g. `"EPSG:3577"` for Australian Albers.
- `tile_size` (pixels): e.g. `224` creates `224 x 224` pixel tiles.

Practical implication: ground footprint per tile is approximately `tile_size * resolution` meters on a side (before projection distortion effects).

## STAC catalog configuration

```json
"catalog": {
  "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
  "modifier": "planetary_computer"
}
```

Rules:
- `url` is the STAC API root.
- `modifier` indicates any special request signing/URL modification logic your code applies (e.g., Planetary Computer signing).

## Minimal checklist for new specs

- Regions: unique names, valid lon/lat bbox order.
- Seasonal windows: correct format, consistent count with model expectation.
- Bands: `bands_s2` contains required bands; `band_map` covers reflectance channels; `band_names` matches your model input.
- Output: `resolution`, `tile_size`, and `output_crs` match what your pipeline expects.
- Catalog: correct STAC endpoint and modifier.