#!python

# Sentinel-2 L2A Collection 0 scaling and offset corrections.
# - Applies to data indexed from https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a
# - The newer https://earth-search.aws.element84.com/v1/collections/sentinel-2-c1-l2a (Collection 1) may not be affected in the same way
#
# TL;DR:
# DN values in COG files have different definitions depending on the processing baseline version
# and whether the offset change has been pre-applied by the cloud data custodian.
#
# Background:
#
# ESA has undertaken a reprocessing of the Sentinel-2 L2A product that includes
# a change to the offset value used to convert digital numbers (in file) to
# scientific values (reflectances).
#
# https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a-algorithms-products
#
# L2A algorithm and products: Starting with the PB 04.00 (25th January 2022), the dynamic
# range of the Level-2A products is shifted by a band-dependent constant: BOA_ADD_OFFSET.
# This offset will allow encoding negative surface reflectances that may occur over very 
# dark surfaces.
#
# L2A_SRi = (L2A_DNi + BOA_ADD_OFFSETi) / QUANTIFICATION_VALUEi
#
# QUANTIFICATION_VALUEi = 10000
# BOA_ADD_OFFSETi = -1000
#
# refl = (dn -1000) / 10000
# refl = dn/10000 - 1000/10000
# refl = dn * 0.0001 - 0.1
# 
# These are the values in the EASI product definition, e.g.
# https://explorer.asia.easi-eo.solutions/products/s2_l2a.odc-product.yaml
#
# Example workflow:
#
# ESA's reprocessing is flowing through to the AWS open data repository of S2 L2A but
# while this stabilises we may see inconsistencies in time series queries due to:
# - More than one processed version of a dataset (scene) in the AWS bucket and indexed in an EASI database
# - Datasets (scenes) that indicate they have an offset applied by ESA but the offset correction
#   has been not been applied to the COG
#
# Element-84 discussion:
# https://github.com/Element84/earth-search/issues/23#issuecomment-1834674853


import xarray as xr
import pandas as pd
import logging
from pathlib import Path
import sys, re

import datacube
from datacube.api.core import output_geobox
from datacube.api.query import SPATIAL_KEYS, CRS_KEYS, OTHER_KEYS
from datacube.utils import masking


# Set logger
log = logging.getLogger(Path(__file__).stem)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

# Constants
search_keys = (
    'product',
    'time',
    'geopolygon',
    'like',
    'limit',
    'ensure_location',
    'dataset_predicate',
) + SPATIAL_KEYS + CRS_KEYS + OTHER_KEYS

# TODO: get measurement aliases from the ODC product record
refl_bands = {
    'coastal','band_01','B01','coastal_aerosol',
    'blue','band_02','B02',
    'green','band_03','B03',
    'red','band_04','B04',
    'rededge1','band_05','B05','red_edge_1',
    'rededge2','band_06','B06','red_edge_2',
    'rededge3','band_07','B07','red_edge_3',
    'nir','band_08','B08','nir_1',
    'nir08','band_8a','B8A','nir_2',
    'nir09','band_09','B09','nir_3',
    'swir16','band_11','B11','swir_1','swir_16',
    'swir22','band_12','B12','swir_2','swir_22',
}
scale_factor = 0.0001
add_offset = -0.1


def highest_sequence_number(matches: list) -> dict:
    """Filter for the highest element84 processing sequence number per scene (scene label excluding the sequence number)
    
    : return : { scene_id_excluding_sequence_number : { highest_sequence_number : datacube.model.Dataset }}
    """
    p = re.compile('(S2.+)_([0-9]+)_(L2A)')
    sorter = {}
    for ds in matches:
        # Separate the scene label from the sequence number
        label = ds.metadata_doc['label']
        m = p.match(label)
        if not m:
            log.warning(f'Dataset label does not match expected pattern: {label}')
            continue
        key = f'{m.group(1)}_{m.group(3)}'
        seq = int(m.group(2))
        # Retain the highest sequence number
        if key in sorter:
            if list(sorter[key])[0] < seq:
                sorter[key] = {seq: ds}
        else:
            sorter[key] = {seq: ds}
    return sorter


def ds_requires_offset(ds: datacube.model.Dataset) -> bool:
    """Return True if a dataset's metadata indicates that the offset correction should be applied"""
    props = ds.metadata_doc['properties']
    
    # If baseline is less than '04.00' then offset correction does not apply
    baseline = props.get('s2:processing_baseline', '0.0')
    p = re.compile('(\d+)\.(\d+)')
    m = p.match(baseline)
    if not m:
        log.warning(f'Dataset processing_baseline does not match expected pattern: {baseline}')
        return None
    if int(m.group(1)) < 4:
        return False

    # If the boa_offset_applied has been applied then offset correction is not required
    boa_offset_applied = props.get('earthsearch:boa_offset_applied', False)
    return not boa_offset_applied


def apply_correction_to_data(ds: xr.Dataset, offset: float = 0) -> xr.Dataset:
    """Apply the scale and offset correction to each reflectance band where there is valid data (not nodata)"""
    refl_vars = [x for x in ds.data_vars if x in refl_bands]
    mask = masking.valid_data_mask(ds[refl_vars])
    # Save on a dask step?
    if offset == 0:
        ds[refl_vars] = ds[refl_vars].where(mask) * scale_factor
    else:
        ds[refl_vars] = ds[refl_vars].where(mask) * scale_factor + offset
    return ds
    

def load_s2l2a_with_offset(
    dc: datacube.Datacube,
    query: dict,
) -> xr.Dataset:
    """
    Replaces datacube.load(**query) for s2_l2a products.

    Method:
    - Find all datasets matching the query (dc.find_datasets)
    - Filter for the highest element84 processing sequence number per scene (scene label excluding the sequence number)
    - Filter into two lists for datasets that have
      - "s2:processing_baseline" >= "04.00" and "earthsearch:boa_offset_applied" == False (offset correction required)
      - everything else (no correction required)
    - If either list is empty then load the non-empty list, apply scale (and offset if required), and return the xarray Dataset
    - Load and combine the two lists of datasets
      - Load each list, apply scale (and offset if required)
      - Concat on time dimension and sort by time
      - Return the combined xarray Dataset
    
    Notes:
    - Any 'groupby' function is applied to each of the xarray Datasets prior to them being combined.
      This could create "extra" (non-grouped) time layers in the combined Dataset if the groupby function
      would have grouped datasets (scenes) from both lists.
    - Scale and offset are applied to the reflectance bands where there is valid data (not `nodata`).
      This includes applying the "scale_factor" even if no datasets require the offset correction.
      Other masks can be applied by the user (e.g. pixel quality or cloud masking).
    """

    product = query.get('product', '<all products>')
    if product != 's2_l2a':
        log.error(f'This function only applies to the "s2_l2a" product, not: {product}')
        return None
    
    # Find all datasets matching the query
    matches = None
    if 'datasets' in query:
        matches = query['datasets']
        del query['datasets']
    if matches is None:
        search_params = {k:v for k,v in query.items() if k in search_keys}
        matches = dc.find_datasets(**search_params)
    if 'skip_broken_datasets' not in query:
        # This helps to avoid data loading error messages
        query['skip_broken_datasets'] = True
    
    # Filter for the highest element84 processing sequence number
    sorter = highest_sequence_number(matches)

    # Filter into two lists
    offset_applied, offset_required = [], []
    for key in sorter.keys():
        ds = list(sorter[key].values())[0]
        isrequired = ds_requires_offset(ds)
        if isrequired is None:
            continue
        elif isrequired:
            offset_required.append(ds)
        else:
            offset_applied.append(ds)
    matches_combined = offset_applied + offset_required

    # If either list is empty then no separation and merge is required
    this_offset = None
    if len(offset_applied) == 0:
        log.info('All datasets require offset correction')
        msg = 'The valid_data_mask, scale and offset have been applied to the reflectance bands'
        this_offset = add_offset
    if len(offset_required) == 0:
        log.info('No datasets require offset correction')
        msg = 'The valid_data_mask and scale (no offset) have been applied to the reflectance bands'
        this_offset = 0
    if this_offset is not None:
        data = dc.load(
            datasets = matches_combined,
            **query
        )
        xx = apply_correction_to_data(data, this_offset)
        log.info(msg)
        return xx

    # DEBUG: What do we have
    # def func(s):
    #     p = re.compile('(\d{8})')
    #     m = p.search(s[0])
    #     if m:
    #         return m.group(1)
    # log.info(f'Number of datasets in initial query: {len(matches)}')
    # log.info(f'{sorted([(x.metadata_doc["label"],x.id) for x in matches], key=func)}')
    # log.info(f'Number of datasets with offset applied: {len(offset_applied)}')
    # log.info(f'{sorted([(x.metadata_doc["label"],x.id) for x in offset_applied], key=func)}')
    # log.info(f'Number of datasets without offset applied: {len(offset_required)}')
    # log.info(f'{sorted( [(x.metadata_doc["label"],x.id) for x in offset_required], key=func)}')
    # return

    # Else, load data into two Datasets
    log.info('Mix of datasets found with either offset required or not.')
    log.info('We will load two xarrays, apply offset where required, and merge into one xarray.')

    # 1. Ensure the target geobox covers all datasets
    target_geobox = output_geobox(
        datasets = matches_combined,
        **query,
    )
    
    # 2. Edit the query for our needs
    # Ensure that dask time chunking = 1
    dask_input = None
    if 'dask_chunks' in query:
        dask_input = query['dask_chunks']  # Save
        if dask_input.get('time', 1) != 1:
            query['dask_chunks'].update({'time': 1})
    # Remove keys that are not compatible with 'like'
    for x in ('output_crs', 'resolution', 'align'):
        if x in query:
            del query[x]
    
    # 3. Load two xarrays
    data_offset_applied = dc.load(
        datasets = offset_applied,
        like = target_geobox,
        **query
    )
    data_offset_required = dc.load(
        datasets = offset_required,
        like = target_geobox,
        **query
    )

    # 4. Apply respective scale and offsets
    data_offset_applied = apply_correction_to_data(data_offset_applied)
    data_offset_required = apply_correction_to_data(data_offset_required, add_offset)
    
    # 5. Combine the two xarrays
    combined = xr.concat([data_offset_applied, data_offset_required], dim='time')
    combined = combined.sortby('time')
    
    # 6. Reapply any time > 1 chunking
    if dask_input is not None:
        if dask_input.get('time', 1) != 1:
            combined = combined.chunk(dask_input)

    log.info('The valid_data_mask, scale and offset have been applied to the reflectance bands')
    return combined
