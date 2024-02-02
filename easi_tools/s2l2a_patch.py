#!python

# Sentinel-2 scaling and offset changes.
#
# ESA has undertaken a reprocessing of the Sentinel-2 L2A product that includes
# a change to the offset value used to convert digital numbers (in file) to
# scientific values (reflectances).
# ESA's reprocessing is flowing through to the AWS open data repository of S2 L2A but
# while this stabilises we may see inconsistencies in the COG files (encoded DNs)
# compared to the STAC metadata (whether offset has been applied or not).
#
# TL;DR: DN values have different defintions depending on the processing baseline version
# and whether the offset change has been pre-applied by the cloud data custodian.
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
# These are the values in the product definition,
# https://explorer.asia.easi-eo.solutions/products/s2_l2a.odc-product.yaml
#
# Example workflow:
# https://github.com/Element84/earth-search/issues/23#issuecomment-1834674853
#
# 1. For each dataset corresponding to a query, filter into lists of
#    offset_applied and offset_notapplied.
# 2. Load each list into separate xarray Datasets.
# 3. Apply the offset to the respective xarray Dataset.
# 4. Merge the two xarray Datasets.
# 5. Apply any "group_by" method given in the query.
# 6. Return the combined xarray Dataset.

import datacube
import xarray as xr
import pandas as pd
import logging
from pathlib import Path
import sys

# Set logger
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def load_s2l2a_with_offset_patch(
    dc: datacube.Datacube,
    query: dict
) -> xr.Dataset:

    if query['product'] != 's2_l2a':
        print(f'This function only applies to "s2_l2a", not: {query["product"]}')
        return None

    search_keys = (
        'product',
        'time',
        'x', 'y',
        'latitude', 'longitude',
        'crs'
    )
    load_keys = (
        'measurements',
        'output_crs',
        'resolution',
        # 'group_by',
        'dask_chunks'
    )
    refl_bands = (
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
    )
    scale_factor = 0.0001
    add_offset = -0.1

    search_params = {k:v for k,v in query.items() if k in search_keys}
    load_params   = {k:v for k,v in query.items() if k in load_keys}

    # Find datasets
    matches = dc.find_datasets(**search_params)

    # Filter into two lists
    offset_applied, offset_notapplied = [], []
    for ds in matches:
        props = ds.metadata_doc['properties']
        if props.get('earthsearch:boa_offset_applied', False):
            offset_applied.append(ds)
        else:
            offset_notapplied.append(ds)

    # If either list is empty then no separation and merge is required
    this_offset = None
    if len(offset_applied) == 0:
        logger.info('No datasets with offset applied. Apply scale and offset to all layers')
        this_offset = add_offset
    if len(offset_notapplied) == 0:
        logger.info('No datasets without offset applied. Apply scale only to all layers')
        this_offset = 0
    if this_offset is not None:
        data = dc.load(**query)
        data_vars = [x for x in data.data_vars if x in refl_bands]
        data[data_vars] = data[data_vars] * scale_factor + this_offset
        return data

    # Else, load data into two Datasets
    logger.info(f'Number of datasets with offset applied: {len(offset_applied)}')
    logger.info(f'Number of datasets without offset applied: {len(offset_notapplied)}')
    data_offset_applied = dc.load(
        datasets = offset_applied,
        **load_params
    )
    data_offset_notapplied = dc.load(
        datasets = offset_notapplied,
        **load_params
    )
    data_vars = [x for x in data_offset_applied.data_vars if x in refl_bands]

    # Apply respective offsets
    data_offset_applied[data_vars] = data_offset_applied[data_vars] * scale_factor
    data_offset_notapplied[data_vars] = data_offset_notapplied[data_vars] * scale_factor + add_offset

    # Merge datasets
    data = xr.merge(
      [data_offset_applied, data_offset_notapplied]
    )

    # Group_by, if required
    if 'group_by' in query:
        # TODO
        pass

    return data
