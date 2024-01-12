#!python3

# A collection of utilities that can be used in Python notebooks.
#
# License: Apache 2.0

# Created for EASI Hub training notebooks, https://dev.azure.com/csiro-easi/easi-hub-public/_git/hub-notebooks

# Data tools
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import datacube
from datacube.utils import masking
from datetime import datetime

# hvPlot, Holoviews, Datashader and Bokeh
import hvplot.pandas
import hvplot.xarray
import panel as pn
import holoviews as hv
# hv.extension("bokeh", logo=False)  # Its likely set from in the notebooks

# Jupyter Lab
from IPython.display import HTML

# Python
import sys, os, re
import logging
from pathlib import Path
from collections import Counter

# Dask
from dask.distributed import Client, LocalCluster
from dask_gateway import Gateway

# Set logger
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def display_table(
    df: pd.DataFrame,
    panel: bool = False,
):
    """Display the full pandas dataframe. If panel is True use a panel object"""
    table = None
    if panel:
        #  Dicts are rendered as "[object Object]". Need to set a formatter, I guess.
        table = pn.widgets.DataFrame(df,
            # sizing_mode='stretch_width',  # equal column widths, full screen
            autosize_mode='fit_viewport',  # fitted columns, about 90-95% width
            # reorderable=True,  # didn't work first try
        )
    else:
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.max_colwidth', -1):
            table = HTML( df.to_html().replace("\\n","<br>") )
    display(table)
    
        
def heading(txt: str):
    """Print a simple HTML heading"""
    display(HTML( f'<h4>{txt}</h4>' ))
    
    
def hv_table_hook(plot, element):
    """Selected options for hv.table() formatting
    
    Use: df.hv.table().opts(hooks=[hv_table_hook])
    """
    plot.handles['table'].autosize_mode="fit_viewport"
    # Other examples
    # plot.handles['table'].row_height = 40
    # from bokeh.models.widgets import DateFormatter
    # plot.handles['table'].columns[6].formatter = DateFormatter(format='%Y-%m-%d')
    

def xarray_object_size(data):
    """Return a formatted string"""
    val, unit = data.nbytes / (1024 ** 2), 'MB'
    if val > 1024:
        val, unit = data.nbytes / (1024 ** 3), 'GB'
    return f'Dataset size: {val:.2f} {unit}'


def mostcommon_crs(dc, query):
    """Adapted from https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Tools/dea_tools/datahandling.py"""
    matching_datasets = dc.find_datasets(**query)
    crs_list = [str(i.crs) for i in matching_datasets]
    crs_mostcommon = None
    if len(crs_list) > 0:
        # Identify most common CRS
        crs_counts = Counter(crs_list)
        crs_mostcommon = crs_counts.most_common(1)[0][0]
    else:
        logger.warning('No data was found for the supplied product query')
    return crs_mostcommon


def initialize_dask(use_gateway=False, workers=(1,2), wait=False, local_port=8786):
    """Initialize a Dask Gateway or Local cluster"""
    # Check inputs
    if isinstance(workers, (int, float)):
        workers = (int(workers), int(workers))
    if len(workers) != 2:
        logger.error('Require workers to be a single integer or a 2-element tuple/list')
        return None, None
    if isinstance(local_port, (str, float)):
        local_port = int(local_port)
    
    # Dask gateway
    if use_gateway:
        gateway = Gateway()
        clusters = gateway.list_clusters()
        if not clusters:
            logger.info('Starting new cluster.')
            cluster = gateway.new_cluster()
        else:
            logger.info(f'An existing cluster was found. Connecting to: {clusters[0].name}')
            cluster = gateway.connect(clusters[0].name)
        client = cluster.get_client()
        cluster.adapt(minimum=workers[0], maximum=workers[1])
        if wait:
            logger.info('Waiting for at least one cluster worker.')
            # client.wait_for_workers(n_workers=1)  # Before release 2023.10.0
            client.sync(client._wait_for_workers,n_workers=1) # Since release 2023.10.0

    # Local cluster
    else:
        try:
            # This creates a new Client connection to an existing Dask scheduler if one exists.
            # There is no practical way to get the LocalCluster object from the existing scheduler,
            # although the scheduler details can be accessed with `client.scheduler`.
            # The LocalCluster object is only available from the notebook that created it.
            # Restart the kernel or `client.close();cluster.close()` in each notebook that
            # created one to remove existing LocalClusters.
            client = Client(f'localhost:{local_port}', timeout='2s', n_workers=workers[0])
            cluster = client.cluster  # None
        except:
            cluster = LocalCluster(
                n_workers=workers[0],
                scheduler_port=local_port
            )
            client = Client(cluster)
        
    return cluster, client


def localcluster_dashboard(client, server='https://hub.csiro.easi-eo.solutions'):
    """Return a dashboard link using jupyter proxy"""
    dashboard_link = client.dashboard_link
    for host in ('127.0.0.1', 'localhost'):
        if host in dashboard_link:
            port = re.search(':(\d+)\/status', dashboard_link).group(1)
            dashboard_link = f'{server}{os.environ["JUPYTERHUB_SERVICE_PREFIX"]}proxy/{port}/status'
            break
    return dashboard_link
