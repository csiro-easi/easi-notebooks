#!python3

from dask_gateway import Gateway


def xarray_object_size(data):
    """Return a formatted string"""
    val, unit = data.nbytes / (1024 ** 2), 'MB'
    if val > 1024:
        val, unit = data.nbytes / (1024 ** 3), 'GB'
    return f'Dataset size: {val:.2f} {unit}'


def init_dask_cluster() -> tuple:
    """Connect to an existing or start a new dask gateway cluster.
    Return (cluster, client)
    """
    gateway = Gateway()
    clusters = gateway.list_clusters()
    if not clusters:
        print('Creating new cluster. Please wait for this to finish.')
        cluster = gateway.new_cluster()
    else:
        print(f'An existing cluster was found. Connecting to: {clusters[0].name}')
        cluster=gateway.connect(clusters[0].name)
    cluster.adapt(minimum=1, maximum=4)  # A default starting point
    client = cluster.get_client()

    return (cluster, client)
