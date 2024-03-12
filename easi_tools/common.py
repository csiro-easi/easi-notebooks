#!python3

import contextlib
import os

def xarray_object_size(data):
    """Return a formatted string"""
    val, unit = data.nbytes / (1024 ** 2), 'MB'
    if val > 1024:
        val, unit = data.nbytes / (1024 ** 3), 'GB'
    return f'Dataset size: {val:.2f} {unit}'


@contextlib.contextmanager
def unset_cachingproxy():
    """Unset the EASI caching proxy with a context manager"""
    # Inspired by https://stackoverflow.com/a/34333710
    env = os.environ
    remove = ('AWS_HTTPS', 'GDAL_HTTP_PROXY')
    update_after = {k: env[k] for k in remove}

    try:
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
