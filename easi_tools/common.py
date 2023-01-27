#!python3

def xarray_object_size(data):
    """Return a formatted string"""
    val, unit = data.nbytes / (1024 ** 2), 'MB'
    if val > 1024:
        val, unit = data.nbytes / (1024 ** 3), 'GB'
    return f'Dataset size: {val:.2f} {unit}'
