#!python3

import sys
import os
import logging

# A class that provides notebook variables for each of the EASI deployments

# Map an internal deployment name to deployment variables and search parameters.
# Update to ensure that the product/space/time parameters are available in the respective databases
deployment_map = {
    'csiro': {
        'domain': 'csiro.easi-eo.solutions',
        'db_database': '',
        'scratch_bucket': '',
        'productmap': {'landsat': 'ga_ls8c_ard_3', 'sentinel-2': 'ga_s2am_ard_3', 'dem': 'copernicus_dem_30'},
        'location': 'Lake Hume, Australia',
        'latitude': (-36.3, -35.8),
        'longitude': (146.8, 147.3),
        'time': ('2020-02-01', '2020-04-01'),
    },
    'asia': {
        'domain': 'asia.easi-eo.solutions',
        'db_database': '',
        'scratch_bucket': '',
        'productmap': {'landsat': 'landsat8_c2l2_sr', 'sentinel-2': 's2_l2a', 'sar': 'asf_s1_grd_gamma0', 'dem': 'copernicus_dem_30'},
        'location': 'Lake Tempe, Indonesia',
        'latitude': (-4.2, -3.9),
        'longitude': (119.8, 120.1),
        'time': ('2020-02-01', '2020-04-01'),
        'proxy': True,
        'target': {
            'landsat': {'crs': 'epsg:32650', 'resolution': (-30,30)},
        },
    },
    'chile': {
        'domain': 'datacubechile.cl',
        'db_database': 'easido_prod_db',
        'scratch_bucket': 'easido-prod-user-scratch',
        'productmap': {'landsat': 'landsat8_c2l2_sr', 'sentinel-2': 's2_l2a', 'sar': 'asf_s1_grd_gamma0', 'dem': 'copernicus_dem_30'},
        'location': '',
        'latitude': (0, 0),
        'longitude': (0, 0),
        'time': ('', ''),
    },
    'adias': {
        'domain': 'adias.aquawatchaus.space',
        'db_database': '',
        'scratch_bucket': '',
        'productmap': {'landsat': 'landsat8_c2l2_sr', 'sentinel-2': 's2_l2a', 'sar': 'asf_s1_grd_gamma0', 'dem': 'copernicus_dem_30'},
        'location': '',
        'latitude': (0, 0),
        'longitude': (0, 0),
        'time': ('', ''),
    },
    'eail': {
        'domain': 'eail.easi-eo.solutions',
        'db_database': 'ceoseail_eail_db',
        'scratch_bucket': 'ceoseail-eail-user-scratch',
        'ows': False,
        'map': False,
        'productmap': {'landsat': 'landsat8_c2l2_sr', 'sentinel-2': 's2_l2a', 'sentinel-1':'s1_rtc', 'dem': 'copernicus_dem_30'},
        'location': 'Newport News, Virginia',
        'latitude': (37.02, 37.12),
        'longitude': (-76.55, -76.45),
        'time': ('2022-01-01', '2022-04-01'),
        'target': {
            'landsat': {'crs': 'epsg:32618', 'resolution': (-30,30)},
        }
    },
    'sub-apse2': {
        'domain': 'sub-apse2.easi-eo.solutions',
        'db_database': '',
        'scratch_bucket': '',
        'ows': False,
        'map': False,
        'productmap': {'landsat': 'ga_ls8c_ard_3', 'sentinel-2': 'ga_s2am_ard_3', 'dem': 'copernicus_dem_30'},
        'location': 'Lake Hume, Australia',
        'latitude': (-36.3, -35.8),
        'longitude': (146.8, 147.3),
        'time': ('2020-02-01', '2020-04-01'),
    },
}


class EasiNotebooks():
    """Provide deployment-specific variables for EASI notebooks"""
    
    def __init__(self, deployment=None):
        """Initialise"""
        self._log = _getlogger(self.__class__.__name__)
        self.name = deployment if deployment else self._find_deployment()
        self.deployment = self._validate(self.name)
        self.proxy = None
        if self.deployment and self.deployment.get('proxy', None):
            self.proxy = EasiCachingProxy()
        if self.deployment:
            self._log.info(f'Successfully found configuration for deployment "{self.name}"')
    
    def _validate(self, deployment):
        """Validate"""
        names = deployment_map.keys()
        # deployment = deployment if deployment else self._find_deployment()
        if deployment is None or deployment not in names:
            self._log.error(f'Deployment name not recognised: {deployment}')
            self._log.error(f'Select one of: {", ".join(names)}')
            return None
        return deployment_map[deployment]
    
    def _find_deployment(self):
        db_database = os.environ['DB_DATABASE']
        deployment_name = [item for item in deployment_map if deployment_map[item]["db_database"] == db_database]
        if len(deployment_name)==1:
            return deployment_name[0]
        elif len(deployment_name)==0:
            self._log.error('Deployment could not be found automatically, try specifying one using EasiNotebooks(deployment="deployment_name").')
            return None
        elif len(deployment_name)>1:
            self._log.error('More than one deployment found')
            return None
        
        
    
    @property
    def domain(self):
        """Deployment domain"""
        return self.deployment['domain']

    @property
    def db_database(self):
        """Database name"""
        return self.deployment['db_database']
    
    @property
    def scratch_bucket(self):
        """Scratch bucket"""
        return self.deployment['scratch_bucket']
    
    @property
    def hub(self):
        """JupyterLab URL"""
        return f'https://hub.{self.domain}'

    @property
    def explorer(self):
        """Explorer URL"""
        return f'https://explorer.{self.domain}'

    @property
    def ows(self):
        """OWS URL"""
        if not self.deployment.get('ows', True):
            self._log.warning(f'Deployment does not have an OWS service: {self.name}')
            return None
        return f'https://ows.{self.domain}'

    @property
    def terria(self):
        """Terria Map URL""" 
        if not self.deployment.get('map', True):
            self._log.warning(f'Deployment does not have a Map service: {self.name}')
            return None
        return f'https://map.{self._domain()}'

    @property
    def location(self):
        """Default location name"""
        return self.deployment['location']

    @property
    def latitude(self):
        """Default latitude name"""
        return self.deployment['latitude']

    @property
    def longitude(self):
        """Default longitude name"""
        return self.deployment['longitude']

    @property
    def time(self):
        """Default time range"""
        return self.deployment['time']

    def product(self, family='landsat'):
        """Product name. Family loosely describes products from a satellite series or product type."""
        p = self.deployment['productmap'].get(family, None)
        if p is None:
            self._log.warning(f'Product family not defined for "{self.name}": {family}')
            out = ', '.join([f'{k} > {v}' for k,v in self.deployment['productmap'].items()])
            self._log.warning(f'{self.name}: {out}')
            return None
        return p

    def crs(self, family='landsat'):
        """Default resolution. Family loosely describes products from a satellite series or product type."""
        return self.deployment.get('target', {}).get(family, {}).get('crs', None)
    
    def resolution(self, family='landsat'):
        """Default resolution. Family loosely describes products from a satellite series or product type."""
        return self.deployment.get('target', {}).get(family, {}).get('resolution', None)


class EasiCachingProxy():
    """Set, unset and return information about the user's caching-proxy configuration"""
    
    def __init__(self):
        pass
    
    
    
def _getlogger(name):
    """Return a logger"""
    # Default logger
    #   log.hasHandlers() = False
    #   log.getEffectiveLevel() = 30 = warning
    #   log.propagate = True
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not len(logger.handlers):
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.propagate = False  # Do not propagate up to root logger, which may have other handlers
    return logger
