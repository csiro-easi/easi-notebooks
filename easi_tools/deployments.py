#!python3

import sys
import logging
import datacube

# A class that provides notebook variables for each of the EASI deployments

# Map an internal deployment name to deployment variables and search parameters.
# Update to ensure that the product/space/time parameters are available in the respective databases
deployment_map = {
    'csiro': {
        'domain': 'csiro.easi-eo.solutions',
        'productmap': {'landsat': 'ga_ls8c_ard_3', 's2': 'ga_s2am_ard_3', 'dem': 'copernicus_dem_30'},
        'location': 'Lake Hume, Australia',
        'latitude': (-36.3, -35.8),
        'longitude': (146.8, 147.3),
        'time': ('2020-02-01', '2020-04-01'),
    },
    'asia': {
        'domain': 'asia.easi-eo.solutions',
        'productmap': {'landsat': 'landsat8_c2l2_sr', 's2': 's2_l2a', 'sar': 'asf_s1_grd_gamma0', 'dem': 'copernicus_dem_30'},
        'location': '',
        'latitude': (0, 0),
        'longitude': (0, 0),
        'time': ('', ''),
    },
    'chile': {
        'domain': 'datacubechile.cl',
        'productmap': {'landsat': 'landsat8_c2l2_sr', 's2': 's2_l2a', 'sar': 'asf_s1_grd_gamma0', 'dem': 'copernicus_dem_30'},
        'location': '',
        'latitude': (0, 0),
        'longitude': (0, 0),
        'time': ('', ''),
    },
    'adias': {
        'domain': 'adias.aquawatchaus.space',
        'productmap': {'landsat': 'landsat8_c2l2_sr', 's2': 's2_l2a', 'sar': 'asf_s1_grd_gamma0', 'dem': 'copernicus_dem_30'},
        'location': '',
        'latitude': (0, 0),
        'longitude': (0, 0),
        'time': ('', ''),
    },
    'eail': {
        'domain': 'eail.easi-eo.solutions',
        'ows': False,
        'map': False,
        'productmap': {},
        'location': '',
        'latitude': (0, 0),
        'longitude': (0, 0),
        'time': ('', ''),
    },
    'sub-apse2': {
        'domain': 'sub-apse2.easi-eo.solutions',
        'ows': False,
        'map': False,
        'productmap': {'landsat': 'ga_ls8c_ard_3', 's2': 'ga_s2am_ard_3', 'dem': 'copernicus_dem_30'},
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
        self._log = self._getlogger(self.__class__.__name__)
        self.name = deployment
        self.deployment = self._validate(deployment)
        self.dc = datacube.Datacube()

    def _getlogger(self, name):
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
    
    def _validate(self, deployment):
        """Validate"""
        names = deployment_map.keys()
        if deployment is None or deployment not in names:
            self._log.error(f'Deployment name not recognised: {deployment}')
            self._log.error(f'Select one of: {", ".join(names)}')
            return None
        return deployment_map[deployment]
    
    @property
    def domain(self):
        """Deployment domain"""
        return self.deployment['domain']

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

    def product(self, family = 'landsat'):
        """Product name. Family loosely describes products from a satellite series or product type."""
        p = self.deployment['productmap'].get(family, None)
        if p is None:
            self._log.warning(f'Product family not defined for "{self.name}": {family}')
            out = ', '.join([f'{k} > {v}' for k,v in self.deployment['productmap'].items()])
            self._log.warning(f'{self.name}: {out}')
            return None
        return p
