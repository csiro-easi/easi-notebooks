#!python3

import sys
import os
import logging
import collections

# A class that provides notebook variables for each of the EASI deployments

# Map an internal deployment name to deployment variables and search parameters.
# Update to ensure that the product/space/time parameters are available in the respective databases
deployment_map = {
    'adias': {
        'domain': 'adias.aquawatchaus.space',
        'db_database': 'adias_prod_db',
        'training_shapefile': '',
        'scratch': 'adias-prod-user-scratch',
        'productmap': {'landsat': 'landsat8_c2l2_sr', 'sentinel-2': 's2_l2a', 'sar': 'asf_s1_grd_gamma0', 'dem': 'copernicus_dem_30'},
        'location': 'Lake Tahoe, California',
        'latitude': (39.0, 39.3),
        'longitude': (-120.2, -119.9),
        'time': ('2022-02-01', '2022-05-01'),
        'target': {
            'landsat': {'crs': 'epsg:32650', 'resolution': (-30,30)},
            'sentinel-2': {'crs': 'epsg:32650', 'resolution': (-10,10)}
        },
        'aliases': {
            'landsat': {'qa_band': 'qa_pixel', 'nir': 'nir08', 'swir1': 'swir16', 'swir2': 'swir22'}
        },
        'qa_mask': {
            'landsat': {'nodata': False, 'water': 'land_or_cloud',
                        'cloud': 'not_high_confidence', 'cloud_shadow': 'not_high_confidence'}
        }
    },
    'asia': {
        'domain': 'asia.easi-eo.solutions',
        'db_database': 'easi_asia_db',
        'training_shapefile': '',
        'scratch': 'easi-asia-user-scratch',
        'productmap': {'landsat': 'landsat8_c2l2_sr', 'sentinel-2': 's2_l2a', 'sar': 'asf_s1_grd_gamma0', 'dem': 'copernicus_dem_30'},
        'location': 'Lake Tempe, Indonesia',
        'latitude': (-4.2, -3.9),
        'longitude': (119.8, 120.1),
        'time': ('2020-02-01', '2020-04-01'),
        'proxy': True,
        'target': {
            'landsat': {'crs': 'epsg:32650', 'resolution': (-30,30)},
            'sentinel-2': {'crs': 'epsg:32650', 'resolution': (-10,10)}
        },
        'aliases': {
            'landsat': {'qa_band': 'qa_pixel', 'nir': 'nir08', 'swir1': 'swir16', 'swir2': 'swir22'}
        },
        'qa_mask': {
            'landsat': {'nodata': False, 'water': 'land_or_cloud',
                        'cloud': 'not_high_confidence', 'cloud_shadow': 'not_high_confidence'}
        }
    },
    'chile': {
        'domain': 'datacubechile.cl',
        'db_database': 'easido_prod_db',
        'training_shapefile': '',
        'scratch': 'easido-prod-user-scratch',
        'productmap': {'landsat': 'landsat8_c2l2_sr', 'sentinel-2': 's2_l2a', 'sar': 'asf_s1_grd_gamma0', 'dem': 'copernicus_dem_30'},
        'location': 'La Serena, Chile',
        'latitude': (-29.95, -29.85),
        'longitude': (-71.3, -71.2),
        'latitude_big': (-29.95, -27.95),
        'longitude_big': (-71.3, -69.3),
        'time': ('2022-02-01', '2022-05-01'),
        'target': {
            'landsat': {'crs': 'epsg:32718', 'resolution': (-30,30)},
            'sentinel-2': {'crs': 'epsg:32718', 'resolution': (-10,10)}
        },
        'aliases': {
            'landsat': {'qa_band': 'qa_pixel', 'nir': 'nir08', 'swir1': 'swir16', 'swir2': 'swir22'}
        },
        'qa_mask': {
            'landsat': {'nodata': False, 'water': 'land_or_cloud',
                        'cloud': 'not_high_confidence', 'cloud_shadow': 'not_high_confidence'}
        }
    },
    'cal': {
        'domain': 'cal.ceos.org',
        'db_database': 'ceoseail_eail_db',
        'training_shapefile': './ancillary_data/VA_Counties_Newport_News.shp',
        'scratch': 'ceoseail-eail-user-scratch',
        'ows': False,
        'map': False,
        'productmap': {'landsat': 'landsat8_c2l2_sr', 'sentinel-2': 's2_l2a', 'sentinel-1':'s1_rtc', 'dem': 'copernicus_dem_30'},
        'location': 'Newport News, Virginia',
        'latitude': (37.02, 37.12),
        'longitude': (-76.55, -76.45),
        'time': ('2022-01-01', '2022-04-01'),
        'target': {
            'landsat': {'crs': 'epsg:32618', 'resolution': (-30,30)},
            'sentinel-2': {'crs': 'epsg:32618', 'resolution': (-10,10)}
        },
        'aliases': {
            'landsat': {'qa_band': 'qa_pixel', 'nir': 'nir08', 'swir1': 'swir16', 'swir2': 'swir22'}
        },
        'qa_mask': {
            'landsat': {'nodata': False, 'water': 'land_or_cloud',
                        'cloud': 'not_high_confidence', 'cloud_shadow': 'not_high_confidence'}
        }
    },
    'csiro': {
        'domain': 'csiro.easi-eo.solutions',
        'db_database': 'easihub_csiro_db',
        'training_shapefile': '',
        'scratch': 'easihub-csiro-user-scratch',
        'productmap': {'landsat': 'ga_ls8c_ard_3', 'sentinel-2': 'ga_s2am_ard_3', 'dem': 'copernicus_dem_30'},
        'location': 'Lake Hume, Australia',
        'latitude': (-36.3, -35.8),
        'longitude': (146.8, 147.3),
        'time': ('2020-02-01', '2020-04-01'),
        'aliases': {
            'landsat': {'red': 'nbart_red', 'green': 'nbart_green', 'blue': 'nbart_blue',
                        'nir': 'nbart_nir', 'swir1': 'nbart_swir_1', 'swir2': 'nbart_swir_2',
                        'qa_band': 'oa_fmask'}
        },
        'qa_mask': {
            'landsat': {'fmask':'valid'}
        }
    },
    'sub-apse2': {
        'domain': 'sub-apse2.easi-eo.solutions',
        'db_database': '',
        'training_shapefile': '',
        'scratch': '',
        'ows': False,
        'map': False,
        'productmap': {'landsat': 'ga_ls8c_ard_3', 'sentinel-2': 'ga_s2am_ard_3', 'dem': 'copernicus_dem_30'},
        'location': 'Lake Hume, Australia',
        'latitude': (-36.3, -35.8),
        'longitude': (146.8, 147.3),
        'time': ('2020-02-01', '2020-04-01'),
        'aliases': {
            'landsat': {'red': 'nbart_red', 'green': 'nbart_green', 'blue': 'nbart_blue',
                        'nir': 'nbart_nir', 'swir1': 'nbart_swir_1', 'swir2': 'nbart_swir_2',
                        'qa_band': 'oa_fmask'}
        },
        'qa_mask': {
            'landsat': {'fmask':'valid'}
        }
    },
}


class EasiDefaults():
    """Provide deployment-specific default variables for EASI notebooks"""
    
    def __init__(self, deployment=None):
        """Initialise"""
        self._log = _getlogger(self.__class__.__name__)
        self.name = deployment if deployment else self._find_deployment()
        self.deployment = self._validate(self.name)
        self.proxy = None
        self._aliases = {}
        if self.deployment and self.deployment.get('proxy', None):
            self.proxy = EasiCachingProxy()
        if self.deployment:
            self._log.info(f'Successfully found configuration for deployment "{self.name}"')
    
    def _validate(self, deployment) -> dict:
        """Return the dict associated with the deployment name"""
        names = deployment_map.keys()
        if deployment is None or deployment not in names:
            self._log.error(f'Deployment name not recognised: {deployment}')
            self._log.error(f'Select one of: {", ".join(names)}')
            return None
        return deployment_map[deployment]
    
    def _find_deployment(self) -> str:
        """Use the deployment's database environment variable as a lookup into the deployment_map dict"""
        db_database = os.environ['DB_DATABASE']
        deployment_name = [item for item in deployment_map if deployment_map[item]["db_database"] == db_database]
        msg = 'Try specifying one using EasiDefaults(deployment="deployment_name").'
        if len(deployment_name) == 0:
            self._log.error(f'Deployment could not be found automatically. {msg}')
            return None
        elif len(deployment_name) > 1:
            self._log.error(f'More than one deployment found. {msg}')
            return None
        return deployment_name[0]

    
    @property
    def domain(self):
        """Deployment domain"""
        return self.deployment['domain']

    @property
    def db_database(self):
        """Database name"""
        return self.deployment['db_database']

    @property
    def training_shapefile(self):
        """A local shapefile"""
        return self.deployment['training_shapefile']
    
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
    def scratch(self):
        """Scratch bucket"""
        return self.deployment['scratch']

    @property
    def location(self):
        """Default location name"""
        return self.deployment['location']

    @property
    def latitude(self):
        """Default latitude range"""
        return self.deployment['latitude']

    @property
    def longitude(self):
        """Default longitude range"""
        return self.deployment['longitude']

    @property
    def latitude_big(self):
        """Default big latitude range"""
        if 'latitude_big' in self.deployment:
            return self.deployment['latitude_big']
        self._log.warning(f'Default big latitude range not defined for "{self.deployment}". Using default latitude range')
        return self.latitude
    
    @property
    def longitude_big(self):
        """Default big longitude range"""
        if 'longitude_big' in self.deployment:
            return self.deployment['longitude_big']
        self._log.warning(f'Default big longitude range not defined for "{self.deployment}". Using default longitude range')
        return self.latitude

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

    def aliases(self, family='landsat') -> collections.UserDict:
        """Return a dict-like object that maps a common name to a specific measurement/alias name.
        Family loosely describes products from a satellite series or product type.
        
        The common name is returned if there is no specific measurement/alias name defined.
        That is, the common name should work as a measurement/alias name for the family in this deployment.
        Else, provide a specific measurement/alias name in the defaults above.
        """
        if family not in self._aliases:
            self._aliases[family] = EasiAlias(self.deployment.get('aliases', {}).get(family, {}))
        return self._aliases[family]

    def qa_mask(self, family='landsat') -> dict:
        """Default QA mask values. Family loosely describes products from a satellite series or product type."""
        return self.deployment.get('qa_mask', {}).get(family, {})


class EasiAlias(collections.UserDict):
    """Custom UserDict that returns a default measurement name for a given key if defined.
    Else returns the key as the value. Items can not be set."""
    def __init__(self, default:dict = {}):
        self.data = default
        self._log = _getlogger(self.__class__.__name__)
    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return key
    def __setitem__(self, key, val):
        self._log.error(f'Error <{self.__class__.__name__}>: Can not set items')


class EasiCachingProxy():
    """Set, unset and return information about the user's caching-proxy configuration"""
    def __init__(self):
        pass
    

def _getlogger(name):
    """Return a logger. Define here to limit external dependecies"""
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
