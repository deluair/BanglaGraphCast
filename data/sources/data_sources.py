"""
Data source management for Bangladesh weather data
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from abc import ABC, abstractmethod

from ...configs.bangladesh_config import DATA_SOURCES, BANGLADESH_DOMAIN

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources"""
    
    @abstractmethod
    def download(self, start_date: datetime, end_date: datetime) -> None:
        pass
    
    @abstractmethod
    def load(self, start_date: datetime, end_date: datetime) -> xr.Dataset:
        pass
    
    @abstractmethod
    def preprocess(self, data: xr.Dataset) -> xr.Dataset:
        pass


class ERA5Source(DataSource):
    """ERA5 Reanalysis data source"""
    
    def __init__(self, data_dir: str = "data/raw/era5"):
        self.data_dir = data_dir
        self.config = DATA_SOURCES['era5']
        os.makedirs(data_dir, exist_ok=True)
    
    def download(self, start_date: datetime, end_date: datetime) -> None:
        """Download ERA5 data using CDS API"""
        try:
            import cdsapi
        except ImportError:
            raise ImportError("Install cdsapi: pip install cdsapi")
        
        c = cdsapi.Client()
        
        # Download surface variables
        self._download_surface_data(c, start_date, end_date)
        
        # Download pressure level data
        self._download_pressure_level_data(c, start_date, end_date)
    
    def _download_surface_data(self, client, start_date: datetime, end_date: datetime):
        """Download surface variables"""
        request = {
            'product_type': 'reanalysis',
            'variable': self.config['variables']['surface'],
            'date': f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': [
                BANGLADESH_DOMAIN['lat_range'][1],  # North
                BANGLADESH_DOMAIN['lon_range'][0],  # West
                BANGLADESH_DOMAIN['lat_range'][0],  # South
                BANGLADESH_DOMAIN['lon_range'][1],  # East
            ],
            'format': 'netcdf',
        }
        
        output_file = os.path.join(
            self.data_dir, 
            f"era5_surface_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"
        )
        
        client.retrieve('reanalysis-era5-single-levels', request, output_file)
        logger.info(f"Downloaded ERA5 surface data to {output_file}")
    
    def _download_pressure_level_data(self, client, start_date: datetime, end_date: datetime):
        """Download pressure level variables"""
        request = {
            'product_type': 'reanalysis',
            'variable': self.config['variables']['pressure_levels'],
            'pressure_level': self.config['levels'],
            'date': f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}",
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': [
                BANGLADESH_DOMAIN['lat_range'][1],
                BANGLADESH_DOMAIN['lon_range'][0],
                BANGLADESH_DOMAIN['lat_range'][0],
                BANGLADESH_DOMAIN['lon_range'][1],
            ],
            'format': 'netcdf',
        }
        
        output_file = os.path.join(
            self.data_dir,
            f"era5_pressure_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"
        )
        
        client.retrieve('reanalysis-era5-pressure-levels', request, output_file)
        logger.info(f"Downloaded ERA5 pressure level data to {output_file}")
    
    def load(self, start_date: datetime, end_date: datetime) -> xr.Dataset:
        """Load ERA5 data from files"""
        surface_files = []
        pressure_files = []
        
        # Find matching files
        for file in os.listdir(self.data_dir):
            if file.startswith("era5_surface_") and file.endswith(".nc"):
                surface_files.append(os.path.join(self.data_dir, file))
            elif file.startswith("era5_pressure_") and file.endswith(".nc"):
                pressure_files.append(os.path.join(self.data_dir, file))
        
        # Load and combine data
        if surface_files and pressure_files:
            surface_data = xr.open_mfdataset(surface_files, combine='by_coords')
            pressure_data = xr.open_mfdataset(pressure_files, combine='by_coords')
            
            # Combine surface and pressure level data
            combined = xr.merge([surface_data, pressure_data])
            
            # Filter by date range
            combined = combined.sel(time=slice(start_date, end_date))
            
            return combined
        else:
            raise FileNotFoundError("ERA5 data files not found")
    
    def preprocess(self, data: xr.Dataset) -> xr.Dataset:
        """Preprocess ERA5 data for Bangladesh domain"""
        # Subset to Bangladesh domain
        data = data.sel(
            latitude=slice(BANGLADESH_DOMAIN['lat_range'][1], BANGLADESH_DOMAIN['lat_range'][0]),
            longitude=slice(BANGLADESH_DOMAIN['lon_range'][0], BANGLADESH_DOMAIN['lon_range'][1])
        )
        
        # Interpolate to target resolution
        target_lat = np.arange(
            BANGLADESH_DOMAIN['lat_range'][0], 
            BANGLADESH_DOMAIN['lat_range'][1] + BANGLADESH_DOMAIN['resolution'], 
            BANGLADESH_DOMAIN['resolution']
        )
        target_lon = np.arange(
            BANGLADESH_DOMAIN['lon_range'][0], 
            BANGLADESH_DOMAIN['lon_range'][1] + BANGLADESH_DOMAIN['resolution'], 
            BANGLADESH_DOMAIN['resolution']
        )
        
        data = data.interp(latitude=target_lat, longitude=target_lon)
        
        # Unit conversions
        if 'tp' in data.variables:  # Total precipitation
            data['tp'] = data['tp'] * 1000  # Convert from m to mm
        
        if 't2m' in data.variables:  # 2m temperature
            data['t2m'] = data['t2m'] - 273.15  # Convert from K to °C
        
        return data


class BMDSource(DataSource):
    """Bangladesh Meteorological Department station data"""
    
    def __init__(self, data_dir: str = "data/raw/bmd"):
        self.data_dir = data_dir
        self.config = DATA_SOURCES['bmd_stations']
        os.makedirs(data_dir, exist_ok=True)
        
        # Station locations (example coordinates)
        self.stations = {
            'dhaka': (23.8103, 90.4125),
            'chittagong': (22.3569, 91.7832),
            'sylhet': (24.8949, 91.8687),
            'khulna': (22.8456, 89.5403),
            'rajshahi': (24.3745, 88.6042),
            'rangpur': (25.7559, 89.2444),
            'barishal': (22.7010, 90.3535),
            'comilla': (23.4607, 91.1809),
            'jessore': (23.1687, 89.2072),
            'mymensingh': (24.7471, 90.4203)
        }
    
    def download(self, start_date: datetime, end_date: datetime) -> None:
        """Download BMD station data (placeholder - requires BMD API access)"""
        logger.warning("BMD data download requires official API access")
        # This would integrate with BMD's data sharing system
        pass
    
    def load(self, start_date: datetime, end_date: datetime) -> xr.Dataset:
        """Load BMD station data"""
        # Placeholder implementation
        # In practice, this would load from BMD database or files
        
        stations_data = []
        for station, (lat, lon) in self.stations.items():
            # Generate synthetic data for demonstration
            times = pd.date_range(start_date, end_date, freq='3H')
            
            station_ds = xr.Dataset({
                'temperature': (['time'], np.random.normal(25, 5, len(times))),
                'precipitation': (['time'], np.random.exponential(2, len(times))),
                'pressure': (['time'], np.random.normal(1013, 10, len(times))),
                'humidity': (['time'], np.random.normal(80, 10, len(times))),
                'wind_speed': (['time'], np.random.exponential(3, len(times))),
                'wind_direction': (['time'], np.random.uniform(0, 360, len(times)))
            }, coords={
                'time': times,
                'latitude': lat,
                'longitude': lon,
                'station_name': station
            })
            
            stations_data.append(station_ds)
        
        # Combine all stations
        combined = xr.concat(stations_data, dim='station')
        return combined
    
    def preprocess(self, data: xr.Dataset) -> xr.Dataset:
        """Preprocess BMD station data"""
        # Quality control
        data = self._quality_control(data)
        
        # Spatial interpolation to grid
        data = self._interpolate_to_grid(data)
        
        return data
    
    def _quality_control(self, data: xr.Dataset) -> xr.Dataset:
        """Apply quality control to station data"""
        # Remove outliers
        for var in ['temperature', 'pressure', 'humidity']:
            if var in data.variables:
                # Remove values outside reasonable bounds
                if var == 'temperature':
                    data[var] = data[var].where((data[var] >= -10) & (data[var] <= 50))
                elif var == 'pressure':
                    data[var] = data[var].where((data[var] >= 950) & (data[var] <= 1050))
                elif var == 'humidity':
                    data[var] = data[var].where((data[var] >= 0) & (data[var] <= 100))
        
        return data
    
    def _interpolate_to_grid(self, data: xr.Dataset) -> xr.Dataset:
        """Interpolate station data to regular grid"""
        # This would use spatial interpolation methods like kriging or IDW
        # For now, return as-is
        return data


class SatelliteSource(DataSource):
    """Satellite data source (GPM IMERG, MODIS, etc.)"""
    
    def __init__(self, data_dir: str = "data/raw/satellite"):
        self.data_dir = data_dir
        self.config = DATA_SOURCES['satellite']
        os.makedirs(data_dir, exist_ok=True)
    
    def download(self, start_date: datetime, end_date: datetime) -> None:
        """Download satellite data"""
        # GPM IMERG precipitation
        self._download_gpm_imerg(start_date, end_date)
        
        # MODIS land surface temperature
        self._download_modis_lst(start_date, end_date)
    
    def _download_gpm_imerg(self, start_date: datetime, end_date: datetime):
        """Download GPM IMERG precipitation data"""
        logger.info("Downloading GPM IMERG data...")
        # This would integrate with NASA EARTHDATA API
        pass
    
    def _download_modis_lst(self, start_date: datetime, end_date: datetime):
        """Download MODIS land surface temperature"""
        logger.info("Downloading MODIS LST data...")
        # This would integrate with NASA LAADS DAAC
        pass
    
    def load(self, start_date: datetime, end_date: datetime) -> xr.Dataset:
        """Load satellite data"""
        # Placeholder implementation
        times = pd.date_range(start_date, end_date, freq='1H')
        
        lat = np.arange(
            BANGLADESH_DOMAIN['lat_range'][0], 
            BANGLADESH_DOMAIN['lat_range'][1], 
            0.1
        )
        lon = np.arange(
            BANGLADESH_DOMAIN['lon_range'][0], 
            BANGLADESH_DOMAIN['lon_range'][1], 
            0.1
        )
        
        # Generate synthetic satellite data
        satellite_data = xr.Dataset({
            'precipitation_rate': (['time', 'latitude', 'longitude'], 
                                 np.random.exponential(0.5, (len(times), len(lat), len(lon)))),
            'land_surface_temperature': (['time', 'latitude', 'longitude'], 
                                       np.random.normal(25, 5, (len(times), len(lat), len(lon))))
        }, coords={
            'time': times,
            'latitude': lat,
            'longitude': lon
        })
        
        return satellite_data
    
    def preprocess(self, data: xr.Dataset) -> xr.Dataset:
        """Preprocess satellite data"""
        # Convert units
        if 'land_surface_temperature' in data.variables:
            # Convert from K to °C if needed
            if data['land_surface_temperature'].max() > 100:
                data['land_surface_temperature'] = data['land_surface_temperature'] - 273.15
        
        # Apply quality flags and masks
        data = self._apply_quality_masks(data)
        
        return data
    
    def _apply_quality_masks(self, data: xr.Dataset) -> xr.Dataset:
        """Apply quality control masks"""
        # Remove poor quality pixels
        # This would use satellite-specific quality flags
        return data


class DataSourceManager:
    """Manages multiple data sources for Bangladesh weather prediction"""
    
    def __init__(self):
        self.sources = {
            'era5': ERA5Source(),
            'bmd': BMDSource(),
            'satellite': SatelliteSource()
        }
    
    def download_all(self, start_date: datetime, end_date: datetime):
        """Download data from all sources"""
        for name, source in self.sources.items():
            logger.info(f"Downloading data from {name}")
            try:
                source.download(start_date, end_date)
            except Exception as e:
                logger.error(f"Failed to download {name} data: {e}")
    
    def load_all(self, start_date: datetime, end_date: datetime) -> Dict[str, xr.Dataset]:
        """Load data from all sources"""
        datasets = {}
        
        for name, source in self.sources.items():
            logger.info(f"Loading data from {name}")
            try:
                data = source.load(start_date, end_date)
                data = source.preprocess(data)
                datasets[name] = data
            except Exception as e:
                logger.error(f"Failed to load {name} data: {e}")
        
        return datasets
    
    def combine_datasets(self, datasets: Dict[str, xr.Dataset]) -> xr.Dataset:
        """Combine datasets from different sources"""
        # This would implement sophisticated data fusion
        # For now, prioritize ERA5 as the base dataset
        
        if 'era5' in datasets:
            combined = datasets['era5'].copy()
            
            # Add satellite precipitation if available
            if 'satellite' in datasets and 'precipitation_rate' in datasets['satellite']:
                combined['satellite_precip'] = datasets['satellite']['precipitation_rate']
            
            # Add station data as point observations
            if 'bmd' in datasets:
                # This would require spatial interpolation or assimilation
                pass
            
            return combined
        
        raise ValueError("No ERA5 data available for combination")
