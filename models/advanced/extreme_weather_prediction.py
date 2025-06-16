"""
Extreme Weather Prediction Module for Bangladesh GraphCast

Specialized module for predicting extreme weather events in Bangladesh:
- Tropical cyclones and storm surge prediction
- Extreme precipitation and flood forecasting
- Heat wave and drought prediction
- Severe thunderstorm and tornado potential
- Multi-hazard risk assessment
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import xarray as xr
from pathlib import Path
import pandas as pd
from scipy import ndimage, signal
from scipy.stats import pearsonr, genextreme

logger = logging.getLogger(__name__)


class ExtremeEventType(Enum):
    """Types of extreme weather events for Bangladesh"""
    TROPICAL_CYCLONE = "tropical_cyclone"
    STORM_SURGE = "storm_surge"
    EXTREME_PRECIPITATION = "extreme_precipitation"
    FLASH_FLOOD = "flash_flood"
    RIVERINE_FLOOD = "riverine_flood"
    DROUGHT = "drought"
    HEAT_WAVE = "heat_wave"
    SEVERE_THUNDERSTORM = "severe_thunderstorm"
    TORNADO = "tornado"
    COASTAL_EROSION = "coastal_erosion"


class ExtremeIntensity(Enum):
    """Intensity classifications for extreme events"""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"
    CATASTROPHIC = "catastrophic"


@dataclass
class ExtremeEventThresholds:
    """Thresholds for extreme event detection in Bangladesh"""
    # Cyclone thresholds (wind speed in m/s)
    cyclone_depression: float = 17.0
    cyclone_storm: float = 24.0
    cyclone_severe: float = 32.0
    cyclone_very_severe: float = 47.0
    cyclone_super: float = 62.0
    
    # Precipitation thresholds (mm/day)
    heavy_rain: float = 64.5
    very_heavy_rain: float = 124.5
    extreme_heavy_rain: float = 244.5
    
    # Temperature thresholds (°C)
    heat_wave_moderate: float = 36.0
    heat_wave_severe: float = 40.0
    heat_wave_extreme: float = 45.0
    
    # Drought indices
    drought_moderate_spi: float = -1.0
    drought_severe_spi: float = -1.5
    drought_extreme_spi: float = -2.0


@dataclass
class ExtremeWeatherConfig:
    """Configuration for extreme weather prediction"""
    # Event types to predict
    event_types: List[ExtremeEventType] = None
    
    # Prediction parameters
    lead_times: List[int] = None  # hours
    ensemble_size: int = 50
    probability_thresholds: List[float] = None
    
    # Detection parameters
    thresholds: ExtremeEventThresholds = None
    spatial_smoothing: bool = True
    temporal_smoothing: bool = True
    
    # Bangladesh-specific regions
    regions_of_interest: List[str] = None
    coastal_focus: bool = True
    river_basin_focus: bool = True
    
    # Model integration
    use_ensemble_spread: bool = True
    use_historical_analogs: bool = True
    use_machine_learning: bool = True
    
    def __post_init__(self):
        if self.event_types is None:
            self.event_types = [
                ExtremeEventType.TROPICAL_CYCLONE,
                ExtremeEventType.EXTREME_PRECIPITATION,
                ExtremeEventType.HEAT_WAVE,
                ExtremeEventType.DROUGHT
            ]
        
        if self.lead_times is None:
            self.lead_times = [6, 12, 24, 48, 72, 120, 168]  # 6h to 7 days
        
        if self.probability_thresholds is None:
            self.probability_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        if self.thresholds is None:
            self.thresholds = ExtremeEventThresholds()
        
        if self.regions_of_interest is None:
            self.regions_of_interest = [
                "dhaka", "chittagong", "khulna", "rajshahi", "barisal",
                "sylhet", "rangpur", "mymensingh", "coastal_zone", "haor_region"
            ]


class CycloneDetector(nn.Module):
    """Neural network for tropical cyclone detection and tracking"""
    
    def __init__(self, input_channels: int = 20, hidden_dim: int = 128):
        super().__init__()
        
        # Feature extraction for cyclone patterns
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16))
        )
        
        # Cyclone classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # No cyclone + 5 intensity categories
        )
        
        # Intensity regression
        self.intensity_regressor = nn.Sequential(
            nn.Linear(256 * 16 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Wind speed
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cyclone detection
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Tuple of (classification_probs, intensity_prediction)
        """
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        
        classification = torch.softmax(self.classifier(features_flat), dim=1)
        intensity = self.intensity_regressor(features_flat)
        
        return classification, intensity


class ExtremeEventTracker:
    """Tracks and analyzes extreme weather events"""
    
    def __init__(self, config: ExtremeWeatherConfig):
        self.config = config
        self.event_history = []
        self.analog_database = {}
    
    def detect_tropical_cyclone(self, 
                              pressure: np.ndarray,
                              wind_speed: np.ndarray,
                              vorticity: np.ndarray,
                              temperature: np.ndarray) -> Dict[str, Any]:
        """
        Detect tropical cyclone presence and characteristics
        
        Args:
            pressure: Sea level pressure field
            wind_speed: Wind speed magnitude
            vorticity: Relative vorticity
            temperature: Temperature field
            
        Returns:
            Dictionary with cyclone detection results
        """
        results = {
            'cyclone_detected': False,
            'intensity_category': None,
            'center_location': None,
            'max_wind_speed': None,
            'minimum_pressure': None,
            'confidence': 0.0
        }
        
        # Find pressure minima
        pressure_smooth = ndimage.gaussian_filter(pressure, sigma=2)
        local_minima = ndimage.minimum_filter(pressure_smooth, size=20) == pressure_smooth
        pressure_minima = np.where(local_minima & (pressure_smooth < np.percentile(pressure_smooth, 5)))
        
        if len(pressure_minima[0]) == 0:
            return results
        
        # Check each minimum for cyclone characteristics
        for i, j in zip(pressure_minima[0], pressure_minima[1]):
            # Extract local region around minimum
            region_size = 10
            i_start, i_end = max(0, i-region_size), min(pressure.shape[0], i+region_size)
            j_start, j_end = max(0, j-region_size), min(pressure.shape[1], j+region_size)
            
            local_pressure = pressure[i_start:i_end, j_start:j_end]
            local_wind = wind_speed[i_start:i_end, j_start:j_end]
            local_vorticity = vorticity[i_start:i_end, j_start:j_end]
            
            # Check cyclone criteria
            pressure_gradient = np.max(local_pressure) - np.min(local_pressure)
            max_wind = np.max(local_wind)
            avg_vorticity = np.mean(local_vorticity)
            
            # Bangladesh-specific cyclone criteria
            is_cyclone = (
                pressure_gradient > 10.0 and  # Significant pressure gradient
                max_wind > self.config.thresholds.cyclone_depression and
                avg_vorticity > 1e-4  # Positive vorticity in Northern Hemisphere
            )
            
            if is_cyclone:
                results['cyclone_detected'] = True
                results['center_location'] = (i, j)
                results['max_wind_speed'] = max_wind
                results['minimum_pressure'] = np.min(local_pressure)
                results['intensity_category'] = self._classify_cyclone_intensity(max_wind)
                results['confidence'] = min(1.0, pressure_gradient / 20.0)
                break
        
        return results
    
    def detect_extreme_precipitation(self, 
                                   precipitation: np.ndarray,
                                   time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Detect extreme precipitation events
        
        Args:
            precipitation: Precipitation rate array
            time_window_hours: Time window for accumulation
            
        Returns:
            Dictionary with extreme precipitation detection results
        """
        # Calculate accumulation
        precip_accum = np.sum(precipitation, axis=0)  # Assuming time is first dimension
        
        results = {
            'extreme_detected': False,
            'max_accumulation': np.max(precip_accum),
            'affected_area_km2': 0.0,
            'intensity_category': None,
            'hotspots': []
        }
        
        # Classify intensity
        max_precip = results['max_accumulation']
        thresholds = self.config.thresholds
        
        if max_precip >= thresholds.extreme_heavy_rain:
            results['extreme_detected'] = True
            results['intensity_category'] = ExtremeIntensity.EXTREME
        elif max_precip >= thresholds.very_heavy_rain:
            results['extreme_detected'] = True
            results['intensity_category'] = ExtremeIntensity.SEVERE
        elif max_precip >= thresholds.heavy_rain:
            results['extreme_detected'] = True
            results['intensity_category'] = ExtremeIntensity.MODERATE
        
        # Find hotspots
        if results['extreme_detected']:
            hotspot_threshold = thresholds.heavy_rain
            hotspots = np.where(precip_accum >= hotspot_threshold)
            results['hotspots'] = list(zip(hotspots[0], hotspots[1]))
            
            # Calculate affected area (assuming 1km grid spacing)
            results['affected_area_km2'] = len(results['hotspots'])
        
        return results
    
    def detect_heat_wave(self, 
                        temperature: np.ndarray,
                        duration_days: int = 3) -> Dict[str, Any]:
        """
        Detect heat wave events
        
        Args:
            temperature: Temperature array [time, lat, lon]
            duration_days: Minimum duration for heat wave
            
        Returns:
            Dictionary with heat wave detection results
        """
        results = {
            'heat_wave_detected': False,
            'max_temperature': np.max(temperature),
            'duration_days': 0,
            'affected_area_km2': 0.0,
            'intensity_category': None
        }
        
        thresholds = self.config.thresholds
        
        # Check for sustained high temperatures
        heat_wave_mask = temperature >= thresholds.heat_wave_moderate
        
        # Count consecutive days above threshold
        for i in range(temperature.shape[1]):
            for j in range(temperature.shape[2]):
                temp_series = temperature[:, i, j]
                consecutive_days = self._count_consecutive_days(
                    temp_series >= thresholds.heat_wave_moderate
                )
                
                if consecutive_days >= duration_days:
                    results['heat_wave_detected'] = True
                    results['duration_days'] = max(results['duration_days'], consecutive_days)
        
        if results['heat_wave_detected']:
            # Classify intensity
            max_temp = results['max_temperature']
            if max_temp >= thresholds.heat_wave_extreme:
                results['intensity_category'] = ExtremeIntensity.EXTREME
            elif max_temp >= thresholds.heat_wave_severe:
                results['intensity_category'] = ExtremeIntensity.SEVERE
            else:
                results['intensity_category'] = ExtremeIntensity.MODERATE
            
            # Calculate affected area
            final_day_mask = heat_wave_mask[-1]  # Last day
            results['affected_area_km2'] = np.sum(final_day_mask)
        
        return results
    
    def _classify_cyclone_intensity(self, max_wind_speed: float) -> str:
        """Classify cyclone intensity based on wind speed"""
        thresholds = self.config.thresholds
        
        if max_wind_speed >= thresholds.cyclone_super:
            return "Super Cyclonic Storm"
        elif max_wind_speed >= thresholds.cyclone_very_severe:
            return "Very Severe Cyclonic Storm"
        elif max_wind_speed >= thresholds.cyclone_severe:
            return "Severe Cyclonic Storm"
        elif max_wind_speed >= thresholds.cyclone_storm:
            return "Cyclonic Storm"
        elif max_wind_speed >= thresholds.cyclone_depression:
            return "Deep Depression"
        else:
            return "Depression"
    
    def _count_consecutive_days(self, mask: np.ndarray) -> int:
        """Count maximum consecutive True values in boolean array"""
        if not np.any(mask):
            return 0
        
        # Find runs of consecutive True values
        diff = np.diff(np.concatenate(([False], mask, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0:
            return 0
        
        return np.max(ends - starts)


class ExtremeWeatherPredictor:
    """Main class for extreme weather prediction"""
    
    def __init__(self, config: ExtremeWeatherConfig):
        self.config = config
        self.cyclone_detector = CycloneDetector()
        self.event_tracker = ExtremeEventTracker(config)
        self.climatology = {}
    
    def predict_extreme_events(self, 
                             forecast_data: Dict[str, np.ndarray],
                             ensemble_forecasts: Optional[List[Dict[str, np.ndarray]]] = None) -> Dict[str, Any]:
        """
        Predict extreme weather events from forecast data
        
        Args:
            forecast_data: Dictionary containing forecast variables
            ensemble_forecasts: Optional ensemble forecast data
            
        Returns:
            Dictionary with extreme event predictions
        """
        predictions = {
            'events': {},
            'probabilities': {},
            'confidence_intervals': {},
            'risk_assessment': {}
        }
        
        # Analyze each requested event type
        for event_type in self.config.event_types:
            if event_type == ExtremeEventType.TROPICAL_CYCLONE:
                result = self._predict_cyclone(forecast_data, ensemble_forecasts)
            elif event_type == ExtremeEventType.EXTREME_PRECIPITATION:
                result = self._predict_extreme_precipitation(forecast_data, ensemble_forecasts)
            elif event_type == ExtremeEventType.HEAT_WAVE:
                result = self._predict_heat_wave(forecast_data, ensemble_forecasts)
            elif event_type == ExtremeEventType.DROUGHT:
                result = self._predict_drought(forecast_data, ensemble_forecasts)
            else:
                result = {'detected': False, 'probability': 0.0}
            
            predictions['events'][event_type.value] = result
        
        # Calculate overall risk assessment
        predictions['risk_assessment'] = self._assess_overall_risk(predictions['events'])
        
        return predictions
    
    def _predict_cyclone(self, 
                        forecast_data: Dict[str, np.ndarray],
                        ensemble_forecasts: Optional[List[Dict[str, np.ndarray]]]) -> Dict[str, Any]:
        """Predict tropical cyclone formation and development"""
        
        # Extract relevant variables
        pressure = forecast_data.get('msl', np.zeros((1, 100, 100)))
        wind_u = forecast_data.get('u10', np.zeros_like(pressure))
        wind_v = forecast_data.get('v10', np.zeros_like(pressure))
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)
        
        # Calculate vorticity
        vorticity = self._calculate_vorticity(wind_u, wind_v)
        
        # Detect cyclone in deterministic forecast
        detection_result = self.event_tracker.detect_tropical_cyclone(
            pressure[0], wind_speed[0], vorticity[0], 
            forecast_data.get('t2m', np.zeros_like(pressure[0]))
        )
        
        # Calculate ensemble probabilities if available
        ensemble_prob = 0.0
        if ensemble_forecasts:
            detections = 0
            for ensemble_member in ensemble_forecasts:
                ens_pressure = ensemble_member.get('msl', np.zeros((1, 100, 100)))
                ens_wind_u = ensemble_member.get('u10', np.zeros_like(ens_pressure))
                ens_wind_v = ensemble_member.get('v10', np.zeros_like(ens_pressure))
                ens_wind_speed = np.sqrt(ens_wind_u**2 + ens_wind_v**2)
                ens_vorticity = self._calculate_vorticity(ens_wind_u, ens_wind_v)
                
                ens_result = self.event_tracker.detect_tropical_cyclone(
                    ens_pressure[0], ens_wind_speed[0], ens_vorticity[0],
                    ensemble_member.get('t2m', np.zeros_like(ens_pressure[0]))
                )
                
                if ens_result['cyclone_detected']:
                    detections += 1
            
            ensemble_prob = detections / len(ensemble_forecasts)
        
        return {
            'detected': detection_result['cyclone_detected'],
            'probability': ensemble_prob if ensemble_forecasts else detection_result['confidence'],
            'intensity_category': detection_result['intensity_category'],
            'max_wind_speed': detection_result['max_wind_speed'],
            'center_location': detection_result['center_location'],
            'minimum_pressure': detection_result['minimum_pressure']
        }
    
    def _predict_extreme_precipitation(self,
                                     forecast_data: Dict[str, np.ndarray],
                                     ensemble_forecasts: Optional[List[Dict[str, np.ndarray]]]) -> Dict[str, Any]:
        """Predict extreme precipitation events"""
        
        precipitation = forecast_data.get('tp', np.zeros((24, 100, 100)))  # 24-hour forecast
        
        # Convert from m to mm
        precipitation_mm = precipitation * 1000
        
        # Detect extreme precipitation
        detection_result = self.event_tracker.detect_extreme_precipitation(precipitation_mm)
        
        # Calculate ensemble probabilities
        ensemble_prob = 0.0
        if ensemble_forecasts:
            detections = 0
            for ensemble_member in ensemble_forecasts:
                ens_precip = ensemble_member.get('tp', np.zeros((24, 100, 100))) * 1000
                ens_result = self.event_tracker.detect_extreme_precipitation(ens_precip)
                
                if ens_result['extreme_detected']:
                    detections += 1
            
            ensemble_prob = detections / len(ensemble_forecasts)
        
        return {
            'detected': detection_result['extreme_detected'],
            'probability': ensemble_prob if ensemble_forecasts else (1.0 if detection_result['extreme_detected'] else 0.0),
            'max_accumulation': detection_result['max_accumulation'],
            'affected_area_km2': detection_result['affected_area_km2'],
            'intensity_category': detection_result['intensity_category'],
            'hotspots': detection_result['hotspots']
        }
    
    def _predict_heat_wave(self,
                          forecast_data: Dict[str, np.ndarray],
                          ensemble_forecasts: Optional[List[Dict[str, np.ndarray]]]) -> Dict[str, Any]:
        """Predict heat wave events"""
        
        temperature = forecast_data.get('t2m', np.zeros((7, 100, 100)))  # 7-day forecast
        
        # Convert from Kelvin to Celsius
        temperature_c = temperature - 273.15
        
        # Detect heat wave
        detection_result = self.event_tracker.detect_heat_wave(temperature_c)
        
        # Calculate ensemble probabilities
        ensemble_prob = 0.0
        if ensemble_forecasts:
            detections = 0
            for ensemble_member in ensemble_forecasts:
                ens_temp = ensemble_member.get('t2m', np.zeros((7, 100, 100))) - 273.15
                ens_result = self.event_tracker.detect_heat_wave(ens_temp)
                
                if ens_result['heat_wave_detected']:
                    detections += 1
            
            ensemble_prob = detections / len(ensemble_forecasts)
        
        return {
            'detected': detection_result['heat_wave_detected'],
            'probability': ensemble_prob if ensemble_forecasts else (1.0 if detection_result['heat_wave_detected'] else 0.0),
            'max_temperature': detection_result['max_temperature'],
            'duration_days': detection_result['duration_days'],
            'affected_area_km2': detection_result['affected_area_km2'],
            'intensity_category': detection_result['intensity_category']
        }
    
    def _predict_drought(self,
                        forecast_data: Dict[str, np.ndarray],
                        ensemble_forecasts: Optional[List[Dict[str, np.ndarray]]]) -> Dict[str, Any]:
        """Predict drought conditions"""
        
        # Simple drought assessment based on precipitation and temperature
        precipitation = forecast_data.get('tp', np.zeros((30, 100, 100))) * 1000  # 30-day
        temperature = forecast_data.get('t2m', np.zeros((30, 100, 100))) - 273.15
        
        # Calculate simple drought index (precipitation deficit + high temperature)
        total_precip = np.sum(precipitation, axis=0)
        avg_temp = np.mean(temperature, axis=0)
        
        # Normalize by climatological values (simplified)
        normal_precip = 150.0  # mm/month for Bangladesh
        normal_temp = 26.0     # °C average
        
        drought_index = (normal_precip - total_precip) / normal_precip + (avg_temp - normal_temp) / 10.0
        
        drought_detected = np.any(drought_index > 0.5)  # Threshold for drought
        max_drought_severity = np.max(drought_index)
        
        return {
            'detected': drought_detected,
            'probability': min(1.0, max(0.0, max_drought_severity)),
            'severity_index': max_drought_severity,
            'affected_area_km2': np.sum(drought_index > 0.5)
        }
    
    def _calculate_vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Calculate relative vorticity from wind components"""
        # Simple finite difference approximation
        du_dy = np.gradient(u, axis=0)
        dv_dx = np.gradient(v, axis=1)
        vorticity = dv_dx - du_dy
        return vorticity
    
    def _assess_overall_risk(self, events: Dict[str, Dict]) -> Dict[str, Any]:
        """Assess overall risk based on all predicted events"""
        
        # Calculate composite risk score
        risk_score = 0.0
        active_events = []
        
        for event_type, event_data in events.items():
            if event_data.get('detected', False):
                active_events.append(event_type)
                
                # Weight different events by severity
                if event_type == 'tropical_cyclone':
                    risk_score += 0.8 * event_data.get('probability', 0.0)
                elif event_type == 'extreme_precipitation':
                    risk_score += 0.6 * event_data.get('probability', 0.0)
                elif event_type == 'heat_wave':
                    risk_score += 0.4 * event_data.get('probability', 0.0)
                elif event_type == 'drought':
                    risk_score += 0.5 * event_data.get('probability', 0.0)
        
        # Normalize risk score
        risk_score = min(1.0, risk_score)
        
        # Categorize risk level
        if risk_score >= 0.8:
            risk_level = "EXTREME"
        elif risk_score >= 0.6:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MODERATE"
        elif risk_score >= 0.2:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'active_events': active_events,
            'recommendations': self._generate_recommendations(active_events, risk_level)
        }
    
    def _generate_recommendations(self, active_events: List[str], risk_level: str) -> List[str]:
        """Generate recommendations based on predicted events and risk level"""
        
        recommendations = []
        
        if 'tropical_cyclone' in active_events:
            recommendations.extend([
                "Activate cyclone early warning systems",
                "Prepare evacuation plans for coastal areas",
                "Secure fishing boats and coastal infrastructure",
                "Stock emergency supplies in cyclone shelters"
            ])
        
        if 'extreme_precipitation' in active_events:
            recommendations.extend([
                "Monitor river levels and dam capacity",
                "Prepare flood management systems",
                "Issue flash flood warnings for urban areas",
                "Clear drainage systems and waterways"
            ])
        
        if 'heat_wave' in active_events:
            recommendations.extend([
                "Issue heat wave advisories",
                "Ensure adequate cooling centers",
                "Monitor vulnerable populations",
                "Implement power demand management"
            ])
        
        if 'drought' in active_events:
            recommendations.extend([
                "Implement water conservation measures",
                "Monitor agricultural water needs",
                "Prepare alternative water sources",
                "Support drought-affected farmers"
            ])
        
        # General recommendations based on risk level
        if risk_level in ["HIGH", "EXTREME"]:
            recommendations.extend([
                "Activate emergency response centers",
                "Coordinate with disaster management authorities",
                "Prepare media communications",
                "Ready emergency response teams"
            ])
        
        return recommendations


def create_extreme_weather_predictor(config: Optional[ExtremeWeatherConfig] = None) -> ExtremeWeatherPredictor:
    """
    Factory function to create an extreme weather predictor
    
    Args:
        config: Optional configuration for the predictor
        
    Returns:
        Configured ExtremeWeatherPredictor instance
    """
    if config is None:
        config = ExtremeWeatherConfig()
    
    return ExtremeWeatherPredictor(config)


# Example usage and testing
if __name__ == "__main__":
    # Create predictor with default configuration
    config = ExtremeWeatherConfig()
    predictor = create_extreme_weather_predictor(config)
    
    # Example forecast data (would come from GraphCast model)
    example_forecast = {
        'msl': np.random.normal(101325, 1000, (1, 100, 100)),  # Sea level pressure
        'u10': np.random.normal(0, 5, (1, 100, 100)),          # U wind component
        'v10': np.random.normal(0, 5, (1, 100, 100)),          # V wind component
        't2m': np.random.normal(298, 5, (7, 100, 100)),        # Temperature
        'tp': np.random.exponential(0.001, (24, 100, 100))     # Precipitation
    }
    
    # Make predictions
    predictions = predictor.predict_extreme_events(example_forecast)
    
    print("Extreme Weather Prediction Results:")
    print("="*50)
    
    for event_type, result in predictions['events'].items():
        print(f"\n{event_type.upper()}:")
        print(f"  Detected: {result['detected']}")
        print(f"  Probability: {result['probability']:.2f}")
    
    print(f"\nOverall Risk Assessment:")
    risk = predictions['risk_assessment']
    print(f"  Risk Level: {risk['risk_level']}")
    print(f"  Risk Score: {risk['risk_score']:.2f}")
    print(f"  Active Events: {', '.join(risk['active_events'])}")
