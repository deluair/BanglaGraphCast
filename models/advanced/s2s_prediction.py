"""
Subseasonal-to-Seasonal (S2S) Prediction System for Bangladesh

Implements extended-range forecasting capabilities:
- Multi-week to multi-month predictions (up to 6 months)
- Teleconnection pattern analysis (ENSO, IOD, MJO)
- Seasonal monsoon prediction
- Agricultural planning forecasts
- Water resource management forecasts
- Climate trend integration
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import xarray as xr
from pathlib import Path

logger = logging.getLogger(__name__)


class S2SForecastType(Enum):
    """Types of S2S forecasts"""
    MONSOON_ONSET = "monsoon_onset"
    MONSOON_WITHDRAWAL = "monsoon_withdrawal"
    SEASONAL_RAINFALL = "seasonal_rainfall"
    TEMPERATURE_ANOMALY = "temperature_anomaly"
    CYCLONE_ACTIVITY = "cyclone_activity"
    DROUGHT_PROBABILITY = "drought_probability"
    FLOOD_RISK = "flood_risk"


class TeleconnectionIndex(Enum):
    """Major teleconnection indices affecting Bangladesh"""
    ENSO = "enso"  # El Niño Southern Oscillation
    IOD = "iod"    # Indian Ocean Dipole
    MJO = "mjo"    # Madden-Julian Oscillation
    AO = "ao"      # Arctic Oscillation
    SAM = "sam"    # Southern Annular Mode


@dataclass
class S2SConfig:
    """Configuration for S2S prediction system"""
    forecast_types: List[S2SForecastType] = None
    lead_times_weeks: List[int] = None
    teleconnection_indices: List[TeleconnectionIndex] = None
    ensemble_size: int = 50
    calibration_period_years: int = 30
    skill_threshold: float = 0.6
    use_climate_trends: bool = True
    
    def __post_init__(self):
        if self.forecast_types is None:
            self.forecast_types = [S2SForecastType.SEASONAL_RAINFALL, 
                                 S2SForecastType.TEMPERATURE_ANOMALY]
        if self.lead_times_weeks is None:
            self.lead_times_weeks = [2, 4, 8, 12, 16, 20, 24]
        if self.teleconnection_indices is None:
            self.teleconnection_indices = [TeleconnectionIndex.ENSO, 
                                         TeleconnectionIndex.IOD, 
                                         TeleconnectionIndex.MJO]


class S2SPredictionSystem:
    """
    Subseasonal-to-Seasonal prediction system for Bangladesh
    
    Features:
    - Multi-timescale prediction framework
    - Teleconnection pattern integration
    - Bangladesh-specific climate drivers
    - Agricultural and water resource applications
    - Uncertainty quantification
    - Skill assessment and calibration
    """
    
    def __init__(self, config: S2SConfig, data_path: str):
        self.config = config
        self.data_path = Path(data_path)
        
        # Initialize prediction models
        self.prediction_models = {}
        self._initialize_prediction_models()
        
        # Teleconnection analyzers
        self.teleconnection_analyzers = {}
        self._initialize_teleconnection_analyzers()
        
        # Climate trend model
        self.climate_trend_model = ClimateTrendModel() if config.use_climate_trends else None
        
        # Skill assessment system
        self.skill_assessor = SkillAssessment()
        
        # Calibration system
        self.calibrator = S2SCalibration()
        
        # Load historical data for training and calibration
        self.historical_data = self._load_historical_data()
        
        # Load climatology
        self.climatology = self._compute_climatology()
        
    def generate_s2s_forecast(self, 
                             initialization_time: datetime,
                             short_range_forecast: torch.Tensor) -> Dict:
        """
        Generate subseasonal-to-seasonal forecast
        
        Args:
            initialization_time: Forecast initialization time
            short_range_forecast: Short-range forecast (up to 2 weeks) for initialization
            
        Returns:
            S2S forecast products and metadata
        """
        logger.info(f"Generating S2S forecast initialized at {initialization_time}")
        
        # Analyze teleconnection patterns
        teleconnection_state = self._analyze_teleconnections(initialization_time)
        
        # Analyze current climate state
        climate_state = self._analyze_climate_state(initialization_time, short_range_forecast)
        
        # Generate forecasts for each type and lead time
        s2s_forecasts = {}
        
        for forecast_type in self.config.forecast_types:
            logger.debug(f"Generating {forecast_type.value} forecasts")
            
            forecast_results = {}
            
            for lead_time_weeks in self.config.lead_times_weeks:
                # Generate forecast for this lead time
                forecast = self._generate_forecast(
                    forecast_type, 
                    lead_time_weeks,
                    initialization_time,
                    climate_state,
                    teleconnection_state
                )
                
                # Apply calibration
                calibrated_forecast = self._apply_calibration(
                    forecast, forecast_type, lead_time_weeks, initialization_time
                )
                
                # Assess skill
                skill_metrics = self._assess_forecast_skill(
                    forecast_type, lead_time_weeks, initialization_time
                )
                
                forecast_results[f"week_{lead_time_weeks}"] = {
                    'forecast': calibrated_forecast,
                    'skill_metrics': skill_metrics,
                    'uncertainty': self._quantify_uncertainty(calibrated_forecast),
                    'valid_time': initialization_time + timedelta(weeks=lead_time_weeks)
                }
            
            s2s_forecasts[forecast_type.value] = forecast_results
        
        # Generate specialized products
        specialized_products = self._generate_specialized_products(
            s2s_forecasts, initialization_time, teleconnection_state
        )
        
        # Generate summary and confidence assessment
        forecast_summary = self._generate_forecast_summary(
            s2s_forecasts, specialized_products, teleconnection_state
        )
        
        return {
            'initialization_time': initialization_time,
            'forecasts': s2s_forecasts,
            'specialized_products': specialized_products,
            'teleconnection_state': teleconnection_state,
            'climate_state': climate_state,
            'forecast_summary': forecast_summary,
            'metadata': {
                'forecast_types': [ft.value for ft in self.config.forecast_types],
                'lead_times_weeks': self.config.lead_times_weeks,
                'ensemble_size': self.config.ensemble_size
            }
        }
    
    def _analyze_teleconnections(self, initialization_time: datetime) -> Dict:
        """Analyze current state of teleconnection patterns"""
        
        teleconnection_state = {}
        
        for index in self.config.teleconnection_indices:
            analyzer = self.teleconnection_analyzers[index]
            
            # Get current index value and forecast
            current_value = analyzer.get_current_value(initialization_time)
            forecast_evolution = analyzer.forecast_evolution(initialization_time)
            phase_classification = analyzer.classify_phase(current_value)
            impact_assessment = analyzer.assess_bangladesh_impact(
                current_value, forecast_evolution
            )
            
            teleconnection_state[index.value] = {
                'current_value': current_value,
                'forecast_evolution': forecast_evolution,
                'phase': phase_classification,
                'bangladesh_impact': impact_assessment,
                'confidence': analyzer.get_forecast_confidence()
            }
        
        # Compute composite teleconnection impact
        composite_impact = self._compute_composite_teleconnection_impact(teleconnection_state)
        teleconnection_state['composite_impact'] = composite_impact
        
        return teleconnection_state
    
    def _analyze_climate_state(self, 
                              initialization_time: datetime,
                              short_range_forecast: torch.Tensor) -> Dict:
        """Analyze current climate state and trends"""
        
        # Extract key climate indicators from short-range forecast
        climate_indicators = self._extract_climate_indicators(short_range_forecast)
        
        # Analyze seasonal cycle context
        seasonal_context = self._analyze_seasonal_context(initialization_time)
        
        # Analyze climate trends if enabled
        climate_trends = None
        if self.climate_trend_model:
            climate_trends = self.climate_trend_model.analyze_trends(
                initialization_time, self.historical_data
            )
        
        # Identify climate anomalies
        climate_anomalies = self._identify_climate_anomalies(
            climate_indicators, seasonal_context
        )
        
        return {
            'climate_indicators': climate_indicators,
            'seasonal_context': seasonal_context,
            'climate_trends': climate_trends,
            'climate_anomalies': climate_anomalies,
            'analysis_time': initialization_time
        }
    
    def _generate_forecast(self,
                          forecast_type: S2SForecastType,
                          lead_time_weeks: int,
                          initialization_time: datetime,
                          climate_state: Dict,
                          teleconnection_state: Dict) -> torch.Tensor:
        """Generate forecast for specific type and lead time"""
        
        # Get appropriate prediction model
        model = self.prediction_models[forecast_type][f"week_{lead_time_weeks}"]
        
        # Prepare input features
        input_features = self._prepare_prediction_features(
            climate_state, teleconnection_state, initialization_time, lead_time_weeks
        )
        
        # Generate ensemble forecast
        with torch.no_grad():
            forecast_ensemble = model(input_features)
        
        return forecast_ensemble
    
    def _generate_specialized_products(self,
                                     s2s_forecasts: Dict,
                                     initialization_time: datetime,
                                     teleconnection_state: Dict) -> Dict:
        """Generate specialized S2S products for specific applications"""
        
        specialized_products = {}
        
        # Agricultural planning forecasts
        agricultural_forecast = self._generate_agricultural_forecast(
            s2s_forecasts, initialization_time
        )
        specialized_products['agricultural'] = agricultural_forecast
        
        # Water resource forecasts
        water_resource_forecast = self._generate_water_resource_forecast(
            s2s_forecasts, initialization_time
        )
        specialized_products['water_resources'] = water_resource_forecast
        
        # Monsoon forecasts
        monsoon_forecast = self._generate_monsoon_forecast(
            s2s_forecasts, initialization_time, teleconnection_state
        )
        specialized_products['monsoon'] = monsoon_forecast
        
        # Cyclone activity forecast
        cyclone_forecast = self._generate_cyclone_activity_forecast(
            s2s_forecasts, initialization_time, teleconnection_state
        )
        specialized_products['cyclone_activity'] = cyclone_forecast
        
        # Energy sector forecast
        energy_forecast = self._generate_energy_sector_forecast(
            s2s_forecasts, initialization_time
        )
        specialized_products['energy'] = energy_forecast
        
        return specialized_products
    
    def _generate_agricultural_forecast(self,
                                      s2s_forecasts: Dict,
                                      initialization_time: datetime) -> Dict:
        """Generate agricultural planning forecasts"""
        
        # Extract relevant forecasts
        rainfall_forecast = s2s_forecasts.get('seasonal_rainfall', {})
        temperature_forecast = s2s_forecasts.get('temperature_anomaly', {})
        
        agricultural_forecast = {
            'planting_recommendations': {},
            'crop_suitability': {},
            'irrigation_needs': {},
            'pest_disease_risk': {},
            'harvest_timing': {}
        }
        
        # Analyze planting windows
        for crop in ['rice', 'wheat', 'jute', 'sugarcane']:
            planting_window = self._analyze_planting_window(
                crop, rainfall_forecast, temperature_forecast, initialization_time
            )
            agricultural_forecast['planting_recommendations'][crop] = planting_window
        
        # Assess crop suitability for different regions
        for region in ['northern', 'central', 'southern', 'coastal']:
            suitability = self._assess_crop_suitability(
                region, rainfall_forecast, temperature_forecast
            )
            agricultural_forecast['crop_suitability'][region] = suitability
        
        return agricultural_forecast
    
    def _generate_water_resource_forecast(self,
                                        s2s_forecasts: Dict,
                                        initialization_time: datetime) -> Dict:
        """Generate water resource management forecasts"""
        
        rainfall_forecast = s2s_forecasts.get('seasonal_rainfall', {})
        
        water_forecast = {
            'reservoir_management': {},
            'groundwater_outlook': {},
            'river_flow_forecast': {},
            'flood_drought_risk': {}
        }
        
        # Major river systems in Bangladesh
        rivers = ['ganges', 'brahmaputra', 'meghna', 'jamuna']
        
        for river in rivers:
            flow_forecast = self._forecast_river_flow(river, rainfall_forecast)
            water_forecast['river_flow_forecast'][river] = flow_forecast
        
        # Reservoir management recommendations
        reservoirs = ['kaptai', 'teesta']
        for reservoir in reservoirs:
            management_plan = self._generate_reservoir_management_plan(
                reservoir, rainfall_forecast
            )
            water_forecast['reservoir_management'][reservoir] = management_plan
        
        return water_forecast
    
    def _generate_monsoon_forecast(self,
                                 s2s_forecasts: Dict,
                                 initialization_time: datetime,
                                 teleconnection_state: Dict) -> Dict:
        """Generate detailed monsoon forecasts"""
        
        month = initialization_time.month
        
        monsoon_forecast = {
            'onset_forecast': None,
            'withdrawal_forecast': None,
            'seasonal_characteristics': {},
            'break_periods': {},
            'intensity_forecast': {}
        }
        
        # Monsoon onset forecast (March-May initialization)
        if month in [3, 4, 5]:
            onset_forecast = self._forecast_monsoon_onset(
                teleconnection_state, initialization_time
            )
            monsoon_forecast['onset_forecast'] = onset_forecast
        
        # Monsoon withdrawal forecast (August-October initialization)
        if month in [8, 9, 10]:
            withdrawal_forecast = self._forecast_monsoon_withdrawal(
                teleconnection_state, initialization_time
            )
            monsoon_forecast['withdrawal_forecast'] = withdrawal_forecast
        
        # Seasonal characteristics
        seasonal_chars = self._forecast_monsoon_characteristics(
            s2s_forecasts, teleconnection_state
        )
        monsoon_forecast['seasonal_characteristics'] = seasonal_chars
        
        return monsoon_forecast
    
    def _initialize_prediction_models(self):
        """Initialize S2S prediction models"""
        
        self.prediction_models = {}
        
        for forecast_type in self.config.forecast_types:
            self.prediction_models[forecast_type] = {}
            
            for lead_time_weeks in self.config.lead_times_weeks:
                # Create model for this forecast type and lead time
                model = S2SPredictionModel(
                    forecast_type=forecast_type,
                    lead_time_weeks=lead_time_weeks,
                    ensemble_size=self.config.ensemble_size
                )
                
                self.prediction_models[forecast_type][f"week_{lead_time_weeks}"] = model
    
    def _initialize_teleconnection_analyzers(self):
        """Initialize teleconnection pattern analyzers"""
        
        for index in self.config.teleconnection_indices:
            if index == TeleconnectionIndex.ENSO:
                self.teleconnection_analyzers[index] = ENSOAnalyzer()
            elif index == TeleconnectionIndex.IOD:
                self.teleconnection_analyzers[index] = IODAnalyzer()
            elif index == TeleconnectionIndex.MJO:
                self.teleconnection_analyzers[index] = MJOAnalyzer()
            elif index == TeleconnectionIndex.AO:
                self.teleconnection_analyzers[index] = AOAnalyzer()
            elif index == TeleconnectionIndex.SAM:
                self.teleconnection_analyzers[index] = SAMAnalyzer()
    
    # Additional helper methods with simplified implementations
    def _load_historical_data(self):
        """Load historical data for training and calibration"""
        return {}  # Simplified
    
    def _compute_climatology(self):
        """Compute climatological statistics"""
        return {}  # Simplified
    
    def _apply_calibration(self, forecast, forecast_type, lead_time_weeks, init_time):
        """Apply calibration to forecast"""
        return forecast  # Simplified
    
    def _assess_forecast_skill(self, forecast_type, lead_time_weeks, init_time):
        """Assess forecast skill"""
        return {'correlation': 0.7, 'rmse': 1.5}  # Simplified
    
    def _quantify_uncertainty(self, forecast):
        """Quantify forecast uncertainty"""
        return torch.std(forecast, dim=0)  # Simplified
    
    def _generate_forecast_summary(self, forecasts, products, teleconnections):
        """Generate forecast summary"""
        return {'summary': 'Forecast generated successfully'}  # Simplified
    
    # Placeholder implementations for other complex methods
    def _extract_climate_indicators(self, forecast):
        return {}
    
    def _analyze_seasonal_context(self, time):
        return {}
    
    def _identify_climate_anomalies(self, indicators, context):
        return {}
    
    def _prepare_prediction_features(self, climate_state, telecon_state, time, lead_time):
        return torch.randn(1, 100)  # Simplified
    
    def _compute_composite_teleconnection_impact(self, telecon_state):
        return {'impact_strength': 'moderate', 'confidence': 0.7}
    
    def _analyze_planting_window(self, crop, rainfall, temperature, time):
        return {'recommended_start': time + timedelta(days=30), 'confidence': 0.8}
    
    def _assess_crop_suitability(self, region, rainfall, temperature):
        return {'rice': 'high', 'wheat': 'medium', 'jute': 'low'}
    
    def _forecast_river_flow(self, river, rainfall_forecast):
        return {'flow_category': 'normal', 'confidence': 0.7}
    
    def _generate_reservoir_management_plan(self, reservoir, rainfall_forecast):
        return {'action': 'maintain_current_level', 'confidence': 0.8}
    
    def _forecast_monsoon_onset(self, telecon_state, time):
        return {'onset_date': time + timedelta(days=60), 'confidence': 0.7}
    
    def _forecast_monsoon_withdrawal(self, telecon_state, time):
        return {'withdrawal_date': time + timedelta(days=45), 'confidence': 0.6}
    
    def _forecast_monsoon_characteristics(self, forecasts, telecon_state):
        return {'total_rainfall': 'above_normal', 'number_of_breaks': 2}


class S2SPredictionModel(nn.Module):
    """
    Neural network model for S2S prediction
    """
    
    def __init__(self, 
                 forecast_type: S2SForecastType,
                 lead_time_weeks: int,
                 ensemble_size: int):
        super().__init__()
        
        self.forecast_type = forecast_type
        self.lead_time_weeks = lead_time_weeks
        self.ensemble_size = ensemble_size
        
        # Input features: climate state + teleconnections
        input_dim = 100  # Simplified
        hidden_dim = 256
        output_dim = 64 * 64  # Spatial grid
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Ensemble prediction heads
        self.ensemble_heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(ensemble_size)
        ])
        
    def forward(self, x):
        """Forward pass to generate ensemble predictions"""
        
        # Encode input features
        encoded = self.encoder(x)
        
        # Generate ensemble predictions
        ensemble_predictions = []
        for head in self.ensemble_heads:
            prediction = head(encoded)
            ensemble_predictions.append(prediction)
        
        # Stack ensemble members
        ensemble_output = torch.stack(ensemble_predictions, dim=0)
        
        return ensemble_output


# Teleconnection analyzers

class ENSOAnalyzer:
    """El Niño Southern Oscillation analyzer"""
    
    def get_current_value(self, time: datetime) -> float:
        # Mock implementation
        return np.random.normal(0, 1)
    
    def forecast_evolution(self, time: datetime) -> Dict:
        # Mock implementation
        return {'3_month': 0.5, '6_month': 0.2}
    
    def classify_phase(self, value: float) -> str:
        if value > 0.5:
            return 'El Niño'
        elif value < -0.5:
            return 'La Niña'
        else:
            return 'Neutral'
    
    def assess_bangladesh_impact(self, value: float, forecast: Dict) -> Dict:
        # Mock implementation
        return {
            'rainfall_impact': 'below_normal' if value > 0.5 else 'above_normal',
            'temperature_impact': 'above_normal' if value > 0.5 else 'below_normal',
            'confidence': 0.7
        }
    
    def get_forecast_confidence(self) -> float:
        return 0.75


class IODAnalyzer:
    """Indian Ocean Dipole analyzer"""
    
    def get_current_value(self, time: datetime) -> float:
        return np.random.normal(0, 0.5)
    
    def forecast_evolution(self, time: datetime) -> Dict:
        return {'3_month': 0.2, '6_month': 0.0}
    
    def classify_phase(self, value: float) -> str:
        if value > 0.4:
            return 'Positive IOD'
        elif value < -0.4:
            return 'Negative IOD'
        else:
            return 'Neutral'
    
    def assess_bangladesh_impact(self, value: float, forecast: Dict) -> Dict:
        return {
            'rainfall_impact': 'above_normal' if value > 0.4 else 'below_normal',
            'confidence': 0.6
        }
    
    def get_forecast_confidence(self) -> float:
        return 0.65


class MJOAnalyzer:
    """Madden-Julian Oscillation analyzer"""
    
    def get_current_value(self, time: datetime) -> float:
        return np.random.uniform(1, 8)  # MJO phase
    
    def forecast_evolution(self, time: datetime) -> Dict:
        return {'2_week': 4.5, '4_week': 7.2}
    
    def classify_phase(self, value: float) -> str:
        phase = int(value)
        phase_names = {
            1: 'Indian Ocean',
            2: 'Maritime Continent',
            3: 'Western Pacific',
            4: 'Eastern Pacific',
            5: 'Atlantic/Africa',
            6: 'Indian Ocean Return',
            7: 'Maritime Continent Return',
            8: 'Western Pacific Return'
        }
        return phase_names.get(phase, 'Undefined')
    
    def assess_bangladesh_impact(self, value: float, forecast: Dict) -> Dict:
        phase = int(value)
        # MJO phases 2-4 typically enhance Bangladesh monsoon
        if phase in [2, 3, 4]:
            impact = 'enhanced_convection'
        else:
            impact = 'suppressed_convection'
        
        return {
            'convection_impact': impact,
            'confidence': 0.8
        }
    
    def get_forecast_confidence(self) -> float:
        return 0.70


class AOAnalyzer:
    """Arctic Oscillation analyzer"""
    
    def get_current_value(self, time: datetime) -> float:
        return np.random.normal(0, 1)
    
    def forecast_evolution(self, time: datetime) -> Dict:
        return {'1_month': 0.3, '3_month': 0.1}
    
    def classify_phase(self, value: float) -> str:
        return 'Positive' if value > 0 else 'Negative'
    
    def assess_bangladesh_impact(self, value: float, forecast: Dict) -> Dict:
        return {
            'winter_temperature_impact': 'warmer' if value > 0 else 'cooler',
            'confidence': 0.5
        }
    
    def get_forecast_confidence(self) -> float:
        return 0.55


class SAMAnalyzer:
    """Southern Annular Mode analyzer"""
    
    def get_current_value(self, time: datetime) -> float:
        return np.random.normal(0, 1)
    
    def forecast_evolution(self, time: datetime) -> Dict:
        return {'1_month': 0.2, '3_month': 0.0}
    
    def classify_phase(self, value: float) -> str:
        return 'Positive' if value > 0 else 'Negative'
    
    def assess_bangladesh_impact(self, value: float, forecast: Dict) -> Dict:
        return {
            'indirect_impact': 'minimal',
            'confidence': 0.3
        }
    
    def get_forecast_confidence(self) -> float:
        return 0.40


class ClimateTrendModel:
    """Climate trend analysis model"""
    
    def analyze_trends(self, time: datetime, historical_data: Dict) -> Dict:
        return {
            'temperature_trend': 0.02,  # °C/year
            'precipitation_trend': -0.5,  # mm/year
            'confidence': 0.9
        }


class SkillAssessment:
    """S2S forecast skill assessment"""
    
    def assess_skill(self, forecast, observations, lead_time):
        return {
            'correlation': 0.7,
            'rmse': 1.5,
            'skill_score': 0.6
        }


class S2SCalibration:
    """S2S forecast calibration system"""
    
    def calibrate(self, forecast, forecast_type, lead_time, season):
        return forecast  # Simplified


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = S2SConfig(
        forecast_types=[S2SForecastType.SEASONAL_RAINFALL, 
                       S2SForecastType.TEMPERATURE_ANOMALY],
        lead_times_weeks=[4, 8, 12, 16],
        teleconnection_indices=[TeleconnectionIndex.ENSO, 
                               TeleconnectionIndex.IOD,
                               TeleconnectionIndex.MJO],
        ensemble_size=20
    )
    
    # Initialize S2S system
    s2s_system = S2SPredictionSystem(config, "/path/to/data")
    
    # Mock short-range forecast
    short_range_forecast = torch.randn(1, 6, 64, 64)
    initialization_time = datetime.now()
    
    # Generate S2S forecast
    logger.info("Testing S2S forecast generation...")
    s2s_result = s2s_system.generate_s2s_forecast(
        initialization_time, short_range_forecast
    )
    
    logger.info(f"Generated S2S forecast for {len(s2s_result['forecasts'])} forecast types")
    logger.info(f"Lead times: {s2s_result['metadata']['lead_times_weeks']} weeks")
    logger.info(f"Specialized products: {list(s2s_result['specialized_products'].keys())}")
    logger.info(f"Teleconnection indices: {list(s2s_result['teleconnection_state'].keys())}")
    
    logger.info("S2S prediction test completed successfully!")
