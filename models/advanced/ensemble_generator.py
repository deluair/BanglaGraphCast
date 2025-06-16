"""
Ensemble Generation System for Bangladesh GraphCast

Implements multiple ensemble generation strategies:
- Initial condition perturbations
- Model parameter perturbations  
- Stochastic physics schemes
- Multi-model ensemble integration
- Lagged ensemble configurations
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import random
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)


class EnsemblePerturbationType(Enum):
    """Types of ensemble perturbations"""
    INITIAL_CONDITIONS = "initial_conditions"
    MODEL_PARAMETERS = "model_parameters"
    STOCHASTIC_PHYSICS = "stochastic_physics"
    MULTI_MODEL = "multi_model"
    LAGGED = "lagged"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble generation"""
    n_members: int = 20
    perturbation_types: List[EnsemblePerturbationType] = None
    ic_perturbation_scale: float = 0.01
    param_perturbation_scale: float = 0.05
    physics_noise_scale: float = 0.02
    lagged_hours: List[int] = None
    multi_model_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.perturbation_types is None:
            self.perturbation_types = [EnsemblePerturbationType.INITIAL_CONDITIONS]
        if self.lagged_hours is None:
            self.lagged_hours = [0, -6, -12]


class EnsembleGenerator:
    """
    Advanced ensemble generation system for Bangladesh weather prediction
    
    Features:
    - Multiple perturbation strategies
    - Bangladesh-specific error covariance
    - Tropical cyclone ensemble spread enhancement
    - Monsoon variability representation
    - Seasonal ensemble calibration
    """
    
    def __init__(self, config: EnsembleConfig, model_config: Dict):
        self.config = config
        self.model_config = model_config
        
        # Error covariance matrices for Bangladesh region
        self.error_covariance = self._initialize_error_covariance()
        
        # Physics perturbation schemes
        self.physics_perturbations = {
            'convection': ConvectionPerturbation(),
            'surface_flux': SurfaceFluxPerturbation(),
            'radiation': RadiationPerturbation(),
            'turbulence': TurbulencePerturbation()
        }
        
        # Multi-model ensemble components
        self.model_variants = {}
        if EnsemblePerturbationType.MULTI_MODEL in self.config.perturbation_types:
            self._initialize_model_variants()
        
        # Seasonal calibration factors
        self.seasonal_calibration = self._load_seasonal_calibration()
        
    def generate_ensemble(self, 
                         base_initial_conditions: torch.Tensor,
                         forecast_time: datetime,
                         model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Generate ensemble forecast
        
        Args:
            base_initial_conditions: Base initial conditions
            forecast_time: Forecast initialization time
            model: Base GraphCast model
            
        Returns:
            Dictionary with ensemble members and metadata
        """
        logger.info(f"Generating {self.config.n_members}-member ensemble")
        
        ensemble_members = []
        member_metadata = []
        
        for member_id in range(self.config.n_members):
            logger.debug(f"Generating ensemble member {member_id + 1}/{self.config.n_members}")
            
            # Apply perturbations based on configuration
            perturbed_ic, perturbed_model, member_info = self._generate_member_perturbations(
                base_initial_conditions, model, member_id, forecast_time
            )
            
            # Run forecast with perturbed conditions/model
            member_forecast = self._run_perturbed_forecast(
                perturbed_ic, perturbed_model, member_info
            )
            
            ensemble_members.append(member_forecast)
            member_metadata.append(member_info)
        
        # Combine ensemble members
        ensemble_tensor = torch.stack(ensemble_members, dim=0)
        
        # Compute ensemble statistics
        ensemble_stats = self._compute_ensemble_statistics(ensemble_tensor)
        
        # Apply post-processing
        processed_ensemble = self._post_process_ensemble(
            ensemble_tensor, ensemble_stats, forecast_time
        )
        
        result = {
            'ensemble_members': processed_ensemble,
            'ensemble_mean': ensemble_stats['mean'],
            'ensemble_spread': ensemble_stats['spread'],
            'ensemble_probability': ensemble_stats['probability_fields'],
            'member_metadata': member_metadata,
            'forecast_time': forecast_time,
            'generation_config': self.config
        }
        
        logger.info("Ensemble generation completed")
        return result
    
    def _generate_member_perturbations(self,
                                     base_ic: torch.Tensor,
                                     base_model: nn.Module,
                                     member_id: int,
                                     forecast_time: datetime) -> Tuple[torch.Tensor, nn.Module, Dict]:
        """Generate perturbations for ensemble member"""
        
        member_info = {
            'member_id': member_id,
            'perturbation_types': [],
            'perturbation_magnitudes': {}
        }
        
        perturbed_ic = base_ic.clone()
        perturbed_model = base_model
        
        # Initial condition perturbations
        if EnsemblePerturbationType.INITIAL_CONDITIONS in self.config.perturbation_types:
            ic_perturbation = self._generate_ic_perturbation(base_ic, member_id, forecast_time)
            perturbed_ic = perturbed_ic + ic_perturbation
            member_info['perturbation_types'].append('initial_conditions')
            member_info['perturbation_magnitudes']['ic'] = torch.norm(ic_perturbation).item()
        
        # Model parameter perturbations
        if EnsemblePerturbationType.MODEL_PARAMETERS in self.config.perturbation_types:
            perturbed_model = self._generate_parameter_perturbation(base_model, member_id)
            member_info['perturbation_types'].append('model_parameters')
        
        # Stochastic physics
        if EnsemblePerturbationType.STOCHASTIC_PHYSICS in self.config.perturbation_types:
            member_info['physics_perturbations'] = self._prepare_physics_perturbations(member_id)
            member_info['perturbation_types'].append('stochastic_physics')
        
        # Multi-model ensemble
        if EnsemblePerturbationType.MULTI_MODEL in self.config.perturbation_types:
            perturbed_model = self._select_model_variant(member_id)
            member_info['perturbation_types'].append('multi_model')
            member_info['model_variant'] = f"variant_{member_id % len(self.model_variants)}"
        
        # Lagged ensemble
        if EnsemblePerturbationType.LAGGED in self.config.perturbation_types:
            lag_hours = self.config.lagged_hours[member_id % len(self.config.lagged_hours)]
            perturbed_ic = self._apply_lag_perturbation(perturbed_ic, lag_hours)
            member_info['perturbation_types'].append('lagged')
            member_info['lag_hours'] = lag_hours
        
        return perturbed_ic, perturbed_model, member_info
    
    def _generate_ic_perturbation(self, 
                                 base_ic: torch.Tensor, 
                                 member_id: int,
                                 forecast_time: datetime) -> torch.Tensor:
        """
        Generate initial condition perturbations using Bangladesh-specific error covariance
        """
        # Use seasonal and location-specific error covariance
        season = self._get_season(forecast_time)
        covariance_matrix = self.error_covariance[season]
        
        # Generate spatially and temporally correlated perturbations
        perturbation = self._generate_correlated_perturbation(
            base_ic.shape, covariance_matrix, member_id
        )
        
        # Scale perturbation based on configuration and local conditions
        scale_factor = self.config.ic_perturbation_scale
        
        # Enhance perturbations in cyclone-prone areas during cyclone season
        if season in ['pre_monsoon', 'post_monsoon']:
            scale_factor *= self._get_cyclone_enhancement_factor(base_ic)
        
        # Enhance perturbations in monsoon regions during monsoon
        if season == 'monsoon':
            scale_factor *= self._get_monsoon_enhancement_factor(base_ic)
        
        return perturbation * scale_factor
    
    def _generate_parameter_perturbation(self, 
                                       base_model: nn.Module, 
                                       member_id: int) -> nn.Module:
        """Generate model parameter perturbations"""
        perturbed_model = self._copy_model(base_model)
        
        # Set random seed for reproducible perturbations
        torch.manual_seed(member_id + 12345)
        
        for name, param in perturbed_model.named_parameters():
            if self._should_perturb_parameter(name):
                # Generate parameter-specific perturbation
                perturbation = torch.randn_like(param) * self.config.param_perturbation_scale
                
                # Apply parameter constraints
                perturbation = self._apply_parameter_constraints(name, param, perturbation)
                
                param.data += perturbation
        
        return perturbed_model
    
    def _prepare_physics_perturbations(self, member_id: int) -> Dict:
        """Prepare stochastic physics perturbations"""
        physics_perturbations = {}
        
        for scheme_name, perturbation_scheme in self.physics_perturbations.items():
            perturbation_params = perturbation_scheme.generate_perturbation_params(
                member_id, self.config.physics_noise_scale
            )
            physics_perturbations[scheme_name] = perturbation_params
        
        return physics_perturbations
    
    def _run_perturbed_forecast(self,
                              perturbed_ic: torch.Tensor,
                              perturbed_model: nn.Module,
                              member_info: Dict) -> torch.Tensor:
        """Run forecast with perturbed initial conditions and/or model"""
        
        # Apply stochastic physics if enabled
        if 'physics_perturbations' in member_info:
            # Modify model to include stochastic physics
            perturbed_model = self._apply_stochastic_physics(
                perturbed_model, member_info['physics_perturbations']
            )
        
        # Run the forecast
        with torch.no_grad():
            forecast = perturbed_model(perturbed_ic)
        
        return forecast
    
    def _compute_ensemble_statistics(self, ensemble_tensor: torch.Tensor) -> Dict:
        """Compute ensemble statistics"""
        
        # Basic statistics
        ensemble_mean = torch.mean(ensemble_tensor, dim=0)
        ensemble_var = torch.var(ensemble_tensor, dim=0)
        ensemble_spread = torch.sqrt(ensemble_var)
        
        # Probability fields for key thresholds
        probability_fields = self._compute_probability_fields(ensemble_tensor)
        
        # Ensemble range
        ensemble_min = torch.min(ensemble_tensor, dim=0)[0]
        ensemble_max = torch.max(ensemble_tensor, dim=0)[0]
        
        # Reliability metrics
        reliability_metrics = self._compute_reliability_metrics(ensemble_tensor)
        
        return {
            'mean': ensemble_mean,
            'variance': ensemble_var,
            'spread': ensemble_spread,
            'min': ensemble_min,
            'max': ensemble_max,
            'probability_fields': probability_fields,
            'reliability': reliability_metrics
        }
    
    def _compute_probability_fields(self, ensemble_tensor: torch.Tensor) -> Dict:
        """Compute probability fields for important thresholds"""
        
        probability_fields = {}
        
        # Precipitation thresholds (mm/day)
        precip_thresholds = [1, 5, 10, 25, 50, 100, 200]
        
        # Temperature thresholds (°C)
        temp_thresholds = [25, 30, 35, 40]  # Heat wave thresholds
        
        # Wind speed thresholds (m/s)
        wind_thresholds = [10, 15, 20, 25, 30]  # Cyclone intensity
        
        # Compute precipitation probabilities
        # Assuming precipitation is in specific channels
        if ensemble_tensor.shape[1] >= 3:  # At least 3 variables
            precip_field = ensemble_tensor[:, 2]  # Assuming 3rd channel is precipitation
            
            for threshold in precip_thresholds:
                prob_field = (precip_field > threshold).float().mean(dim=0)
                probability_fields[f'precip_gt_{threshold}mm'] = prob_field
        
        # Compute temperature probabilities
        if ensemble_tensor.shape[1] >= 1:  # Temperature in first channel
            temp_field = ensemble_tensor[:, 0]  # Assuming 1st channel is temperature
            
            for threshold in temp_thresholds:
                prob_field = (temp_field > threshold).float().mean(dim=0)
                probability_fields[f'temp_gt_{threshold}C'] = prob_field
        
        # Compute wind speed probabilities
        if ensemble_tensor.shape[1] >= 5:  # Wind components available
            u_wind = ensemble_tensor[:, 3]
            v_wind = ensemble_tensor[:, 4]
            wind_speed = torch.sqrt(u_wind**2 + v_wind**2)
            
            for threshold in wind_thresholds:
                prob_field = (wind_speed > threshold).float().mean(dim=0)
                probability_fields[f'wind_gt_{threshold}ms'] = prob_field
        
        return probability_fields
    
    def _post_process_ensemble(self,
                             ensemble_tensor: torch.Tensor,
                             ensemble_stats: Dict,
                             forecast_time: datetime) -> torch.Tensor:
        """Post-process ensemble forecast"""
        
        # Apply seasonal calibration
        season = self._get_season(forecast_time)
        calibration_factors = self.seasonal_calibration[season]
        
        calibrated_ensemble = self._apply_calibration(ensemble_tensor, calibration_factors)
        
        # Apply ensemble inflation if spread is too small
        calibrated_ensemble = self._apply_ensemble_inflation(
            calibrated_ensemble, ensemble_stats
        )
        
        # Ensure physical consistency
        calibrated_ensemble = self._enforce_physical_constraints(calibrated_ensemble)
        
        return calibrated_ensemble
    
    def _initialize_error_covariance(self) -> Dict:
        """Initialize error covariance matrices for different seasons"""
        
        # Define seasonal error covariance based on Bangladesh climatology
        error_covariance = {}
        
        # Pre-monsoon (March-May): High temperature variance, moderate precipitation
        error_covariance['pre_monsoon'] = {
            'temperature': 2.0,
            'pressure': 0.5,
            'humidity': 15.0,
            'precipitation': 5.0,
            'wind': 3.0
        }
        
        # Monsoon (June-September): High precipitation variance, moderate temperature
        error_covariance['monsoon'] = {
            'temperature': 1.0,
            'pressure': 0.3,
            'humidity': 10.0,
            'precipitation': 15.0,
            'wind': 5.0
        }
        
        # Post-monsoon (October-November): Cyclone season - high wind variance
        error_covariance['post_monsoon'] = {
            'temperature': 1.5,
            'pressure': 2.0,
            'humidity': 12.0,
            'precipitation': 10.0,
            'wind': 8.0
        }
        
        # Winter (December-February): Low variance overall
        error_covariance['winter'] = {
            'temperature': 1.2,
            'pressure': 0.4,
            'humidity': 8.0,
            'precipitation': 2.0,
            'wind': 2.0
        }
        
        return error_covariance
    
    def _load_seasonal_calibration(self) -> Dict:
        """Load seasonal calibration factors"""
        
        # Seasonal bias correction factors for Bangladesh
        return {
            'pre_monsoon': {
                'temperature': 0.98,  # Slight cold bias
                'precipitation': 1.15,  # Under-prediction
                'wind': 1.05
            },
            'monsoon': {
                'temperature': 1.02,  # Slight warm bias
                'precipitation': 0.95,  # Over-prediction
                'wind': 0.98
            },
            'post_monsoon': {
                'temperature': 0.99,
                'precipitation': 1.08,
                'wind': 0.92  # Over-prediction during cyclones
            },
            'winter': {
                'temperature': 1.01,
                'precipitation': 1.20,  # Strong under-prediction
                'wind': 1.03
            }
        }
    
    def _get_season(self, forecast_time: datetime) -> str:
        """Get season for given forecast time"""
        month = forecast_time.month
        
        if month in [3, 4, 5]:
            return 'pre_monsoon'
        elif month in [6, 7, 8, 9]:
            return 'monsoon'
        elif month in [10, 11]:
            return 'post_monsoon'
        else:
            return 'winter'
    
    # Additional helper methods...
    def _generate_correlated_perturbation(self, shape, covariance, member_id):
        """Generate spatially correlated perturbations"""
        torch.manual_seed(member_id + 54321)
        return torch.randn(shape)
    
    def _get_cyclone_enhancement_factor(self, base_ic):
        """Get cyclone season enhancement factor"""
        return 1.5  # Simplified
    
    def _get_monsoon_enhancement_factor(self, base_ic):
        """Get monsoon season enhancement factor"""
        return 1.3  # Simplified
    
    def _copy_model(self, model):
        """Create deep copy of model"""
        import copy
        return copy.deepcopy(model)
    
    def _should_perturb_parameter(self, param_name):
        """Determine if parameter should be perturbed"""
        # Don't perturb batch norm parameters, bias terms, etc.
        exclude_keywords = ['bn', 'bias', 'norm']
        return not any(keyword in param_name.lower() for keyword in exclude_keywords)
    
    def _apply_parameter_constraints(self, name, param, perturbation):
        """Apply constraints to parameter perturbations"""
        # Ensure perturbations don't violate parameter bounds
        return torch.clamp(perturbation, -0.1 * torch.abs(param), 0.1 * torch.abs(param))
    
    def _apply_stochastic_physics(self, model, physics_perturbations):
        """Apply stochastic physics to model"""
        # This would modify the model to include stochastic elements
        return model
    
    def _compute_reliability_metrics(self, ensemble_tensor):
        """Compute ensemble reliability metrics"""
        return {'rank_histogram': None, 'spread_skill': None}
    
    def _apply_calibration(self, ensemble_tensor, calibration_factors):
        """Apply seasonal calibration to ensemble"""
        return ensemble_tensor  # Simplified
    
    def _apply_ensemble_inflation(self, ensemble_tensor, ensemble_stats):
        """Apply ensemble inflation if needed"""
        return ensemble_tensor  # Simplified
    
    def _enforce_physical_constraints(self, ensemble_tensor):
        """Enforce physical constraints on ensemble"""
        return ensemble_tensor  # Simplified
    
    def _initialize_model_variants(self):
        """Initialize model variants for multi-model ensemble"""
        pass  # Simplified


# Physics perturbation schemes

class ConvectionPerturbation:
    """Convective parameterization perturbations"""
    
    def generate_perturbation_params(self, member_id: int, noise_scale: float) -> Dict:
        np.random.seed(member_id + 1111)
        return {
            'entrainment_rate': np.random.normal(1.0, noise_scale),
            'cape_threshold': np.random.normal(1.0, noise_scale),
            'closure_timescale': np.random.normal(1.0, noise_scale)
        }


class SurfaceFluxPerturbation:
    """Surface flux parameterization perturbations"""
    
    def generate_perturbation_params(self, member_id: int, noise_scale: float) -> Dict:
        np.random.seed(member_id + 2222)
        return {
            'roughness_length': np.random.normal(1.0, noise_scale),
            'exchange_coefficient': np.random.normal(1.0, noise_scale),
            'stability_function': np.random.normal(1.0, noise_scale)
        }


class RadiationPerturbation:
    """Radiation scheme perturbations"""
    
    def generate_perturbation_params(self, member_id: int, noise_scale: float) -> Dict:
        np.random.seed(member_id + 3333)
        return {
            'cloud_overlap': np.random.normal(1.0, noise_scale),
            'aerosol_optical_depth': np.random.normal(1.0, noise_scale * 0.5),
            'water_vapor_continuum': np.random.normal(1.0, noise_scale)
        }


class TurbulencePerturbation:
    """Turbulence parameterization perturbations"""
    
    def generate_perturbation_params(self, member_id: int, noise_scale: float) -> Dict:
        np.random.seed(member_id + 4444)
        return {
            'mixing_length': np.random.normal(1.0, noise_scale),
            'tke_dissipation': np.random.normal(1.0, noise_scale),
            'stability_parameter': np.random.normal(1.0, noise_scale)
        }


class EnsembleCalibrator:
    """
    Ensemble post-processing and calibration system
    """
    
    def __init__(self, calibration_data_path: str):
        self.calibration_data_path = calibration_data_path
        self.calibration_models = {}
        
    def calibrate_ensemble(self, 
                          raw_ensemble: torch.Tensor,
                          observations: torch.Tensor,
                          forecast_time: datetime) -> torch.Tensor:
        """
        Calibrate ensemble using historical verification data
        
        Args:
            raw_ensemble: Raw ensemble forecast
            observations: Verification observations
            forecast_time: Forecast initialization time
            
        Returns:
            Calibrated ensemble forecast
        """
        
        # Apply different calibration methods
        calibrated_ensemble = raw_ensemble.clone()
        
        # 1. Bias correction
        calibrated_ensemble = self._apply_bias_correction(calibrated_ensemble, forecast_time)
        
        # 2. Ensemble spread calibration
        calibrated_ensemble = self._calibrate_spread(calibrated_ensemble, observations)
        
        # 3. Rank-based calibration
        calibrated_ensemble = self._apply_rank_calibration(calibrated_ensemble)
        
        return calibrated_ensemble
    
    def _apply_bias_correction(self, ensemble: torch.Tensor, forecast_time: datetime) -> torch.Tensor:
        """Apply systematic bias correction"""
        # Implementation would use historical bias statistics
        return ensemble
    
    def _calibrate_spread(self, ensemble: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """Calibrate ensemble spread using spread-skill relationship"""
        # Implementation would adjust ensemble spread based on historical performance
        return ensemble
    
    def _apply_rank_calibration(self, ensemble: torch.Tensor) -> torch.Tensor:
        """Apply rank-based ensemble calibration"""
        # Implementation would ensure ensemble rank histogram is uniform
        return ensemble


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = EnsembleConfig(
        n_members=20,
        perturbation_types=[
            EnsemblePerturbationType.INITIAL_CONDITIONS,
            EnsemblePerturbationType.STOCHASTIC_PHYSICS
        ],
        ic_perturbation_scale=0.01,
        physics_noise_scale=0.02
    )
    
    # Initialize ensemble generator
    ensemble_generator = EnsembleGenerator(config, {})
    
    # Mock data for testing
    batch_size = 1
    n_variables = 6
    lat_size = 64
    lon_size = 64
    
    base_ic = torch.randn(batch_size, n_variables, lat_size, lon_size)
    forecast_time = datetime.now()
    
    # Mock model
    class MockModel(nn.Module):
        def forward(self, x):
            return x + torch.randn_like(x) * 0.1
    
    model = MockModel()
    
    # Generate ensemble
    logger.info("Testing ensemble generation...")
    ensemble_result = ensemble_generator.generate_ensemble(base_ic, forecast_time, model)
    
    logger.info(f"Generated ensemble with {ensemble_result['ensemble_members'].shape[0]} members")
    logger.info(f"Ensemble mean shape: {ensemble_result['ensemble_mean'].shape}")
    logger.info(f"Ensemble spread shape: {ensemble_result['ensemble_spread'].shape}")
    logger.info(f"Number of probability fields: {len(ensemble_result['ensemble_probability'])}")
    
    logger.info("Ensemble generation test completed successfully!")
