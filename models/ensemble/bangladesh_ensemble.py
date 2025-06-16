"""
Bangladesh-specific ensemble forecasting system for GraphCast.

This module implements multiple ensemble generation approaches tailored for
Bangladesh's weather challenges, including initial condition perturbations,
model parameter uncertainty, and stochastic physics.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble generation."""
    n_members: int = 50
    perturbation_methods: List[str] = None
    parameter_uncertainty: bool = True
    stochastic_physics: bool = True
    initial_condition_spread: float = 0.01
    model_uncertainty_std: float = 0.005
    
    def __post_init__(self):
        if self.perturbation_methods is None:
            self.perturbation_methods = [
                'initial_conditions',
                'model_parameters',
                'stochastic_physics',
                'boundary_conditions'
            ]


class PerturbationGenerator(ABC):
    """Abstract base class for ensemble perturbation generators."""
    
    @abstractmethod
    def generate_perturbations(self, 
                             base_state: torch.Tensor, 
                             n_members: int) -> torch.Tensor:
        """Generate ensemble perturbations."""
        pass


class InitialConditionPerturber(PerturbationGenerator):
    """Generate perturbations in initial conditions."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.climatological_std = None
        self._load_climatology()
    
    def _load_climatology(self):
        """Load climatological standard deviations for perturbations."""
        # In practice, this would load from historical data
        # For now, using typical values for Bangladesh
        self.climatological_std = {
            'temperature': 2.0,  # 2K
            'pressure': 500.0,   # 5 hPa
            'humidity': 0.1,     # 10%
            'wind_u': 3.0,       # 3 m/s
            'wind_v': 3.0,       # 3 m/s
            'precipitation': 0.5  # Factor for log-normal
        }
    
    def generate_perturbations(self, 
                             base_state: torch.Tensor, 
                             n_members: int) -> torch.Tensor:
        """
        Generate ensemble perturbations using bred vectors approach.
        
        Args:
            base_state: Base atmospheric state [batch, nodes, features]
            n_members: Number of ensemble members
            
        Returns:
            Perturbed ensemble states [n_members, batch, nodes, features]
        """
        device = base_state.device
        batch_size, n_nodes, n_features = base_state.shape
        
        # Initialize ensemble array
        ensemble = torch.zeros(n_members, batch_size, n_nodes, n_features, device=device)
        
        for member in range(n_members):
            perturbed_state = base_state.clone()
            
            # Generate spatially correlated perturbations
            for feat_idx, (var_name, std) in enumerate(self.climatological_std.items()):
                if feat_idx >= n_features:
                    break
                    
                # Generate random field with spatial correlation
                perturbation = self._generate_correlated_field(
                    n_nodes, std, correlation_length=200.0  # 200 km
                )
                
                # Apply perturbation with respect to variable type
                if var_name == 'precipitation':
                    # Log-normal perturbation for precipitation
                    perturbed_state[:, :, feat_idx] *= torch.exp(
                        perturbation * self.config.initial_condition_spread
                    )
                else:
                    # Additive Gaussian perturbation
                    perturbed_state[:, :, feat_idx] += (
                        perturbation * self.config.initial_condition_spread
                    )
            
            ensemble[member] = perturbed_state
        
        return ensemble
    
    def _generate_correlated_field(self, 
                                 n_nodes: int, 
                                 std: float, 
                                 correlation_length: float) -> torch.Tensor:
        """Generate spatially correlated random field."""
        # Simplified implementation - in practice would use more sophisticated
        # spatial correlation based on actual mesh geometry
        base_field = torch.randn(n_nodes) * std
        
        # Apply spatial smoothing to create correlation
        # This is a placeholder for proper spatial correlation
        kernel_size = max(1, int(correlation_length / 100))  # Rough approximation
        if kernel_size > 1:
            # Simple moving average for correlation
            padded = torch.cat([base_field[-kernel_size//2:], 
                              base_field, 
                              base_field[:kernel_size//2]])
            smoothed = torch.zeros_like(base_field)
            for i in range(n_nodes):
                smoothed[i] = padded[i:i+kernel_size].mean()
            return smoothed
        
        return base_field


class ModelParameterPerturber(PerturbationGenerator):
    """Generate perturbations in model parameters."""
    
    def __init__(self, model: nn.Module, config: EnsembleConfig):
        self.model = model
        self.config = config
        self.parameter_samples = []
        self._prepare_parameter_distributions()
    
    def _prepare_parameter_distributions(self):
        """Prepare parameter uncertainty distributions."""
        self.parameter_info = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store original parameter and define uncertainty
                param_info = {
                    'name': name,
                    'original': param.data.clone(),
                    'std': param.data.abs() * self.config.model_uncertainty_std,
                    'shape': param.shape
                }
                self.parameter_info.append(param_info)
    
    def generate_parameter_ensemble(self, n_members: int) -> List[Dict]:
        """Generate ensemble of model parameters."""
        ensemble_params = []
        
        for member in range(n_members):
            member_params = {}
            
            for param_info in self.parameter_info:
                # Generate perturbed parameters
                perturbation = torch.randn_like(param_info['original']) * param_info['std']
                perturbed_param = param_info['original'] + perturbation
                member_params[param_info['name']] = perturbed_param
            
            ensemble_params.append(member_params)
        
        return ensemble_params
    
    def generate_perturbations(self, 
                             base_state: torch.Tensor, 
                             n_members: int) -> torch.Tensor:
        """Apply parameter perturbations to model predictions."""
        # This method would typically be called during inference
        # with different parameter sets
        return self.generate_parameter_ensemble(n_members)


class StochasticPhysicsModule(nn.Module):
    """Implement stochastic physics perturbations."""
    
    def __init__(self, config: EnsembleConfig):
        super().__init__()
        self.config = config
        
        # Stochastic parameterizations for Bangladesh-specific processes
        self.convection_noise = ConvectionNoise()
        self.boundary_layer_noise = BoundaryLayerNoise()
        self.monsoon_variability = MonsoonVariabilityNoise()
    
    def forward(self, state: torch.Tensor, member_id: int = 0) -> torch.Tensor:
        """Apply stochastic physics perturbations."""
        if not self.config.stochastic_physics:
            return state
        
        perturbed_state = state.clone()
        
        # Apply different noise patterns based on member ID
        torch.manual_seed(member_id)  # Ensure reproducible perturbations
        
        # Convective perturbations (important for monsoon)
        conv_noise = self.convection_noise(state)
        perturbed_state = perturbed_state + conv_noise
        
        # Boundary layer perturbations (urban heat island effects)
        bl_noise = self.boundary_layer_noise(state)
        perturbed_state = perturbed_state + bl_noise
        
        # Monsoon-specific variability
        monsoon_noise = self.monsoon_variability(state)
        perturbed_state = perturbed_state + monsoon_noise
        
        return perturbed_state


class ConvectionNoise(nn.Module):
    """Stochastic convection parameterization."""
    
    def __init__(self):
        super().__init__()
        self.noise_scale = 0.01
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Generate convection noise based on atmospheric instability."""
        # Simplified implementation
        # In practice, this would depend on CAPE, wind shear, etc.
        batch_size, n_nodes, n_features = state.shape
        
        # Generate noise proportional to atmospheric instability
        instability_proxy = torch.abs(state[:, :, 0])  # Using temperature as proxy
        noise_magnitude = self.noise_scale * instability_proxy.unsqueeze(-1)
        
        noise = torch.randn_like(state) * noise_magnitude
        return noise


class BoundaryLayerNoise(nn.Module):
    """Stochastic boundary layer parameterization."""
    
    def __init__(self):
        super().__init__()
        self.noise_scale = 0.005
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Generate boundary layer turbulence noise."""
        # Simplified implementation focusing on surface effects
        batch_size, n_nodes, n_features = state.shape
        
        # Surface-concentrated noise (stronger near surface)
        surface_factor = torch.ones(n_features)
        surface_factor[:2] *= 2.0  # Stronger for temperature and humidity
        
        noise = torch.randn_like(state) * self.noise_scale
        noise = noise * surface_factor.view(1, 1, -1)
        
        return noise


class MonsoonVariabilityNoise(nn.Module):
    """Stochastic monsoon variability parameterization."""
    
    def __init__(self):
        super().__init__()
        self.noise_scale = 0.008
        self.seasonal_factor = 1.0
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Generate monsoon-related variability noise."""
        # This would include MJO, monsoon breaks, etc.
        batch_size, n_nodes, n_features = state.shape
        
        # Moisture-dependent noise (stronger during active monsoon)
        moisture_proxy = torch.abs(state[:, :, 2])  # Assuming humidity is feature 2
        noise_magnitude = self.noise_scale * (1 + moisture_proxy.unsqueeze(-1))
        
        noise = torch.randn_like(state) * noise_magnitude * self.seasonal_factor
        return noise


class BanglaGraphCastEnsemble(nn.Module):
    """
    Main ensemble forecasting system for Bangladesh GraphCast.
    
    Combines multiple perturbation approaches:
    1. Initial condition perturbations
    2. Model parameter uncertainty
    3. Stochastic physics
    4. Multi-model ensemble capability
    """
    
    def __init__(self, 
                 base_model: nn.Module, 
                 config: EnsembleConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # Initialize perturbation generators
        self.ic_perturber = InitialConditionPerturber(config)
        self.param_perturber = ModelParameterPerturber(base_model, config)
        self.stochastic_physics = StochasticPhysicsModule(config)
        
        # Ensemble statistics
        self.ensemble_stats = EnsembleStatistics()
    
    def forward(self, 
                initial_conditions: torch.Tensor,
                lead_times: List[int],
                return_members: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate ensemble forecast.
        
        Args:
            initial_conditions: Initial atmospheric state
            lead_times: Forecast lead times (hours)
            return_members: Whether to return individual members
            
        Returns:
            Dictionary containing ensemble mean, spread, and optionally members
        """
        n_members = self.config.n_members
        device = initial_conditions.device
        
        # Generate perturbed initial conditions
        if 'initial_conditions' in self.config.perturbation_methods:
            ic_ensemble = self.ic_perturber.generate_perturbations(
                initial_conditions, n_members
            )
        else:
            ic_ensemble = initial_conditions.unsqueeze(0).repeat(n_members, 1, 1, 1)
        
        # Generate parameter ensemble
        if 'model_parameters' in self.config.perturbation_methods:
            param_ensemble = self.param_perturber.generate_parameter_ensemble(n_members)
        else:
            param_ensemble = [None] * n_members
        
        # Run ensemble forecast
        ensemble_forecasts = []
        
        for member in range(n_members):
            logger.info(f"Running ensemble member {member + 1}/{n_members}")
            
            # Set perturbed parameters if available
            if param_ensemble[member] is not None:
                self._set_model_parameters(param_ensemble[member])
            
            # Run forecast for this member
            member_ic = ic_ensemble[member]
            member_forecast = self._run_member_forecast(
                member_ic, lead_times, member
            )
            
            ensemble_forecasts.append(member_forecast)
            
            # Reset parameters
            if param_ensemble[member] is not None:
                self._reset_model_parameters()
        
        # Stack ensemble forecasts
        ensemble_tensor = torch.stack(ensemble_forecasts, dim=0)
        
        # Compute ensemble statistics
        ensemble_stats = self.ensemble_stats.compute_statistics(ensemble_tensor)
        
        if return_members:
            ensemble_stats['members'] = ensemble_tensor
        
        return ensemble_stats
    
    def _run_member_forecast(self, 
                           initial_conditions: torch.Tensor,
                           lead_times: List[int],
                           member_id: int) -> torch.Tensor:
        """Run forecast for a single ensemble member."""
        current_state = initial_conditions
        forecasts = []
        
        for lead_time in lead_times:
            # Apply stochastic physics if enabled
            if 'stochastic_physics' in self.config.perturbation_methods:
                current_state = self.stochastic_physics(current_state, member_id)
            
            # Run model forward
            next_state = self.base_model(current_state)
            forecasts.append(next_state)
            current_state = next_state
        
        return torch.stack(forecasts, dim=0)
    
    def _set_model_parameters(self, params: Dict[str, torch.Tensor]):
        """Set model parameters for ensemble member."""
        for name, param in self.base_model.named_parameters():
            if name in params:
                param.data = params[name]
    
    def _reset_model_parameters(self):
        """Reset model parameters to original values."""
        for param_info in self.param_perturber.parameter_info:
            param = dict(self.base_model.named_parameters())[param_info['name']]
            param.data = param_info['original']


class EnsembleStatistics:
    """Compute ensemble statistics and diagnostics."""
    
    def compute_statistics(self, ensemble: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute ensemble statistics.
        
        Args:
            ensemble: Ensemble forecasts [n_members, lead_times, batch, nodes, features]
            
        Returns:
            Dictionary with ensemble mean, spread, percentiles, etc.
        """
        stats = {}
        
        # Basic statistics
        stats['mean'] = ensemble.mean(dim=0)
        stats['std'] = ensemble.std(dim=0)
        stats['var'] = ensemble.var(dim=0)
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'p{p}'] = torch.quantile(ensemble, p/100.0, dim=0)
        
        # Ensemble spread metrics
        stats['range'] = ensemble.max(dim=0)[0] - ensemble.min(dim=0)[0]
        stats['iqr'] = stats['p75'] - stats['p25']  # Interquartile range
        
        # Probability of exceedance for critical thresholds
        stats['prob_heavy_rain'] = self._probability_of_exceedance(
            ensemble, threshold=50.0, variable_idx=5  # Assuming precip is index 5
        )
        stats['prob_strong_wind'] = self._probability_of_exceedance(
            ensemble, threshold=17.2, variable_idx=3  # Assuming wind is index 3
        )
        
        return stats
    
    def _probability_of_exceedance(self, 
                                 ensemble: torch.Tensor,
                                 threshold: float,
                                 variable_idx: int) -> torch.Tensor:
        """Compute probability of exceeding threshold."""
        exceedance = ensemble[:, :, :, :, variable_idx] > threshold
        return exceedance.float().mean(dim=0)


class EnsembleCalibration:
    """Ensemble post-processing and calibration."""
    
    def __init__(self):
        self.bias_correction = None
        self.reliability_correction = None
    
    def fit_calibration(self, 
                       ensemble_forecasts: torch.Tensor,
                       observations: torch.Tensor):
        """Fit ensemble calibration parameters."""
        # Implement ensemble model output statistics (EMOS)
        # or other calibration methods
        pass
    
    def apply_calibration(self, 
                         raw_ensemble: torch.Tensor) -> torch.Tensor:
        """Apply calibration to raw ensemble."""
        # Apply learned bias and reliability corrections
        calibrated = raw_ensemble.clone()
        
        if self.bias_correction is not None:
            calibrated = calibrated - self.bias_correction
        
        if self.reliability_correction is not None:
            calibrated = calibrated * self.reliability_correction
        
        return calibrated


# Example usage and configuration
def create_bangladesh_ensemble(base_model: nn.Module) -> BanglaGraphCastEnsemble:
    """Create configured ensemble system for Bangladesh."""
    config = EnsembleConfig(
        n_members=50,
        perturbation_methods=[
            'initial_conditions',
            'model_parameters', 
            'stochastic_physics'
        ],
        parameter_uncertainty=True,
        stochastic_physics=True,
        initial_condition_spread=0.01,
        model_uncertainty_std=0.005
    )
    
    ensemble_system = BanglaGraphCastEnsemble(base_model, config)
    return ensemble_system


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock base model for testing
    class MockGraphCast(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create ensemble system
    base_model = MockGraphCast()
    ensemble = create_bangladesh_ensemble(base_model)
    
    # Mock initial conditions
    initial_conditions = torch.randn(1, 100, 10)  # [batch, nodes, features]
    lead_times = [6, 12, 24, 48, 72]  # hours
    
    # Generate ensemble forecast
    forecast_stats = ensemble(
        initial_conditions, 
        lead_times, 
        return_members=False
    )
    
    print(f"Ensemble mean shape: {forecast_stats['mean'].shape}")
    print(f"Ensemble std shape: {forecast_stats['std'].shape}")
    print("Ensemble generation completed successfully!")
