"""
Climate downscaling and projection system for Bangladesh.

This module provides statistical and dynamical downscaling capabilities
for climate projections, focusing on high-resolution climate impacts
for Bangladesh's vulnerable regions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DownscalingMethod(Enum):
    """Downscaling methodology options."""
    STATISTICAL = "statistical"
    DYNAMICAL = "dynamical"
    HYBRID = "hybrid"
    SUPER_RESOLUTION = "super_resolution"


class ClimateScenario(Enum):
    """Climate scenarios for projections."""
    SSP1_19 = "ssp1_1.9"    # 1.5°C pathway
    SSP1_26 = "ssp1_2.6"    # 2°C pathway
    SSP2_45 = "ssp2_4.5"    # Middle pathway
    SSP3_70 = "ssp3_7.0"    # High emissions
    SSP5_85 = "ssp5_8.5"    # Very high emissions


@dataclass
class DownscalingConfig:
    """Configuration for climate downscaling."""
    # Spatial configuration
    target_resolution_km: float = 1.0  # Target resolution
    coarse_resolution_km: float = 25.0  # Input resolution
    spatial_domain: Dict[str, float] = None  # lat/lon bounds
    
    # Temporal configuration
    time_aggregation: str = "daily"  # daily, monthly, seasonal
    bias_correction: bool = True
    trend_preservation: bool = True
    
    # Downscaling method
    method: DownscalingMethod = DownscalingMethod.HYBRID
    
    # Climate scenarios
    scenarios: List[ClimateScenario] = None
    
    # Variables to downscale
    variables: List[str] = None
    
    # Bangladesh-specific options
    urban_heat_island: bool = True
    coastal_effects: bool = True
    topographic_effects: bool = True
    land_use_effects: bool = True
    
    def __post_init__(self):
        if self.spatial_domain is None:
            # Bangladesh domain with buffer
            self.spatial_domain = {
                "lat_min": 20.5, "lat_max": 26.8,
                "lon_min": 87.0, "lon_max": 93.0
            }
        
        if self.scenarios is None:
            self.scenarios = [
                ClimateScenario.SSP1_26,
                ClimateScenario.SSP2_45,
                ClimateScenario.SSP5_85
            ]
        
        if self.variables is None:
            self.variables = [
                "temperature_2m", "precipitation", "humidity",
                "wind_speed", "wind_direction", "pressure",
                "solar_radiation", "cloud_cover"
            ]


class StatisticalDownscaler(nn.Module):
    """Statistical downscaling using deep learning."""
    
    def __init__(self, config: DownscalingConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale feature extractor
        self.coarse_encoder = self._build_coarse_encoder()
        
        # High-resolution decoder
        self.fine_decoder = self._build_fine_decoder()
        
        # Variable-specific processing
        self.variable_processors = nn.ModuleDict({
            var: VariableProcessor(var) for var in config.variables
        })
        
        # Bias correction module
        if config.bias_correction:
            self.bias_corrector = BiasCorrector()
    
    def _build_coarse_encoder(self) -> nn.Module:
        """Build encoder for coarse-resolution input."""
        return nn.Sequential(
            # Multi-scale convolutions
            nn.Conv2d(len(self.config.variables), 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            
            # Attention mechanism for important patterns
            SpatialAttention(128),
            
            # Further encoding
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU()
        )
    
    def _build_fine_decoder(self) -> nn.Module:
        """Build decoder for high-resolution output."""
        scale_factor = int(self.config.coarse_resolution_km / self.config.target_resolution_km)
        
        layers = []
        in_channels = 512
        
        # Progressive upsampling
        while scale_factor > 1:
            layers.extend([
                nn.ConvTranspose2d(in_channels, in_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
                nn.ReLU()
            ])
            in_channels //= 2
            scale_factor //= 2
        
        # Final output layer
        layers.append(nn.Conv2d(in_channels, len(self.config.variables), 3, padding=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, 
                coarse_input: torch.Tensor,
                auxiliary_data: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Downscale coarse-resolution input to high resolution.
        
        Args:
            coarse_input: Coarse resolution climate data [batch, vars, lat, lon]
            auxiliary_data: Additional data (topography, land use, etc.)
            
        Returns:
            High-resolution downscaled data
        """
        # Encode coarse-resolution features
        encoded = self.coarse_encoder(coarse_input)
        
        # Incorporate auxiliary data if available
        if auxiliary_data is not None:
            encoded = self._incorporate_auxiliary_data(encoded, auxiliary_data)
        
        # Decode to high resolution
        downscaled = self.fine_decoder(encoded)
        
        # Apply variable-specific processing
        processed_vars = []
        for i, var in enumerate(self.config.variables):
            var_data = downscaled[:, i:i+1, :, :]
            processed_var = self.variable_processors[var](var_data)
            processed_vars.append(processed_var)
        
        result = torch.cat(processed_vars, dim=1)
        
        # Apply bias correction if enabled
        if self.config.bias_correction and hasattr(self, 'bias_corrector'):
            result = self.bias_corrector(result, coarse_input)
        
        return result
    
    def _incorporate_auxiliary_data(self, 
                                  encoded: torch.Tensor,
                                  auxiliary_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Incorporate auxiliary data into encoding."""
        # Simple concatenation approach
        # In practice, would use more sophisticated fusion
        aux_features = []
        for key, data in auxiliary_data.items():
            # Resize to match encoded resolution
            resized = nn.functional.interpolate(
                data, size=encoded.shape[2:], mode='bilinear', align_corners=False
            )
            aux_features.append(resized)
        
        if aux_features:
            aux_combined = torch.cat(aux_features, dim=1)
            encoded = torch.cat([encoded, aux_combined], dim=1)
        
        return encoded


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for important climate patterns."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights


class VariableProcessor(nn.Module):
    """Variable-specific post-processing."""
    
    def __init__(self, variable_name: str):
        super().__init__()
        self.variable_name = variable_name
        
        # Variable-specific constraints and transformations
        if variable_name == "precipitation":
            self.activation = self._precipitation_activation
        elif variable_name in ["humidity"]:
            self.activation = self._bounded_activation
        elif variable_name in ["wind_speed"]:
            self.activation = self._non_negative_activation
        else:
            self.activation = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)
    
    def _precipitation_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure non-negative precipitation with realistic distribution."""
        return torch.relu(x)
    
    def _bounded_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Bound values between 0 and 1 for humidity."""
        return torch.sigmoid(x)
    
    def _non_negative_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure non-negative values."""
        return torch.relu(x)


class BiasCorrector(nn.Module):
    """Bias correction for downscaled outputs."""
    
    def __init__(self):
        super().__init__()
        self.correction_network = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),  # Input + output channels
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1)   # Output channels
        )
    
    def forward(self, 
                downscaled: torch.Tensor,
                coarse_input: torch.Tensor) -> torch.Tensor:
        """Apply bias correction based on input-output relationship."""
        # Resize coarse input to match downscaled resolution
        coarse_resized = nn.functional.interpolate(
            coarse_input, size=downscaled.shape[2:], 
            mode='bilinear', align_corners=False
        )
        
        # Concatenate for correction network
        combined = torch.cat([downscaled, coarse_resized], dim=1)
        correction = self.correction_network(combined)
        
        return downscaled + correction


class DynamicalDownscaler(nn.Module):
    """Dynamical downscaling with physics constraints."""
    
    def __init__(self, config: DownscalingConfig):
        super().__init__()
        self.config = config
        
        # Physics-informed components
        self.topographic_processor = TopographicProcessor()
        self.land_use_processor = LandUseProcessor()
        self.coastal_processor = CoastalProcessor()
        
        # Dynamical constraints
        self.physics_constraints = PhysicsConstraints()
    
    def forward(self, 
                coarse_input: torch.Tensor,
                topography: torch.Tensor,
                land_use: torch.Tensor,
                coastline_distance: torch.Tensor) -> torch.Tensor:
        """Apply dynamical downscaling with physics constraints."""
        
        # Apply topographic effects
        topo_adjusted = self.topographic_processor(coarse_input, topography)
        
        # Apply land use effects
        landuse_adjusted = self.land_use_processor(topo_adjusted, land_use)
        
        # Apply coastal effects
        coastal_adjusted = self.coastal_processor(landuse_adjusted, coastline_distance)
        
        # Enforce physics constraints
        physics_constrained = self.physics_constraints(coastal_adjusted)
        
        return physics_constrained


class TopographicProcessor(nn.Module):
    """Process topographic effects on climate variables."""
    
    def __init__(self):
        super().__init__()
        self.elevation_effects = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1),  # Climate vars + elevation
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 8, 3, padding=1)  # Climate vars output
        )
    
    def forward(self, 
                climate_data: torch.Tensor,
                elevation: torch.Tensor) -> torch.Tensor:
        """Apply topographic modifications."""
        # Combine climate data with elevation
        combined = torch.cat([climate_data, elevation], dim=1)
        
        # Apply elevation-dependent adjustments
        adjustments = self.elevation_effects(combined)
        
        return climate_data + adjustments


class LandUseProcessor(nn.Module):
    """Process land use effects on local climate."""
    
    def __init__(self):
        super().__init__()
        # Different effects for different land use types
        self.urban_effects = UrbanHeatIslandProcessor()
        self.forest_effects = ForestCanopyProcessor()
        self.agriculture_effects = CroplandProcessor()
        self.water_effects = WaterBodyProcessor()
    
    def forward(self, 
                climate_data: torch.Tensor,
                land_use: torch.Tensor) -> torch.Tensor:
        """Apply land use specific modifications."""
        result = climate_data.clone()
        
        # Extract land use categories (assuming one-hot encoding)
        urban_mask = land_use[:, 0:1, :, :]
        forest_mask = land_use[:, 1:2, :, :]
        agriculture_mask = land_use[:, 2:3, :, :]
        water_mask = land_use[:, 3:4, :, :]
        
        # Apply effects based on land use
        result = result + self.urban_effects(climate_data) * urban_mask
        result = result + self.forest_effects(climate_data) * forest_mask
        result = result + self.agriculture_effects(climate_data) * agriculture_mask
        result = result + self.water_effects(climate_data) * water_mask
        
        return result


class UrbanHeatIslandProcessor(nn.Module):
    """Model urban heat island effects."""
    
    def __init__(self):
        super().__init__()
        self.uhi_network = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1)
        )
    
    def forward(self, climate_data: torch.Tensor) -> torch.Tensor:
        """Apply urban heat island effects."""
        uhi_effect = self.uhi_network(climate_data)
        
        # Stronger effects on temperature and humidity
        uhi_effect[:, 0, :, :] *= 2.0  # Temperature enhancement
        uhi_effect[:, 2, :, :] *= -0.5  # Humidity reduction
        
        return uhi_effect


class ForestCanopyProcessor(nn.Module):
    """Model forest canopy effects."""
    
    def __init__(self):
        super().__init__()
        self.canopy_network = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1)
        )
    
    def forward(self, climate_data: torch.Tensor) -> torch.Tensor:
        """Apply forest canopy effects."""
        canopy_effect = self.canopy_network(climate_data)
        
        # Cooling and humidifying effects
        canopy_effect[:, 0, :, :] *= -0.5  # Temperature reduction
        canopy_effect[:, 2, :, :] *= 0.3   # Humidity increase
        canopy_effect[:, 3, :, :] *= -0.2  # Wind speed reduction
        
        return canopy_effect


class CroplandProcessor(nn.Module):
    """Model agricultural land effects."""
    
    def __init__(self):
        super().__init__()
        self.crop_network = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1)
        )
    
    def forward(self, climate_data: torch.Tensor) -> torch.Tensor:
        """Apply agricultural land effects."""
        return self.crop_network(climate_data) * 0.5  # Moderate effects


class WaterBodyProcessor(nn.Module):
    """Model water body effects."""
    
    def __init__(self):
        super().__init__()
        self.water_network = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1)
        )
    
    def forward(self, climate_data: torch.Tensor) -> torch.Tensor:
        """Apply water body effects."""
        water_effect = self.water_network(climate_data)
        
        # Moderating effects
        water_effect[:, 0, :, :] *= -0.3  # Temperature moderation
        water_effect[:, 2, :, :] *= 0.5   # Humidity increase
        
        return water_effect


class CoastalProcessor(nn.Module):
    """Process coastal effects on climate."""
    
    def __init__(self):
        super().__init__()
        self.coastal_network = nn.Sequential(
            nn.Conv2d(9, 32, 3, padding=1),  # Climate + distance
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1)
        )
    
    def forward(self, 
                climate_data: torch.Tensor,
                coastline_distance: torch.Tensor) -> torch.Tensor:
        """Apply coastal modifications."""
        # Combine with distance to coast
        combined = torch.cat([climate_data, coastline_distance], dim=1)
        
        # Apply distance-dependent coastal effects
        coastal_effect = self.coastal_network(combined)
        
        return climate_data + coastal_effect


class PhysicsConstraints(nn.Module):
    """Enforce physical constraints on downscaled variables."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, climate_data: torch.Tensor) -> torch.Tensor:
        """Apply physics constraints."""
        constrained = climate_data.clone()
        
        # Temperature constraints (reasonable bounds)
        constrained[:, 0, :, :] = torch.clamp(constrained[:, 0, :, :], -50, 60)
        
        # Precipitation constraints (non-negative)
        constrained[:, 1, :, :] = torch.relu(constrained[:, 1, :, :])
        
        # Humidity constraints (0-100%)
        constrained[:, 2, :, :] = torch.clamp(constrained[:, 2, :, :], 0, 100)
        
        # Wind speed constraints (non-negative, reasonable max)
        constrained[:, 3, :, :] = torch.clamp(constrained[:, 3, :, :], 0, 100)
        
        # Pressure constraints
        constrained[:, 5, :, :] = torch.clamp(constrained[:, 5, :, :], 800, 1100)
        
        return constrained


class ClimateProjectionSystem(nn.Module):
    """
    Complete climate projection and downscaling system for Bangladesh.
    
    Combines statistical and dynamical downscaling with climate projections
    for multiple scenarios and time periods.
    """
    
    def __init__(self, config: DownscalingConfig):
        super().__init__()
        self.config = config
        
        # Downscaling modules
        if config.method in [DownscalingMethod.STATISTICAL, DownscalingMethod.HYBRID]:
            self.statistical_downscaler = StatisticalDownscaler(config)
        
        if config.method in [DownscalingMethod.DYNAMICAL, DownscalingMethod.HYBRID]:
            self.dynamical_downscaler = DynamicalDownscaler(config)
        
        # Trend preservation
        if config.trend_preservation:
            self.trend_preserver = TrendPreserver()
        
        # Uncertainty quantification
        self.uncertainty_estimator = DownscalingUncertaintyEstimator()
    
    def forward(self, 
                coarse_projections: Dict[str, torch.Tensor],
                auxiliary_data: Dict[str, torch.Tensor],
                scenario: ClimateScenario,
                time_period: str) -> Dict[str, torch.Tensor]:
        """
        Generate high-resolution climate projections.
        
        Args:
            coarse_projections: Coarse-resolution climate model output
            auxiliary_data: Topography, land use, coastline data
            scenario: Climate scenario (SSP)
            time_period: Time period (e.g., "2030-2050")
            
        Returns:
            High-resolution downscaled projections with uncertainty
        """
        results = {}
        
        for var_name, coarse_data in coarse_projections.items():
            if var_name in self.config.variables:
                # Apply appropriate downscaling method
                if self.config.method == DownscalingMethod.STATISTICAL:
                    downscaled = self.statistical_downscaler(coarse_data, auxiliary_data)
                
                elif self.config.method == DownscalingMethod.DYNAMICAL:
                    downscaled = self.dynamical_downscaler(
                        coarse_data,
                        auxiliary_data["topography"],
                        auxiliary_data["land_use"],
                        auxiliary_data["coastline_distance"]
                    )
                
                elif self.config.method == DownscalingMethod.HYBRID:
                    # Combine statistical and dynamical approaches
                    stat_result = self.statistical_downscaler(coarse_data, auxiliary_data)
                    dyn_result = self.dynamical_downscaler(
                        coarse_data,
                        auxiliary_data["topography"],
                        auxiliary_data["land_use"],
                        auxiliary_data["coastline_distance"]
                    )
                    downscaled = 0.6 * stat_result + 0.4 * dyn_result
                
                # Preserve long-term trends if required
                if self.config.trend_preservation:
                    downscaled = self.trend_preserver(downscaled, coarse_data)
                
                # Estimate uncertainty
                uncertainty = self.uncertainty_estimator(
                    downscaled, coarse_data, scenario, time_period
                )
                
                results[var_name] = {
                    "downscaled": downscaled,
                    "uncertainty": uncertainty,
                    "scenario": scenario.value,
                    "time_period": time_period
                }
        
        return results


class TrendPreserver(nn.Module):
    """Preserve large-scale trends in downscaled data."""
    
    def __init__(self):
        super().__init__()
        self.trend_network = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1)
        )
    
    def forward(self, 
                downscaled: torch.Tensor,
                coarse_data: torch.Tensor) -> torch.Tensor:
        """Preserve trends from coarse to downscaled data."""
        # Upscale coarse data
        coarse_upscaled = nn.functional.interpolate(
            coarse_data, size=downscaled.shape[2:],
            mode='bilinear', align_corners=False
        )
        
        # Compute trend adjustment
        combined = torch.cat([downscaled, coarse_upscaled], dim=1)
        trend_adjustment = self.trend_network(combined)
        
        return downscaled + trend_adjustment


class DownscalingUncertaintyEstimator(nn.Module):
    """Estimate uncertainty in downscaled projections."""
    
    def __init__(self):
        super().__init__()
        self.uncertainty_network = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self,
                downscaled: torch.Tensor,
                coarse_data: torch.Tensor,
                scenario: ClimateScenario,
                time_period: str) -> torch.Tensor:
        """Estimate downscaling uncertainty."""
        # Base uncertainty from downscaling process
        base_uncertainty = self.uncertainty_network(downscaled)
        
        # Scenario-dependent scaling
        scenario_factors = {
            ClimateScenario.SSP1_19: 1.0,
            ClimateScenario.SSP1_26: 1.1,
            ClimateScenario.SSP2_45: 1.3,
            ClimateScenario.SSP3_70: 1.6,
            ClimateScenario.SSP5_85: 2.0
        }
        
        scenario_factor = scenario_factors.get(scenario, 1.5)
        
        return base_uncertainty * scenario_factor


# Configuration and usage
def create_bangladesh_downscaling_system() -> ClimateProjectionSystem:
    """Create configured downscaling system for Bangladesh."""
    config = DownscalingConfig(
        target_resolution_km=1.0,
        coarse_resolution_km=25.0,
        spatial_domain={
            "lat_min": 20.5, "lat_max": 26.8,
            "lon_min": 87.0, "lon_max": 93.0
        },
        time_aggregation="daily",
        bias_correction=True,
        trend_preservation=True,
        method=DownscalingMethod.HYBRID,
        scenarios=[
            ClimateScenario.SSP1_26,
            ClimateScenario.SSP2_45,
            ClimateScenario.SSP5_85
        ],
        variables=[
            "temperature_2m", "precipitation", "humidity",
            "wind_speed", "wind_direction", "pressure",
            "solar_radiation", "cloud_cover"
        ],
        urban_heat_island=True,
        coastal_effects=True,
        topographic_effects=True,
        land_use_effects=True
    )
    
    downscaling_system = ClimateProjectionSystem(config)
    return downscaling_system


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create downscaling system
    downscaling_system = create_bangladesh_downscaling_system()
    
    # Mock coarse-resolution projections
    coarse_projections = {
        "temperature_2m": torch.randn(1, 1, 32, 32),
        "precipitation": torch.randn(1, 1, 32, 32),
        "humidity": torch.randn(1, 1, 32, 32),
        "wind_speed": torch.randn(1, 1, 32, 32)
    }
    
    # Mock auxiliary data
    auxiliary_data = {
        "topography": torch.randn(1, 1, 128, 128),
        "land_use": torch.randn(1, 4, 128, 128),  # 4 land use categories
        "coastline_distance": torch.randn(1, 1, 128, 128)
    }
    
    # Generate downscaled projections
    projections = downscaling_system(
        coarse_projections,
        auxiliary_data,
        ClimateScenario.SSP2_45,
        "2030-2050"
    )
    
    print("Downscaled projection variables:")
    for var_name, data in projections.items():
        downscaled_shape = data["downscaled"].shape
        uncertainty_shape = data["uncertainty"].shape
        print(f"  {var_name}: {downscaled_shape}, uncertainty: {uncertainty_shape}")
    
    print("Climate downscaling system created successfully!")
