"""
Subseasonal-to-Seasonal (S2S) forecasting system for Bangladesh.

This module extends GraphCast for extended-range predictions (weeks to months),
focusing on monsoon prediction, cyclone seasonal activity, and climate patterns
affecting Bangladesh.
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


class S2STimeScale(Enum):
    """Time scales for S2S prediction."""
    WEEK_1_2 = "week_1_2"      # 1-2 weeks
    WEEK_3_4 = "week_3_4"      # 3-4 weeks  
    MONTH_1 = "month_1"        # Month 1
    MONTH_2_3 = "month_2_3"    # Months 2-3
    SEASONAL = "seasonal"       # Full season


@dataclass
class S2SConfig:
    """Configuration for S2S forecasting system."""
    # Temporal configuration
    max_lead_time_days: int = 90
    prediction_intervals: List[str] = None
    ensemble_size: int = 30
    
    # Climate indices to track
    climate_indices: List[str] = None
    
    # Model configuration
    use_teleconnections: bool = True
    use_soil_memory: bool = True
    use_sst_forcing: bool = True
    use_mjo_tracking: bool = True
    
    # Bangladesh-specific features
    monsoon_phases: List[str] = None
    cyclone_climatology: bool = True
    river_discharge_memory: bool = True
    
    def __post_init__(self):
        if self.prediction_intervals is None:
            self.prediction_intervals = [
                "week_1", "week_2", "week_3", "week_4",
                "month_1", "month_2", "month_3"
            ]
        
        if self.climate_indices is None:
            self.climate_indices = [
                "mjo", "enso", "iod", "sam", "ao", "pna"
            ]
        
        if self.monsoon_phases is None:
            self.monsoon_phases = [
                "pre_monsoon", "onset", "active", "break", "withdrawal"
            ]


class ClimateIndexPredictor(nn.Module):
    """Predictor for large-scale climate indices."""
    
    def __init__(self, config: S2SConfig):
        super().__init__()
        self.config = config
        
        # Index-specific predictors
        self.mjo_predictor = MJOPredictor()
        self.enso_predictor = ENSOPredictor()
        self.iod_predictor = IODPredictor()
        
        # Teleconnection encoder
        self.teleconnection_encoder = TeleconnectionEncoder(
            n_indices=len(config.climate_indices),
            hidden_dim=256
        )
    
    def forward(self, 
                global_state: torch.Tensor,
                lead_time_days: int) -> Dict[str, torch.Tensor]:
        """
        Predict climate indices at given lead time.
        
        Args:
            global_state: Global atmospheric state
            lead_time_days: Forecast lead time in days
            
        Returns:
            Dictionary of predicted climate indices
        """
        indices = {}
        
        # MJO prediction (crucial for monsoon)
        if "mjo" in self.config.climate_indices:
            mjo_amplitude, mjo_phase = self.mjo_predictor(global_state, lead_time_days)
            indices["mjo_amplitude"] = mjo_amplitude
            indices["mjo_phase"] = mjo_phase
        
        # ENSO prediction
        if "enso" in self.config.climate_indices:
            indices["enso"] = self.enso_predictor(global_state, lead_time_days)
        
        # Indian Ocean Dipole
        if "iod" in self.config.climate_indices:
            indices["iod"] = self.iod_predictor(global_state, lead_time_days)
        
        # Encode teleconnections
        teleconnection_state = self.teleconnection_encoder(indices)
        indices["teleconnection_state"] = teleconnection_state
        
        return indices


class MJOPredictor(nn.Module):
    """Madden-Julian Oscillation predictor."""
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(10, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 64, 256),
            nn.ReLU()
        )
        
        # MJO amplitude and phase predictors
        self.amplitude_head = nn.Linear(256, 1)
        self.phase_head = nn.Linear(256, 8)  # 8 MJO phases
        
    def forward(self, 
                global_state: torch.Tensor, 
                lead_time_days: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict MJO amplitude and phase."""
        # Extract features relevant to MJO
        # Focus on tropical convection and circulation patterns
        features = self.feature_extractor(global_state)
        
        # Predict amplitude (0-3 scale)
        amplitude = torch.sigmoid(self.amplitude_head(features)) * 3.0
        
        # Predict phase (circular embedding)
        phase_logits = self.phase_head(features)
        phase = torch.softmax(phase_logits, dim=-1)
        
        return amplitude, phase


class ENSOPredictor(nn.Module):
    """El Niño-Southern Oscillation predictor."""
    
    def __init__(self):
        super().__init__()
        # Focus on Pacific SST patterns and Walker circulation
        self.pacific_encoder = nn.Sequential(
            nn.Conv2d(5, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 20)),  # Pacific region
            nn.Flatten(),
            nn.Linear(64 * 200, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, global_state: torch.Tensor, lead_time_days: int) -> torch.Tensor:
        """Predict ENSO index (ONI-like)."""
        # Extract Pacific SST and circulation patterns
        enso_index = self.pacific_encoder(global_state)
        return torch.tanh(enso_index) * 3.0  # Scale to typical ENSO range


class IODPredictor(nn.Module):
    """Indian Ocean Dipole predictor."""
    
    def __init__(self):
        super().__init__()
        # Focus on Indian Ocean SST gradients
        self.indian_ocean_encoder = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 12)),  # Indian Ocean region
            nn.Flatten(),
            nn.Linear(64 * 96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, global_state: torch.Tensor, lead_time_days: int) -> torch.Tensor:
        """Predict IOD index."""
        iod_index = self.indian_ocean_encoder(global_state)
        return torch.tanh(iod_index) * 2.0  # Scale to typical IOD range


class TeleconnectionEncoder(nn.Module):
    """Encode teleconnection patterns for regional impact."""
    
    def __init__(self, n_indices: int, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_indices, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, indices: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode climate indices into teleconnection state."""
        # Concatenate available indices
        index_values = []
        for key, value in indices.items():
            if key != "teleconnection_state" and value is not None:
                if value.dim() > 1:
                    value = value.flatten(start_dim=1)
                index_values.append(value)
        
        if index_values:
            combined = torch.cat(index_values, dim=-1)
            return self.encoder(combined)
        else:
            # Return zero tensor if no indices available
            return torch.zeros(1, 256)


class MonsoonPhasePredictor(nn.Module):
    """Predict monsoon phases for Bangladesh."""
    
    def __init__(self, config: S2SConfig):
        super().__init__()
        self.config = config
        self.n_phases = len(config.monsoon_phases)
        
        # Multi-scale feature extraction
        self.regional_encoder = nn.Sequential(
            nn.Conv2d(15, 64, 3, padding=1),  # Regional South Asian features
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten()
        )
        
        # Temporal encoder for monsoon evolution
        self.temporal_encoder = nn.LSTM(
            input_size=128 * 256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Phase classifier
        self.phase_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.n_phases)
        )
        
        # Onset/withdrawal timing predictor
        self.timing_predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # onset and withdrawal day-of-year
        )
    
    def forward(self, 
                regional_state: torch.Tensor,
                climate_indices: Dict[str, torch.Tensor],
                lead_time_days: int) -> Dict[str, torch.Tensor]:
        """
        Predict monsoon phases and timing.
        
        Args:
            regional_state: Regional atmospheric state over South Asia
            climate_indices: Large-scale climate indices
            lead_time_days: Forecast lead time
            
        Returns:
            Dictionary with phase probabilities and timing predictions
        """
        # Extract regional features
        regional_features = self.regional_encoder(regional_state)
        
        # Add temporal dimension for LSTM
        if regional_features.dim() == 2:
            regional_features = regional_features.unsqueeze(1)
        
        # Process temporal evolution
        lstm_out, _ = self.temporal_encoder(regional_features)
        final_state = lstm_out[:, -1, :]  # Take last time step
        
        # Predict phase probabilities
        phase_logits = self.phase_classifier(final_state)
        phase_probs = torch.softmax(phase_logits, dim=-1)
        
        # Predict onset/withdrawal timing
        timing = self.timing_predictor(final_state)
        onset_day = torch.sigmoid(timing[:, 0]) * 365  # Day of year
        withdrawal_day = torch.sigmoid(timing[:, 1]) * 365
        
        return {
            "phase_probabilities": phase_probs,
            "onset_day": onset_day,
            "withdrawal_day": withdrawal_day,
            "phase_names": self.config.monsoon_phases
        }


class SeasonalCyclonePredictor(nn.Module):
    """Predict seasonal cyclone activity in Bay of Bengal."""
    
    def __init__(self):
        super().__init__()
        # Features relevant to cyclogenesis
        self.cyclone_predictor = nn.Sequential(
            nn.Linear(512, 256),  # Climate + oceanic features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # [total_count, intense_count, peak_month, avg_intensity]
        )
    
    def forward(self, 
                sst_state: torch.Tensor,
                climate_indices: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict seasonal cyclone characteristics."""
        # Combine SST and climate index information
        features = []
        features.append(sst_state.flatten())
        
        for key, value in climate_indices.items():
            if value is not None and key != "teleconnection_state":
                features.append(value.flatten())
        
        combined_features = torch.cat(features, dim=-1)
        
        # Predict cyclone statistics
        predictions = self.cyclone_predictor(combined_features)
        
        # Apply appropriate activations
        total_count = torch.relu(predictions[:, 0])  # Non-negative count
        intense_count = torch.relu(predictions[:, 1])
        peak_month = torch.sigmoid(predictions[:, 2]) * 12  # Month 0-12
        avg_intensity = torch.sigmoid(predictions[:, 3]) * 5  # Category 0-5
        
        return {
            "seasonal_count": total_count,
            "intense_count": intense_count,
            "peak_activity_month": peak_month,
            "average_intensity": avg_intensity
        }


class LandSurfaceMemory(nn.Module):
    """Model land surface memory effects for S2S prediction."""
    
    def __init__(self):
        super().__init__()
        # Soil moisture memory
        self.soil_memory_encoder = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),  # Multi-layer soil
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Flatten()
        )
        
        # Vegetation state encoder
        self.vegetation_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # NDVI, LAI, etc.
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Flatten()
        )
        
        # Memory state predictor
        self.memory_predictor = nn.Sequential(
            nn.Linear(64 * 1024 + 32 * 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # Memory state encoding
        )
    
    def forward(self, 
                soil_state: torch.Tensor,
                vegetation_state: torch.Tensor) -> torch.Tensor:
        """Encode land surface memory state."""
        soil_features = self.soil_memory_encoder(soil_state)
        veg_features = self.vegetation_encoder(vegetation_state)
        
        combined = torch.cat([soil_features, veg_features], dim=-1)
        memory_state = self.memory_predictor(combined)
        
        return memory_state


class BangladeshS2SModel(nn.Module):
    """
    Complete S2S forecasting model for Bangladesh.
    
    Integrates:
    1. Climate index prediction
    2. Monsoon phase forecasting
    3. Seasonal cyclone activity
    4. Land surface memory
    5. Extended-range weather patterns
    """
    
    def __init__(self, 
                 base_graphcast: nn.Module,
                 config: S2SConfig):
        super().__init__()
        self.base_graphcast = base_graphcast
        self.config = config
        
        # S2S-specific components
        self.climate_predictor = ClimateIndexPredictor(config)
        self.monsoon_predictor = MonsoonPhasePredictor(config)
        self.cyclone_predictor = SeasonalCyclonePredictor()
        self.land_memory = LandSurfaceMemory()
        
        # Multi-time scale fusion
        self.temporal_fusion = TemporalScaleFusion()
        
        # Uncertainty quantification
        self.uncertainty_estimator = S2SUncertaintyEstimator()
    
    def forward(self, 
                initial_state: torch.Tensor,
                target_lead_times: List[int],
                return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Generate S2S forecast for Bangladesh.
        
        Args:
            initial_state: Initial atmospheric state
            target_lead_times: Lead times in days
            return_components: Whether to return individual components
            
        Returns:
            Dictionary with S2S predictions and uncertainties
        """
        results = {}
        
        # Extract different spatial scales
        global_state = self._extract_global_features(initial_state)
        regional_state = self._extract_regional_features(initial_state)
        
        # Predict climate indices
        climate_indices = self.climate_predictor(global_state, max(target_lead_times))
        results["climate_indices"] = climate_indices
        
        # Predict monsoon evolution
        monsoon_forecast = self.monsoon_predictor(
            regional_state, climate_indices, max(target_lead_times)
        )
        results["monsoon"] = monsoon_forecast
        
        # Predict seasonal cyclone activity
        if max(target_lead_times) >= 30:  # Only for monthly+ forecasts
            sst_state = self._extract_sst_features(initial_state)
            cyclone_forecast = self.cyclone_predictor(sst_state, climate_indices)
            results["cyclones"] = cyclone_forecast
        
        # Incorporate land surface memory
        soil_state = self._extract_soil_features(initial_state)
        vegetation_state = self._extract_vegetation_features(initial_state)
        land_memory_state = self.land_memory(soil_state, vegetation_state)
        results["land_memory"] = land_memory_state
        
        # Generate multi-time scale forecasts
        for lead_time in target_lead_times:
            # Determine time scale
            time_scale = self._determine_time_scale(lead_time)
            
            # Fuse information across scales
            fused_state = self.temporal_fusion(
                base_state=initial_state,
                climate_state=climate_indices["teleconnection_state"],
                monsoon_state=monsoon_forecast["phase_probabilities"],
                land_memory_state=land_memory_state,
                time_scale=time_scale
            )
            
            # Generate forecast for this lead time
            forecast = self._generate_lead_time_forecast(fused_state, lead_time)
            
            # Estimate uncertainty
            uncertainty = self.uncertainty_estimator(
                forecast, time_scale, lead_time
            )
            
            results[f"day_{lead_time}"] = {
                "forecast": forecast,
                "uncertainty": uncertainty,
                "time_scale": time_scale
            }
        
        return results
    
    def _extract_global_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extract global-scale features."""
        # Placeholder - would extract global circulation patterns
        return state.mean(dim=(2, 3), keepdim=True).expand(-1, -1, 64, 128)
    
    def _extract_regional_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extract South Asian regional features."""
        # Placeholder - would focus on South Asian domain
        return state[:, :, 20:44, 60:100]  # Rough South Asia bounds
    
    def _extract_sst_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extract sea surface temperature features."""
        # Placeholder - would extract SST from state
        return state[:, 0, :, :].flatten()  # Temperature field
    
    def _extract_soil_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extract soil state features."""
        # Placeholder - would extract soil layers
        return state[:, :5, :, :]  # First 5 variables as soil
    
    def _extract_vegetation_features(self, state: torch.Tensor) -> torch.Tensor:
        """Extract vegetation state features."""
        # Placeholder - would extract vegetation indices
        return state[:, -3:, :, :]  # Last 3 variables as vegetation
    
    def _determine_time_scale(self, lead_time_days: int) -> S2STimeScale:
        """Determine appropriate time scale for lead time."""
        if lead_time_days <= 14:
            return S2STimeScale.WEEK_1_2
        elif lead_time_days <= 28:
            return S2STimeScale.WEEK_3_4
        elif lead_time_days <= 31:
            return S2STimeScale.MONTH_1
        elif lead_time_days <= 90:
            return S2STimeScale.MONTH_2_3
        else:
            return S2STimeScale.SEASONAL
    
    def _generate_lead_time_forecast(self, 
                                   fused_state: torch.Tensor,
                                   lead_time: int) -> torch.Tensor:
        """Generate forecast for specific lead time."""
        # Use appropriate model based on lead time
        # This would involve the base GraphCast model
        return self.base_graphcast(fused_state)


class TemporalScaleFusion(nn.Module):
    """Fuse information across different temporal scales."""
    
    def __init__(self):
        super().__init__()
        self.fusion_network = nn.Sequential(
            nn.Linear(1024, 512),  # Adjust based on input sizes
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self,
                base_state: torch.Tensor,
                climate_state: torch.Tensor,
                monsoon_state: torch.Tensor,
                land_memory_state: torch.Tensor,
                time_scale: S2STimeScale) -> torch.Tensor:
        """Fuse multi-scale information."""
        # Combine different temporal scale information
        # This is a simplified version
        features = torch.cat([
            base_state.flatten(),
            climate_state.flatten(),
            monsoon_state.flatten(),
            land_memory_state.flatten()
        ], dim=-1)
        
        return self.fusion_network(features)


class S2SUncertaintyEstimator(nn.Module):
    """Estimate uncertainty for S2S forecasts."""
    
    def __init__(self):
        super().__init__()
        self.uncertainty_network = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self,
                forecast: torch.Tensor,
                time_scale: S2STimeScale,
                lead_time: int) -> torch.Tensor:
        """Estimate forecast uncertainty."""
        # Simple uncertainty model - grows with lead time
        base_uncertainty = self.uncertainty_network(forecast.flatten())
        
        # Scale uncertainty based on lead time and time scale
        scale_factor = {
            S2STimeScale.WEEK_1_2: 1.0,
            S2STimeScale.WEEK_3_4: 1.5,
            S2STimeScale.MONTH_1: 2.0,
            S2STimeScale.MONTH_2_3: 3.0,
            S2STimeScale.SEASONAL: 4.0
        }
        
        scaled_uncertainty = base_uncertainty * scale_factor[time_scale]
        return torch.sigmoid(scaled_uncertainty)


# Configuration and usage example
def create_bangladesh_s2s_system(base_graphcast: nn.Module) -> BangladeshS2SModel:
    """Create configured S2S system for Bangladesh."""
    config = S2SConfig(
        max_lead_time_days=90,
        prediction_intervals=[
            "week_1", "week_2", "week_3", "week_4",
            "month_1", "month_2", "month_3"
        ],
        ensemble_size=30,
        climate_indices=["mjo", "enso", "iod", "sam"],
        use_teleconnections=True,
        use_soil_memory=True,
        use_sst_forcing=True,
        monsoon_phases=[
            "pre_monsoon", "onset", "active", "break", "withdrawal"
        ],
        cyclone_climatology=True,
        river_discharge_memory=True
    )
    
    s2s_model = BangladeshS2SModel(base_graphcast, config)
    return s2s_model


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock base GraphCast model
    class MockGraphCast(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 128)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create S2S system
    base_model = MockGraphCast()
    s2s_system = create_bangladesh_s2s_system(base_model)
    
    # Mock initial conditions
    initial_state = torch.randn(1, 10, 64, 128)  # [batch, vars, lat, lon]
    target_lead_times = [7, 14, 21, 28, 45, 60, 90]  # days
    
    # Generate S2S forecast
    s2s_forecast = s2s_system(
        initial_state, 
        target_lead_times,
        return_components=True
    )
    
    print("S2S forecast components:")
    for key in s2s_forecast.keys():
        if isinstance(s2s_forecast[key], dict):
            print(f"  {key}: {list(s2s_forecast[key].keys())}")
        else:
            print(f"  {key}: {type(s2s_forecast[key])}")
    
    print("S2S system created successfully!")
