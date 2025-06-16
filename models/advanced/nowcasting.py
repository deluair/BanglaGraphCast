"""
Nowcasting Module for Bangladesh GraphCast

High-resolution nowcasting (0-6 hours) for immediate weather threats:
- Radar-based precipitation nowcasting
- Thunderstorm initiation and evolution
- Flash flood prediction
- Severe weather onset
- Air quality forecasting
- Urban microclimate predictions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import cv2
from scipy import ndimage
from pathlib import Path

logger = logging.getLogger(__name__)


class NowcastType(Enum):
    """Types of nowcasting products"""
    PRECIPITATION = "precipitation"
    THUNDERSTORM = "thunderstorm"
    FLASH_FLOOD = "flash_flood"
    VISIBILITY = "visibility"
    AIR_QUALITY = "air_quality"
    URBAN_HEAT = "urban_heat"
    WIND_GUST = "wind_gust"


class NowcastTimeScale(Enum):
    """Nowcast time scales"""
    MINUTES_15 = "15min"
    MINUTES_30 = "30min"
    HOUR_1 = "1hr"
    HOURS_2 = "2hr"
    HOURS_3 = "3hr"
    HOURS_6 = "6hr"


@dataclass
class NowcastConfig:
    """Configuration for nowcasting system"""
    # Temporal settings
    nowcast_types: List[NowcastType] = None
    time_scales: List[NowcastTimeScale] = None
    update_frequency_minutes: int = 15
    max_lead_time_hours: int = 6
    
    # Spatial settings
    spatial_resolution_km: float = 1.0
    domain_bounds: Dict[str, float] = None
    urban_high_res_km: float = 0.25
    
    # Input data sources
    use_radar: bool = True
    use_satellite: bool = True
    use_surface_obs: bool = True
    use_lightning: bool = True
    use_air_quality_sensors: bool = True
    
    # Model settings
    use_optical_flow: bool = True
    use_deep_learning: bool = True
    ensemble_nowcasts: int = 10
    
    # Bangladesh-specific settings
    monsoon_aware: bool = True
    urban_focus_cities: List[str] = None
    river_basin_focus: bool = True
    
    def __post_init__(self):
        if self.nowcast_types is None:
            self.nowcast_types = [
                NowcastType.PRECIPITATION,
                NowcastType.THUNDERSTORM,
                NowcastType.FLASH_FLOOD
            ]
        
        if self.time_scales is None:
            self.time_scales = [
                NowcastTimeScale.MINUTES_15,
                NowcastTimeScale.MINUTES_30,
                NowcastTimeScale.HOUR_1,
                NowcastTimeScale.HOURS_3
            ]
        
        if self.domain_bounds is None:
            # Bangladesh domain with high resolution
            self.domain_bounds = {
                "lat_min": 20.5, "lat_max": 26.8,
                "lon_min": 87.0, "lon_max": 93.0
            }
        
        if self.urban_focus_cities is None:
            self.urban_focus_cities = [
                "dhaka", "chittagong", "khulna", "rajshahi", 
                "sylhet", "barisal", "rangpur", "comilla"
            ]


class RadarNowcaster(nn.Module):
    """Deep learning nowcaster using radar data"""
    
    def __init__(self, 
                 input_channels: int = 6,  # Multiple radar variables
                 sequence_length: int = 8,  # 2 hours of 15-min data
                 hidden_dim: int = 64):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Convolutional encoder for each time step
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        
        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTMCell(hidden_dim, hidden_dim, 3)
        
        # Attention mechanism for important patterns
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Decoder for future predictions
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, 3, padding=1),
            nn.ReLU()  # Ensure positive precipitation
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                radar_sequence: torch.Tensor,
                forecast_steps: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for radar nowcasting
        
        Args:
            radar_sequence: Input sequence [batch, time, channels, height, width]
            forecast_steps: Number of future time steps to predict
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        batch_size, seq_len, channels, height, width = radar_sequence.shape
        
        # Initialize LSTM hidden states
        h_t = torch.zeros(batch_size, self.hidden_dim, height, width, 
                         device=radar_sequence.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, height, width,
                         device=radar_sequence.device)
        
        # Encode input sequence
        encoded_sequence = []
        for t in range(seq_len):
            # Encode current frame
            encoded = self.conv_encoder(radar_sequence[:, t])
            
            # Update LSTM state
            h_t, c_t = self.conv_lstm(encoded, (h_t, c_t))
            encoded_sequence.append(h_t)
        
        # Apply temporal attention
        encoded_stack = torch.stack(encoded_sequence, dim=1)
        attended_features = self.temporal_attention(encoded_stack)
        
        # Generate future predictions
        predictions = []
        uncertainties = []
        
        current_state = attended_features
        for step in range(forecast_steps):
            # Decode current state
            pred = self.decoder(current_state)
            uncertainty = self.uncertainty_head(current_state)
            
            predictions.append(pred)
            uncertainties.append(uncertainty)
            
            # Update state for next prediction (autoregressive)
            encoded_pred = self.conv_encoder(pred)
            h_t, c_t = self.conv_lstm(encoded_pred, (h_t, c_t))
            current_state = h_t
        
        predictions = torch.stack(predictions, dim=1)
        uncertainties = torch.stack(uncertainties, dim=1)
        
        return predictions, uncertainties


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for spatiotemporal modeling"""
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 
            4 * hidden_dim,
            kernel_size, 
            padding=padding
        )
    
    def forward(self, 
                input_tensor: torch.Tensor,
                hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for ConvLSTM cell"""
        h_cur, c_cur = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        
        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply gates
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # New cell content
        
        # Update cell state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequence modeling"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.attention = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention to sequence
        
        Args:
            sequence: Input sequence [batch, time, channels, height, width]
            
        Returns:
            Attended features [batch, channels, height, width]
        """
        # Calculate attention weights
        attention_weights = self.attention(sequence.permute(0, 2, 1, 3, 4))
        attention_weights = attention_weights.permute(0, 2, 1, 3, 4)
        
        # Apply attention
        attended = sequence * attention_weights
        
        # Weighted sum over time dimension
        output = torch.sum(attended, dim=1)
        
        return output


class OpticalFlowNowcaster:
    """Optical flow-based precipitation nowcasting"""
    
    def __init__(self, config: NowcastConfig):
        self.config = config
        
    def calculate_motion_vectors(self, 
                                radar_frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate motion vectors using optical flow
        
        Args:
            radar_frames: Sequence of radar frames [time, height, width]
            
        Returns:
            Tuple of (u_motion, v_motion) velocity components
        """
        if len(radar_frames) < 2:
            raise ValueError("Need at least 2 frames for motion calculation")
        
        # Use the last two frames
        frame1 = radar_frames[-2].astype(np.float32)
        frame2 = radar_frames[-1].astype(np.float32)
        
        # Normalize frames
        frame1 = (frame1 - frame1.min()) / (frame1.max() - frame1.min() + 1e-8)
        frame2 = (frame2 - frame2.min()) / (frame2.max() - frame2.min() + 1e-8)
        
        # Calculate optical flow using Lucas-Kanade method
        flow = cv2.calcOpticalFlowPyrLK(
            (frame1 * 255).astype(np.uint8),
            (frame2 * 255).astype(np.uint8),
            None, None
        )
        
        # Alternative: Dense optical flow
        flow_dense = cv2.calcOpticalFlowFarneback(
            (frame1 * 255).astype(np.uint8),
            (frame2 * 255).astype(np.uint8),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        u_motion = flow_dense[:, :, 0]
        v_motion = flow_dense[:, :, 1]
        
        return u_motion, v_motion
    
    def advect_field(self, 
                    field: np.ndarray,
                    u_motion: np.ndarray,
                    v_motion: np.ndarray,
                    dt_minutes: float = 15) -> np.ndarray:
        """
        Advect precipitation field using motion vectors
        
        Args:
            field: Current precipitation field
            u_motion: East-West motion component (pixels/timestep)
            v_motion: North-South motion component (pixels/timestep)
            dt_minutes: Time step in minutes
            
        Returns:
            Advected precipitation field
        """
        height, width = field.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Calculate new positions
        x_new = x_coords + u_motion * (dt_minutes / 15.0)  # Normalize to 15-min timestep
        y_new = y_coords + v_motion * (dt_minutes / 15.0)
        
        # Interpolate field to new positions
        from scipy.interpolate import RegularGridInterpolator
        
        interpolator = RegularGridInterpolator(
            (np.arange(height), np.arange(width)),
            field,
            bounds_error=False,
            fill_value=0.0
        )
        
        # Clip coordinates to valid range
        y_new = np.clip(y_new, 0, height - 1)
        x_new = np.clip(x_new, 0, width - 1)
        
        # Interpolate
        coords = np.stack([y_new.ravel(), x_new.ravel()], axis=1)
        advected = interpolator(coords).reshape(height, width)
        
        return advected
    
    def predict_sequence(self,
                        radar_frames: np.ndarray,
                        forecast_steps: int = 12) -> np.ndarray:
        """
        Generate nowcast sequence using optical flow
        
        Args:
            radar_frames: Input radar sequence [time, height, width]
            forecast_steps: Number of future time steps
            
        Returns:
            Predicted sequence [time, height, width]
        """
        # Calculate motion vectors
        u_motion, v_motion = self.calculate_motion_vectors(radar_frames)
        
        # Apply motion smoothing
        u_motion = ndimage.gaussian_filter(u_motion, sigma=1.0)
        v_motion = ndimage.gaussian_filter(v_motion, sigma=1.0)
        
        # Initialize prediction with last frame
        current_frame = radar_frames[-1].copy()
        predictions = []
        
        for step in range(forecast_steps):
            # Advect current frame
            advected_frame = self.advect_field(current_frame, u_motion, v_motion)
            
            # Apply growth/decay models (simplified)
            decay_factor = np.exp(-step * 0.1)  # Exponential decay
            advected_frame *= decay_factor
            
            predictions.append(advected_frame)
            current_frame = advected_frame
        
        return np.array(predictions)


class ThunderstormNowcaster:
    """Nowcasting for thunderstorm initiation and evolution"""
    
    def __init__(self, config: NowcastConfig):
        self.config = config
        
    def detect_initiation_areas(self,
                              radar_data: np.ndarray,
                              temperature: np.ndarray,
                              humidity: np.ndarray,
                              cape: np.ndarray) -> np.ndarray:
        """
        Detect areas favorable for thunderstorm initiation
        
        Args:
            radar_data: Current radar reflectivity
            temperature: Surface temperature
            humidity: Surface humidity
            cape: Convective Available Potential Energy
            
        Returns:
            Thunderstorm initiation probability map
        """
        # Calculate convective parameters
        initiation_prob = np.zeros_like(radar_data)
        
        # High CAPE areas
        cape_factor = np.clip(cape / 2500.0, 0, 1)  # Normalize CAPE
        
        # High humidity areas
        humidity_factor = np.clip((humidity - 70) / 30.0, 0, 1)  # RH > 70%
        
        # Temperature gradient (convergence zones)
        temp_grad = np.gradient(temperature)
        temp_gradient_mag = np.sqrt(temp_grad[0]**2 + temp_grad[1]**2)
        gradient_factor = np.clip(temp_gradient_mag / np.percentile(temp_gradient_mag, 90), 0, 1)
        
        # Existing convection enhancement
        existing_conv = np.clip(radar_data / 35.0, 0, 1)  # 35 dBZ threshold
        
        # Combine factors
        initiation_prob = (
            0.4 * cape_factor +
            0.3 * humidity_factor +
            0.2 * gradient_factor +
            0.1 * existing_conv
        )
        
        # Apply smoothing
        initiation_prob = ndimage.gaussian_filter(initiation_prob, sigma=2.0)
        
        return initiation_prob
    
    def track_storm_cells(self,
                         radar_sequence: np.ndarray,
                         threshold_dbz: float = 35.0) -> List[Dict[str, Any]]:
        """
        Track individual storm cells through time
        
        Args:
            radar_sequence: Sequence of radar data [time, height, width]
            threshold_dbz: Minimum reflectivity threshold
            
        Returns:
            List of tracked storm cells with properties
        """
        storms = []
        
        for t, radar_frame in enumerate(radar_sequence):
            # Identify storm cells
            storm_mask = radar_frame >= threshold_dbz
            
            # Label connected components
            labeled_storms, n_storms = ndimage.label(storm_mask)
            
            # Extract storm properties
            for storm_id in range(1, n_storms + 1):
                storm_pixels = labeled_storms == storm_id
                
                if np.sum(storm_pixels) < 10:  # Minimum size filter
                    continue
                
                # Calculate storm properties
                storm_props = self._calculate_storm_properties(
                    radar_frame, storm_pixels, t
                )
                storms.append(storm_props)
        
        return storms
    
    def _calculate_storm_properties(self,
                                  radar_frame: np.ndarray,
                                  storm_mask: np.ndarray,
                                  time_index: int) -> Dict[str, Any]:
        """Calculate properties of individual storm cell"""
        
        # Center of mass
        y_coords, x_coords = np.where(storm_mask)
        center_y = np.mean(y_coords)
        center_x = np.mean(x_coords)
        
        # Maximum reflectivity
        max_dbz = np.max(radar_frame[storm_mask])
        
        # Storm area
        area_pixels = np.sum(storm_mask)
        
        # Storm intensity (volume)
        volume = np.sum(radar_frame[storm_mask])
        
        return {
            'time_index': time_index,
            'center_y': center_y,
            'center_x': center_x,
            'max_reflectivity': max_dbz,
            'area_pixels': area_pixels,
            'volume': volume,
            'storm_mask': storm_mask
        }


class FlashFloodNowcaster:
    """Flash flood nowcasting for urban and rural areas"""
    
    def __init__(self, config: NowcastConfig):
        self.config = config
        
    def calculate_runoff_potential(self,
                                 precipitation: np.ndarray,
                                 elevation: np.ndarray,
                                 land_use: np.ndarray,
                                 soil_moisture: np.ndarray) -> np.ndarray:
        """
        Calculate flash flood runoff potential
        
        Args:
            precipitation: Precipitation accumulation
            elevation: Digital elevation model
            land_use: Land use classification
            soil_moisture: Current soil moisture
            
        Returns:
            Flash flood risk map
        """
        # Calculate slope from elevation
        elevation_grad = np.gradient(elevation)
        slope = np.sqrt(elevation_grad[0]**2 + elevation_grad[1]**2)
        
        # Runoff coefficients by land use type
        runoff_coeffs = {
            0: 0.1,  # Forest
            1: 0.3,  # Grassland
            2: 0.5,  # Agriculture
            3: 0.8,  # Urban
            4: 0.0   # Water
        }
        
        # Create runoff coefficient map
        runoff_coeff_map = np.zeros_like(land_use, dtype=float)
        for land_type, coeff in runoff_coeffs.items():
            runoff_coeff_map[land_use == land_type] = coeff
        
        # Adjust for soil moisture (higher moisture = more runoff)
        soil_factor = np.clip(soil_moisture / 0.4, 0.5, 1.5)  # Normalize soil moisture
        
        # Calculate effective runoff
        effective_precip = precipitation * runoff_coeff_map * soil_factor
        
        # Slope enhancement (steeper = more runoff concentration)
        slope_factor = 1.0 + np.clip(slope / np.percentile(slope, 95), 0, 2)
        
        # Flash flood potential
        flood_potential = effective_precip * slope_factor
        
        # Apply watershed accumulation (simplified)
        flood_potential = ndimage.gaussian_filter(flood_potential, sigma=2.0)
        
        return flood_potential


class BangladeshNowcaster:
    """Main nowcasting system for Bangladesh"""
    
    def __init__(self, config: NowcastConfig):
        self.config = config
        
        # Initialize component nowcasters
        self.radar_nowcaster = RadarNowcaster()
        self.optical_flow_nowcaster = OpticalFlowNowcaster(config)
        self.thunderstorm_nowcaster = ThunderstormNowcaster(config)
        self.flash_flood_nowcaster = FlashFloodNowcaster(config)
        
        # Load Bangladesh-specific parameters
        self._load_local_climatology()
    
    def generate_nowcast(self,
                        input_data: Dict[str, np.ndarray],
                        nowcast_type: NowcastType,
                        lead_time_hours: float = 3.0) -> Dict[str, Any]:
        """
        Generate nowcast for specified type and lead time
        
        Args:
            input_data: Dictionary of input data arrays
            nowcast_type: Type of nowcast to generate
            lead_time_hours: Forecast lead time in hours
            
        Returns:
            Nowcast results dictionary
        """
        forecast_steps = int(lead_time_hours * 4)  # 15-minute intervals
        
        if nowcast_type == NowcastType.PRECIPITATION:
            return self._nowcast_precipitation(input_data, forecast_steps)
        elif nowcast_type == NowcastType.THUNDERSTORM:
            return self._nowcast_thunderstorms(input_data, forecast_steps)
        elif nowcast_type == NowcastType.FLASH_FLOOD:
            return self._nowcast_flash_floods(input_data, forecast_steps)
        else:
            raise ValueError(f"Nowcast type {nowcast_type} not implemented")
    
    def _nowcast_precipitation(self,
                             input_data: Dict[str, np.ndarray],
                             forecast_steps: int) -> Dict[str, Any]:
        """Generate precipitation nowcast"""
        
        radar_data = input_data.get('radar', np.zeros((8, 256, 256)))
        
        # Deep learning nowcast
        radar_tensor = torch.from_numpy(radar_data).unsqueeze(0).unsqueeze(2)
        dl_predictions, uncertainties = self.radar_nowcaster(radar_tensor, forecast_steps)
        dl_nowcast = dl_predictions.squeeze().detach().numpy()
        
        # Optical flow nowcast
        of_nowcast = self.optical_flow_nowcaster.predict_sequence(
            radar_data, forecast_steps
        )
        
        # Ensemble combination
        ensemble_nowcast = 0.7 * dl_nowcast + 0.3 * of_nowcast
        
        # Calculate confidence
        confidence = 1.0 - uncertainties.squeeze().detach().numpy()
        
        return {
            'precipitation_forecast': ensemble_nowcast,
            'confidence': confidence,
            'method': 'ensemble_dl_optical_flow',
            'lead_time_hours': forecast_steps / 4.0,
            'spatial_resolution_km': self.config.spatial_resolution_km
        }
    
    def _nowcast_thunderstorms(self,
                             input_data: Dict[str, np.ndarray],
                             forecast_steps: int) -> Dict[str, Any]:
        """Generate thunderstorm nowcast"""
        
        radar_data = input_data.get('radar', np.zeros((8, 256, 256)))
        temperature = input_data.get('temperature', np.zeros((256, 256)))
        humidity = input_data.get('humidity', np.zeros((256, 256)))
        cape = input_data.get('cape', np.zeros((256, 256)))
        
        # Detect initiation areas
        initiation_prob = self.thunderstorm_nowcaster.detect_initiation_areas(
            radar_data[-1], temperature, humidity, cape
        )
        
        # Track existing storms
        storm_tracks = self.thunderstorm_nowcaster.track_storm_cells(radar_data)
        
        return {
            'initiation_probability': initiation_prob,
            'storm_tracks': storm_tracks,
            'lead_time_hours': forecast_steps / 4.0
        }
    
    def _nowcast_flash_floods(self,
                            input_data: Dict[str, np.ndarray],
                            forecast_steps: int) -> Dict[str, Any]:
        """Generate flash flood nowcast"""
        
        precipitation = input_data.get('precipitation_accum', np.zeros((256, 256)))
        elevation = input_data.get('elevation', np.zeros((256, 256)))
        land_use = input_data.get('land_use', np.zeros((256, 256)))
        soil_moisture = input_data.get('soil_moisture', np.zeros((256, 256)))
        
        # Calculate flood potential
        flood_risk = self.flash_flood_nowcaster.calculate_runoff_potential(
            precipitation, elevation, land_use, soil_moisture
        )
        
        return {
            'flood_risk_map': flood_risk,
            'high_risk_areas': flood_risk > np.percentile(flood_risk, 90),
            'lead_time_hours': forecast_steps / 4.0
        }
    
    def _load_local_climatology(self):
        """Load Bangladesh-specific climatological parameters"""
        # This would typically load from data files
        self.local_climate = {
            'monsoon_months': [6, 7, 8, 9],  # June-September
            'pre_monsoon_months': [3, 4, 5],  # March-May
            'post_monsoon_months': [10, 11],  # October-November
            'winter_months': [12, 1, 2],      # December-February
            
            'typical_precip_rates': {
                'light': 2.5,      # mm/hr
                'moderate': 10.0,  # mm/hr
                'heavy': 50.0,     # mm/hr
                'extreme': 100.0   # mm/hr
            }
        }


def create_bangladesh_nowcaster(config: Optional[NowcastConfig] = None) -> BangladeshNowcaster:
    """
    Factory function to create Bangladesh nowcaster
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured BangladeshNowcaster instance
    """
    if config is None:
        config = NowcastConfig()
    
    return BangladeshNowcaster(config)


# Example usage
if __name__ == "__main__":
    # Create nowcaster
    config = NowcastConfig()
    nowcaster = create_bangladesh_nowcaster(config)
    
    # Example input data
    input_data = {
        'radar': np.random.rand(8, 256, 256) * 50,  # 8 time steps of radar data
        'temperature': np.random.rand(256, 256) * 10 + 25,  # 25-35°C
        'humidity': np.random.rand(256, 256) * 30 + 60,     # 60-90% RH
        'cape': np.random.rand(256, 256) * 3000,            # 0-3000 J/kg
        'precipitation_accum': np.random.rand(256, 256) * 50,  # 0-50mm
        'elevation': np.random.rand(256, 256) * 100,        # 0-100m
        'land_use': np.random.randint(0, 5, (256, 256)),    # 5 land use types
        'soil_moisture': np.random.rand(256, 256) * 0.5     # 0-0.5 m³/m³
    }
    
    # Generate precipitation nowcast
    precip_nowcast = nowcaster.generate_nowcast(
        input_data, NowcastType.PRECIPITATION, lead_time_hours=3.0
    )
    
    print("Precipitation Nowcast Generated:")
    print(f"  Forecast shape: {precip_nowcast['precipitation_forecast'].shape}")
    print(f"  Lead time: {precip_nowcast['lead_time_hours']} hours")
    print(f"  Spatial resolution: {precip_nowcast['spatial_resolution_km']} km")
    
    # Generate thunderstorm nowcast
    storm_nowcast = nowcaster.generate_nowcast(
        input_data, NowcastType.THUNDERSTORM, lead_time_hours=2.0
    )
    
    print(f"\nThunderstorm Nowcast Generated:")
    print(f"  Initiation probability shape: {storm_nowcast['initiation_probability'].shape}")
    print(f"  Number of tracked storms: {len(storm_nowcast['storm_tracks'])}")
    
    print("\nBangladesh Nowcasting System Ready!")
