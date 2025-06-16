"""
Advanced bias correction module for BanglaGraphCast.

This module implements systematic bias correction techniques including
quantile mapping, machine learning-based correction, and adaptive bias correction
specifically tailored for Bangladesh weather patterns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BiasConfig:
    """Configuration for bias correction."""
    
    # Correction methods
    methods: List[str] = None
    primary_method: str = "quantile_mapping"
    
    # Quantile mapping settings
    quantile_method: str = "linear"  # linear, cubic, monotonic
    n_quantiles: int = 100
    extrapolation: str = "constant"  # constant, linear
    
    # ML correction settings
    correction_model_type: str = "neural_network"  # neural_network, random_forest, xgboost
    ml_features: List[str] = None
    hidden_sizes: List[int] = None
    
    # Adaptive correction settings
    adaptation_window: int = 30  # days
    update_frequency: str = "daily"  # daily, weekly, monthly
    memory_factor: float = 0.95
    
    # Regional settings
    use_regional_correction: bool = True
    regions: List[str] = None
    
    # Variable-specific settings
    variables: List[str] = None
    variable_weights: Dict[str, float] = None
    
    # Output settings
    save_correction_maps: bool = True
    save_diagnostics: bool = True
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = [
                "quantile_mapping", "linear_scaling", "delta_method",
                "neural_network", "adaptive"
            ]
        
        if self.ml_features is None:
            self.ml_features = [
                "raw_prediction", "time_of_day", "day_of_year",
                "elevation", "distance_to_coast", "land_use"
            ]
        
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64, 32]
        
        if self.regions is None:
            self.regions = [
                "dhaka", "chittagong", "sylhet", "rajshahi", 
                "barisal", "rangpur", "mymensingh", "khulna"
            ]
        
        if self.variables is None:
            self.variables = [
                "temperature_2m", "precipitation", "relative_humidity",
                "wind_speed", "pressure", "solar_radiation"
            ]
        
        if self.variable_weights is None:
            self.variable_weights = {
                "temperature_2m": 1.0,
                "precipitation": 2.0,  # Higher weight for precipitation
                "relative_humidity": 0.8,
                "wind_speed": 0.6,
                "pressure": 0.5,
                "solar_radiation": 0.7
            }


class QuantileMapping:
    """Quantile mapping bias correction."""
    
    def __init__(self, config: BiasConfig):
        self.config = config
        self.quantile_maps = {}
        self.fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
        variable: str = "default"
    ):
        """
        Fit quantile mapping correction.
        
        Args:
            predictions: Historical predictions
            observations: Corresponding observations
            variable: Variable name for separate fitting
        """
        from scipy import interpolate, stats
        
        # Remove NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(observations))
        pred_clean = predictions[valid_mask]
        obs_clean = observations[valid_mask]
        
        if len(pred_clean) < 10:
            logger.warning(f"Insufficient data for quantile mapping: {len(pred_clean)} points")
            return
        
        # Calculate quantiles
        quantiles = np.linspace(0, 1, self.config.n_quantiles)
        pred_quantiles = np.quantile(pred_clean, quantiles)
        obs_quantiles = np.quantile(obs_clean, quantiles)
        
        # Create interpolation function
        if self.config.quantile_method == "linear":
            interp_func = interpolate.interp1d(
                pred_quantiles, obs_quantiles,
                kind='linear',
                bounds_error=False,
                fill_value=(obs_quantiles[0], obs_quantiles[-1])
            )
        elif self.config.quantile_method == "cubic":
            interp_func = interpolate.interp1d(
                pred_quantiles, obs_quantiles,
                kind='cubic',
                bounds_error=False,
                fill_value=(obs_quantiles[0], obs_quantiles[-1])
            )
        else:  # monotonic
            interp_func = interpolate.PchipInterpolator(
                pred_quantiles, obs_quantiles,
                extrapolate=True
            )
        
        self.quantile_maps[variable] = {
            'interpolator': interp_func,
            'pred_quantiles': pred_quantiles,
            'obs_quantiles': obs_quantiles,
            'pred_min': np.min(pred_clean),
            'pred_max': np.max(pred_clean),
            'obs_min': np.min(obs_clean),
            'obs_max': np.max(obs_clean)
        }
        
        self.fitted = True
    
    def correct(
        self,
        predictions: np.ndarray,
        variable: str = "default"
    ) -> np.ndarray:
        """
        Apply quantile mapping correction.
        
        Args:
            predictions: Predictions to correct
            variable: Variable name
        
        Returns:
            Corrected predictions
        """
        if not self.fitted or variable not in self.quantile_maps:
            logger.warning(f"Quantile mapping not fitted for variable: {variable}")
            return predictions
        
        qmap = self.quantile_maps[variable]
        
        # Handle extrapolation
        pred_clipped = np.clip(
            predictions,
            qmap['pred_min'] if self.config.extrapolation == "constant" else -np.inf,
            qmap['pred_max'] if self.config.extrapolation == "constant" else np.inf
        )
        
        # Apply correction
        corrected = qmap['interpolator'](pred_clipped)
        
        return corrected


class LinearScaling:
    """Linear scaling bias correction."""
    
    def __init__(self):
        self.scaling_params = {}
        self.fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
        variable: str = "default"
    ):
        """Fit linear scaling parameters."""
        valid_mask = ~(np.isnan(predictions) | np.isnan(observations))
        pred_clean = predictions[valid_mask]
        obs_clean = observations[valid_mask]
        
        if len(pred_clean) < 10:
            logger.warning(f"Insufficient data for linear scaling: {len(pred_clean)} points")
            return
        
        # Calculate scaling parameters
        obs_mean = np.mean(obs_clean)
        pred_mean = np.mean(pred_clean)
        
        obs_std = np.std(obs_clean)
        pred_std = np.std(pred_clean)
        
        # Scaling factor and offset
        scale = obs_std / pred_std if pred_std > 0 else 1.0
        offset = obs_mean - scale * pred_mean
        
        self.scaling_params[variable] = {
            'scale': scale,
            'offset': offset,
            'obs_mean': obs_mean,
            'pred_mean': pred_mean,
            'obs_std': obs_std,
            'pred_std': pred_std
        }
        
        self.fitted = True
    
    def correct(
        self,
        predictions: np.ndarray,
        variable: str = "default"
    ) -> np.ndarray:
        """Apply linear scaling correction."""
        if not self.fitted or variable not in self.scaling_params:
            logger.warning(f"Linear scaling not fitted for variable: {variable}")
            return predictions
        
        params = self.scaling_params[variable]
        corrected = params['scale'] * predictions + params['offset']
        
        return corrected


class DeltaMethod:
    """Delta method bias correction (additive/multiplicative)."""
    
    def __init__(self, method_type: str = "additive"):
        self.method_type = method_type  # additive or multiplicative
        self.delta_params = {}
        self.fitted = False
    
    def fit(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
        variable: str = "default"
    ):
        """Fit delta method parameters."""
        valid_mask = ~(np.isnan(predictions) | np.isnan(observations))
        pred_clean = predictions[valid_mask]
        obs_clean = observations[valid_mask]
        
        if len(pred_clean) < 10:
            logger.warning(f"Insufficient data for delta method: {len(pred_clean)} points")
            return
        
        if self.method_type == "additive":
            delta = np.mean(obs_clean - pred_clean)
        else:  # multiplicative
            # Avoid division by zero
            pred_nonzero = pred_clean[pred_clean != 0]
            obs_nonzero = obs_clean[pred_clean != 0]
            
            if len(pred_nonzero) > 0:
                delta = np.mean(obs_nonzero / pred_nonzero)
            else:
                delta = 1.0
        
        self.delta_params[variable] = {
            'delta': delta,
            'method': self.method_type
        }
        
        self.fitted = True
    
    def correct(
        self,
        predictions: np.ndarray,
        variable: str = "default"
    ) -> np.ndarray:
        """Apply delta method correction."""
        if not self.fitted or variable not in self.delta_params:
            logger.warning(f"Delta method not fitted for variable: {variable}")
            return predictions
        
        params = self.delta_params[variable]
        
        if params['method'] == "additive":
            corrected = predictions + params['delta']
        else:  # multiplicative
            corrected = predictions * params['delta']
        
        return corrected


class NeuralNetworkCorrection(nn.Module):
    """Neural network-based bias correction."""
    
    def __init__(self, config: BiasConfig):
        super().__init__()
        self.config = config
        
        # Network architecture
        input_size = len(config.ml_features)
        hidden_sizes = config.hidden_sizes
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Feature statistics for normalization
        self.feature_stats = {}
        self.fitted = False
    
    def prepare_features(
        self,
        predictions: torch.Tensor,
        metadata: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Prepare input features for the network.
        
        Args:
            predictions: Raw predictions
            metadata: Additional metadata (time, location, etc.)
        
        Returns:
            Feature tensor
        """
        features = [predictions.flatten()]
        
        # Add time features if available
        if 'time_of_day' in metadata:
            features.append(metadata['time_of_day'].flatten())
        
        if 'day_of_year' in metadata:
            features.append(metadata['day_of_year'].flatten())
        
        # Add spatial features if available
        if 'elevation' in metadata:
            features.append(metadata['elevation'].flatten())
        
        if 'distance_to_coast' in metadata:
            features.append(metadata['distance_to_coast'].flatten())
        
        # Stack features
        feature_tensor = torch.stack(features, dim=1)
        
        return feature_tensor
    
    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features using stored statistics."""
        if not self.fitted:
            return features
        
        normalized = features.clone()
        for i, (mean, std) in enumerate(zip(
            self.feature_stats['mean'],
            self.feature_stats['std']
        )):
            normalized[:, i] = (normalized[:, i] - mean) / (std + 1e-8)
        
        return normalized
    
    def fit(
        self,
        predictions: torch.Tensor,
        observations: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
        epochs: int = 100
    ):
        """Train the neural network correction."""
        # Prepare features
        features = self.prepare_features(predictions, metadata)
        
        # Calculate feature statistics
        self.feature_stats = {
            'mean': torch.mean(features, dim=0),
            'std': torch.std(features, dim=0)
        }
        
        # Normalize features
        features_norm = self.normalize_features(features)
        
        # Target is the correction (observation - prediction)
        targets = observations.flatten() - predictions.flatten()
        
        # Training setup
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            corrections = self.network(features_norm).squeeze()
            loss = criterion(corrections, targets)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Neural correction epoch {epoch}, loss: {loss.item():.4f}")
        
        self.fitted = True
    
    def correct(
        self,
        predictions: torch.Tensor,
        metadata: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply neural network correction."""
        if not self.fitted:
            logger.warning("Neural network correction not fitted")
            return predictions
        
        self.eval()
        
        with torch.no_grad():
            features = self.prepare_features(predictions, metadata)
            features_norm = self.normalize_features(features)
            
            corrections = self.network(features_norm).squeeze()
            corrected = predictions.flatten() + corrections
            
            # Reshape to original shape
            corrected = corrected.reshape(predictions.shape)
        
        return corrected


class AdaptiveBiasCorrection:
    """Adaptive bias correction that updates over time."""
    
    def __init__(self, config: BiasConfig):
        self.config = config
        self.correction_history = {}
        self.recent_errors = {}
        self.adaptation_weights = {}
        self.last_update = {}
    
    def update(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
        timestamp: datetime,
        variable: str = "default"
    ):
        """
        Update adaptive correction with new data.
        
        Args:
            predictions: Recent predictions
            observations: Corresponding observations
            timestamp: Current timestamp
            variable: Variable name
        """
        # Calculate recent errors
        errors = observations - predictions
        valid_errors = errors[~np.isnan(errors)]
        
        if len(valid_errors) == 0:
            return
        
        # Initialize if first update
        if variable not in self.correction_history:
            self.correction_history[variable] = []
            self.recent_errors[variable] = []
            self.adaptation_weights[variable] = []
            self.last_update[variable] = timestamp
        
        # Add to recent errors with time decay
        time_diff = (timestamp - self.last_update[variable]).total_seconds() / 86400  # days
        decay_factor = self.config.memory_factor ** time_diff
        
        # Update running statistics
        current_bias = np.mean(valid_errors)
        current_std = np.std(valid_errors)
        
        self.recent_errors[variable].append({
            'timestamp': timestamp,
            'bias': current_bias,
            'std': current_std,
            'weight': decay_factor
        })
        
        # Keep only recent data
        cutoff_date = timestamp - timedelta(days=self.config.adaptation_window)
        self.recent_errors[variable] = [
            err for err in self.recent_errors[variable]
            if err['timestamp'] > cutoff_date
        ]
        
        self.last_update[variable] = timestamp
    
    def get_adaptive_correction(
        self,
        predictions: np.ndarray,
        timestamp: datetime,
        variable: str = "default"
    ) -> np.ndarray:
        """
        Get adaptive bias correction.
        
        Args:
            predictions: Current predictions
            timestamp: Current timestamp
            variable: Variable name
        
        Returns:
            Corrected predictions
        """
        if variable not in self.recent_errors or not self.recent_errors[variable]:
            return predictions
        
        # Calculate weighted bias correction
        total_weight = 0
        weighted_bias = 0
        
        for error_info in self.recent_errors[variable]:
            time_diff = (timestamp - error_info['timestamp']).total_seconds() / 86400
            weight = error_info['weight'] * (self.config.memory_factor ** time_diff)
            
            weighted_bias += error_info['bias'] * weight
            total_weight += weight
        
        if total_weight > 0:
            adaptive_bias = weighted_bias / total_weight
            corrected = predictions + adaptive_bias
        else:
            corrected = predictions
        
        return corrected


class BiasCorrection:
    """Main bias correction system."""
    
    def __init__(self, config: BiasConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize correction methods
        self.methods = {}
        self._initialize_methods()
        
        # Regional correction models
        self.regional_models = {}
        
        # Performance tracking
        self.correction_stats = {}
    
    def _initialize_methods(self):
        """Initialize bias correction methods."""
        if "quantile_mapping" in self.config.methods:
            self.methods["quantile_mapping"] = QuantileMapping(self.config)
        
        if "linear_scaling" in self.config.methods:
            self.methods["linear_scaling"] = LinearScaling()
        
        if "delta_method" in self.config.methods:
            self.methods["delta_method"] = DeltaMethod()
        
        if "neural_network" in self.config.methods:
            self.methods["neural_network"] = NeuralNetworkCorrection(self.config)
        
        if "adaptive" in self.config.methods:
            self.methods["adaptive"] = AdaptiveBiasCorrection(self.config)
    
    def fit_correction(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        observations: Union[np.ndarray, torch.Tensor],
        variable: str = "default",
        region: str = "global",
        metadata: Dict = None
    ):
        """
        Fit bias correction models.
        
        Args:
            predictions: Historical predictions
            observations: Corresponding observations
            variable: Weather variable name
            region: Geographic region
            metadata: Additional metadata for ML methods
        """
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()
        
        # Fit each method
        for method_name, method in self.methods.items():
            try:
                if method_name == "neural_network" and metadata is not None:
                    # Convert metadata to torch tensors
                    torch_metadata = {}
                    for key, value in metadata.items():
                        if isinstance(value, np.ndarray):
                            torch_metadata[key] = torch.from_numpy(value).float()
                        else:
                            torch_metadata[key] = value
                    
                    method.fit(
                        torch.from_numpy(predictions).float(),
                        torch.from_numpy(observations).float(),
                        torch_metadata
                    )
                else:
                    method.fit(predictions, observations, variable)
                
                logger.info(f"Fitted {method_name} for variable {variable}, region {region}")
                
            except Exception as e:
                logger.error(f"Failed to fit {method_name}: {str(e)}")
    
    def apply_correction(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        variable: str = "default",
        region: str = "global",
        method: str = None,
        metadata: Dict = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply bias correction to predictions.
        
        Args:
            predictions: Predictions to correct
            variable: Weather variable name
            region: Geographic region
            method: Specific method to use (default: primary_method)
            metadata: Additional metadata for ML methods
        
        Returns:
            Corrected predictions
        """
        if method is None:
            method = self.config.primary_method
        
        if method not in self.methods:
            logger.warning(f"Method {method} not available")
            return predictions
        
        # Convert to appropriate format
        original_type = type(predictions)
        if isinstance(predictions, torch.Tensor):
            pred_array = predictions.cpu().numpy()
        else:
            pred_array = predictions
        
        # Apply correction
        try:
            if method == "neural_network" and metadata is not None:
                # Convert to torch tensors
                torch_pred = torch.from_numpy(pred_array).float()
                torch_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, np.ndarray):
                        torch_metadata[key] = torch.from_numpy(value).float()
                    else:
                        torch_metadata[key] = value
                
                corrected = self.methods[method].correct(torch_pred, torch_metadata)
                corrected_array = corrected.cpu().numpy()
            
            elif method == "adaptive":
                # Need timestamp for adaptive method
                timestamp = metadata.get('timestamp', datetime.now()) if metadata else datetime.now()
                corrected_array = self.methods[method].get_adaptive_correction(
                    pred_array, timestamp, variable
                )
            
            else:
                corrected_array = self.methods[method].correct(pred_array, variable)
            
            # Convert back to original type
            if original_type == torch.Tensor:
                return torch.from_numpy(corrected_array).float()
            else:
                return corrected_array
                
        except Exception as e:
            logger.error(f"Failed to apply {method} correction: {str(e)}")
            return predictions
    
    def ensemble_correction(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        variable: str = "default",
        weights: Dict[str, float] = None,
        metadata: Dict = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply ensemble of correction methods.
        
        Args:
            predictions: Predictions to correct
            variable: Weather variable name
            weights: Method weights for ensemble
            metadata: Additional metadata
        
        Returns:
            Ensemble corrected predictions
        """
        if weights is None:
            # Equal weights
            weights = {method: 1.0 / len(self.methods) for method in self.methods.keys()}
        
        corrected_ensemble = []
        total_weight = 0
        
        for method_name, weight in weights.items():
            if method_name in self.methods:
                corrected = self.apply_correction(
                    predictions, variable, method=method_name, metadata=metadata
                )
                corrected_ensemble.append(corrected * weight)
                total_weight += weight
        
        if corrected_ensemble:
            # Combine weighted corrections
            if isinstance(predictions, torch.Tensor):
                ensemble_result = torch.stack(corrected_ensemble).sum(dim=0) / total_weight
            else:
                ensemble_result = np.sum(corrected_ensemble, axis=0) / total_weight
            
            return ensemble_result
        else:
            return predictions
    
    def evaluate_correction(
        self,
        original_predictions: Union[np.ndarray, torch.Tensor],
        corrected_predictions: Union[np.ndarray, torch.Tensor],
        observations: Union[np.ndarray, torch.Tensor],
        variable: str = "default"
    ) -> Dict[str, float]:
        """
        Evaluate bias correction performance.
        
        Args:
            original_predictions: Original model predictions
            corrected_predictions: Bias-corrected predictions
            observations: True observations
            variable: Weather variable name
        
        Returns:
            Evaluation metrics
        """
        # Convert to numpy
        if isinstance(original_predictions, torch.Tensor):
            orig_pred = original_predictions.cpu().numpy()
        else:
            orig_pred = original_predictions
        
        if isinstance(corrected_predictions, torch.Tensor):
            corr_pred = corrected_predictions.cpu().numpy()
        else:
            corr_pred = corrected_predictions
        
        if isinstance(observations, torch.Tensor):
            obs = observations.cpu().numpy()
        else:
            obs = observations
        
        # Remove NaN values
        valid_mask = ~(np.isnan(orig_pred) | np.isnan(corr_pred) | np.isnan(obs))
        orig_clean = orig_pred[valid_mask]
        corr_clean = corr_pred[valid_mask]
        obs_clean = obs[valid_mask]
        
        if len(obs_clean) == 0:
            return {}
        
        # Calculate metrics
        metrics = {}
        
        # Bias
        orig_bias = np.mean(orig_clean - obs_clean)
        corr_bias = np.mean(corr_clean - obs_clean)
        metrics['original_bias'] = float(orig_bias)
        metrics['corrected_bias'] = float(corr_bias)
        metrics['bias_reduction'] = float(abs(orig_bias) - abs(corr_bias))
        
        # RMSE
        orig_rmse = np.sqrt(np.mean((orig_clean - obs_clean) ** 2))
        corr_rmse = np.sqrt(np.mean((corr_clean - obs_clean) ** 2))
        metrics['original_rmse'] = float(orig_rmse)
        metrics['corrected_rmse'] = float(corr_rmse)
        metrics['rmse_improvement'] = float(orig_rmse - corr_rmse)
        
        # MAE
        orig_mae = np.mean(np.abs(orig_clean - obs_clean))
        corr_mae = np.mean(np.abs(corr_clean - obs_clean))
        metrics['original_mae'] = float(orig_mae)
        metrics['corrected_mae'] = float(corr_mae)
        metrics['mae_improvement'] = float(orig_mae - corr_mae)
        
        # Correlation
        if np.std(obs_clean) > 0:
            orig_corr = np.corrcoef(orig_clean, obs_clean)[0, 1]
            corr_corr = np.corrcoef(corr_clean, obs_clean)[0, 1]
            metrics['original_correlation'] = float(orig_corr)
            metrics['corrected_correlation'] = float(corr_corr)
            metrics['correlation_improvement'] = float(corr_corr - orig_corr)
        
        return metrics
    
    def save_correction_models(self, save_path: str):
        """Save fitted correction models."""
        save_data = {
            'config': self.config.__dict__,
            'methods': {},
            'correction_stats': self.correction_stats
        }
        
        for method_name, method in self.methods.items():
            if method_name == "neural_network":
                # Save neural network state
                save_data['methods'][method_name] = {
                    'state_dict': method.state_dict(),
                    'feature_stats': method.feature_stats,
                    'fitted': method.fitted
                }
            elif hasattr(method, '__dict__'):
                # Save other method parameters
                save_data['methods'][method_name] = method.__dict__
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved bias correction models to {save_path}")
    
    def load_correction_models(self, load_path: str):
        """Load fitted correction models."""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore methods
        for method_name, method_data in save_data['methods'].items():
            if method_name == "neural_network" and method_name in self.methods:
                # Restore neural network
                self.methods[method_name].load_state_dict(method_data['state_dict'])
                self.methods[method_name].feature_stats = method_data['feature_stats']
                self.methods[method_name].fitted = method_data['fitted']
            elif method_name in self.methods:
                # Restore other methods
                self.methods[method_name].__dict__.update(method_data)
        
        self.correction_stats = save_data.get('correction_stats', {})
        
        logger.info(f"Loaded bias correction models from {load_path}")


# Bangladesh-specific bias correction
class BangladeshBiasCorrection(BiasCorrection):
    """Bias correction specifically tuned for Bangladesh weather patterns."""
    
    def __init__(self, config: BiasConfig):
        super().__init__(config)
        
        # Bangladesh-specific regions
        self.bangladesh_regions = {
            'dhaka': {'lat_range': (23.5, 24.2), 'lon_range': (90.0, 90.8)},
            'chittagong': {'lat_range': (22.0, 23.0), 'lon_range': (91.0, 92.0)},
            'sylhet': {'lat_range': (24.5, 25.2), 'lon_range': (91.3, 92.2)},
            'rajshahi': {'lat_range': (24.0, 25.0), 'lon_range': (88.0, 89.0)},
            'barisal': {'lat_range': (22.3, 23.0), 'lon_range': (90.0, 91.0)},
            'rangpur': {'lat_range': (25.4, 26.0), 'lon_range': (89.0, 90.0)},
            'mymensingh': {'lat_range': (24.5, 25.5), 'lon_range': (90.0, 91.0)},
            'khulna': {'lat_range': (22.5, 23.5), 'lon_range': (89.0, 90.0)}
        }
        
        # Seasonal patterns requiring different corrections
        self.seasonal_patterns = {
            'monsoon': {
                'months': [6, 7, 8, 9],
                'precipitation_bias': 'high',
                'temperature_bias': 'low',
                'humidity_bias': 'medium'
            },
            'winter': {
                'months': [12, 1, 2],
                'precipitation_bias': 'low',
                'temperature_bias': 'high',
                'humidity_bias': 'low'
            },
            'pre_monsoon': {
                'months': [3, 4, 5],
                'precipitation_bias': 'medium',
                'temperature_bias': 'high',
                'humidity_bias': 'medium'
            },
            'post_monsoon': {
                'months': [10, 11],
                'precipitation_bias': 'medium',
                'temperature_bias': 'medium',
                'humidity_bias': 'medium'
            }
        }
    
    def get_seasonal_weights(self, month: int) -> Dict[str, float]:
        """Get seasonal correction weights."""
        # Determine season
        season = None
        for season_name, season_info in self.seasonal_patterns.items():
            if month in season_info['months']:
                season = season_name
                break
        
        if season is None:
            return {'quantile_mapping': 0.4, 'linear_scaling': 0.3, 'adaptive': 0.3}
        
        # Season-specific weights
        if season == 'monsoon':
            # More emphasis on quantile mapping for extreme precipitation
            return {'quantile_mapping': 0.5, 'neural_network': 0.3, 'adaptive': 0.2}
        elif season == 'winter':
            # Linear methods work well for stable winter conditions
            return {'linear_scaling': 0.4, 'delta_method': 0.3, 'quantile_mapping': 0.3}
        else:
            # Balanced approach for transition seasons
            return {'quantile_mapping': 0.35, 'neural_network': 0.35, 'adaptive': 0.3}
    
    def regional_correction(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        coordinates: Union[np.ndarray, torch.Tensor],
        variable: str = "default",
        month: int = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply region-specific bias correction.
        
        Args:
            predictions: Predictions to correct
            coordinates: Lat/lon coordinates
            variable: Weather variable
            month: Month for seasonal adjustment
        
        Returns:
            Regionally corrected predictions
        """
        # Convert coordinates to numpy
        if isinstance(coordinates, torch.Tensor):
            coords = coordinates.cpu().numpy()
        else:
            coords = coordinates
        
        corrected = predictions.copy()
        
        # Apply correction for each region
        for region_name, region_bounds in self.bangladesh_regions.items():
            # Find points in this region
            lat_mask = ((coords[:, 0] >= region_bounds['lat_range'][0]) & 
                       (coords[:, 0] <= region_bounds['lat_range'][1]))
            lon_mask = ((coords[:, 1] >= region_bounds['lon_range'][0]) & 
                       (coords[:, 1] <= region_bounds['lon_range'][1]))
            region_mask = lat_mask & lon_mask
            
            if np.sum(region_mask) > 0:
                # Get seasonal weights if month provided
                weights = self.get_seasonal_weights(month) if month else None
                
                # Apply ensemble correction to this region
                region_preds = predictions[region_mask] if isinstance(predictions, np.ndarray) else predictions[region_mask]
                
                region_corrected = self.ensemble_correction(
                    region_preds,
                    variable=f"{variable}_{region_name}",
                    weights=weights
                )
                
                if isinstance(corrected, torch.Tensor):
                    corrected[region_mask] = region_corrected
                else:
                    corrected[region_mask] = region_corrected
        
        return corrected
