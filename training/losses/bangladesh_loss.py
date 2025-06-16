"""
Bangladesh-specific loss functions for GraphCast training
"""

import math
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BangladeshLoss:
    """
    Multi-objective loss function tailored for Bangladesh weather prediction
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Loss component weights
        self.weights = config.get('loss_weights', {
            'standard_vars': 1.0,
            'precipitation': 2.5,      # Critical for agriculture and flooding
            'cyclone_track': 5.0,      # High impact events
            'cyclone_intensity': 4.0,  # Life-threatening parameter
            'monsoon_onset': 3.0,      # Agricultural planning
            'extreme_precip': 3.5,     # Flooding prediction
            'heat_stress': 2.0,        # Health impacts
            'flood_risk': 3.0,         # Composite flooding risk
            'coastal_surge': 4.5,      # Storm surge prediction
            'wind_damage': 3.5         # Infrastructure damage
        })
        
        # Evaluation thresholds
        self.thresholds = config.get('evaluation_thresholds', {
            'precipitation': [1, 5, 10, 25, 50, 100],  # mm
            'wind_speed': [10, 15, 25, 35],             # m/s
            'temperature': [30, 35, 40],                # °C extreme heat
            'storm_surge': [0.5, 1.0, 2.0, 3.0]       # m above MSL
        })
        
        # Temporal weights (forecast lead time)
        self.temporal_weights = self._create_temporal_weights()
        
        # Spatial weights (geographic importance)
        self.spatial_weights = self._create_spatial_weights()
    
    def __call__(self, predictions: Dict, targets: Dict, metadata: Dict) -> Dict:
        """
        Calculate total weighted loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            metadata: Additional information (masks, time, location, etc.)
        
        Returns:
            Dictionary containing total loss and component losses
        """
        total_loss = 0.0
        component_losses = {}
        
        # 1. Standard meteorological variables loss
        standard_loss = self.standard_loss(predictions, targets, metadata)
        total_loss += self.weights['standard_vars'] * standard_loss
        component_losses['standard'] = standard_loss
        
        # 2. Precipitation-specific loss
        precip_loss = self.precipitation_loss(predictions, targets, metadata)
        total_loss += self.weights['precipitation'] * precip_loss
        component_losses['precipitation'] = precip_loss
        
        # 3. Cyclone-related losses
        if metadata.get('has_cyclone', False):
            cyclone_losses = self.cyclone_losses(predictions, targets, metadata)
            
            total_loss += self.weights['cyclone_track'] * cyclone_losses['track']
            total_loss += self.weights['cyclone_intensity'] * cyclone_losses['intensity']
            total_loss += self.weights['coastal_surge'] * cyclone_losses['surge']
            total_loss += self.weights['wind_damage'] * cyclone_losses['wind_damage']
            
            component_losses.update({
                'cyclone_track': cyclone_losses['track'],
                'cyclone_intensity': cyclone_losses['intensity'],
                'coastal_surge': cyclone_losses['surge'],
                'wind_damage': cyclone_losses['wind_damage']
            })
        
        # 4. Monsoon onset/withdrawal loss
        if metadata.get('monsoon_period', False):
            monsoon_loss = self.monsoon_timing_loss(predictions, targets, metadata)
            total_loss += self.weights['monsoon_onset'] * monsoon_loss
            component_losses['monsoon_timing'] = monsoon_loss
        
        # 5. Extreme precipitation loss
        extreme_precip_loss = self.extreme_precipitation_loss(predictions, targets, metadata)
        total_loss += self.weights['extreme_precip'] * extreme_precip_loss
        component_losses['extreme_precipitation'] = extreme_precip_loss
        
        # 6. Heat stress loss
        heat_loss = self.heat_stress_loss(predictions, targets, metadata)
        total_loss += self.weights['heat_stress'] * heat_loss
        component_losses['heat_stress'] = heat_loss
        
        # 7. Compound flood risk loss
        if metadata.get('flood_risk_available', False):
            flood_loss = self.flood_risk_loss(predictions, targets, metadata)
            total_loss += self.weights['flood_risk'] * flood_loss
            component_losses['flood_risk'] = flood_loss
        
        # Apply temporal and spatial weighting
        total_loss = self._apply_temporal_weighting(total_loss, metadata)
        total_loss = self._apply_spatial_weighting(total_loss, metadata)
        
        return {
            'total_loss': total_loss,
            'component_losses': component_losses,
            'loss_metadata': {
                'lead_time': metadata.get('lead_time', 0),
                'season': metadata.get('season', 'unknown'),
                'has_cyclone': metadata.get('has_cyclone', False),
                'num_grid_points': metadata.get('num_grid_points', 0)
            }
        }
    
    def standard_loss(self, predictions: Dict, targets: Dict, metadata: Dict) -> float:
        """
        Loss for standard meteorological variables (temperature, pressure, humidity, wind)
        """
        total_loss = 0.0
        num_vars = 0
        
        standard_vars = ['temperature', 'pressure', 'humidity', 'u_wind', 'v_wind']
        
        for var in standard_vars:
            if var in predictions and var in targets:
                # Mean Squared Error with spatial weighting
                var_loss = self._weighted_mse(
                    predictions[var], 
                    targets[var], 
                    metadata.get('spatial_mask', None)
                )
                
                # Variable-specific scaling
                if var == 'temperature':
                    var_loss *= 0.1  # Temperature errors in °C
                elif var == 'pressure':
                    var_loss *= 0.001  # Pressure errors in hPa
                elif var in ['u_wind', 'v_wind']:
                    var_loss *= 0.01  # Wind errors in m/s
                
                total_loss += var_loss
                num_vars += 1
        
        return total_loss / max(num_vars, 1)
    
    def precipitation_loss(self, predictions: Dict, targets: Dict, metadata: Dict) -> float:
        """
        Enhanced loss for precipitation with spatial attention
        """
        if 'precipitation' not in predictions or 'precipitation' not in targets:
            return 0.0
        
        pred_precip = predictions['precipitation']
        target_precip = targets['precipitation']
        
        # Basic MSE loss
        mse_loss = self._weighted_mse(pred_precip, target_precip, metadata.get('spatial_mask'))
        
        # Threshold-based losses for different precipitation intensities
        threshold_loss = 0.0
        for threshold in self.thresholds['precipitation']:
            # Binary classification loss for exceeding threshold
            pred_binary = self._threshold_binary(pred_precip, threshold)
            target_binary = self._threshold_binary(target_precip, threshold)
            
            # Weighted binary cross-entropy
            bce_loss = self._binary_cross_entropy(pred_binary, target_binary)
            
            # Higher weight for extreme precipitation thresholds
            threshold_weight = 1.0 + (threshold / 50.0)  # Increases with threshold
            threshold_loss += threshold_weight * bce_loss
        
        # Frequency bias penalty (to avoid systematic over/under-prediction)
        freq_bias_loss = self._frequency_bias_loss(pred_precip, target_precip)
        
        # Spatial gradient loss (preserve precipitation patterns)
        gradient_loss = self._spatial_gradient_loss(pred_precip, target_precip)
        
        # Combine losses
        total_precip_loss = (
            mse_loss + 
            0.5 * threshold_loss + 
            0.3 * freq_bias_loss + 
            0.2 * gradient_loss
        )
        
        return total_precip_loss
    
    def cyclone_losses(self, predictions: Dict, targets: Dict, metadata: Dict) -> Dict:
        """
        Comprehensive cyclone-related losses
        """
        cyclone_losses = {
            'track': 0.0,
            'intensity': 0.0,
            'surge': 0.0,
            'wind_damage': 0.0
        }
        
        # Extract cyclone information
        pred_cyclones = predictions.get('cyclones', [])
        target_cyclones = targets.get('cyclones', [])
        
        if not pred_cyclones or not target_cyclones:
            return cyclone_losses
        
        # Track error (position)
        track_error = self._cyclone_track_error(pred_cyclones, target_cyclones)
        cyclone_losses['track'] = track_error
        
        # Intensity error (maximum wind speed)
        intensity_error = self._cyclone_intensity_error(pred_cyclones, target_cyclones)
        cyclone_losses['intensity'] = intensity_error
        
        # Storm surge prediction error
        surge_error = self._storm_surge_error(predictions, targets, metadata)
        cyclone_losses['surge'] = surge_error
        
        # Wind damage potential error
        wind_damage_error = self._wind_damage_error(predictions, targets, metadata)
        cyclone_losses['wind_damage'] = wind_damage_error
        
        return cyclone_losses
    
    def monsoon_timing_loss(self, predictions: Dict, targets: Dict, metadata: Dict) -> float:
        """
        Loss for monsoon onset and withdrawal timing
        """
        # Check if this is a critical monsoon period
        current_date = metadata.get('date')
        if not current_date:
            return 0.0
        
        # Monsoon onset period (May-June)
        if current_date.month in [5, 6]:
            return self._monsoon_onset_loss(predictions, targets, metadata)
        
        # Monsoon withdrawal period (September-October)
        elif current_date.month in [9, 10]:
            return self._monsoon_withdrawal_loss(predictions, targets, metadata)
        
        return 0.0
    
    def extreme_precipitation_loss(self, predictions: Dict, targets: Dict, metadata: Dict) -> float:
        """
        Specialized loss for extreme precipitation events (>50mm/day)
        """
        if 'precipitation' not in predictions or 'precipitation' not in targets:
            return 0.0
        
        pred_precip = predictions['precipitation']
        target_precip = targets['precipitation']
        
        # Identify extreme precipitation events
        extreme_threshold = 50.0  # mm/day
        extreme_mask = self._create_extreme_precipitation_mask(target_precip, extreme_threshold)
        
        if not any(extreme_mask):
            return 0.0  # No extreme events
        
        # Focus loss on extreme events
        extreme_pred = [p for i, p in enumerate(pred_precip) if extreme_mask[i]]
        extreme_target = [t for i, t in enumerate(target_precip) if extreme_mask[i]]
        
        # Enhanced MSE for extreme events
        extreme_mse = self._mse(extreme_pred, extreme_target)
        
        # Penalty for missing extreme events (false negatives)
        false_negative_penalty = self._false_negative_penalty(
            pred_precip, target_precip, extreme_threshold
        )
        
        # Penalty for false extreme events (false positives)
        false_positive_penalty = self._false_positive_penalty(
            pred_precip, target_precip, extreme_threshold
        )
        
        return extreme_mse + false_negative_penalty + false_positive_penalty
    
    def heat_stress_loss(self, predictions: Dict, targets: Dict, metadata: Dict) -> float:
        """
        Loss for heat stress prediction (temperature > 35°C with high humidity)
        """
        if 'temperature' not in predictions or 'humidity' not in predictions:
            return 0.0
        
        pred_temp = predictions['temperature']
        pred_humidity = predictions.get('humidity', [50] * len(pred_temp))
        target_temp = targets['temperature']
        target_humidity = targets.get('humidity', [50] * len(target_temp))
        
        # Calculate heat index for predictions and targets
        pred_heat_index = self._calculate_heat_index(pred_temp, pred_humidity)
        target_heat_index = self._calculate_heat_index(target_temp, target_humidity)
        
        # Focus on dangerous heat index values (>40°C equivalent)
        dangerous_threshold = 40.0
        dangerous_mask = [hi > dangerous_threshold for hi in target_heat_index]
        
        if not any(dangerous_mask):
            return 0.0
        
        # Enhanced loss for dangerous heat conditions
        dangerous_pred = [p for i, p in enumerate(pred_heat_index) if dangerous_mask[i]]
        dangerous_target = [t for i, t in enumerate(target_heat_index) if dangerous_mask[i]]
        
        return self._mse(dangerous_pred, dangerous_target)
    
    def flood_risk_loss(self, predictions: Dict, targets: Dict, metadata: Dict) -> float:
        """
        Loss for compound flood risk prediction
        """
        if 'flood_risk' not in predictions or 'flood_risk' not in targets:
            return 0.0
        
        pred_risk = predictions['flood_risk']
        target_risk = targets['flood_risk']
        
        # Multi-class classification loss for flood risk levels
        risk_levels = ['low', 'moderate', 'high', 'extreme']
        
        # Convert risk levels to probabilities
        pred_probs = self._risk_to_probabilities(pred_risk, risk_levels)
        target_probs = self._risk_to_probabilities(target_risk, risk_levels)
        
        # Cross-entropy loss
        ce_loss = self._cross_entropy_loss(pred_probs, target_probs)
        
        # Additional penalty for missing high-risk events
        high_risk_penalty = self._high_risk_penalty(pred_risk, target_risk)
        
        return ce_loss + high_risk_penalty
    
    def _weighted_mse(self, predictions: List[float], targets: List[float], 
                     spatial_mask: Optional[List[float]] = None) -> float:
        """Calculate weighted Mean Squared Error"""
        if len(predictions) != len(targets):
            return float('inf')
        
        if spatial_mask is None:
            spatial_mask = [1.0] * len(predictions)
        
        total_error = 0.0
        total_weight = 0.0
        
        for pred, target, weight in zip(predictions, targets, spatial_mask):
            error = (pred - target) ** 2
            total_error += weight * error
            total_weight += weight
        
        return total_error / max(total_weight, 1e-6)
    
    def _mse(self, predictions: List[float], targets: List[float]) -> float:
        """Calculate Mean Squared Error"""
        if len(predictions) != len(targets) or len(predictions) == 0:
            return 0.0
        
        total_error = sum((p - t) ** 2 for p, t in zip(predictions, targets))
        return total_error / len(predictions)
    
    def _threshold_binary(self, values: List[float], threshold: float) -> List[float]:
        """Convert values to binary based on threshold"""
        return [1.0 if v > threshold else 0.0 for v in values]
    
    def _binary_cross_entropy(self, pred_binary: List[float], target_binary: List[float]) -> float:
        """Calculate binary cross-entropy loss"""
        if len(pred_binary) != len(target_binary) or len(pred_binary) == 0:
            return 0.0
        
        eps = 1e-6  # Small value to prevent log(0)
        total_loss = 0.0
        
        for pred, target in zip(pred_binary, target_binary):
            pred_clipped = max(eps, min(1 - eps, pred))  # Clip to avoid log(0)
            loss = -(target * math.log(pred_clipped) + (1 - target) * math.log(1 - pred_clipped))
            total_loss += loss
        
        return total_loss / len(pred_binary)
    
    def _frequency_bias_loss(self, predictions: List[float], targets: List[float]) -> float:
        """Calculate frequency bias loss to prevent systematic errors"""
        if len(predictions) == 0:
            return 0.0
        
        pred_mean = sum(predictions) / len(predictions)
        target_mean = sum(targets) / len(targets)
        
        # Penalize systematic bias
        bias = abs(pred_mean - target_mean) / max(target_mean, 1.0)
        return bias
    
    def _spatial_gradient_loss(self, predictions: List[float], targets: List[float]) -> float:
        """Calculate spatial gradient loss to preserve patterns"""
        if len(predictions) < 2:
            return 0.0
        
        # Simple gradient approximation
        pred_gradients = [predictions[i+1] - predictions[i] for i in range(len(predictions)-1)]
        target_gradients = [targets[i+1] - targets[i] for i in range(len(targets)-1)]
        
        return self._mse(pred_gradients, target_gradients)
    
    def _cyclone_track_error(self, pred_cyclones: List[Dict], target_cyclones: List[Dict]) -> float:
        """Calculate cyclone track error"""
        if not pred_cyclones or not target_cyclones:
            return 0.0
        
        # Match cyclones (simplified - use first cyclone)
        pred_center = pred_cyclones[0].get('center', (0, 0))
        target_center = target_cyclones[0].get('center', (0, 0))
        
        # Great circle distance
        distance_error = self._great_circle_distance(pred_center, target_center)
        
        # Convert to normalized error (0-1 scale)
        max_error = 500.0  # km
        return min(distance_error / max_error, 1.0)
    
    def _cyclone_intensity_error(self, pred_cyclones: List[Dict], target_cyclones: List[Dict]) -> float:
        """Calculate cyclone intensity error"""
        if not pred_cyclones or not target_cyclones:
            return 0.0
        
        pred_intensity = pred_cyclones[0].get('max_wind', 0)
        target_intensity = target_cyclones[0].get('max_wind', 0)
        
        # Normalized intensity error
        intensity_error = abs(pred_intensity - target_intensity)
        max_intensity = 100.0  # m/s
        return intensity_error / max_intensity
    
    def _storm_surge_error(self, predictions: Dict, targets: Dict, metadata: Dict) -> float:
        """Calculate storm surge prediction error"""
        if 'storm_surge' not in predictions or 'storm_surge' not in targets:
            return 0.0
        
        pred_surge = predictions['storm_surge']
        target_surge = targets['storm_surge']
        
        return self._mse(pred_surge, target_surge)
    
    def _wind_damage_error(self, predictions: Dict, targets: Dict, metadata: Dict) -> float:
        """Calculate wind damage potential error"""
        if 'wind_speed' not in predictions or 'wind_speed' not in targets:
            return 0.0
        
        pred_wind = predictions['wind_speed']
        target_wind = targets['wind_speed']
        
        # Focus on damaging wind speeds (>25 m/s)
        damage_threshold = 25.0
        damage_mask = [w > damage_threshold for w in target_wind]
        
        if not any(damage_mask):
            return 0.0
        
        damage_pred = [p for i, p in enumerate(pred_wind) if damage_mask[i]]
        damage_target = [t for i, t in enumerate(target_wind) if damage_mask[i]]
        
        return self._mse(damage_pred, damage_target)
    
    def _monsoon_onset_loss(self, predictions: Dict, targets: Dict, metadata: Dict) -> float:
        """Calculate monsoon onset timing loss"""
        # Simplified onset detection based on precipitation patterns
        if 'precipitation' not in predictions:
            return 0.0
        
        pred_precip = predictions['precipitation']
        target_precip = targets['precipitation']
        
        # Check for onset criteria (>5mm/day average)
        onset_threshold = 5.0
        
        pred_onset = sum(pred_precip) / len(pred_precip) > onset_threshold
        target_onset = sum(target_precip) / len(target_precip) > onset_threshold
        
        # Binary classification loss for onset detection
        return float(pred_onset != target_onset)
    
    def _monsoon_withdrawal_loss(self, predictions: Dict, targets: Dict, metadata: Dict) -> float:
        """Calculate monsoon withdrawal timing loss"""
        # Similar to onset but for withdrawal detection
        if 'precipitation' not in predictions:
            return 0.0
        
        pred_precip = predictions['precipitation']
        target_precip = targets['precipitation']
        
        withdrawal_threshold = 3.0  # Lower threshold for withdrawal
        
        pred_withdrawal = sum(pred_precip) / len(pred_precip) < withdrawal_threshold
        target_withdrawal = sum(target_precip) / len(target_precip) < withdrawal_threshold
        
        return float(pred_withdrawal != target_withdrawal)
    
    def _create_extreme_precipitation_mask(self, precipitation: List[float], threshold: float) -> List[bool]:
        """Create mask for extreme precipitation events"""
        return [p > threshold for p in precipitation]
    
    def _false_negative_penalty(self, predictions: List[float], targets: List[float], threshold: float) -> float:
        """Penalty for missing extreme events"""
        false_negatives = sum(1 for p, t in zip(predictions, targets) 
                            if t > threshold and p <= threshold)
        total_extremes = sum(1 for t in targets if t > threshold)
        
        if total_extremes == 0:
            return 0.0
        
        return false_negatives / total_extremes
    
    def _false_positive_penalty(self, predictions: List[float], targets: List[float], threshold: float) -> float:
        """Penalty for false extreme events"""
        false_positives = sum(1 for p, t in zip(predictions, targets) 
                            if p > threshold and t <= threshold)
        total_predicted_extremes = sum(1 for p in predictions if p > threshold)
        
        if total_predicted_extremes == 0:
            return 0.0
        
        return false_positives / total_predicted_extremes
    
    def _calculate_heat_index(self, temperature: List[float], humidity: List[float]) -> List[float]:
        """Calculate heat index from temperature and humidity"""
        heat_index = []
        
        for T, RH in zip(temperature, humidity):
            # Heat index calculation (simplified Rothfusz equation)
            if T < 27:  # Below threshold, use temperature
                hi = T
            else:
                # Full heat index calculation
                hi = (-8.78469475556 + 
                     1.61139411 * T + 
                     2.33854883889 * RH + 
                     -0.14611605 * T * RH + 
                     -0.012308094 * T**2 + 
                     -0.0164248277778 * RH**2 + 
                     0.002211732 * T**2 * RH + 
                     0.00072546 * T * RH**2 + 
                     -0.000003582 * T**2 * RH**2)
            
            heat_index.append(hi)
        
        return heat_index
    
    def _risk_to_probabilities(self, risk_levels: List[str], level_names: List[str]) -> List[List[float]]:
        """Convert risk level names to probability distributions"""
        probabilities = []
        
        for risk in risk_levels:
            if risk in level_names:
                idx = level_names.index(risk)
                prob = [0.0] * len(level_names)
                prob[idx] = 1.0
                probabilities.append(prob)
            else:
                # Unknown risk level - uniform distribution
                uniform_prob = [1.0 / len(level_names)] * len(level_names)
                probabilities.append(uniform_prob)
        
        return probabilities
    
    def _cross_entropy_loss(self, pred_probs: List[List[float]], target_probs: List[List[float]]) -> float:
        """Calculate cross-entropy loss for multi-class classification"""
        if len(pred_probs) != len(target_probs) or len(pred_probs) == 0:
            return 0.0
        
        eps = 1e-6
        total_loss = 0.0
        
        for pred_dist, target_dist in zip(pred_probs, target_probs):
            loss = 0.0
            for p_pred, p_target in zip(pred_dist, target_dist):
                p_pred_clipped = max(eps, min(1 - eps, p_pred))
                loss -= p_target * math.log(p_pred_clipped)
            total_loss += loss
        
        return total_loss / len(pred_probs)
    
    def _high_risk_penalty(self, pred_risk: List[str], target_risk: List[str]) -> float:
        """Additional penalty for missing high-risk events"""
        high_risk_levels = ['high', 'extreme']
        
        false_negatives = sum(1 for p, t in zip(pred_risk, target_risk)
                            if t in high_risk_levels and p not in high_risk_levels)
        total_high_risk = sum(1 for t in target_risk if t in high_risk_levels)
        
        if total_high_risk == 0:
            return 0.0
        
        return false_negatives / total_high_risk
    
    def _great_circle_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate great circle distance between two points"""
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        R = 6371  # Earth radius in km
        return R * c
    
    def _create_temporal_weights(self) -> Dict[int, float]:
        """Create temporal weights based on forecast lead time"""
        # Higher weights for shorter lead times
        temporal_weights = {}
        for lead_time in range(0, 241, 6):  # 0 to 240 hours, every 6 hours
            if lead_time <= 24:
                weight = 1.0
            elif lead_time <= 72:
                weight = 0.8
            elif lead_time <= 120:
                weight = 0.6
            else:
                weight = 0.4
            
            temporal_weights[lead_time] = weight
        
        return temporal_weights
    
    def _create_spatial_weights(self) -> Dict[str, float]:
        """Create spatial weights based on geographic importance"""
        return {
            'coastal_zone': 1.5,      # Higher weight for coastal areas
            'river_confluence': 1.3,   # Important for flood prediction
            'urban_areas': 1.2,        # Population centers
            'agricultural': 1.1,       # Agricultural regions
            'default': 1.0
        }
    
    def _apply_temporal_weighting(self, loss: float, metadata: Dict) -> float:
        """Apply temporal weighting to loss"""
        lead_time = metadata.get('lead_time', 0)
        weight = self.temporal_weights.get(lead_time, 1.0)
        return loss * weight
    
    def _apply_spatial_weighting(self, loss: float, metadata: Dict) -> float:
        """Apply spatial weighting to loss"""
        region_type = metadata.get('region_type', 'default')
        weight = self.spatial_weights.get(region_type, 1.0)
        return loss * weight


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    config = {
        'loss_weights': {
            'standard_vars': 1.0,
            'precipitation': 2.5,
            'cyclone_track': 5.0,
            'cyclone_intensity': 4.0,
            'monsoon_onset': 3.0,
            'extreme_precip': 3.5,
            'heat_stress': 2.0
        },
        'evaluation_thresholds': {
            'precipitation': [1, 5, 10, 25, 50, 100],
            'wind_speed': [10, 15, 25, 35],
            'temperature': [30, 35, 40]
        }
    }
    
    # Initialize loss function
    loss_fn = BangladeshLoss(config)
    
    # Example predictions and targets
    predictions = {
        'temperature': [25.0, 30.0, 35.0],
        'precipitation': [2.0, 15.0, 60.0],
        'pressure': [1013.0, 1010.0, 1005.0],
        'cyclones': [{'center': (22.0, 90.0), 'max_wind': 45.0}]
    }
    
    targets = {
        'temperature': [24.0, 32.0, 37.0],
        'precipitation': [1.0, 12.0, 55.0],
        'pressure': [1014.0, 1008.0, 1003.0],
        'cyclones': [{'center': (21.8, 90.2), 'max_wind': 42.0}]
    }
    
    metadata = {
        'has_cyclone': True,
        'lead_time': 24,
        'season': 'monsoon',
        'num_grid_points': 3
    }
    
    # Calculate loss
    loss_result = loss_fn(predictions, targets, metadata)
    
    logger.info(f"Total loss: {loss_result['total_loss']:.4f}")
    logger.info("Component losses:")
    for component, value in loss_result['component_losses'].items():
        logger.info(f"  {component}: {value:.4f}")
    
    logger.info("Bangladesh-specific loss function test completed successfully")
