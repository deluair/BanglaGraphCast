"""
Physics-informed modules for Bangladesh weather prediction
"""

import math
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TropicalCycloneModule:
    """
    Enhanced prediction module for Bay of Bengal tropical cyclones
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.physics_constraints = config.get('cyclone', {})
        
        # Cyclone detection and tracking
        self.vortex_tracker = VortexDetection()
        self.intensity_predictor = CycloneIntensityGNN()
        self.landfall_module = LandfallPrediction()
        
        # Physical constants
        self.constants = {
            'coriolis_parameter': 2 * 7.2921e-5,  # Earth's angular velocity
            'gas_constant': 287.0,  # J/(kg·K)
            'cp': 1004.0,  # Specific heat at constant pressure
            'latent_heat': 2.5e6,  # Latent heat of vaporization
            'stefan_boltzmann': 5.67e-8  # Stefan-Boltzmann constant
        }
        
    def apply_cyclone_physics(self, state: Dict) -> Dict:
        """
        Apply cyclone physics constraints to model state
        
        Enforces:
        - Gradient wind balance in cyclone core
        - Warm core structure
        - Angular momentum conservation
        - SST-intensity relationship
        - Land friction parameterization
        """
        # Detect active cyclones
        cyclones = self.vortex_tracker.detect_cyclones(state)
        
        if cyclones:
            logger.info(f"Detected {len(cyclones)} active cyclone(s)")
            
            for cyclone in cyclones:
                # Apply physics constraints to each cyclone
                state = self._apply_single_cyclone_physics(state, cyclone)
        
        return state
    
    def _apply_single_cyclone_physics(self, state: Dict, cyclone: Dict) -> Dict:
        """Apply physics constraints to a single cyclone"""
        center_lat, center_lon = cyclone['center']
        intensity = cyclone.get('intensity', 0)
        
        # 1. Gradient wind balance
        state = self._enforce_gradient_wind_balance(state, cyclone)
        
        # 2. Warm core structure
        state = self._enforce_warm_core_structure(state, cyclone)
        
        # 3. Angular momentum conservation
        state = self._conserve_angular_momentum(state, cyclone)
        
        # 4. SST-intensity relationship
        state = self._apply_sst_intensity_relationship(state, cyclone)
        
        # 5. Land friction effects
        state = self._apply_land_friction(state, cyclone)
        
        return state
    
    def _enforce_gradient_wind_balance(self, state: Dict, cyclone: Dict) -> Dict:
        """
        Enforce gradient wind balance in cyclone core
        
        The gradient wind equation:
        V²/r + fV = (1/ρ)(dp/dr)
        
        Where V is tangential wind, r is radius, f is Coriolis parameter,
        ρ is density, and dp/dr is radial pressure gradient
        """
        center_lat, center_lon = cyclone['center']
        max_wind = cyclone.get('max_wind', 0)
        
        if max_wind < 17:  # Below tropical storm strength
            return state
        
        # Calculate Coriolis parameter at cyclone center
        f = 2 * self.constants['coriolis_parameter'] * math.sin(math.radians(center_lat))
        
        # Apply gradient wind adjustment in cyclone vicinity
        cyclone_region = self._get_cyclone_region(state, cyclone)
        
        for point in cyclone_region:
            lat, lon = point['coordinates']
            distance = self._calculate_distance(center_lat, center_lon, lat, lon)
            
            if distance < 300:  # Within 300 km of center
                # Calculate theoretical wind from pressure gradient
                if 'pressure' in point and 'wind_speed' in point:
                    pressure_gradient = self._calculate_pressure_gradient(point, cyclone)
                    density = self._calculate_air_density(point)
                    
                    # Solve gradient wind equation
                    theoretical_wind = self._solve_gradient_wind(
                        pressure_gradient, f, density, distance * 1000
                    )
                    
                    # Adjust wind speed toward theoretical value
                    adjustment_factor = 0.1  # Gentle adjustment
                    point['wind_speed'] = (
                        (1 - adjustment_factor) * point['wind_speed'] +
                        adjustment_factor * theoretical_wind
                    )
        
        return state
    
    def _enforce_warm_core_structure(self, state: Dict, cyclone: Dict) -> Dict:
        """
        Enforce warm core temperature structure
        
        Tropical cyclones have warm cores due to latent heat release
        """
        center_lat, center_lon = cyclone['center']
        intensity = cyclone.get('intensity', 0)
        
        if intensity < 25:  # Weak system
            return state
        
        # Calculate warm core temperature anomaly
        max_warming = min(15.0, intensity * 0.2)  # Max 15°C warming
        
        cyclone_region = self._get_cyclone_region(state, cyclone)
        
        for point in cyclone_region:
            lat, lon = point['coordinates']
            distance = self._calculate_distance(center_lat, center_lon, lat, lon)
            
            if distance < 100:  # Within eye wall region
                # Apply maximum warming
                warming = max_warming * math.exp(-distance / 30)
                
                if 'temperature' in point:
                    point['temperature'] += warming
                
                # Adjust pressure levels accordingly
                self._adjust_pressure_levels_for_warming(point, warming)
        
        return state
    
    def _conserve_angular_momentum(self, state: Dict, cyclone: Dict) -> Dict:
        """
        Apply angular momentum conservation constraints
        
        As air spirals inward, it must spin faster (like ice skater)
        """
        center_lat, center_lon = cyclone['center']
        
        cyclone_region = self._get_cyclone_region(state, cyclone)
        
        for point in cyclone_region:
            lat, lon = point['coordinates']
            distance = self._calculate_distance(center_lat, center_lon, lat, lon)
            
            if distance < 200:  # Within significant influence region
                # Calculate expected tangential wind from angular momentum conservation
                # Simplified: V*r = constant
                
                reference_distance = 100  # km
                reference_wind = cyclone.get('max_wind', 0)
                
                if distance > 10 and reference_wind > 0:  # Avoid division by zero
                    expected_wind = reference_wind * reference_distance / distance
                    
                    # Apply constraint gradually
                    if 'wind_speed' in point:
                        constraint_weight = 0.05
                        point['wind_speed'] = (
                            (1 - constraint_weight) * point['wind_speed'] +
                            constraint_weight * expected_wind
                        )
        
        return state
    
    def _apply_sst_intensity_relationship(self, state: Dict, cyclone: Dict) -> Dict:
        """
        Apply sea surface temperature - intensity relationship
        
        Cyclones intensify over warm water (>26.5°C) and weaken over cool water
        """
        center_lat, center_lon = cyclone['center']
        
        # Get SST at cyclone center
        center_sst = self._get_sst_at_location(state, center_lat, center_lon)
        
        if center_sst is None:
            return state
        
        # Calculate potential intensity based on SST
        potential_intensity = self._calculate_potential_intensity(center_sst)
        
        # Adjust cyclone intensity toward potential intensity
        current_intensity = cyclone.get('max_wind', 0)
        
        if center_sst >= 26.5:  # Favorable for intensification
            intensity_tendency = 0.1 * (potential_intensity - current_intensity)
        else:  # Unfavorable - weakening
            intensity_tendency = -0.2 * current_intensity
        
        # Apply intensity change
        new_intensity = max(0, current_intensity + intensity_tendency)
        cyclone['max_wind'] = new_intensity
        
        # Update wind field accordingly
        state = self._update_wind_field_for_intensity(state, cyclone, new_intensity)
        
        return state
    
    def _apply_land_friction(self, state: Dict, cyclone: Dict) -> Dict:
        """
        Apply land friction effects when cyclone moves over land
        """
        center_lat, center_lon = cyclone['center']
        
        # Check if cyclone center is over land
        is_over_land = self._is_location_over_land(center_lat, center_lon)
        
        if is_over_land:
            # Apply friction-induced weakening
            current_intensity = cyclone.get('max_wind', 0)
            
            # Land friction causes rapid weakening
            friction_factor = 0.95  # 5% weakening per time step
            new_intensity = current_intensity * friction_factor
            
            cyclone['max_wind'] = new_intensity
            
            # Update wind field
            state = self._update_wind_field_for_intensity(state, cyclone, new_intensity)
            
            logger.debug(f"Applied land friction: intensity reduced from {current_intensity:.1f} to {new_intensity:.1f}")
        
        return state
    
    def _get_cyclone_region(self, state: Dict, cyclone: Dict) -> List[Dict]:
        """Get grid points in cyclone influence region"""
        # Simplified implementation
        return state.get('grid_points', [])
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km"""
        # Haversine formula
        R = 6371  # Earth radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _calculate_pressure_gradient(self, point: Dict, cyclone: Dict) -> float:
        """Calculate pressure gradient at point"""
        # Simplified pressure gradient calculation
        return 0.001  # Placeholder
    
    def _calculate_air_density(self, point: Dict) -> float:
        """Calculate air density at point"""
        # Using ideal gas law: ρ = p/(RT)
        pressure = point.get('pressure', 101325)  # Pa
        temperature = point.get('temperature', 15) + 273.15  # K
        
        density = pressure / (self.constants['gas_constant'] * temperature)
        return density
    
    def _solve_gradient_wind(self, pressure_gradient: float, f: float, 
                           density: float, radius: float) -> float:
        """Solve gradient wind equation for wind speed"""
        # Simplified solution
        # V²/r + fV = (1/ρ)(dp/dr)
        # This is a quadratic equation in V
        
        if radius == 0:
            return 0
        
        a = 1 / radius
        b = f
        c = -pressure_gradient / density
        
        # Solve quadratic equation
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return 0
        
        # Take positive solution
        v = (-b + math.sqrt(discriminant)) / (2*a)
        return max(0, v)
    
    def _adjust_pressure_levels_for_warming(self, point: Dict, warming: float):
        """Adjust pressure levels for warm core warming"""
        # Simplified hydrostatic adjustment
        if 'pressure' in point:
            # Pressure decreases with warming aloft
            pressure_reduction = warming * 2.0  # hPa per °C
            point['pressure'] = max(900, point['pressure'] - pressure_reduction)
    
    def _get_sst_at_location(self, state: Dict, lat: float, lon: float) -> Optional[float]:
        """Get sea surface temperature at location"""
        # Simplified SST lookup
        # In practice, would interpolate from SST field
        
        # Check if over water
        if self._is_location_over_land(lat, lon):
            return None
        
        # Return climatological SST for Bay of Bengal
        return 28.5  # °C - typical Bay of Bengal SST
    
    def _calculate_potential_intensity(self, sst: float) -> float:
        """Calculate potential intensity from SST"""
        # Simplified potential intensity formula
        # Based on Emanuel's formula
        
        if sst < 26.5:
            return 0
        
        # Potential intensity increases with SST
        potential_intensity = 30 + (sst - 26.5) * 15
        return min(potential_intensity, 85)  # Cap at 85 m/s
    
    def _update_wind_field_for_intensity(self, state: Dict, cyclone: Dict, new_intensity: float) -> Dict:
        """Update wind field based on new intensity"""
        center_lat, center_lon = cyclone['center']
        
        cyclone_region = self._get_cyclone_region(state, cyclone)
        
        for point in cyclone_region:
            lat, lon = point['coordinates']
            distance = self._calculate_distance(center_lat, center_lon, lat, lon)
            
            if distance < 300:  # Within cyclone influence
                # Scale wind speed based on new intensity
                intensity_ratio = new_intensity / max(cyclone.get('max_wind', 1), 1)
                
                if 'wind_speed' in point:
                    point['wind_speed'] *= intensity_ratio
        
        return state
    
    def _is_location_over_land(self, lat: float, lon: float) -> bool:
        """Check if location is over land"""
        # Simplified land mask for Bangladesh region
        # In practice, would use high-resolution land mask
        
        # Bangladesh mainland
        if 20.5 <= lat <= 26.5 and 88.0 <= lon <= 92.5:
            # Exclude major water bodies
            if lat < 21.5 and 89.0 <= lon <= 92.0:  # Bay of Bengal
                return False
            return True
        
        # Surrounding land areas
        if lat > 26.5:  # India
            return True
        if lon < 88.0:  # India
            return True
        
        return False  # Default to water


class MonsoonPhysics:
    """
    Physics module for monsoon dynamics specific to Bangladesh
    """
    
    def __init__(self):
        self.onset_criteria = {
            'rainfall_threshold': 5.0,  # mm/day
            'consecutive_days': 5,
            'wind_shear_reversal': True,
            'moisture_threshold': 40.0  # mm column water vapor
        }
        
        self.monsoon_indices = {
            'all_india_monsoon_index': 0.0,
            'bay_of_bengal_index': 0.0,
            'heat_low_strength': 0.0
        }
    
    def enforce_monsoon_constraints(self, predictions: Dict, current_date: datetime) -> Dict:
        """
        Enforce known monsoon relationships and constraints
        
        Applies:
        - Monsoon trough position
        - Heat low over northwestern India
        - Cross-equatorial flow strength
        - Tibetan anticyclone influence
        """
        # Determine monsoon phase
        monsoon_phase = self._get_monsoon_phase(current_date)
        
        if monsoon_phase == 'active':
            predictions = self._apply_active_monsoon_physics(predictions)
        elif monsoon_phase == 'break':
            predictions = self._apply_break_monsoon_physics(predictions)
        elif monsoon_phase == 'onset':
            predictions = self._apply_onset_physics(predictions)
        elif monsoon_phase == 'withdrawal':
            predictions = self._apply_withdrawal_physics(predictions)
        
        return predictions
    
    def _get_monsoon_phase(self, date: datetime) -> str:
        """Determine current monsoon phase"""
        month = date.month
        day = date.day
        
        if month < 6:
            return 'pre_monsoon'
        elif month == 6 and day < 15:
            return 'onset'
        elif month in [7, 8] or (month == 6 and day >= 15):
            # Determine if active or break phase
            return self._determine_active_break_phase(date)
        elif month == 9:
            return 'withdrawal'
        else:
            return 'post_monsoon'
    
    def _determine_active_break_phase(self, date: datetime) -> str:
        """Determine if monsoon is in active or break phase"""
        # Simplified determination based on climatology
        # In practice, would use real-time indices
        
        # Simplified: alternate between active and break
        week_of_year = date.timetuple().tm_yday // 7
        return 'active' if week_of_year % 2 == 0 else 'break'
    
    def _apply_active_monsoon_physics(self, predictions: Dict) -> Dict:
        """Apply physics for active monsoon phase"""
        # Enhanced southwest flow
        predictions = self._enhance_southwest_flow(predictions)
        
        # Strengthen monsoon trough
        predictions = self._strengthen_monsoon_trough(predictions)
        
        # Increase convective activity
        predictions = self._enhance_convective_activity(predictions)
        
        return predictions
    
    def _apply_break_monsoon_physics(self, predictions: Dict) -> Dict:
        """Apply physics for break monsoon phase"""
        # Weaken southwest flow
        predictions = self._weaken_southwest_flow(predictions)
        
        # Northward shift of monsoon trough
        predictions = self._shift_monsoon_trough_north(predictions)
        
        # Suppress convective activity over Bangladesh
        predictions = self._suppress_convective_activity(predictions)
        
        return predictions
    
    def _apply_onset_physics(self, predictions: Dict) -> Dict:
        """Apply physics for monsoon onset"""
        # Gradual establishment of southwest flow
        predictions = self._establish_southwest_flow(predictions)
        
        # Moisture buildup
        predictions = self._increase_atmospheric_moisture(predictions)
        
        return predictions
    
    def _apply_withdrawal_physics(self, predictions: Dict) -> Dict:
        """Apply physics for monsoon withdrawal"""
        # Weakening of southwest flow
        predictions = self._weaken_southwest_flow(predictions)
        
        # Establishment of northeast flow
        predictions = self._establish_northeast_flow(predictions)
        
        return predictions
    
    def _enhance_southwest_flow(self, predictions: Dict) -> Dict:
        """Enhance southwest monsoon flow"""
        # Strengthen southwest wind component
        for point in predictions.get('grid_points', []):
            lat, lon = point['coordinates']
            
            if 20 <= lat <= 26 and 88 <= lon <= 93:  # Bangladesh region
                if 'u_wind' in point and 'v_wind' in point:
                    # Enhance southwest component
                    point['u_wind'] += 2.0  # Increase westerly component
                    point['v_wind'] += 1.0  # Increase southerly component
        
        return predictions
    
    def _strengthen_monsoon_trough(self, predictions: Dict) -> Dict:
        """Strengthen monsoon trough"""
        # Lower pressure along trough line
        trough_latitude = 25.0  # Approximate trough position
        
        for point in predictions.get('grid_points', []):
            lat, lon = point['coordinates']
            
            # Points near trough line
            if abs(lat - trough_latitude) < 2.0:
                if 'pressure' in point:
                    point['pressure'] -= 2.0  # Lower pressure by 2 hPa
        
        return predictions
    
    def _enhance_convective_activity(self, predictions: Dict) -> Dict:
        """Enhance convective activity during active monsoon"""
        for point in predictions.get('grid_points', []):
            lat, lon = point['coordinates']
            
            if 20 <= lat <= 26 and 88 <= lon <= 93:  # Bangladesh region
                if 'precipitation' in point:
                    # Increase precipitation
                    point['precipitation'] *= 1.5
                
                if 'temperature' in point:
                    # Slight cooling due to increased convection
                    point['temperature'] -= 1.0
        
        return predictions
    
    def _weaken_southwest_flow(self, predictions: Dict) -> Dict:
        """Weaken southwest monsoon flow"""
        for point in predictions.get('grid_points', []):
            lat, lon = point['coordinates']
            
            if 20 <= lat <= 26 and 88 <= lon <= 93:
                if 'u_wind' in point and 'v_wind' in point:
                    # Reduce southwest component
                    point['u_wind'] *= 0.7
                    point['v_wind'] *= 0.7
        
        return predictions
    
    def _shift_monsoon_trough_north(self, predictions: Dict) -> Dict:
        """Shift monsoon trough northward during break phase"""
        # Trough shifts to foothills
        new_trough_latitude = 28.0
        
        for point in predictions.get('grid_points', []):
            lat, lon = point['coordinates']
            
            if abs(lat - new_trough_latitude) < 2.0:
                if 'pressure' in point:
                    point['pressure'] -= 1.0
        
        return predictions
    
    def _suppress_convective_activity(self, predictions: Dict) -> Dict:
        """Suppress convective activity during break phase"""
        for point in predictions.get('grid_points', []):
            lat, lon = point['coordinates']
            
            if 20 <= lat <= 26 and 88 <= lon <= 93:
                if 'precipitation' in point:
                    point['precipitation'] *= 0.3  # Significant reduction
        
        return predictions
    
    def _establish_southwest_flow(self, predictions: Dict) -> Dict:
        """Gradually establish southwest flow during onset"""
        # Similar to enhance but more gradual
        for point in predictions.get('grid_points', []):
            lat, lon = point['coordinates']
            
            if 20 <= lat <= 26 and 88 <= lon <= 93:
                if 'u_wind' in point and 'v_wind' in point:
                    point['u_wind'] += 1.0
                    point['v_wind'] += 0.5
        
        return predictions
    
    def _increase_atmospheric_moisture(self, predictions: Dict) -> Dict:
        """Increase atmospheric moisture during onset"""
        for point in predictions.get('grid_points', []):
            if 'humidity' in point:
                point['humidity'] = min(100, point['humidity'] + 10)
        
        return predictions
    
    def _establish_northeast_flow(self, predictions: Dict) -> Dict:
        """Establish northeast flow during withdrawal"""
        for point in predictions.get('grid_points', []):
            lat, lon = point['coordinates']
            
            if 20 <= lat <= 26 and 88 <= lon <= 93:
                if 'u_wind' in point and 'v_wind' in point:
                    # Reverse to northeast flow
                    point['u_wind'] = -abs(point['u_wind']) * 0.5
                    point['v_wind'] = -abs(point['v_wind']) * 0.5
        
        return predictions


class HydroMetCoupler:
    """
    Couple atmospheric predictions with hydrological model for flood prediction
    """
    
    def __init__(self):
        self.routing_model = MuskingumCunge()
        self.groundwater = MODFLOWSimplified()
        self.tidal_model = TidalBoundary()
        
        # Major rivers in Bangladesh
        self.river_network = {
            'ganges': {'upstream_flow': 0, 'routing_params': {}},
            'brahmaputra': {'upstream_flow': 0, 'routing_params': {}},
            'meghna': {'upstream_flow': 0, 'routing_params': {}}
        }
    
    def predict_compound_flooding(self, weather_pred: Dict) -> Dict:
        """
        Predict compound flooding combining multiple sources:
        - Rainfall-driven riverine flooding
        - Storm surge from cyclones
        - High tide timing
        - Upstream discharge from India
        """
        flood_prediction = {
            'riverine_flood_risk': {},
            'coastal_flood_risk': {},
            'compound_flood_risk': {},
            'evacuation_zones': []
        }
        
        # 1. Riverine flooding from precipitation
        riverine_risk = self._predict_riverine_flooding(weather_pred)
        flood_prediction['riverine_flood_risk'] = riverine_risk
        
        # 2. Coastal flooding from storm surge
        coastal_risk = self._predict_coastal_flooding(weather_pred)
        flood_prediction['coastal_flood_risk'] = coastal_risk
        
        # 3. Combined compound risk
        compound_risk = self._combine_flood_risks(riverine_risk, coastal_risk)
        flood_prediction['compound_flood_risk'] = compound_risk
        
        # 4. Identify evacuation zones
        evacuation_zones = self._identify_evacuation_zones(compound_risk)
        flood_prediction['evacuation_zones'] = evacuation_zones
        
        return flood_prediction
    
    def _predict_riverine_flooding(self, weather_pred: Dict) -> Dict:
        """Predict riverine flooding from precipitation"""
        riverine_risk = {}
        
        # Calculate basin-averaged precipitation
        basin_precip = self._calculate_basin_precipitation(weather_pred)
        
        # Route through river network
        for river_name, river_data in self.river_network.items():
            if river_name in basin_precip:
                # Simple routing model
                upstream_flow = river_data['upstream_flow']
                local_runoff = basin_precip[river_name] * 0.1  # Simple runoff coefficient
                
                total_flow = upstream_flow + local_runoff
                
                # Determine flood risk level
                if total_flow > 50000:  # m³/s
                    risk_level = 'extreme'
                elif total_flow > 30000:
                    risk_level = 'high'
                elif total_flow > 15000:
                    risk_level = 'moderate'
                else:
                    risk_level = 'low'
                
                riverine_risk[river_name] = {
                    'flow_rate': total_flow,
                    'risk_level': risk_level,
                    'flood_stage': total_flow > 20000
                }
        
        return riverine_risk
    
    def _predict_coastal_flooding(self, weather_pred: Dict) -> Dict:
        """Predict coastal flooding from storm surge"""
        coastal_risk = {}
        
        # Check for active cyclones
        cyclones = self._detect_cyclones_in_prediction(weather_pred)
        
        for cyclone in cyclones:
            surge_height = self._calculate_storm_surge(cyclone)
            
            # Get tidal information
            tide_level = self.tidal_model.get_current_tide()
            
            # Combined water level
            total_water_level = surge_height + tide_level
            
            # Determine coastal flood risk
            if total_water_level > 3.0:  # meters above mean sea level
                risk_level = 'extreme'
            elif total_water_level > 2.0:
                risk_level = 'high'
            elif total_water_level > 1.0:
                risk_level = 'moderate'
            else:
                risk_level = 'low'
            
            coastal_risk[f"cyclone_{cyclone.get('id', 'unknown')}"] = {
                'surge_height': surge_height,
                'tide_level': tide_level,
                'total_water_level': total_water_level,
                'risk_level': risk_level
            }
        
        return coastal_risk
    
    def _combine_flood_risks(self, riverine_risk: Dict, coastal_risk: Dict) -> Dict:
        """Combine riverine and coastal flood risks"""
        compound_risk = {}
        
        # Risk level scoring
        risk_scores = {'low': 1, 'moderate': 2, 'high': 3, 'extreme': 4}
        
        # Combine risks for each region
        regions = ['dhaka', 'chittagong', 'khulna', 'barishal', 'sylhet']
        
        for region in regions:
            riverine_score = 0
            coastal_score = 0
            
            # Get riverine risk for region
            for river, risk_data in riverine_risk.items():
                if self._river_affects_region(river, region):
                    riverine_score = max(riverine_score, 
                                       risk_scores.get(risk_data['risk_level'], 0))
            
            # Get coastal risk for region
            for cyclone, risk_data in coastal_risk.items():
                if self._cyclone_affects_region(cyclone, region):
                    coastal_score = max(coastal_score,
                                      risk_scores.get(risk_data['risk_level'], 0))
            
            # Combined score (non-linear combination)
            if riverine_score > 0 and coastal_score > 0:
                # Compound effect is worse than individual risks
                combined_score = min(4, riverine_score + coastal_score - 1)
            else:
                combined_score = max(riverine_score, coastal_score)
            
            # Convert back to risk level
            if combined_score >= 4:
                risk_level = 'extreme'
            elif combined_score >= 3:
                risk_level = 'high'
            elif combined_score >= 2:
                risk_level = 'moderate'
            else:
                risk_level = 'low'
            
            compound_risk[region] = {
                'riverine_component': riverine_score,
                'coastal_component': coastal_score,
                'combined_score': combined_score,
                'risk_level': risk_level
            }
        
        return compound_risk
    
    def _calculate_basin_precipitation(self, weather_pred: Dict) -> Dict:
        """Calculate basin-averaged precipitation"""
        basin_precip = {
            'ganges': 0,
            'brahmaputra': 0,
            'meghna': 0
        }
        
        # Simple basin averaging
        for point in weather_pred.get('grid_points', []):
            lat, lon = point['coordinates']
            precip = point.get('precipitation', 0)
            
            # Assign to basins based on location
            if 23 <= lat <= 26 and 88 <= lon <= 90:
                basin_precip['ganges'] += precip
            elif 24 <= lat <= 26 and 90 <= lon <= 92:
                basin_precip['brahmaputra'] += precip
            elif 22 <= lat <= 24 and 90 <= lon <= 92:
                basin_precip['meghna'] += precip
        
        return basin_precip
    
    def _detect_cyclones_in_prediction(self, weather_pred: Dict) -> List[Dict]:
        """Detect cyclones in weather prediction"""
        # Simplified cyclone detection
        return weather_pred.get('cyclones', [])
    
    def _calculate_storm_surge(self, cyclone: Dict) -> float:
        """Calculate storm surge height from cyclone parameters"""
        max_wind = cyclone.get('max_wind', 0)
        
        # Simplified surge calculation
        # Surge height increases with wind speed
        if max_wind < 20:
            return 0.5
        elif max_wind < 35:
            return 1.0
        elif max_wind < 50:
            return 2.0
        elif max_wind < 65:
            return 3.5
        else:
            return 5.0
    
    def _identify_evacuation_zones(self, compound_risk: Dict) -> List[str]:
        """Identify areas requiring evacuation"""
        evacuation_zones = []
        
        for region, risk_data in compound_risk.items():
            if risk_data['risk_level'] in ['high', 'extreme']:
                evacuation_zones.append(region)
        
        return evacuation_zones
    
    def _river_affects_region(self, river: str, region: str) -> bool:
        """Check if river affects specific region"""
        # Simplified mapping
        river_regions = {
            'ganges': ['dhaka', 'khulna'],
            'brahmaputra': ['dhaka', 'sylhet'],
            'meghna': ['dhaka', 'barishal', 'chittagong']
        }
        
        return region in river_regions.get(river, [])
    
    def _cyclone_affects_region(self, cyclone: str, region: str) -> bool:
        """Check if cyclone affects specific region"""
        # Simplified - assume all cyclones affect coastal regions
        coastal_regions = ['chittagong', 'khulna', 'barishal']
        return region in coastal_regions


# Supporting classes (simplified implementations)

class VortexDetection:
    """Cyclone vortex detection"""
    
    def detect_cyclones(self, state: Dict) -> List[Dict]:
        """Detect cyclone vortices in the state"""
        # Simplified detection
        return state.get('cyclones', [])


class CycloneIntensityGNN:
    """Graph neural network for cyclone intensity prediction"""
    
    def predict_intensity(self, cyclone: Dict, environment: Dict) -> float:
        """Predict cyclone intensity"""
        return cyclone.get('max_wind', 0)


class LandfallPrediction:
    """Cyclone landfall prediction"""
    
    def predict_landfall(self, cyclone: Dict) -> Dict:
        """Predict landfall location and timing"""
        return {'location': None, 'time': None}


class MuskingumCunge:
    """Muskingum-Cunge river routing model"""
    
    def route_flow(self, inflow: float, reach_params: Dict) -> float:
        """Route flow through river reach"""
        return inflow * 0.9  # Simplified routing


class MODFLOWSimplified:
    """Simplified groundwater model"""
    
    def calculate_baseflow(self, precipitation: float) -> float:
        """Calculate baseflow contribution"""
        return precipitation * 0.1


class TidalBoundary:
    """Tidal boundary conditions"""
    
    def get_current_tide(self) -> float:
        """Get current tide level"""
        return 0.5  # Simplified tide level


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test cyclone physics
    config = {'cyclone': {'min_pressure': 900}}
    cyclone_module = TropicalCycloneModule(config)
    
    # Test monsoon physics
    monsoon_physics = MonsoonPhysics()
    
    # Test hydro-met coupling
    hydro_met = HydroMetCoupler()
    
    logger.info("Physics modules initialized successfully")
