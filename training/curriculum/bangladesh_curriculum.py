"""
Curriculum learning system for progressive training of Bangladesh GraphCast
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TrainingStage(Enum):
    """Training curriculum stages"""
    BASIC_DYNAMICS = "basic_dynamics"
    PRECIPITATION_PATTERNS = "precipitation_patterns"
    EXTREME_EVENTS = "extreme_events"
    FULL_COMPLEXITY = "full_complexity"


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage"""
    name: str
    stage: TrainingStage
    lead_time_hours: int
    focus_variables: List[str]
    loss_weights: Dict[str, float]
    data_complexity: str
    min_epochs: int
    max_epochs: int
    success_criteria: Dict[str, float]
    description: str


class BangladeshCurriculum:
    """
    Curriculum learning system tailored for Bangladesh weather prediction
    
    Progressive training stages:
    1. Basic Dynamics: Learn fundamental atmospheric patterns
    2. Precipitation Patterns: Focus on monsoon and precipitation
    3. Extreme Events: Master cyclones and extreme weather
    4. Full Complexity: All phenomena with full lead time
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.current_stage_idx = 0
        self.stage_history = []
        
        # Define curriculum stages
        self.stages = [
            CurriculumStage(
                name="Stage 1: Basic Dynamics",
                stage=TrainingStage.BASIC_DYNAMICS,
                lead_time_hours=24,
                focus_variables=['temperature', 'pressure', 'humidity', 'wind'],
                loss_weights={
                    'standard_vars': 1.0,
                    'precipitation': 0.5,
                    'cyclone_track': 0.0,
                    'cyclone_intensity': 0.0,
                    'monsoon_onset': 0.0,
                    'extreme_precip': 0.0,
                    'heat_stress': 0.5
                },
                data_complexity='simple',
                min_epochs=10,
                max_epochs=30,
                success_criteria={
                    'rmse_temperature': 2.0,
                    'rmse_pressure': 5.0,
                    'acc_geopotential': 0.8
                },
                description="Learn basic atmospheric dynamics and thermodynamics"
            ),
            
            CurriculumStage(
                name="Stage 2: Precipitation Patterns",
                stage=TrainingStage.PRECIPITATION_PATTERNS,
                lead_time_hours=48,
                focus_variables=['precipitation', 'humidity', 'temperature', 'wind'],
                loss_weights={
                    'standard_vars': 1.0,
                    'precipitation': 2.0,
                    'cyclone_track': 0.0,
                    'cyclone_intensity': 0.0,
                    'monsoon_onset': 1.5,
                    'extreme_precip': 1.0,
                    'heat_stress': 1.0
                },
                data_complexity='moderate',
                min_epochs=15,
                max_epochs=40,
                success_criteria={
                    'ets_precipitation_10mm': 0.3,
                    'bias_precipitation': 1.2,
                    'rmse_humidity': 10.0
                },
                description="Master precipitation patterns and monsoon dynamics"
            ),
            
            CurriculumStage(
                name="Stage 3: Extreme Events",
                stage=TrainingStage.EXTREME_EVENTS,
                lead_time_hours=72,
                focus_variables=['all'],
                loss_weights={
                    'standard_vars': 1.0,
                    'precipitation': 2.5,
                    'cyclone_track': 4.0,
                    'cyclone_intensity': 3.5,
                    'monsoon_onset': 2.0,
                    'extreme_precip': 3.0,
                    'heat_stress': 2.0,
                    'coastal_surge': 3.5,
                    'wind_damage': 3.0
                },
                data_complexity='high',
                min_epochs=20,
                max_epochs=50,
                success_criteria={
                    'cyclone_track_error': 100.0,  # km
                    'cyclone_intensity_mae': 10.0,  # m/s
                    'ets_extreme_precip': 0.2,
                    'pod_storm_surge': 0.7
                },
                description="Handle extreme events: cyclones, floods, heat waves"
            ),
            
            CurriculumStage(
                name="Stage 4: Full Complexity",
                stage=TrainingStage.FULL_COMPLEXITY,
                lead_time_hours=240,
                focus_variables=['all'],
                loss_weights={
                    'standard_vars': 1.0,
                    'precipitation': 2.5,
                    'cyclone_track': 5.0,
                    'cyclone_intensity': 4.0,
                    'monsoon_onset': 3.0,
                    'extreme_precip': 3.5,
                    'heat_stress': 2.0,
                    'flood_risk': 3.0,
                    'coastal_surge': 4.5,
                    'wind_damage': 3.5
                },
                data_complexity='full',
                min_epochs=30,
                max_epochs=100,
                success_criteria={
                    'overall_skill_score': 0.6,
                    'cyclone_track_error': 80.0,
                    'monsoon_onset_accuracy': 0.8,
                    'flood_prediction_f1': 0.7
                },
                description="Full operational capability with extended lead times"
            )
        ]
        
        # Data selection strategies for each stage
        self.data_strategies = {
            TrainingStage.BASIC_DYNAMICS: self._select_basic_dynamics_data,
            TrainingStage.PRECIPITATION_PATTERNS: self._select_precipitation_data,
            TrainingStage.EXTREME_EVENTS: self._select_extreme_events_data,
            TrainingStage.FULL_COMPLEXITY: self._select_full_complexity_data
        }
        
        # Track stage performance
        self.stage_metrics = {}
        
    def get_current_stage(self) -> CurriculumStage:
        """Get current training stage configuration"""
        if self.current_stage_idx >= len(self.stages):
            return self.stages[-1]  # Stay at final stage
        
        return self.stages[self.current_stage_idx]
    
    def should_advance_stage(self, current_metrics: Dict[str, float]) -> bool:
        """
        Determine if model is ready to advance to next stage
        
        Args:
            current_metrics: Current model performance metrics
            
        Returns:
            True if ready to advance, False otherwise
        """
        current_stage = self.get_current_stage()
        
        # Check if all success criteria are met
        criteria_met = True
        for criterion, threshold in current_stage.success_criteria.items():
            if criterion not in current_metrics:
                logger.warning(f"Missing metric {criterion} for stage advancement")
                criteria_met = False
                continue
            
            current_value = current_metrics[criterion]
            
            # Determine if criterion is met (depends on metric type)
            if criterion.startswith('rmse') or criterion.endswith('error'):
                # Lower is better
                if current_value > threshold:
                    criteria_met = False
                    logger.debug(f"Criterion {criterion}: {current_value:.3f} > {threshold:.3f} (not met)")
                else:
                    logger.debug(f"Criterion {criterion}: {current_value:.3f} <= {threshold:.3f} (met)")
            else:
                # Higher is better (accuracy, skill scores, etc.)
                if current_value < threshold:
                    criteria_met = False
                    logger.debug(f"Criterion {criterion}: {current_value:.3f} < {threshold:.3f} (not met)")
                else:
                    logger.debug(f"Criterion {criterion}: {current_value:.3f} >= {threshold:.3f} (met)")
        
        if criteria_met:
            logger.info(f"All criteria met for {current_stage.name}. Ready to advance!")
            return True
        else:
            logger.info(f"Criteria not yet met for {current_stage.name}. Continuing training...")
            return False
    
    def advance_stage(self) -> bool:
        """
        Advance to next curriculum stage
        
        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage_idx >= len(self.stages) - 1:
            logger.info("Already at final curriculum stage")
            return False
        
        # Record stage completion
        old_stage = self.get_current_stage()
        self.stage_history.append({
            'stage': old_stage.name,
            'completed_at': datetime.now(),
            'metrics': self.stage_metrics.get(self.current_stage_idx, {})
        })
        
        # Advance to next stage
        self.current_stage_idx += 1
        new_stage = self.get_current_stage()
        
        logger.info(f"Advanced from '{old_stage.name}' to '{new_stage.name}'")
        logger.info(f"New stage focus: {new_stage.description}")
        
        return True
    
    def get_training_data(self, available_data: Dict) -> Dict:
        """
        Select appropriate training data for current stage
        
        Args:
            available_data: All available training data
            
        Returns:
            Filtered data appropriate for current stage
        """
        current_stage = self.get_current_stage()
        strategy = self.data_strategies[current_stage.stage]
        
        selected_data = strategy(available_data, current_stage)
        
        logger.debug(f"Selected {len(selected_data.get('samples', []))} samples for {current_stage.name}")
        
        return selected_data
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get loss weights for current stage"""
        current_stage = self.get_current_stage()
        return current_stage.loss_weights.copy()
    
    def get_training_config(self) -> Dict:
        """Get training configuration for current stage"""
        current_stage = self.get_current_stage()
        
        return {
            'stage_name': current_stage.name,
            'stage_type': current_stage.stage.value,
            'max_lead_time_hours': current_stage.lead_time_hours,
            'focus_variables': current_stage.focus_variables,
            'loss_weights': current_stage.loss_weights,
            'min_epochs': current_stage.min_epochs,
            'max_epochs': current_stage.max_epochs,
            'data_complexity': current_stage.data_complexity
        }
    
    def update_stage_metrics(self, metrics: Dict[str, float]):
        """Update metrics for current stage"""
        self.stage_metrics[self.current_stage_idx] = metrics.copy()
    
    def get_curriculum_progress(self) -> Dict:
        """Get overall curriculum progress"""
        return {
            'current_stage_idx': self.current_stage_idx,
            'total_stages': len(self.stages),
            'current_stage_name': self.get_current_stage().name,
            'stages_completed': len(self.stage_history),
            'stage_history': self.stage_history,
            'progress_percentage': (self.current_stage_idx / len(self.stages)) * 100
        }
    
    def _select_basic_dynamics_data(self, available_data: Dict, stage: CurriculumStage) -> Dict:
        """
        Select data for basic dynamics learning
        - Focus on clear weather patterns
        - Exclude extreme events
        - Short lead times only
        """
        selected_data = {'samples': [], 'metadata': []}
        
        for sample, metadata in zip(available_data.get('samples', []), 
                                  available_data.get('metadata', [])):
            
            # Skip extreme events
            if metadata.get('has_cyclone', False):
                continue
            
            if metadata.get('extreme_precipitation', False):
                continue
            
            # Only include samples within lead time limit
            if metadata.get('lead_time_hours', 0) > stage.lead_time_hours:
                continue
            
            # Prefer samples with clear atmospheric patterns
            if metadata.get('weather_regime') in ['clear', 'stable', 'normal']:
                selected_data['samples'].append(sample)
                selected_data['metadata'].append(metadata)
        
        return selected_data
    
    def _select_precipitation_data(self, available_data: Dict, stage: CurriculumStage) -> Dict:
        """
        Select data for precipitation pattern learning
        - Include monsoon periods
        - Moderate precipitation events
        - Gradual introduction of complexity
        """
        selected_data = {'samples': [], 'metadata': []}
        
        for sample, metadata in zip(available_data.get('samples', []), 
                                  available_data.get('metadata', [])):
            
            # Include previous stage data
            if metadata.get('lead_time_hours', 0) <= 24:
                selected_data['samples'].append(sample)
                selected_data['metadata'].append(metadata)
                continue
            
            # Skip very extreme events (save for next stage)
            if metadata.get('max_wind_speed', 0) > 25:  # Strong winds
                continue
            
            if metadata.get('max_precipitation', 0) > 100:  # Extreme precipitation
                continue
            
            # Within lead time limit
            if metadata.get('lead_time_hours', 0) > stage.lead_time_hours:
                continue
            
            # Focus on precipitation-related events
            if (metadata.get('has_precipitation', False) or 
                metadata.get('monsoon_period', False)):
                selected_data['samples'].append(sample)
                selected_data['metadata'].append(metadata)
        
        return selected_data
    
    def _select_extreme_events_data(self, available_data: Dict, stage: CurriculumStage) -> Dict:
        """
        Select data for extreme events learning
        - Include cyclones
        - Extreme precipitation
        - Heat waves
        """
        selected_data = {'samples': [], 'metadata': []}
        
        for sample, metadata in zip(available_data.get('samples', []), 
                                  available_data.get('metadata', [])):
            
            # Include all previous data (reduced weight)
            if metadata.get('lead_time_hours', 0) <= 48:
                selected_data['samples'].append(sample)
                selected_data['metadata'].append(metadata)
                continue
            
            # Within lead time limit
            if metadata.get('lead_time_hours', 0) > stage.lead_time_hours:
                continue
            
            # Focus on extreme events
            is_extreme = (
                metadata.get('has_cyclone', False) or
                metadata.get('extreme_precipitation', False) or
                metadata.get('extreme_temperature', False) or
                metadata.get('max_wind_speed', 0) > 15
            )
            
            if is_extreme:
                selected_data['samples'].append(sample)
                selected_data['metadata'].append(metadata)
        
        return selected_data
    
    def _select_full_complexity_data(self, available_data: Dict, stage: CurriculumStage) -> Dict:
        """
        Select all available data for full complexity training
        """
        # Include all data
        return available_data.copy()


class WeatherAugmentation:
    """
    Data augmentation techniques for weather prediction training
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.augmentation_strategies = {
            'cyclone': self.augment_cyclones,
            'monsoon': self.augment_monsoon,
            'precipitation': self.augment_precipitation,
            'temperature': self.augment_temperature
        }
    
    def augment_batch(self, batch_data: Dict, augmentation_types: List[str]) -> Dict:
        """
        Apply data augmentation to a training batch
        
        Args:
            batch_data: Original batch data
            augmentation_types: List of augmentation types to apply
            
        Returns:
            Augmented batch data
        """
        augmented_data = batch_data.copy()
        
        for aug_type in augmentation_types:
            if aug_type in self.augmentation_strategies:
                augmented_data = self.augmentation_strategies[aug_type](augmented_data)
                logger.debug(f"Applied {aug_type} augmentation")
        
        return augmented_data
    
    def augment_cyclones(self, data: Dict) -> Dict:
        """
        Augment cyclone data to increase variability
        
        Synthetic cyclone track variations:
        - Vary landfall location ±200km
        - Intensity perturbations ±10 m/s
        - Translation speed variations ±30%
        """
        import random
        
        augmented_data = data.copy()
        
        if 'cyclones' in data:
            for cyclone in data['cyclones']:
                # Position perturbation
                center_lat, center_lon = cyclone.get('center', (23.0, 90.0))
                lat_perturbation = random.uniform(-1.8, 1.8)  # ~200km at this latitude
                lon_perturbation = random.uniform(-1.8, 1.8)
                
                cyclone['center'] = (
                    center_lat + lat_perturbation,
                    center_lon + lon_perturbation
                )
                
                # Intensity perturbation
                intensity = cyclone.get('max_wind', 30)
                intensity_perturbation = random.uniform(-10, 10)
                cyclone['max_wind'] = max(0, intensity + intensity_perturbation)
                
                # Translation speed perturbation
                speed = cyclone.get('translation_speed', 15)
                speed_perturbation = random.uniform(-0.3, 0.3)
                cyclone['translation_speed'] = speed * (1 + speed_perturbation)
        
        return augmented_data
    
    def augment_monsoon(self, data: Dict) -> Dict:
        """
        Augment monsoon data for onset/withdrawal variability
        
        Monsoon variability:
        - Early/late onset scenarios (±2 weeks)
        - Active/break phase modulation
        - Intensity variations (±20%)
        """
        import random
        
        augmented_data = data.copy()
        
        if 'monsoon_phase' in data:
            # Phase timing perturbation
            phase = data['monsoon_phase']
            if phase == 'onset':
                # Vary onset timing
                timing_shift = random.uniform(-14, 14)  # ±2 weeks
                data['monsoon_onset_day'] = data.get('monsoon_onset_day', 160) + timing_shift
            
            # Intensity modulation
            if 'precipitation' in data:
                intensity_factor = random.uniform(0.8, 1.2)  # ±20%
                data['precipitation'] = [p * intensity_factor for p in data['precipitation']]
        
        return augmented_data
    
    def augment_precipitation(self, data: Dict) -> Dict:
        """
        Augment precipitation patterns
        
        - Spatial shifts (±50km)
        - Intensity scaling (±30%)
        - Temporal shifts (±3 hours)
        """
        import random
        
        augmented_data = data.copy()
        
        if 'precipitation' in data:
            # Intensity scaling
            intensity_factor = random.uniform(0.7, 1.3)
            data['precipitation'] = [p * intensity_factor for p in data['precipitation']]
            
            # Add spatial noise (simplified)
            noise_factor = random.uniform(-0.1, 0.1)
            data['precipitation'] = [max(0, p + p * noise_factor) for p in data['precipitation']]
        
        return augmented_data
    
    def augment_temperature(self, data: Dict) -> Dict:
        """
        Augment temperature data
        
        - Diurnal cycle variations
        - Urban heat island effects
        - Seasonal bias adjustments
        """
        import random
        
        augmented_data = data.copy()
        
        if 'temperature' in data:
            # Small temperature perturbations
            temp_perturbation = random.uniform(-1.0, 1.0)  # ±1°C
            data['temperature'] = [t + temp_perturbation for t in data['temperature']]
            
            # Enhance urban heat island (for urban areas)
            if data.get('is_urban', False):
                urban_enhancement = random.uniform(0, 2.0)  # 0-2°C additional warming
                data['temperature'] = [t + urban_enhancement for t in data['temperature']]
        
        return augmented_data


def create_bangladesh_curriculum(config: Dict) -> BangladeshCurriculum:
    """Create curriculum learning system for Bangladesh weather prediction"""
    return BangladeshCurriculum(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test curriculum system
    config = {'stages': 4}
    curriculum = create_bangladesh_curriculum(config)
    
    logger.info("Created Bangladesh curriculum learning system")
    logger.info(f"Number of stages: {len(curriculum.stages)}")
    
    # Show all stages
    for i, stage in enumerate(curriculum.stages):
        logger.info(f"Stage {i+1}: {stage.name}")
        logger.info(f"  Lead time: {stage.lead_time_hours}h")
        logger.info(f"  Focus: {stage.focus_variables}")
        logger.info(f"  Description: {stage.description}")
    
    # Test stage progression
    current_stage = curriculum.get_current_stage()
    logger.info(f"Current stage: {current_stage.name}")
    
    # Simulate metrics for advancement
    test_metrics = {
        'rmse_temperature': 1.5,
        'rmse_pressure': 4.0,
        'acc_geopotential': 0.85
    }
    
    if curriculum.should_advance_stage(test_metrics):
        curriculum.advance_stage()
        logger.info(f"Advanced to: {curriculum.get_current_stage().name}")
    
    # Test data augmentation
    augmenter = WeatherAugmentation({})
    test_data = {
        'temperature': [25.0, 30.0, 35.0],
        'precipitation': [5.0, 15.0, 25.0],
        'cyclones': [{'center': (22.0, 90.0), 'max_wind': 40.0}]
    }
    
    augmented = augmenter.augment_batch(test_data, ['cyclone', 'temperature'])
    logger.info("Data augmentation test completed")
    
    logger.info("Curriculum learning system test completed successfully")
