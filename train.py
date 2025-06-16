"""
Integrated training system for BanglaGraphCast.

This module provides a unified training workflow that integrates all components:
- Core GraphCast model
- Bangladesh-specific physics
- Ensemble generation
- S2S prediction
- Climate downscaling
- Multi-objective optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import os
import json
from pathlib import Path

# Local imports
from models.core.graphcast_bangladesh import BangladeshGraphCast
from models.ensemble.bangladesh_ensemble import BanglaGraphCastEnsemble, EnsembleConfig
from models.s2s.bangladesh_s2s import BangladeshS2SModel, S2SConfig
from models.climate.bangladesh_climate_downscaling import ClimateProjectionSystem, DownscalingConfig
from models.advanced.extreme_weather_prediction import ExtremeWeatherPredictor
from models.advanced.nowcasting import NowcastingSystem
from models.advanced.uncertainty_quantification import UncertaintyQuantificationSystem
from models.advanced.bias_correction import BiasCorrection
from training.losses.bangladesh_loss import BangladeshLoss
from training.curriculum.bangladesh_curriculum import BangladeshCurriculum
from training.evaluation.bangladesh_metrics import BangladeshMetrics
from configs.bangladesh_config import BangladeshConfig

logger = logging.getLogger(__name__)


class IntegratedTrainingSystem:
    """
    Integrated training system for all BanglaGraphCast components.
    
    Manages training progression from basic weather prediction to advanced
    ensemble, S2S, and climate downscaling capabilities.
    """
    
    def __init__(self, config: BangladeshConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.core_model = None
        self.ensemble_model = None
        self.s2s_model = None
        self.downscaling_model = None
        self.extreme_weather_model = None
        self.nowcasting_model = None
        self.uncertainty_model = None
        self.bias_correction = None
        
        # Training components
        self.loss_function = None
        self.curriculum = None
        self.metrics = None
        self.optimizers = {}
        self.schedulers = {}
        
        # Training state
        self.current_stage = "initialization"
        self.epoch = 0
        self.global_step = 0
        self.best_metrics = {}
        
        self._setup_models()
        self._setup_training_components()
    
    def _setup_models(self):
        """Initialize all model components."""
        logger.info("Setting up models...")
        
        # Core GraphCast model
        self.core_model = BangladeshGraphCast(self.config).to(self.device)
        logger.info(f"Core model parameters: {sum(p.numel() for p in self.core_model.parameters()):,}")
        
        # Ensemble system
        ensemble_config = EnsembleConfig(
            n_members=self.config.training_config.ensemble_size,
            perturbation_methods=["initial_conditions", "model_parameters", "stochastic_physics"],
            parameter_uncertainty=True,
            stochastic_physics=True
        )
        self.ensemble_model = BanglaGraphCastEnsemble(self.core_model, ensemble_config).to(self.device)
        
        # S2S system
        s2s_config = S2SConfig(
            max_lead_time_days=90,
            ensemble_size=20,
            use_teleconnections=True,
            use_soil_memory=True,
            use_mjo_tracking=True
        )
        self.s2s_model = BangladeshS2SModel(self.core_model, s2s_config).to(self.device)
        
        # Climate downscaling system
        downscaling_config = DownscalingConfig(
            target_resolution_km=1.0,
            bias_correction=True,
            statistical_methods=['quantile_mapping', 'bias_correction'],
            dynamic_methods=['spectral_nudging'],
            physics_parameterizations=True
        )
        self.downscaling_model = ClimateProjectionSystem(downscaling_config).to(self.device)
        
        # Advanced modules
        # Extreme weather prediction
        self.extreme_weather_model = ExtremeWeatherPredictor(self.config).to(self.device)
        
        # Nowcasting system
        self.nowcasting_model = NowcastingSystem(self.config).to(self.device)
        
        # Uncertainty quantification
        self.uncertainty_model = UncertaintyQuantificationSystem(self.config).to(self.device)
        
        # Bias correction
        self.bias_correction = BiasCorrection(self.config)
        
        logger.info("All model components initialized successfully")
    
    def _setup_training_components(self):
        """Initialize training components."""
        logger.info("Setting up training components...")
        
        # Loss function
        self.loss_function = BangladeshLoss(self.config.training_config)
        
        # Curriculum learning
        self.curriculum = BangladeshCurriculum(self.config.training_config)
        
        # Metrics
        self.metrics = BangladeshMetrics(self.config.domain_config)
        
        # Optimizers for different components
        self._setup_optimizers()
        
        logger.info("Training components initialized")
    
    def _setup_optimizers(self):
        """Setup optimizers and schedulers for different model components."""
        lr_config = self.config.training_config.learning_rate
        
        # Core model optimizer
        self.optimizers['core'] = optim.AdamW(
            self.core_model.parameters(),
            lr=lr_config.initial_lr,
            weight_decay=lr_config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Ensemble components optimizer
        ensemble_params = list(self.ensemble_model.ic_perturber.parameters()) + \
                        list(self.ensemble_model.stochastic_physics.parameters())
        if ensemble_params:
            self.optimizers['ensemble'] = optim.AdamW(
                ensemble_params,
                lr=lr_config.initial_lr * 0.5,
                weight_decay=lr_config.weight_decay
            )
        
        # S2S components optimizer
        s2s_params = list(self.s2s_model.climate_predictor.parameters()) + \
                    list(self.s2s_model.monsoon_predictor.parameters()) + \
                    list(self.s2s_model.cyclone_predictor.parameters())
        if s2s_params:
            self.optimizers['s2s'] = optim.AdamW(
                s2s_params,
                lr=lr_config.initial_lr * 0.3,
                weight_decay=lr_config.weight_decay
            )
        
        # Downscaling optimizer
        self.optimizers['downscaling'] = optim.AdamW(
            self.downscaling_model.parameters(),
            lr=lr_config.initial_lr * 0.2,
            weight_decay=lr_config.weight_decay
        )
        
        # Extreme weather model optimizer
        self.optimizers['extreme_weather'] = optim.AdamW(
            self.extreme_weather_model.parameters(),
            lr=lr_config.initial_lr * 0.1,
            weight_decay=lr_config.weight_decay
        )
        
        # Nowcasting model optimizer
        self.optimizers['nowcasting'] = optim.AdamW(
            self.nowcasting_model.parameters(),
            lr=lr_config.initial_lr * 0.1,
            weight_decay=lr_config.weight_decay
        )
        
        # Uncertainty quantification model optimizer
        self.optimizers['uncertainty_quantification'] = optim.AdamW(
            self.uncertainty_model.parameters(),
            lr=lr_config.initial_lr * 0.1,
            weight_decay=lr_config.weight_decay
        )
        
        # Bias correction model optimizer
        self.optimizers['bias_correction'] = optim.AdamW(
            self.bias_correction.parameters(),
            lr=lr_config.initial_lr * 0.1,
            weight_decay=lr_config.weight_decay
        )
        
        # Set up learning rate schedulers
        self.schedulers = {}
        for name, optimizer in self.optimizers.items():
            self.schedulers[name] = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training_config.max_epochs,
                eta_min=1e-6
            )
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Main training loop with progressive capability development.
        
        Training stages:
        1. Core weather prediction
        2. Physics-enhanced prediction
        3. Ensemble generation
        4. S2S prediction
        5. Climate downscaling
        """
        logger.info("Starting integrated training...")
          # Training stages with advanced modules
        stages = [
            ("core_weather", 0.15),         # Basic weather prediction
            ("physics_enhanced", 0.15),     # Add physics constraints
            ("nowcasting_training", 0.10),  # Very short-term prediction
            ("ensemble_training", 0.15),    # Ensemble capabilities
            ("extreme_weather_training", 0.10), # Extreme events
            ("s2s_training", 0.10),         # Subseasonal-to-seasonal
            ("uncertainty_training", 0.10), # Uncertainty quantification
            ("bias_correction_training", 0.05), # Bias correction
            ("downscaling_training", 0.10)  # Climate downscaling
        ]
        
        total_epochs = self.config.training_config.epochs
        
        for stage_name, stage_fraction in stages:
            stage_epochs = int(total_epochs * stage_fraction)
            logger.info(f"Starting stage: {stage_name} ({stage_epochs} epochs)")
            
            self.current_stage = stage_name
            
            # Update curriculum for current stage
            self.curriculum.set_stage(stage_name)
            
            # Train for this stage
            for epoch in range(stage_epochs):
                self.epoch += 1
                
                # Training epoch
                train_metrics = self._train_epoch(train_loader, stage_name)
                
                # Validation epoch
                val_metrics = self._validate_epoch(val_loader, stage_name)
                
                # Update curriculum
                self.curriculum.update(epoch, val_metrics)
                
                # Logging
                self._log_epoch_results(train_metrics, val_metrics, stage_name)
                
                # Save checkpoint
                if self.epoch % self.config.training_config.save_interval == 0:
                    self._save_checkpoint()
                
                # Early stopping check
                if self._should_stop_early(val_metrics):
                    logger.info("Early stopping triggered")
                    break
        
        logger.info("Training completed!")
        self._save_final_model()
    
    def _train_epoch(self, train_loader: DataLoader, stage: str) -> Dict:
        """Train for one epoch based on current stage."""        # Set models to training mode
        self.core_model.train()
        if stage in ["ensemble_training", "s2s_training", "downscaling_training", 
                    "extreme_weather_training", "nowcasting_training", "uncertainty_training"]:
            if self.ensemble_model and stage == "ensemble_training":
                self.ensemble_model.train()
            if self.s2s_model and stage == "s2s_training":
                self.s2s_model.train()
            if self.downscaling_model and stage == "downscaling_training":
                self.downscaling_model.train()
            if self.extreme_weather_model and stage == "extreme_weather_training":
                self.extreme_weather_model.train()
            if self.nowcasting_model and stage == "nowcasting_training":
                self.nowcasting_model.train()
            if self.uncertainty_model and stage == "uncertainty_training":
                self.uncertainty_model.train()
        if stage in ["extreme_weather_prediction", "nowcasting", "uncertainty_quantification", "bias_correction"]:
            if self.extreme_weather_model:
                self.extreme_weather_model.train()
            if self.nowcasting_model:
                self.nowcasting_model.train()
            if self.uncertainty_model:
                self.uncertainty_model.train()
            if self.bias_correction:
                self.bias_correction.train()
        
        epoch_metrics = {
            "total_loss": 0.0,
            "weather_loss": 0.0,
            "physics_loss": 0.0,
            "ensemble_loss": 0.0,
            "s2s_loss": 0.0,
            "downscaling_loss": 0.0,
            "extreme_weather_loss": 0.0,
            "nowcasting_loss": 0.0,
            "uncertainty_quantification_loss": 0.0,
            "bias_correction_loss": 0.0,
            "batch_count": 0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Get curriculum parameters
            curriculum_params = self.curriculum.get_curriculum_parameters(
                self.epoch, batch_idx, len(train_loader)
            )
            
            # Forward pass and loss computation based on stage
            losses = self._compute_stage_losses(batch, stage, curriculum_params)
            
            # Backward pass
            self._backward_pass(losses, stage)
            
            # Update metrics
            for key, value in losses.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value.item() if torch.is_tensor(value) else value
            epoch_metrics["batch_count"] += 1
            
            self.global_step += 1
            
            # Progress logging
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {self.epoch}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {losses['total_loss']:.4f}")
        
        # Average metrics
        for key in epoch_metrics:
            if key != "batch_count":
                epoch_metrics[key] /= epoch_metrics["batch_count"]
        
        return epoch_metrics
    
    def _compute_stage_losses(self, 
                            batch: Dict, 
                            stage: str, 
                            curriculum_params: Dict) -> Dict:
        """Compute losses based on training stage."""
        losses = {"total_loss": 0.0}
        
        # Core weather prediction (always computed)
        if stage in ["core_weather", "physics_enhanced", "ensemble_training", "s2s_training"]:
            core_predictions = self.core_model(
                batch["input_state"],
                batch["target_times"]
            )
            
            core_loss = self.loss_function.compute_weather_loss(
                core_predictions, batch["target_state"]
            )
            losses["weather_loss"] = core_loss
            losses["total_loss"] += core_loss * curriculum_params.get("weather_weight", 1.0)
        
        # Physics-enhanced prediction
        if stage in ["physics_enhanced", "ensemble_training", "s2s_training"]:
            physics_loss = self.loss_function.compute_physics_loss(
                core_predictions, batch
            )
            losses["physics_loss"] = physics_loss
            losses["total_loss"] += physics_loss * curriculum_params.get("physics_weight", 0.5)
        
        # Ensemble training
        if stage == "ensemble_training":
            ensemble_forecast = self.ensemble_model(
                batch["input_state"],
                batch["target_times"],
                return_members=True
            )
            
            ensemble_loss = self.loss_function.compute_ensemble_loss(
                ensemble_forecast, batch["target_state"]
            )
            losses["ensemble_loss"] = ensemble_loss
            losses["total_loss"] += ensemble_loss * curriculum_params.get("ensemble_weight", 0.3)
        
        # S2S training
        if stage == "s2s_training":
            s2s_forecast = self.s2s_model(
                batch["input_state"],
                target_lead_times=[7, 14, 28, 45, 90]
            )
            
            s2s_loss = self._compute_s2s_loss(s2s_forecast, batch)
            losses["s2s_loss"] = s2s_loss
            losses["total_loss"] += s2s_loss * curriculum_params.get("s2s_weight", 0.2)
        
        # Climate downscaling training
        if stage == "downscaling_training":
            if "coarse_projections" in batch:
                downscaled = self.downscaling_model(
                    batch["coarse_projections"],
                    batch["auxiliary_data"],
                    batch["scenario"],
                    batch["time_period"]
                )
                
                downscaling_loss = self._compute_downscaling_loss(downscaled, batch)
                losses["downscaling_loss"] = downscaling_loss
                losses["total_loss"] += downscaling_loss
        
        # Extreme weather prediction
        if stage == "extreme_weather_prediction":
            extreme_weather_forecast = self.extreme_weather_model(
                batch["input_state"],
                batch["target_times"]
            )
            
            extreme_weather_loss = self.loss_function.compute_extreme_weather_loss(
                extreme_weather_forecast, batch["target_extreme_weather"]
            )
            losses["extreme_weather_loss"] = extreme_weather_loss
            losses["total_loss"] += extreme_weather_loss * curriculum_params.get("extreme_weather_weight", 0.1)
        
        # Nowcasting
        if stage == "nowcasting":
            nowcasting_forecast = self.nowcasting_model(
                batch["input_state"],
                batch["target_times"]
            )
            
            nowcasting_loss = self.loss_function.compute_nowcasting_loss(
                nowcasting_forecast, batch["target_nowcasting"]
            )
            losses["nowcasting_loss"] = nowcasting_loss
            losses["total_loss"] += nowcasting_loss * curriculum_params.get("nowcasting_weight", 0.1)
        
        # Uncertainty quantification
        if stage == "uncertainty_quantification":
            uq_forecast = self.uncertainty_model(
                batch["input_state"],
                batch["target_times"]
            )
            
            uq_loss = self.loss_function.compute_uncertainty_quantification_loss(
                uq_forecast, batch["target_uncertainty"]
            )
            losses["uncertainty_quantification_loss"] = uq_loss
            losses["total_loss"] += uq_loss * curriculum_params.get("uncertainty_weight", 0.1)
        
        # Bias correction
        if stage == "bias_correction":
            bias_corrected = self.bias_correction(
                batch["input_state"],
                batch["target_times"]
            )
            
            bias_correction_loss = self.loss_function.compute_bias_correction_loss(
                bias_corrected, batch["target_bias"]
            )
            losses["bias_correction_loss"] = bias_correction_loss
            losses["total_loss"] += bias_correction_loss * curriculum_params.get("bias_weight", 0.1)
        
        # Nowcasting training
        if stage == "nowcasting_training":
            nowcast_predictions = self.nowcasting_model(
                batch["input_state"],
                target_lead_times=[0.25, 0.5, 1, 2, 3]  # hours
            )
            
            nowcast_loss = self._compute_nowcast_loss(nowcast_predictions, batch)
            losses["nowcasting_loss"] = nowcast_loss
            losses["total_loss"] += nowcast_loss * curriculum_params.get("nowcast_weight", 1.0)
        
        # Extreme weather training
        if stage == "extreme_weather_training":
            extreme_predictions = self.extreme_weather_model(
                batch["input_state"],
                batch["target_times"]
            )
            
            extreme_loss = self._compute_extreme_weather_loss(extreme_predictions, batch)
            losses["extreme_weather_loss"] = extreme_loss
            losses["total_loss"] += extreme_loss * curriculum_params.get("extreme_weight", 1.0)
        
        # Uncertainty quantification training
        if stage == "uncertainty_training":
            uncertainty_output = self.uncertainty_model(
                batch["input_state"],
                batch["target_times"]
            )
            
            uncertainty_loss = self._compute_uncertainty_loss(uncertainty_output, batch)
            losses["uncertainty_loss"] = uncertainty_loss
            losses["total_loss"] += uncertainty_loss * curriculum_params.get("uncertainty_weight", 0.5)
        
        # Bias correction training
        if stage == "bias_correction_training":
            # Apply bias correction to existing predictions
            if "weather_loss" in losses:
                corrected_predictions = self.bias_correction.correct_forecast(
                    core_predictions, batch["auxiliary_data"]
                )
                
                bias_correction_loss = self._compute_bias_correction_loss(
                    corrected_predictions, batch["target_state"]
                )
                losses["bias_correction_loss"] = bias_correction_loss
                losses["total_loss"] += bias_correction_loss * curriculum_params.get("bias_weight", 0.3)
        
        return losses
    
    def _compute_s2s_loss(self, s2s_forecast: Dict, batch: Dict) -> torch.Tensor:
        """Compute S2S-specific losses."""
        total_loss = 0.0
        
        # Climate index loss
        if "climate_indices" in s2s_forecast and "target_climate_indices" in batch:
            climate_loss = nn.MSELoss()(
                s2s_forecast["climate_indices"]["enso"],
                batch["target_climate_indices"]["enso"]
            )
            total_loss += climate_loss
        
        # Monsoon phase loss
        if "monsoon" in s2s_forecast and "target_monsoon_phase" in batch:
            monsoon_loss = nn.CrossEntropyLoss()(
                s2s_forecast["monsoon"]["phase_probabilities"],
                batch["target_monsoon_phase"]
            )
            total_loss += monsoon_loss
        
        return total_loss
    
    def _compute_downscaling_loss(self, downscaled: Dict, batch: Dict) -> torch.Tensor:
        """Compute climate downscaling losses."""
        total_loss = 0.0
        
        for var_name, data in downscaled.items():
            if f"target_{var_name}" in batch:
                var_loss = nn.MSELoss()(
                    data["downscaled"],
                    batch[f"target_{var_name}"]
                )
                total_loss += var_loss
        
        return total_loss
    
    def _compute_nowcast_loss(self, nowcast_predictions: Dict, batch: Dict) -> torch.Tensor:
        """Compute nowcasting-specific losses with high temporal resolution focus."""
        total_loss = 0.0
        
        # High-resolution precipitation loss (critical for very short-term)
        if "precipitation" in nowcast_predictions and "target_precipitation" in batch:
            precip_loss = nn.MSELoss()(
                nowcast_predictions["precipitation"],
                batch["target_precipitation"]
            )
            # Emphasize accuracy in first hour
            time_weights = torch.tensor([3.0, 2.0, 1.5, 1.0, 1.0], device=self.device)
            total_loss += torch.sum(precip_loss * time_weights)
        
        # Convection initiation loss
        if "convection_probability" in nowcast_predictions:
            conv_loss = nn.BCEWithLogitsLoss()(
                nowcast_predictions["convection_probability"],
                batch.get("target_convection", torch.zeros_like(nowcast_predictions["convection_probability"]))
            )
            total_loss += conv_loss * 2.0
        
        return total_loss
    
    def _compute_extreme_weather_loss(self, extreme_predictions: Dict, batch: Dict) -> torch.Tensor:
        """Compute extreme weather prediction losses."""
        total_loss = 0.0
        
        # Cyclone detection and tracking loss
        if "cyclone_probability" in extreme_predictions:
            cyclone_loss = nn.BCEWithLogitsLoss()(
                extreme_predictions["cyclone_probability"],
                batch.get("target_cyclone_presence", torch.zeros_like(extreme_predictions["cyclone_probability"]))
            )
            total_loss += cyclone_loss * 5.0  # High weight for cyclones
        
        # Extreme precipitation loss
        if "extreme_precip_probability" in extreme_predictions:
            extreme_precip_loss = nn.BCEWithLogitsLoss()(
                extreme_predictions["extreme_precip_probability"],
                batch.get("target_extreme_precip", torch.zeros_like(extreme_predictions["extreme_precip_probability"]))
            )
            total_loss += extreme_precip_loss * 3.0
        
        # Heat wave detection loss
        if "heat_wave_probability" in extreme_predictions:
            heat_loss = nn.BCEWithLogitsLoss()(
                extreme_predictions["heat_wave_probability"],
                batch.get("target_heat_wave", torch.zeros_like(extreme_predictions["heat_wave_probability"]))
            )
            total_loss += heat_loss * 2.0
        
        return total_loss
    
    def _compute_uncertainty_loss(self, uncertainty_output: Dict, batch: Dict) -> torch.Tensor:
        """Compute uncertainty quantification losses."""
        total_loss = 0.0
        
        # Ensemble variance loss (should match observed prediction errors)
        if "prediction_variance" in uncertainty_output:
            # Calibration loss - variance should match squared errors
            if "prediction_errors" in batch:
                variance_loss = nn.MSELoss()(
                    uncertainty_output["prediction_variance"],
                    batch["prediction_errors"] ** 2
                )
                total_loss += variance_loss
        
        # Confidence interval coverage loss
        if "confidence_intervals" in uncertainty_output:
            coverage_loss = self._compute_coverage_loss(
                uncertainty_output["confidence_intervals"],
                batch.get("target_state")
            )
            total_loss += coverage_loss
        
        # Sharpness penalty (encourage narrow but well-calibrated intervals)
        if "interval_width" in uncertainty_output:
            sharpness_loss = torch.mean(uncertainty_output["interval_width"])
            total_loss += sharpness_loss * 0.1
        
        return total_loss
    
    def _compute_bias_correction_loss(self, corrected_predictions: Dict, targets: Dict) -> torch.Tensor:
        """Compute bias correction loss."""
        total_loss = 0.0
        
        for var_name in corrected_predictions:
            if var_name in targets:
                var_loss = nn.MSELoss()(
                    corrected_predictions[var_name],
                    targets[var_name]
                )
                total_loss += var_loss
        
        return total_loss
    
    def _compute_coverage_loss(self, confidence_intervals: Dict, targets: Dict) -> torch.Tensor:
        """Compute loss for confidence interval coverage."""
        total_loss = 0.0
        
        for var_name, intervals in confidence_intervals.items():
            if var_name in targets:
                lower_bound = intervals["lower"]
                upper_bound = intervals["upper"]
                target_values = targets[var_name]
                
                # Penalty for targets outside intervals
                below_lower = torch.relu(lower_bound - target_values)
                above_upper = torch.relu(target_values - upper_bound)
                
                coverage_loss = torch.mean(below_lower + above_upper)
                total_loss += coverage_loss
        
        return total_loss
    
    def _backward_pass(self, losses: Dict, stage: str):
        """Perform backward pass for appropriate optimizers."""
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # Backward pass
        losses["total_loss"].backward()
        
        # Gradient clipping
        max_grad_norm = self.config.training_config.max_grad_norm
        
        if stage in ["core_weather", "physics_enhanced"]:
            torch.nn.utils.clip_grad_norm_(self.core_model.parameters(), max_grad_norm)
            self.optimizers["core"].step()
            self.schedulers["core"].step()
        
        if stage == "ensemble_training" and "ensemble" in self.optimizers:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizers["ensemble"].param_groups for p in group["params"]], 
                max_grad_norm
            )
            self.optimizers["ensemble"].step()
            self.schedulers["ensemble"].step()
        
        if stage == "s2s_training" and "s2s" in self.optimizers:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizers["s2s"].param_groups for p in group["params"]], 
                max_grad_norm
            )
            self.optimizers["s2s"].step()
            self.schedulers["s2s"].step()
        
        if stage == "downscaling_training":
            torch.nn.utils.clip_grad_norm_(self.downscaling_model.parameters(), max_grad_norm)
            self.optimizers["downscaling"].step()
            self.schedulers["downscaling"].step()
        
        if stage == "extreme_weather_prediction" and "extreme_weather" in self.optimizers:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizers["extreme_weather"].param_groups for p in group["params"]], 
                max_grad_norm
            )
            self.optimizers["extreme_weather"].step()
            self.schedulers["extreme_weather"].step()
        
        if stage == "nowcasting" and "nowcasting" in self.optimizers:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizers["nowcasting"].param_groups for p in group["params"]], 
                max_grad_norm
            )
            self.optimizers["nowcasting"].step()
            self.schedulers["nowcasting"].step()
        
        if stage == "uncertainty_quantification" and "uncertainty_quantification" in self.optimizers:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizers["uncertainty_quantification"].param_groups for p in group["params"]], 
                max_grad_norm
            )
            self.optimizers["uncertainty_quantification"].step()
            self.schedulers["uncertainty_quantification"].step()
        
        if stage == "bias_correction" and "bias_correction" in self.optimizers:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizers["bias_correction"].param_groups for p in group["params"]], 
                max_grad_norm
            )
            self.optimizers["bias_correction"].step()
            self.schedulers["bias_correction"].step()
        
        # Nowcasting training
        if stage == "nowcasting_training" and "nowcasting" in self.optimizers:
            torch.nn.utils.clip_grad_norm_(self.nowcasting_model.parameters(), max_grad_norm)
            self.optimizers["nowcasting"].step()
            self.schedulers["nowcasting"].step()
        
        # Extreme weather training
        if stage == "extreme_weather_training" and "extreme_weather" in self.optimizers:
            torch.nn.utils.clip_grad_norm_(self.extreme_weather_model.parameters(), max_grad_norm)
            self.optimizers["extreme_weather"].step()
            self.schedulers["extreme_weather"].step()
        
        # Uncertainty quantification training
        if stage == "uncertainty_training" and "uncertainty" in self.optimizers:
            torch.nn.utils.clip_grad_norm_(self.uncertainty_model.parameters(), max_grad_norm)
            self.optimizers["uncertainty"].step()
            self.schedulers["uncertainty"].step()
        
        # Bias correction training (uses existing core model, no separate optimizer needed)
        if stage == "bias_correction_training":
            # Bias correction may update core model parameters
            torch.nn.utils.clip_grad_norm_(self.core_model.parameters(), max_grad_norm)
            self.optimizers["core"].step()
            self.schedulers["core"].step()
    
    def _validate_epoch(self, val_loader: DataLoader, stage: str) -> Dict:
        """Validate for one epoch."""        # Set models to evaluation mode
        self.core_model.eval()
        if self.ensemble_model:
            self.ensemble_model.eval()
        if self.s2s_model:
            self.s2s_model.eval()
        if self.downscaling_model:
            self.downscaling_model.eval()
        if self.extreme_weather_model:
            self.extreme_weather_model.eval()
        if self.nowcasting_model:
            self.nowcasting_model.eval()
        if self.uncertainty_model:
            self.uncertainty_model.eval()
        if self.extreme_weather_model:
            self.extreme_weather_model.eval()
        if self.nowcasting_model:
            self.nowcasting_model.eval()
        if self.uncertainty_model:
            self.uncertainty_model.eval()
        if self.bias_correction:
            self.bias_correction.eval()
        
        val_metrics = {
            "total_loss": 0.0,
            "weather_metrics": {},
            "physics_metrics": {},
            "ensemble_metrics": {},
            "s2s_metrics": {},
            "batch_count": 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                
                # Compute validation losses and metrics
                losses = self._compute_stage_losses(batch, stage, {})
                
                # Compute detailed metrics
                if stage in ["core_weather", "physics_enhanced"]:
                    predictions = self.core_model(batch["input_state"], batch["target_times"])
                    weather_metrics = self.metrics.compute_standard_metrics(
                        predictions, batch["target_state"]
                    )
                    val_metrics["weather_metrics"] = weather_metrics
                
                val_metrics["total_loss"] += losses["total_loss"].item()
                val_metrics["batch_count"] += 1
        
        # Average metrics
        val_metrics["total_loss"] /= val_metrics["batch_count"]
        
        return val_metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        moved_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved_batch[key] = value.to(self.device)
            else:
                moved_batch[key] = value
        return moved_batch
    
    def _log_epoch_results(self, train_metrics: Dict, val_metrics: Dict, stage: str):
        """Log epoch results."""
        logger.info(f"Epoch {self.epoch} ({stage}) - "
                   f"Train Loss: {train_metrics['total_loss']:.4f}, "
                   f"Val Loss: {val_metrics['total_loss']:.4f}")
        
        # Log to tensorboard/wandb if available
        # Implementation would depend on chosen logging framework
    
    def _should_stop_early(self, val_metrics: Dict) -> bool:
        """Check early stopping criteria."""
        # Simple implementation - could be more sophisticated
        current_loss = val_metrics["total_loss"]
        
        if "best_val_loss" not in self.best_metrics:
            self.best_metrics["best_val_loss"] = current_loss
            return False
        
        if current_loss < self.best_metrics["best_val_loss"]:
            self.best_metrics["best_val_loss"] = current_loss
            self.best_metrics["patience_counter"] = 0
        else:
            self.best_metrics["patience_counter"] = self.best_metrics.get("patience_counter", 0) + 1
        
        return self.best_metrics["patience_counter"] >= self.config.training_config.patience
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.training_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "current_stage": self.current_stage,
            "core_model_state": self.core_model.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "schedulers": {name: sch.state_dict() for name, sch in self.schedulers.items()},
            "best_metrics": self.best_metrics,
            "config": self.config.to_dict()
        }
        
        # Save ensemble model if available
        if self.ensemble_model:
            checkpoint["ensemble_model_state"] = self.ensemble_model.state_dict()
        
        # Save S2S model if available  
        if self.s2s_model:
            checkpoint["s2s_model_state"] = self.s2s_model.state_dict()
        
        # Save downscaling model if available
        if self.downscaling_model:
            checkpoint["downscaling_model_state"] = self.downscaling_model.state_dict()
        
        # Save extreme weather model if available
        if self.extreme_weather_model:
            checkpoint["extreme_weather_model_state"] = self.extreme_weather_model.state_dict()
        
        # Save nowcasting model if available
        if self.nowcasting_model:
            checkpoint["nowcasting_model_state"] = self.nowcasting_model.state_dict()
        
        # Save uncertainty quantification model if available
        if self.uncertainty_model:
            checkpoint["uncertainty_quantification_model_state"] = self.uncertainty_model.state_dict()
        
        # Save bias correction model if available
        if self.bias_correction:
            checkpoint["bias_correction_model_state"] = self.bias_correction.state_dict()
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{self.epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self):
        """Save final trained model."""
        model_dir = Path(self.config.training_config.model_save_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual model components
        torch.save(self.core_model.state_dict(), model_dir / "core_model.pt")
        
        if self.ensemble_model:
            torch.save(self.ensemble_model.state_dict(), model_dir / "ensemble_model.pt")
        
        if self.s2s_model:
            torch.save(self.s2s_model.state_dict(), model_dir / "s2s_model.pt")
        
        if self.downscaling_model:
            torch.save(self.downscaling_model.state_dict(), model_dir / "downscaling_model.pt")
        
        if self.extreme_weather_model:
            torch.save(self.extreme_weather_model.state_dict(), model_dir / "extreme_weather_model.pt")
        
        if self.nowcasting_model:
            torch.save(self.nowcasting_model.state_dict(), model_dir / "nowcasting_model.pt")
        
        if self.uncertainty_model:
            torch.save(self.uncertainty_model.state_dict(), model_dir / "uncertainty_quantification_model.pt")
        
        if self.bias_correction:
            torch.save(self.bias_correction.state_dict(), model_dir / "bias_correction_model.pt")
        
        # Save configuration
        with open(model_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Final models saved to: {model_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.current_stage = checkpoint["current_stage"]
        self.best_metrics = checkpoint["best_metrics"]
        
        # Load model states
        self.core_model.load_state_dict(checkpoint["core_model_state"])
        
        if "ensemble_model_state" in checkpoint and self.ensemble_model:
            self.ensemble_model.load_state_dict(checkpoint["ensemble_model_state"])
        
        if "s2s_model_state" in checkpoint and self.s2s_model:
            self.s2s_model.load_state_dict(checkpoint["s2s_model_state"])
        
        if "downscaling_model_state" in checkpoint and self.downscaling_model:
            self.downscaling_model.load_state_dict(checkpoint["downscaling_model_state"])
        
        if "extreme_weather_model_state" in checkpoint and self.extreme_weather_model:
            self.extreme_weather_model.load_state_dict(checkpoint["extreme_weather_model_state"])
        
        if "nowcasting_model_state" in checkpoint and self.nowcasting_model:
            self.nowcasting_model.load_state_dict(checkpoint["nowcasting_model_state"])
        
        if "uncertainty_quantification_model_state" in checkpoint and self.uncertainty_model:
            self.uncertainty_model.load_state_dict(checkpoint["uncertainty_quantification_model_state"])
        
        if "bias_correction_model_state" in checkpoint and self.bias_correction:
            self.bias_correction.load_state_dict(checkpoint["bias_correction_model_state"])
        
        # Load optimizer and scheduler states
        for name, state in checkpoint["optimizers"].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state)
        
        for name, state in checkpoint["schedulers"].items():
            if name in self.schedulers:
                self.schedulers[name].load_state_dict(state)
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")


def main():
    """Example training script."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = BangladeshConfig()
    
    # Create training system
    training_system = IntegratedTrainingSystem(config)
    
    # Create mock data loaders (replace with actual data loading)
    train_loader = None  # Your training data loader
    val_loader = None    # Your validation data loader
    
    # Start training
    if train_loader and val_loader:
        training_system.train(train_loader, val_loader)
    else:
        logger.info("Training system initialized successfully!")
        logger.info("Please provide actual data loaders to start training.")


if __name__ == "__main__":
    main()
