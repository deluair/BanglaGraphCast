"""
Advanced uncertainty quantification module for BanglaGraphCast.

This module implements Bayesian neural networks, ensemble variance,
and prediction interval estimation for weather forecast uncertainty.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification."""
    
    # Bayesian network settings
    use_bayesian: bool = True
    num_mc_samples: int = 100
    prior_variance: float = 1.0
    
    # Ensemble settings
    use_ensemble_variance: bool = True
    ensemble_size: int = 10
    
    # Prediction interval settings
    confidence_levels: List[float] = None
    calibration_method: str = "temperature_scaling"  # temperature_scaling, platt_scaling
    
    # Output settings
    save_uncertainty_maps: bool = True
    uncertainty_metrics: List[str] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.5, 0.8, 0.9, 0.95, 0.99]
        
        if self.uncertainty_metrics is None:
            self.uncertainty_metrics = [
                "epistemic", "aleatoric", "total", 
                "prediction_interval", "calibration"
            ]


class BayesianLayer(nn.Module):
    """Bayesian linear layer with weight uncertainty."""
    
    def __init__(self, in_features: int, out_features: int, prior_variance: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_variance = prior_variance
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights and biases
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for variational inference."""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # KL divergence between posterior and prior
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu**2 + weight_sigma**2) / self.prior_variance - 
            torch.log(weight_sigma**2 / self.prior_variance) - 1
        )
        
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu**2 + bias_sigma**2) / self.prior_variance - 
            torch.log(bias_sigma**2 / self.prior_variance) - 1
        )
        
        return weight_kl + bias_kl


class BayesianGraphCast(nn.Module):
    """Bayesian version of GraphCast for uncertainty quantification."""
    
    def __init__(self, base_model: nn.Module, num_bayesian_layers: int = 3):
        super().__init__()
        self.base_model = base_model
        self.num_bayesian_layers = num_bayesian_layers
        
        # Replace some layers with Bayesian versions
        self._add_bayesian_layers()
    
    def _add_bayesian_layers(self):
        """Add Bayesian layers to the model."""
        # This would be implemented based on the specific architecture
        # For now, we'll add uncertainty to the final layers
        if hasattr(self.base_model, 'output_layers'):
            original_layers = self.base_model.output_layers
            bayesian_layers = nn.ModuleList()
            
            for layer in original_layers:
                if isinstance(layer, nn.Linear):
                    bayesian_layers.append(
                        BayesianLayer(layer.in_features, layer.out_features)
                    )
                else:
                    bayesian_layers.append(layer)
            
            self.base_model.output_layers = bayesian_layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
    
    def get_kl_loss(self) -> torch.Tensor:
        """Get KL divergence loss for all Bayesian layers."""
        kl_loss = 0
        for module in self.modules():
            if isinstance(module, BayesianLayer):
                kl_loss += module.kl_divergence()
        return kl_loss


class UncertaintyQuantification:
    """Main uncertainty quantification system."""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.bayesian_model = None
        self.calibration_model = None
        self.uncertainty_cache = {}
    
    def setup_bayesian_model(self, base_model: nn.Module) -> BayesianGraphCast:
        """Setup Bayesian version of the model."""
        self.bayesian_model = BayesianGraphCast(base_model)
        return self.bayesian_model
    
    def monte_carlo_prediction(
        self,
        model: nn.Module,
        x: torch.Tensor,
        num_samples: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform Monte Carlo prediction for uncertainty estimation.
        
        Args:
            model: Model to use for prediction
            x: Input tensor
            num_samples: Number of MC samples
        
        Returns:
            Dictionary with mean, variance, and samples
        """
        if num_samples is None:
            num_samples = self.config.num_mc_samples
        
        model.train()  # Enable dropout/stochastic layers
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        std_pred = torch.sqrt(var_pred)
        
        return {
            'mean': mean_pred,
            'variance': var_pred,
            'std': std_pred,
            'samples': predictions,
            'epistemic_uncertainty': var_pred
        }
    
    def ensemble_uncertainty(
        self,
        ensemble_predictions: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate uncertainty from ensemble predictions.
        
        Args:
            ensemble_predictions: List of predictions from ensemble members
        
        Returns:
            Uncertainty statistics
        """
        predictions = torch.stack(ensemble_predictions)
        
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        std_pred = torch.sqrt(var_pred)
        
        # Calculate ensemble spread
        ensemble_spread = torch.std(predictions, dim=0)
        
        return {
            'mean': mean_pred,
            'variance': var_pred,
            'std': std_pred,
            'ensemble_spread': ensemble_spread,
            'epistemic_uncertainty': var_pred
        }
    
    def prediction_intervals(
        self,
        predictions: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence_levels: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate prediction intervals.
        
        Args:
            predictions: Mean predictions
            uncertainty: Uncertainty estimates (std)
            confidence_levels: Confidence levels for intervals
        
        Returns:
            Prediction intervals for each confidence level
        """
        if confidence_levels is None:
            confidence_levels = self.config.confidence_levels
        
        intervals = {}
        
        for conf in confidence_levels:
            # Calculate z-score for confidence level
            z_score = stats.norm.ppf((1 + conf) / 2)
            
            # Calculate intervals
            lower = predictions - z_score * uncertainty
            upper = predictions + z_score * uncertainty
            
            intervals[f'interval_{int(conf*100)}'] = {
                'lower': lower,
                'upper': upper,
                'width': upper - lower
            }
        
        return intervals
    
    def temperature_scaling(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        validation_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> nn.Module:
        """
        Calibrate model predictions using temperature scaling.
        
        Args:
            logits: Model logits
            targets: Target values
            validation_data: Validation data for calibration
        
        Returns:
            Calibrated model
        """
        class TemperatureScaling(nn.Module):
            def __init__(self):
                super().__init__()
                self.temperature = nn.Parameter(torch.ones(1))
            
            def forward(self, logits):
                return logits / self.temperature
        
        temp_model = TemperatureScaling()
        optimizer = torch.optim.Adam(temp_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Calibration on validation set
        val_logits, val_targets = validation_data
        
        for epoch in range(100):
            optimizer.zero_grad()
            calibrated_logits = temp_model(val_logits)
            loss = criterion(calibrated_logits, val_targets)
            loss.backward()
            optimizer.step()
        
        self.calibration_model = temp_model
        return temp_model
    
    def calculate_calibration_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate calibration metrics.
        
        Args:
            predictions: Model predictions
            targets: True targets
            uncertainty: Uncertainty estimates
        
        Returns:
            Calibration metrics
        """
        # Convert to numpy for easier computation
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        unc_np = uncertainty.cpu().numpy().flatten()
        
        # Calculate coverage for different confidence levels
        metrics = {}
        
        for conf in self.config.confidence_levels:
            z_score = stats.norm.ppf((1 + conf) / 2)
            
            # Calculate if targets fall within prediction intervals
            lower = pred_np - z_score * unc_np
            upper = pred_np + z_score * unc_np
            
            coverage = np.mean((target_np >= lower) & (target_np <= upper))
            metrics[f'coverage_{int(conf*100)}'] = coverage
            
            # Expected coverage vs actual coverage
            metrics[f'calibration_error_{int(conf*100)}'] = abs(coverage - conf)
        
        # Average calibration error
        cal_errors = [v for k, v in metrics.items() if 'calibration_error' in k]
        metrics['average_calibration_error'] = np.mean(cal_errors)
        
        return metrics
    
    def uncertainty_decomposition(
        self,
        ensemble_predictions: List[torch.Tensor],
        mc_predictions: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose uncertainty into epistemic and aleatoric components.
        
        Args:
            ensemble_predictions: Predictions from ensemble members
            mc_predictions: Monte Carlo predictions (optional)
        
        Returns:
            Decomposed uncertainty components
        """
        # Ensemble-based epistemic uncertainty
        ensemble_preds = torch.stack(ensemble_predictions)
        epistemic = torch.var(ensemble_preds, dim=0)
        
        # If MC predictions available, use for aleatoric uncertainty
        if mc_predictions is not None:
            total_uncertainty = torch.var(mc_predictions, dim=0)
            aleatoric = total_uncertainty - epistemic
        else:
            # Estimate aleatoric from individual model uncertainties
            aleatoric = torch.zeros_like(epistemic)
        
        total = epistemic + aleatoric
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total,
            'epistemic_ratio': epistemic / (total + 1e-8),
            'aleatoric_ratio': aleatoric / (total + 1e-8)
        }
    
    def visualize_uncertainty(
        self,
        predictions: torch.Tensor,
        uncertainty: torch.Tensor,
        targets: torch.Tensor = None,
        save_path: str = None
    ):
        """
        Visualize uncertainty estimates.
        
        Args:
            predictions: Model predictions
            uncertainty: Uncertainty estimates
            targets: True targets (optional)
            save_path: Path to save plots
        """
        if not self.config.save_uncertainty_maps:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        unc_np = uncertainty.cpu().numpy()
        
        # Prediction map
        im1 = axes[0, 0].imshow(pred_np[0, 0], cmap='viridis')
        axes[0, 0].set_title('Predictions')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Uncertainty map
        im2 = axes[0, 1].imshow(unc_np[0, 0], cmap='plasma')
        axes[0, 1].set_title('Uncertainty')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Prediction vs uncertainty scatter
        pred_flat = pred_np.flatten()
        unc_flat = unc_np.flatten()
        
        axes[1, 0].scatter(pred_flat, unc_flat, alpha=0.5)
        axes[1, 0].set_xlabel('Predictions')
        axes[1, 0].set_ylabel('Uncertainty')
        axes[1, 0].set_title('Prediction vs Uncertainty')
        
        # Uncertainty histogram
        axes[1, 1].hist(unc_flat, bins=50, alpha=0.7)
        axes[1, 1].set_xlabel('Uncertainty')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Uncertainty Distribution')
        
        if targets is not None:
            target_np = targets.cpu().numpy()
            
            # Add prediction vs target plot
            fig2, ax = plt.subplots(figsize=(8, 6))
            target_flat = target_np.flatten()
            
            ax.scatter(pred_flat, target_flat, c=unc_flat, cmap='plasma', alpha=0.6)
            ax.plot([target_flat.min(), target_flat.max()], 
                   [target_flat.min(), target_flat.max()], 'r--')
            ax.set_xlabel('Predictions')
            ax.set_ylabel('Targets')
            ax.set_title('Predictions vs Targets (colored by uncertainty)')
            plt.colorbar(ax.collections[0], label='Uncertainty')
            
            if save_path:
                fig2.savefig(save_path.replace('.png', '_pred_vs_target.png'))
            plt.close(fig2)
        
        if save_path:
            fig.savefig(save_path)
        else:
            plt.show()
        
        plt.close(fig)
    
    def generate_uncertainty_report(
        self,
        predictions: torch.Tensor,
        uncertainty: torch.Tensor,
        targets: torch.Tensor = None
    ) -> Dict:
        """
        Generate comprehensive uncertainty report.
        
        Args:
            predictions: Model predictions
            uncertainty: Uncertainty estimates
            targets: True targets (optional)
        
        Returns:
            Uncertainty analysis report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'statistics': {}
        }
        
        # Basic statistics
        pred_np = predictions.cpu().numpy()
        unc_np = uncertainty.cpu().numpy()
        
        report['statistics']['prediction_stats'] = {
            'mean': float(np.mean(pred_np)),
            'std': float(np.std(pred_np)),
            'min': float(np.min(pred_np)),
            'max': float(np.max(pred_np))
        }
        
        report['statistics']['uncertainty_stats'] = {
            'mean': float(np.mean(unc_np)),
            'std': float(np.std(unc_np)),
            'min': float(np.min(unc_np)),
            'max': float(np.max(unc_np)),
            'median': float(np.median(unc_np))
        }
        
        # Calibration metrics if targets available
        if targets is not None:
            calibration_metrics = self.calculate_calibration_metrics(
                predictions, targets, uncertainty
            )
            report['calibration'] = calibration_metrics
        
        return report


# Bangladesh-specific uncertainty analysis
class BangladeshUncertaintyAnalyzer:
    """Uncertainty analysis specific to Bangladesh weather patterns."""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.uncertainty_system = UncertaintyQuantification(config)
        
        # Bangladesh-specific regions for analysis
        self.regions = {
            'dhaka': {'lat': 23.8103, 'lon': 90.4125},
            'chittagong': {'lat': 22.3569, 'lon': 91.7832},
            'sylhet': {'lat': 24.8949, 'lon': 91.8687},
            'rajshahi': {'lat': 24.3745, 'lon': 88.6042},
            'barisal': {'lat': 22.7010, 'lon': 90.3535},
            'rangpur': {'lat': 25.7439, 'lon': 89.2752}
        }
        
        # Weather patterns with high uncertainty
        self.high_uncertainty_patterns = [
            'monsoon_onset', 'cyclone_formation', 'flash_flood',
            'heat_wave', 'fog_formation', 'thunderstorm'
        ]
    
    def analyze_regional_uncertainty(
        self,
        predictions: torch.Tensor,
        uncertainty: torch.Tensor,
        coordinates: torch.Tensor
    ) -> Dict[str, Dict]:
        """
        Analyze uncertainty patterns across Bangladesh regions.
        
        Args:
            predictions: Weather predictions
            uncertainty: Uncertainty estimates
            coordinates: Lat/lon coordinates
        
        Returns:
            Regional uncertainty analysis
        """
        regional_analysis = {}
        
        for region_name, region_coords in self.regions.items():
            # Find grid points near this region
            lat_mask = torch.abs(coordinates[:, 0] - region_coords['lat']) < 0.5
            lon_mask = torch.abs(coordinates[:, 1] - region_coords['lon']) < 0.5
            region_mask = lat_mask & lon_mask
            
            if region_mask.sum() > 0:
                region_pred = predictions[region_mask]
                region_unc = uncertainty[region_mask]
                
                regional_analysis[region_name] = {
                    'mean_uncertainty': float(torch.mean(region_unc)),
                    'max_uncertainty': float(torch.max(region_unc)),
                    'uncertainty_std': float(torch.std(region_unc)),
                    'prediction_range': float(torch.max(region_pred) - torch.min(region_pred)),
                    'reliability_score': self._calculate_reliability_score(region_unc)
                }
        
        return regional_analysis
    
    def _calculate_reliability_score(self, uncertainty: torch.Tensor) -> float:
        """Calculate reliability score based on uncertainty distribution."""
        # Lower uncertainty = higher reliability
        mean_unc = torch.mean(uncertainty)
        std_unc = torch.std(uncertainty)
        
        # Normalize to 0-1 scale (1 = most reliable)
        reliability = 1.0 / (1.0 + mean_unc + std_unc)
        return float(reliability)
    
    def seasonal_uncertainty_analysis(
        self,
        uncertainty_data: Dict[str, torch.Tensor],
        season: str
    ) -> Dict:
        """
        Analyze uncertainty patterns for specific seasons.
        
        Args:
            uncertainty_data: Dictionary of uncertainty data by time
            season: Season name ('winter', 'pre_monsoon', 'monsoon', 'post_monsoon')
        
        Returns:
            Seasonal uncertainty analysis
        """
        season_patterns = {
            'winter': {
                'high_uncertainty_events': ['fog', 'cold_wave'],
                'typical_uncertainty': 'low',
                'reliability': 'high'
            },
            'pre_monsoon': {
                'high_uncertainty_events': ['thunderstorm', 'heat_wave'],
                'typical_uncertainty': 'medium',
                'reliability': 'medium'
            },
            'monsoon': {
                'high_uncertainty_events': ['cyclone', 'flood', 'heavy_rain'],
                'typical_uncertainty': 'high',
                'reliability': 'low'
            },
            'post_monsoon': {
                'high_uncertainty_events': ['cyclone', 'retreat_timing'],
                'typical_uncertainty': 'medium',
                'reliability': 'medium'
            }
        }
        
        analysis = {
            'season': season,
            'characteristics': season_patterns.get(season, {}),
            'uncertainty_statistics': {},
            'recommendations': []
        }
        
        # Add uncertainty statistics if data provided
        if uncertainty_data:
            all_uncertainty = torch.cat(list(uncertainty_data.values()))
            analysis['uncertainty_statistics'] = {
                'mean': float(torch.mean(all_uncertainty)),
                'std': float(torch.std(all_uncertainty)),
                'percentiles': {
                    '25': float(torch.quantile(all_uncertainty, 0.25)),
                    '50': float(torch.quantile(all_uncertainty, 0.50)),
                    '75': float(torch.quantile(all_uncertainty, 0.75)),
                    '95': float(torch.quantile(all_uncertainty, 0.95))
                }
            }
        
        # Generate recommendations
        if season == 'monsoon':
            analysis['recommendations'] = [
                "Increase ensemble size during monsoon season",
                "Use probabilistic forecasts for heavy rain events",
                "Consider multiple model consensus for cyclone predictions"
            ]
        elif season == 'winter':
            analysis['recommendations'] = [
                "Focus uncertainty analysis on fog prediction",
                "Use deterministic forecasts for most events",
                "Maintain standard ensemble size"
            ]
        
        return analysis
