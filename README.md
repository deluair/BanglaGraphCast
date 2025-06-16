# BanglaGraphCast: Advanced Weather Forecasting for Bangladesh

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌦️ Overview

BanglaGraphCast is a state-of-the-art weather forecasting system specifically designed for Bangladesh and the Bay of Bengal region. Built on Google's GraphCast architecture, it provides comprehensive weather prediction capabilities from nowcasting (0-3 hours) to climate projections (decades).

### 🎯 Key Features

- **Multi-Scale Predictions**: Nowcasting, short-term, medium-range, S2S, and climate projections
- **Ensemble Forecasting**: Multi-member ensemble with uncertainty quantification
- **Extreme Weather Focus**: Specialized modules for cyclones, floods, and heat waves
- **High-Resolution Downscaling**: 1km resolution climate projections
- **Bangladesh-Specific Physics**: Monsoon dynamics, coastal processes, and orographic effects
- **Operational Ready**: Real-time data integration and automated forecasting

## 🏗️ System Architecture

### Integrated Advanced Components

```
BanglaGraphCast/
├── models/
│   ├── core/                    # Core GraphCast implementation
│   │   └── graphcast_bangladesh.py
│   ├── ensemble/                # Ensemble generation system
│   │   └── bangladesh_ensemble.py
│   ├── s2s/                     # Subseasonal-to-seasonal prediction
│   │   └── bangladesh_s2s.py
│   ├── climate/                 # Climate downscaling
│   │   └── bangladesh_climate_downscaling.py
│   ├── advanced/                # Advanced modules
│   │   ├── extreme_weather_prediction.py
│   │   ├── nowcasting.py
│   │   ├── uncertainty_quantification.py
│   │   └── bias_correction.py
│   └── physics/                 # Bangladesh-specific physics
│       └── bangladesh_physics.py
├── training/
│   ├── curriculum/              # Curriculum learning system
│   │   └── bangladesh_curriculum.py
│   ├── evaluation/              # Bangladesh-specific metrics
│   │   └── bangladesh_metrics.py
│   └── losses/                  # Multi-objective loss functions
│       └── bangladesh_loss.py
├── operational/                 # Real-time forecasting system
├── data/                        # Data processing and management
├── configs/                     # Configuration files
│   └── bangladesh_config.py
└── train.py                     # Integrated training system
```

### Prediction Capabilities

| Component | Time Range | Resolution | Key Features |
|-----------|------------|------------|--------------|
| **Nowcasting** | 0-3 hours | 1km, 5min | Convection, precipitation intensity |
| **Short-term** | 3-72 hours | 5km, 1h | Standard weather variables |
| **Medium-range** | 3-10 days | 10km, 3h | Ensemble forecasting |
| **S2S Prediction** | 2-12 weeks | 25km, daily | Monsoon, teleconnections |
| **Climate Projections** | Years-decades | 1-5km | Downscaled scenarios |

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/deluair/BanglaGraphCast.git
cd BanglaGraphCast
```

2. **Create conda environment**
```bash
conda create -n banglagraphcast python=3.9
conda activate banglagraphcast
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from train import IntegratedTrainingSystem
from configs.bangladesh_config import BangladeshConfig

# Initialize system
config = BangladeshConfig()
training_system = IntegratedTrainingSystem(config)

# Train the system (with your data loaders)
training_system.train(train_loader, val_loader)
```

## 🧠 Training System

### Progressive Curriculum Learning

The system uses a 9-stage curriculum learning approach:

1. **Core Weather (15%)** - Basic atmospheric dynamics
2. **Physics Enhanced (15%)** - Physical constraints and conservation laws
3. **Nowcasting Training (10%)** - Very short-term high-resolution prediction
4. **Ensemble Training (15%)** - Multi-member ensemble generation
5. **Extreme Weather Training (10%)** - Cyclones, floods, heat waves
6. **S2S Training (10%)** - Long-range prediction with teleconnections
7. **Uncertainty Training (10%)** - Confidence estimation and calibration
8. **Bias Correction Training (5%)** - Systematic error correction
9. **Downscaling Training (10%)** - High-resolution climate projections

### Multi-Objective Loss Function

```python
total_loss = (
    1.0 * standard_loss +          # Basic meteorological variables
    2.5 * precipitation_loss +     # Critical for agriculture/flooding
    5.0 * cyclone_track_loss +     # High-impact events
    4.0 * cyclone_intensity_loss + # Life-threatening parameter
    3.0 * monsoon_onset_loss +     # Agricultural planning
    3.5 * extreme_precip_loss +    # Flooding prediction
    2.0 * heat_stress_loss +       # Health impacts
    3.0 * flood_risk_loss +        # Composite flooding risk
    4.5 * coastal_surge_loss +     # Storm surge prediction
    3.5 * wind_damage_loss         # Infrastructure damage
)
```

## 📊 Evaluation Metrics

### Bangladesh-Specific Metrics

- **Precipitation**: ETS, POD, FAR for thresholds [1, 5, 10, 25, 50, 100] mm
- **Cyclone Tracking**: Track error, intensity MAE, landfall timing
- **Monsoon**: Onset/withdrawal date accuracy, seasonal rainfall
- **Impact-Based**: Flood risk score, agricultural relevance, heat stress accuracy

## 🌊 Advanced Features

### Extreme Weather Prediction

```python
from models.advanced.extreme_weather_prediction import ExtremeWeatherPredictor

# Initialize extreme weather predictor
extreme_predictor = ExtremeWeatherPredictor(config)

# Predict cyclone formation and track
cyclone_forecast = extreme_predictor.predict_cyclone_formation(
    atmospheric_state, lead_time_hours=120
)
```

### Ensemble Uncertainty Quantification

```python
from models.ensemble.bangladesh_ensemble import BanglaGraphCastEnsemble

# Generate ensemble forecast
ensemble = BanglaGraphCastEnsemble(core_model, ensemble_config)
ensemble_forecast = ensemble.generate_ensemble_forecast(
    initial_state, n_members=20, lead_time_hours=240
)
```

### Climate Downscaling

```python
from models.climate.bangladesh_climate_downscaling import ClimateProjectionSystem

# Downscale climate projections
downscaler = ClimateProjectionSystem(downscaling_config)
high_res_projections = downscaler.downscale_projection(
    coarse_gcm_data, scenario='SSP2-4.5', target_resolution_km=1.0
)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Run tests: `python test_system.py`
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **DeepMind/Google**: Original GraphCast architecture
- **Bangladesh Meteorological Department**: Domain expertise and data
- **University of Tennessee**: Research support

---

**Made with ❤️ for Bangladesh 🇧🇩**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   ```bash
   python scripts/download_era5.py --domain bangladesh
   python scripts/process_bmd_data.py
   ```

3. **Training**:
   ```bash
   python train.py --config configs/bangladesh_config.yaml
   ```

4. **Inference**:
   ```bash
   python inference.py --model_path checkpoints/best_model.pt
   ```

## Data Sources

- **ERA5 Reanalysis**: High-resolution atmospheric data
- **BMD Stations**: Local observational data
- **Satellite Data**: GPM IMERG, MODIS, Sentinel-1
- **Topographical**: SRTM DEM, land use/cover
- **Hydrological**: River gauge data, coastal boundaries

## Model Innovations

1. **Adaptive Mesh**: Higher resolution over coastal areas, river confluences, and urban centers
2. **Cyclone Module**: Specialized tropical cyclone prediction with physics constraints
3. **Monsoon Physics**: Captures intra-seasonal oscillations and onset/withdrawal
4. **Hydrological Coupling**: Integrated flood prediction capabilities

## Evaluation Metrics

- Standard meteorological metrics (RMSE, ACC, Bias)
- Precipitation-specific scores (ETS, Frequency Bias)
- Cyclone metrics (Track Error, Intensity MAE)
- Impact-based metrics (Flood Risk, Agricultural Relevance)

## Deployment

The system is designed for operational deployment at Bangladesh Meteorological Department (BMD) with:
- Real-time data ingestion
- Automated forecast generation
- Multi-sector product dissemination
- Web-based visualization interface

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google DeepMind for the original GraphCast architecture
- Bangladesh Meteorological Department for data and domain expertise
- European Centre for Medium-Range Weather Forecasts (ECMWF) for ERA5 data
