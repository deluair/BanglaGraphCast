"""
Bangladesh-specific GraphCast configuration
"""

# Domain Configuration
BANGLADESH_DOMAIN = {
    'name': 'bangladesh_extended',
    'lat_range': (20.0, 27.0),  # Extended buffer for boundary conditions
    'lon_range': (88.0, 93.0),
    'resolution': 0.1,  # Target resolution in degrees
    'center': (23.8103, 90.4125),  # Dhaka coordinates
}

# Data Sources
DATA_SOURCES = {
    'era5': {
        'variables': {
            'surface': [
                '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
                'surface_pressure', 'total_precipitation', 'sea_surface_temperature'
            ],
            'pressure_levels': [
                'temperature', 'u_component_of_wind', 'v_component_of_wind',
                'geopotential', 'specific_humidity'
            ],
            'levels': [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
        },
        'temporal_resolution': '6H',
        'spatial_resolution': 0.25
    },
    'bmd_stations': {
        'synoptic': 35,
        'rainfall': 95,
        'temporal_resolution': '3H'
    },
    'satellite': {
        'gpm_imerg': {'resolution': 0.1, 'temporal': 'hourly'},
        'modis_lst': {'resolution': 1000, 'temporal': 'daily'},
        'sentinel1': {'resolution': 10, 'temporal': 'weekly'}
    }
}

# Model Architecture
MODEL_CONFIG = {
    'mesh_levels': {
        'global': 6,     # Icosahedron refinement for global context
        'regional': 8,   # South Asia region
        'local': 10      # Bangladesh focus area
    },
    'encoder_layers': 16,
    'processor_layers': 16,
    'decoder_layers': 16,
    'hidden_dim': 512,
    'num_heads': 8,
    'dropout': 0.1,
    'activation': 'gelu'
}

# Physics Constraints
PHYSICS_CONFIG = {
    'cyclone': {
        'min_pressure': 900,  # hPa
        'max_wind_speed': 85,  # m/s
        'eye_radius_range': (10, 80),  # km
        'track_constraints': True
    },
    'monsoon': {
        'onset_criteria': {
            'rainfall_threshold': 5.0,  # mm/day
            'consecutive_days': 5,
            'wind_shear_reversal': True
        },
        'seasonal_memory': True
    },
    'orographic': {
        'enable_blocking': True,
        'precipitation_enhancement': True,
        'lee_wave_effects': True
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'max_epochs': 100,
    'patience': 10,
    'gradient_clip': 1.0,
    'accumulate_grad_batches': 4,
    'precision': 16,
    'curriculum_learning': True
}

# Loss Weights
LOSS_WEIGHTS = {
    'standard_vars': 1.0,
    'precipitation': 2.5,
    'cyclone_track': 5.0,
    'cyclone_intensity': 4.0,
    'monsoon_onset': 3.0,
    'extreme_precip': 3.5,
    'heat_stress': 2.0,
    'flood_risk': 3.0
}

# Evaluation Thresholds
EVALUATION_THRESHOLDS = {
    'precipitation': [1, 5, 10, 25, 50, 100],  # mm
    'wind_speed': [10, 15, 25, 35],  # m/s
    'temperature': [30, 35, 40],  # °C
    'cyclone_categories': [
        (119, 'Depression'),
        (88, 'Cyclonic Storm'),
        (117, 'Severe Cyclonic Storm'),
        (165, 'Very Severe Cyclonic Storm'),
        (221, 'Extremely Severe Cyclonic Storm'),
        (float('inf'), 'Super Cyclonic Storm')
    ]
}

# Operational Settings
OPERATIONAL_CONFIG = {
    'forecast_horizon': 240,  # hours (10 days)
    'update_frequency': 6,    # hours
    'data_latency': 3,        # hours
    'ensemble_members': 20,
    'products': {
        'public_warning': True,
        'marine_forecast': True,
        'aviation': True,
        'agriculture': True,
        'renewable_energy': True
    }
}

# File Paths
PATHS = {
    'data_root': 'data/',
    'raw_data': 'data/raw/',
    'processed_data': 'data/processed/',
    'models': 'models/',
    'checkpoints': 'checkpoints/',
    'outputs': 'outputs/',
    'logs': 'logs/',
    'configs': 'configs/'
}
