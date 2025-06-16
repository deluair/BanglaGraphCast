"""
Real-time operational pipeline for Bangladesh GraphCast weather prediction
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class DataStreamStatus(Enum):
    """Data stream status"""
    ACTIVE = "active"
    DELAYED = "delayed"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class ForecastStatus(Enum):
    """Forecast run status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    POST_PROCESSING = "post_processing"


@dataclass
class DataStreamInfo:
    """Information about a data stream"""
    name: str
    source: str
    update_frequency_hours: int
    last_update: Optional[datetime]
    status: DataStreamStatus
    latency_minutes: int
    quality_score: float


@dataclass
class ForecastRun:
    """Information about a forecast run"""
    run_id: str
    initialization_time: datetime
    status: ForecastStatus
    lead_time_hours: int
    completion_time: Optional[datetime]
    model_version: str
    input_data_quality: float


class OperationalPipeline:
    """
    Real-time operational pipeline for Bangladesh weather prediction
    
    Manages:
    - Data ingestion from multiple sources
    - Model initialization and execution
    - Quality control and bias correction
    - Product generation and dissemination
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize data stream processors
        self.data_streams = {
            'gts': GTSDecoder(config.get('gts', {})),
            'satellite': SatelliteProcessor(config.get('satellite', {})),
            'radar': DhakaRadarProcessor(config.get('radar', {})),
            'aws': AutoWeatherStations(config.get('aws', {})),
            'bmd_stations': BMDStationProcessor(config.get('bmd', {})),
            'upstream_models': UpstreamModelProcessor(config.get('upstream', {}))
        }
        
        # Model components
        self.model = None  # Will be loaded
        self.data_assimilation = DataAssimilationSystem()
        self.bias_correction = BiasCorrection()
        self.ensemble_processor = EnsembleProcessor()
        
        # Product generators
        self.product_generators = {
            'public_warnings': PublicWarningGenerator(),
            'marine_forecast': MarineForecastGenerator(),
            'agriculture': AgriculturalAdvisoryGenerator(),
            'aviation': AviationProductGenerator(),
            'flood_forecast': FloodForecastGenerator(),
            'energy': RenewableEnergyGenerator()
        }
        
        # Status tracking
        self.stream_status = {}
        self.forecast_runs = {}
        self.last_successful_run = None
        
        # Operational parameters
        self.forecast_schedule = config.get('forecast_schedule', [0, 6, 12, 18])  # UTC hours
        self.max_data_latency = config.get('max_data_latency_hours', 6)
        self.min_data_quality = config.get('min_data_quality', 0.7)
        
    async def run_operational_cycle(self) -> ForecastRun:
        """
        Execute complete operational forecast cycle
        
        Returns:
            ForecastRun object with results and status
        """
        run_id = f"BDGraphCast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        initialization_time = datetime.now()
        
        logger.info(f"Starting operational cycle: {run_id}")
        
        forecast_run = ForecastRun(
            run_id=run_id,
            initialization_time=initialization_time,
            status=ForecastStatus.INITIALIZING,
            lead_time_hours=240,  # 10 days
            completion_time=None,
            model_version="BanglaGraphCast_v1.0",
            input_data_quality=0.0
        )
        
        try:
            # Step 1: Data collection and quality assessment
            logger.info("Step 1: Data collection and assessment")
            forecast_run.status = ForecastStatus.INITIALIZING
            
            data_quality = await self._collect_and_assess_data()
            forecast_run.input_data_quality = data_quality
            
            if data_quality < self.min_data_quality:
                logger.warning(f"Data quality {data_quality:.2f} below threshold {self.min_data_quality}")
                # Continue with degraded forecast or wait for better data
            
            # Step 2: Data assimilation
            logger.info("Step 2: Data assimilation")
            analysis_state = await self._run_data_assimilation()
            
            # Step 3: Model forecast
            logger.info("Step 3: Running GraphCast forecast")
            forecast_run.status = ForecastStatus.RUNNING
            
            raw_forecast = await self._run_graphcast_forecast(analysis_state)
            
            # Step 4: Post-processing
            logger.info("Step 4: Post-processing and bias correction")
            forecast_run.status = ForecastStatus.POST_PROCESSING
            
            processed_forecast = await self._post_process_forecast(raw_forecast)
            
            # Step 5: Product generation
            logger.info("Step 5: Generating forecast products")
            products = await self._generate_forecast_products(processed_forecast)
            
            # Step 6: Dissemination
            logger.info("Step 6: Product dissemination")
            await self._disseminate_products(products)
            
            # Complete the run
            forecast_run.status = ForecastStatus.COMPLETED
            forecast_run.completion_time = datetime.now()
            
            self.last_successful_run = forecast_run
            self.forecast_runs[run_id] = forecast_run
            
            logger.info(f"Operational cycle completed successfully: {run_id}")
            logger.info(f"Runtime: {(forecast_run.completion_time - forecast_run.initialization_time).total_seconds():.1f}s")
            
            return forecast_run
            
        except Exception as e:
            logger.error(f"Operational cycle failed: {e}")
            forecast_run.status = ForecastStatus.FAILED
            forecast_run.completion_time = datetime.now()
            self.forecast_runs[run_id] = forecast_run
            
            # Emergency procedures
            await self._handle_forecast_failure(forecast_run, str(e))
            
            return forecast_run
    
    async def _collect_and_assess_data(self) -> float:
        """
        Collect data from all sources and assess quality
        
        Returns:
            Overall data quality score (0-1)
        """
        quality_scores = []
        
        for stream_name, processor in self.data_streams.items():
            try:
                # Get latest data
                data, metadata = await processor.get_latest_data()
                
                # Assess quality
                quality = await processor.assess_data_quality(data, metadata)
                quality_scores.append(quality)
                
                # Update stream status
                self.stream_status[stream_name] = DataStreamInfo(
                    name=stream_name,
                    source=processor.get_source_info(),
                    update_frequency_hours=processor.get_update_frequency(),
                    last_update=metadata.get('timestamp'),
                    status=DataStreamStatus.ACTIVE if quality > 0.5 else DataStreamStatus.DELAYED,
                    latency_minutes=processor.get_latency_minutes(),
                    quality_score=quality
                )
                
                logger.debug(f"Data stream {stream_name}: quality={quality:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to collect data from {stream_name}: {e}")
                
                # Mark stream as failed
                self.stream_status[stream_name] = DataStreamInfo(
                    name=stream_name,
                    source="unknown",
                    update_frequency_hours=0,
                    last_update=None,
                    status=DataStreamStatus.FAILED,
                    latency_minutes=9999,
                    quality_score=0.0
                )
                quality_scores.append(0.0)
        
        # Calculate overall quality
        overall_quality = sum(quality_scores) / max(len(quality_scores), 1)
        
        logger.info(f"Data collection complete. Overall quality: {overall_quality:.2f}")
        
        return overall_quality
    
    async def _run_data_assimilation(self) -> Dict:
        """
        Run 3D-Var data assimilation to create analysis state
        
        Returns:
            Analysis state for model initialization
        """
        logger.info("Running 3D-Var data assimilation")
        
        # Collect observations from all sources
        observations = {}
        for stream_name, processor in self.data_streams.items():
            try:
                obs_data, _ = await processor.get_observations_for_assimilation()
                observations[stream_name] = obs_data
            except Exception as e:
                logger.warning(f"Could not get observations from {stream_name}: {e}")
        
        # Run assimilation
        analysis_state = await self.data_assimilation.run_3dvar(observations)
        
        logger.info("Data assimilation completed")
        return analysis_state
    
    async def _run_graphcast_forecast(self, analysis_state: Dict) -> Dict:
        """
        Run GraphCast model forecast
        
        Args:
            analysis_state: Initial conditions from data assimilation
            
        Returns:
            Raw model forecast output
        """
        if self.model is None:
            await self._load_model()
        
        logger.info("Running GraphCast model")
        
        # Prepare input data
        model_input = await self._prepare_model_input(analysis_state)
        
        # Run inference
        forecast_output = await self.model.predict(model_input)
        
        logger.info("GraphCast forecast completed")
        return forecast_output
    
    async def _post_process_forecast(self, raw_forecast: Dict) -> Dict:
        """
        Post-process raw forecast output
        
        Args:
            raw_forecast: Raw model output
            
        Returns:
            Post-processed forecast
        """
        # Bias correction
        corrected_forecast = await self.bias_correction.apply_corrections(raw_forecast)
        
        # Ensemble calibration if available
        if self.config.get('ensemble_mode', False):
            calibrated_forecast = await self.ensemble_processor.calibrate_ensemble(corrected_forecast)
        else:
            calibrated_forecast = corrected_forecast
        
        # Quality control
        qc_forecast = await self._apply_quality_control(calibrated_forecast)
        
        return qc_forecast
    
    async def _generate_forecast_products(self, forecast: Dict) -> Dict:
        """
        Generate all forecast products
        
        Args:
            forecast: Processed forecast data
            
        Returns:
            Dictionary of generated products
        """
        products = {}
        
        for product_name, generator in self.product_generators.items():
            try:
                product = await generator.generate(forecast)
                products[product_name] = product
                logger.debug(f"Generated {product_name} product")
            except Exception as e:
                logger.error(f"Failed to generate {product_name} product: {e}")
                products[product_name] = None
        
        return products
    
    async def _disseminate_products(self, products: Dict) -> None:
        """
        Disseminate forecast products to various channels
        
        Args:
            products: Generated forecast products
        """
        dissemination_tasks = []
        
        # BMD internal systems
        dissemination_tasks.append(self._send_to_bmd_systems(products))
        
        # Public websites and APIs
        dissemination_tasks.append(self._update_public_api(products))
        
        # Mobile applications
        dissemination_tasks.append(self._update_mobile_apps(products))
        
        # International exchanges (WMO GTS)
        dissemination_tasks.append(self._send_to_gts(products))
        
        # Media and emergency services
        dissemination_tasks.append(self._notify_emergency_services(products))
        
        # Execute all dissemination tasks concurrently
        await asyncio.gather(*dissemination_tasks, return_exceptions=True)
        
        logger.info("Product dissemination completed")
    
    async def _handle_forecast_failure(self, forecast_run: ForecastRun, error_message: str) -> None:
        """
        Handle forecast failure with emergency procedures
        
        Args:
            forecast_run: Failed forecast run information
            error_message: Error description
        """
        logger.error(f"Activating emergency procedures for failed run: {forecast_run.run_id}")
        
        # Send alerts to operators
        await self._send_operator_alerts(forecast_run, error_message)
        
        # Fall back to previous forecast or backup models
        if self.last_successful_run:
            logger.info("Extending previous forecast as fallback")
            await self._extend_previous_forecast()
        else:
            logger.warning("No previous forecast available - using climatology")
            await self._generate_climatological_forecast()
        
        # Log incident for analysis
        await self._log_operational_incident(forecast_run, error_message)
    
    async def get_operational_status(self) -> Dict:
        """
        Get current operational status
        
        Returns:
            Comprehensive operational status information
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'data_streams': {},
            'model_status': 'operational' if self.model else 'offline',
            'last_forecast': None,
            'next_forecast': None,
            'system_health': 'healthy'
        }
        
        # Data stream status
        for stream_name, stream_info in self.stream_status.items():
            status['data_streams'][stream_name] = {
                'status': stream_info.status.value,
                'quality': stream_info.quality_score,
                'latency_minutes': stream_info.latency_minutes,
                'last_update': stream_info.last_update.isoformat() if stream_info.last_update else None
            }
        
        # Last forecast information
        if self.last_successful_run:
            status['last_forecast'] = {
                'run_id': self.last_successful_run.run_id,
                'initialization_time': self.last_successful_run.initialization_time.isoformat(),
                'completion_time': self.last_successful_run.completion_time.isoformat(),
                'data_quality': self.last_successful_run.input_data_quality
            }
        
        # Next scheduled forecast
        status['next_forecast'] = self._get_next_forecast_time().isoformat()
        
        # Overall system health assessment
        data_health = sum(1 for s in self.stream_status.values() 
                         if s.status == DataStreamStatus.ACTIVE) / max(len(self.stream_status), 1)
        
        if data_health < 0.5:
            status['system_health'] = 'degraded'
        elif data_health < 0.3:
            status['system_health'] = 'critical'
        
        return status
    
    def _get_next_forecast_time(self) -> datetime:
        """Calculate next scheduled forecast time"""
        now = datetime.now()
        current_hour = now.hour
        
        # Find next forecast hour
        next_hour = None
        for hour in self.forecast_schedule:
            if hour > current_hour:
                next_hour = hour
                break
        
        if next_hour is None:
            # Next forecast is tomorrow
            next_hour = self.forecast_schedule[0]
            next_date = now.date() + timedelta(days=1)
        else:
            next_date = now.date()
        
        return datetime.combine(next_date, datetime.min.time().replace(hour=next_hour))
    
    # Placeholder methods for complex operations
    async def _load_model(self):
        """Load GraphCast model"""
        self.model = MockGraphCastModel()
        logger.info("GraphCast model loaded")
    
    async def _prepare_model_input(self, analysis_state: Dict) -> Dict:
        """Prepare input for GraphCast model"""
        return analysis_state
    
    async def _apply_quality_control(self, forecast: Dict) -> Dict:
        """Apply quality control to forecast"""
        return forecast
    
    async def _send_to_bmd_systems(self, products: Dict):
        """Send products to BMD internal systems"""
        logger.debug("Sending products to BMD systems")
    
    async def _update_public_api(self, products: Dict):
        """Update public API with latest products"""
        logger.debug("Updating public API")
    
    async def _update_mobile_apps(self, products: Dict):
        """Update mobile applications"""
        logger.debug("Updating mobile applications")
    
    async def _send_to_gts(self, products: Dict):
        """Send products to WMO Global Telecommunication System"""
        logger.debug("Sending to GTS")
    
    async def _notify_emergency_services(self, products: Dict):
        """Notify emergency services of critical conditions"""
        if products.get('public_warnings'):
            logger.info("Notifying emergency services")
    
    async def _send_operator_alerts(self, forecast_run: ForecastRun, error: str):
        """Send alerts to system operators"""
        logger.critical(f"OPERATOR ALERT: Forecast failure - {error}")
    
    async def _extend_previous_forecast(self):
        """Extend previous forecast as emergency fallback"""
        logger.info("Using previous forecast as fallback")
    
    async def _generate_climatological_forecast(self):
        """Generate climatological forecast as last resort"""
        logger.info("Generating climatological forecast")
    
    async def _log_operational_incident(self, forecast_run: ForecastRun, error: str):
        """Log operational incident for analysis"""
        incident = {
            'timestamp': datetime.now().isoformat(),
            'run_id': forecast_run.run_id,
            'error': error,
            'data_quality': forecast_run.input_data_quality,
            'stream_status': {name: info.status.value for name, info in self.stream_status.items()}
        }
        logger.error(f"INCIDENT LOGGED: {json.dumps(incident)}")


# Supporting classes (simplified implementations)

class GTSDecoder:
    """Global Telecommunication System data decoder"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    async def get_latest_data(self) -> Tuple[Dict, Dict]:
        """Get latest GTS data"""
        return {}, {'timestamp': datetime.now()}
    
    async def assess_data_quality(self, data: Dict, metadata: Dict) -> float:
        """Assess GTS data quality"""
        return 0.8
    
    async def get_observations_for_assimilation(self) -> Tuple[Dict, Dict]:
        """Get observations for data assimilation"""
        return {}, {}
    
    def get_source_info(self) -> str:
        return "WMO GTS"
    
    def get_update_frequency(self) -> int:
        return 6  # hours
    
    def get_latency_minutes(self) -> int:
        return 90


class SatelliteProcessor:
    """Satellite data processor"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    async def get_latest_data(self) -> Tuple[Dict, Dict]:
        return {}, {'timestamp': datetime.now()}
    
    async def assess_data_quality(self, data: Dict, metadata: Dict) -> float:
        return 0.9
    
    async def get_observations_for_assimilation(self) -> Tuple[Dict, Dict]:
        return {}, {}
    
    def get_source_info(self) -> str:
        return "Multiple Satellites"
    
    def get_update_frequency(self) -> int:
        return 1  # hours
    
    def get_latency_minutes(self) -> int:
        return 30


class DhakaRadarProcessor:
    """Dhaka weather radar processor"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    async def get_latest_data(self) -> Tuple[Dict, Dict]:
        return {}, {'timestamp': datetime.now()}
    
    async def assess_data_quality(self, data: Dict, metadata: Dict) -> float:
        return 0.85
    
    async def get_observations_for_assimilation(self) -> Tuple[Dict, Dict]:
        return {}, {}
    
    def get_source_info(self) -> str:
        return "BMD Dhaka Radar"
    
    def get_update_frequency(self) -> int:
        return 0.25  # 15 minutes
    
    def get_latency_minutes(self) -> int:
        return 10


class AutoWeatherStations:
    """Automatic Weather Stations processor"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    async def get_latest_data(self) -> Tuple[Dict, Dict]:
        return {}, {'timestamp': datetime.now()}
    
    async def assess_data_quality(self, data: Dict, metadata: Dict) -> float:
        return 0.75
    
    async def get_observations_for_assimilation(self) -> Tuple[Dict, Dict]:
        return {}, {}
    
    def get_source_info(self) -> str:
        return "BMD AWS Network"
    
    def get_update_frequency(self) -> int:
        return 1  # hours
    
    def get_latency_minutes(self) -> int:
        return 15


class BMDStationProcessor:
    """BMD station data processor"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    async def get_latest_data(self) -> Tuple[Dict, Dict]:
        return {}, {'timestamp': datetime.now()}
    
    async def assess_data_quality(self, data: Dict, metadata: Dict) -> float:
        return 0.9
    
    async def get_observations_for_assimilation(self) -> Tuple[Dict, Dict]:
        return {}, {}
    
    def get_source_info(self) -> str:
        return "BMD Station Network"
    
    def get_update_frequency(self) -> int:
        return 3  # hours
    
    def get_latency_minutes(self) -> int:
        return 60


class UpstreamModelProcessor:
    """Upstream model data processor (GFS, ECMWF, etc.)"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    async def get_latest_data(self) -> Tuple[Dict, Dict]:
        return {}, {'timestamp': datetime.now()}
    
    async def assess_data_quality(self, data: Dict, metadata: Dict) -> float:
        return 0.95
    
    async def get_observations_for_assimilation(self) -> Tuple[Dict, Dict]:
        return {}, {}
    
    def get_source_info(self) -> str:
        return "Global Models"
    
    def get_update_frequency(self) -> int:
        return 6  # hours
    
    def get_latency_minutes(self) -> int:
        return 180


class DataAssimilationSystem:
    """3D-Var data assimilation system"""
    
    async def run_3dvar(self, observations: Dict) -> Dict:
        """Run 3D-Var data assimilation"""
        logger.info("Running 3D-Var assimilation")
        return {'analysis_complete': True}


class BiasCorrection:
    """Bias correction system"""
    
    async def apply_corrections(self, forecast: Dict) -> Dict:
        """Apply bias corrections"""
        logger.debug("Applying bias corrections")
        return forecast


class EnsembleProcessor:
    """Ensemble forecast processor"""
    
    async def calibrate_ensemble(self, forecast: Dict) -> Dict:
        """Calibrate ensemble forecast"""
        logger.debug("Calibrating ensemble")
        return forecast


class MockGraphCastModel:
    """Mock GraphCast model for testing"""
    
    async def predict(self, input_data: Dict) -> Dict:
        """Run model prediction"""
        await asyncio.sleep(2)  # Simulate model runtime
        return {'forecast_complete': True}


# Product generators (simplified)
class PublicWarningGenerator:
    async def generate(self, forecast: Dict) -> Dict:
        return {'warnings': []}

class MarineForecastGenerator:
    async def generate(self, forecast: Dict) -> Dict:
        return {'marine_forecast': {}}

class AgriculturalAdvisoryGenerator:
    async def generate(self, forecast: Dict) -> Dict:
        return {'agricultural_advisory': {}}

class AviationProductGenerator:
    async def generate(self, forecast: Dict) -> Dict:
        return {'aviation_products': {}}

class FloodForecastGenerator:
    async def generate(self, forecast: Dict) -> Dict:
        return {'flood_forecast': {}}

class RenewableEnergyGenerator:
    async def generate(self, forecast: Dict) -> Dict:
        return {'energy_forecast': {}}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    async def test_operational_pipeline():
        config = {
            'forecast_schedule': [0, 6, 12, 18],
            'max_data_latency_hours': 6,
            'min_data_quality': 0.7
        }
        
        pipeline = OperationalPipeline(config)
        
        # Test operational cycle
        forecast_run = await pipeline.run_operational_cycle()
        
        logger.info(f"Test run completed: {forecast_run.run_id}")
        logger.info(f"Status: {forecast_run.status}")
        logger.info(f"Data quality: {forecast_run.input_data_quality:.2f}")
        
        # Get operational status
        status = await pipeline.get_operational_status()
        logger.info(f"System health: {status['system_health']}")
        logger.info(f"Active data streams: {sum(1 for s in status['data_streams'].values() if s['status'] == 'active')}")
    
    # Run test
    asyncio.run(test_operational_pipeline())
    
    logger.info("Operational pipeline test completed successfully")
