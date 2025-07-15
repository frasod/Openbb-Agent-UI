from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging
from enum import Enum

from ..processors.error_handler import ErrorHandler, error_boundary
from ..processors.data_validator import DataValidator

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "idle"
    INITIALIZING = "initializing"
    TRAINING = "training"
    PREDICTING = "predicting"
    ERROR = "error"


@dataclass
class AgentMetadata:
    agent_id: str
    name: str
    version: str
    created_at: datetime = field(default_factory=datetime.now)
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    state: AgentState = AgentState.IDLE
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentConfig:
    enable_caching: bool = True
    cache_ttl: int = 300
    max_retries: int = 3
    timeout: float = 30.0
    debug_mode: bool = False
    auto_validate: bool = True
    enable_monitoring: bool = True


class DataScienceAgent(ABC):
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.metadata = self._initialize_metadata()
        self.error_handler = ErrorHandler()
        self.validator = DataValidator()
        self._cache = {}
        self._performance_tracker = {}

    @abstractmethod
    def _initialize_metadata(self) -> AgentMetadata:
        pass

    @abstractmethod
    async def process_data(self, data: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def train_model(self, training_data: Any) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def predict(self, input_data: Any) -> Dict[str, Any]:
        pass

    async def initialize(self) -> bool:
        self.metadata.state = AgentState.INITIALIZING
        try:
            await self._setup_dependencies()
            await self._validate_configuration()
            await self._perform_health_checks()
            self.metadata.state = AgentState.IDLE
            logger.info(f"Agent {self.metadata.name} initialized successfully")
            return True
        except Exception as e:
            self.metadata.state = AgentState.ERROR
            logger.error(f"Agent initialization failed: {e}")
            return False

    async def _setup_dependencies(self) -> None:
        for dependency in self.metadata.dependencies:
            try:
                __import__(dependency)
                logger.debug(f"Successfully imported {dependency}")
            except ImportError as e:
                raise ImportError(f"Required dependency {dependency} not available: {e}")

    async def _validate_configuration(self) -> None:
        required_configs = ['enable_caching', 'max_retries', 'timeout']
        for config_key in required_configs:
            if not hasattr(self.config, config_key):
                raise ValueError(f"Missing required configuration: {config_key}")

    async def _perform_health_checks(self) -> None:
        health_checks = [
            self._check_memory_usage,
            self._check_system_resources,
            self._verify_data_access
        ]
        
        for check in health_checks:
            try:
                await check()
            except Exception as e:
                logger.warning(f"Health check failed: {e}")

    async def _check_memory_usage(self) -> None:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:
            logger.warning(f"High memory usage: {memory_percent}%")

    async def _check_system_resources(self) -> None:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            logger.warning(f"High CPU usage: {cpu_percent}%")

    async def _verify_data_access(self) -> None:
        pass

    @error_boundary(fallback=None)
    async def safe_process_data(self, data: Any) -> Optional[Dict[str, Any]]:
        if self.config.auto_validate:
            await self._validate_input_data(data)
        
        cache_key = self._generate_cache_key(data)
        if self.config.enable_caching and cache_key in self._cache:
            logger.debug(f"Cache hit for key: {cache_key}")
            return self._cache[cache_key]

        start_time = datetime.now()
        result = await self.error_handler.retry_with_backoff(
            self.process_data,
            data,
            max_retries=self.config.max_retries
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_performance_metrics('processing_time', processing_time)
        
        if self.config.enable_caching and result:
            self._cache[cache_key] = result
            asyncio.create_task(self._expire_cache_entry(cache_key))
        
        return result

    @error_boundary(fallback=None)
    async def safe_train_model(self, training_data: Any) -> Optional[Dict[str, Any]]:
        if self.config.auto_validate:
            await self._validate_training_data(training_data)
        
        self.metadata.state = AgentState.TRAINING
        start_time = datetime.now()
        
        try:
            result = await self.error_handler.retry_with_backoff(
                self.train_model,
                training_data,
                max_retries=self.config.max_retries
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics('training_time', training_time)
            self.metadata.state = AgentState.IDLE
            
            return result
            
        except Exception as e:
            self.metadata.state = AgentState.ERROR
            await self.error_handler.log_error_with_context(e, training_data)
            raise

    @error_boundary(fallback=None)
    async def safe_predict(self, input_data: Any) -> Optional[Dict[str, Any]]:
        if self.config.auto_validate:
            await self._validate_prediction_input(input_data)
        
        self.metadata.state = AgentState.PREDICTING
        start_time = datetime.now()
        
        try:
            result = await self.error_handler.retry_with_backoff(
                self.predict,
                input_data,
                max_retries=self.config.max_retries
            )
            
            prediction_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics('prediction_time', prediction_time)
            self.metadata.state = AgentState.IDLE
            
            return result
            
        except Exception as e:
            self.metadata.state = AgentState.ERROR
            await self.error_handler.log_error_with_context(e, input_data)
            raise

    async def _validate_input_data(self, data: Any) -> None:
        if hasattr(data, '__dict__') or isinstance(data, dict):
            data_dict = data.__dict__ if hasattr(data, '__dict__') else data
            
            issues = self.validator.detect_data_quality_issues(data_dict)
            if issues:
                error_issues = [issue for issue in issues if issue.severity == 'error']
                if error_issues:
                    raise ValueError(f"Data validation failed: {error_issues}")
                
                warning_issues = [issue for issue in issues if issue.severity == 'warning']
                if warning_issues:
                    logger.warning(f"Data quality warnings: {warning_issues}")

    async def _validate_training_data(self, training_data: Any) -> None:
        if not training_data:
            raise ValueError("Training data cannot be empty")
        
        if isinstance(training_data, (list, tuple)) and len(training_data) < 10:
            logger.warning("Training data has fewer than 10 samples")

    async def _validate_prediction_input(self, input_data: Any) -> None:
        if not input_data:
            raise ValueError("Prediction input cannot be empty")

    def _generate_cache_key(self, data: Any) -> str:
        import hashlib
        data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()

    async def _expire_cache_entry(self, cache_key: str) -> None:
        await asyncio.sleep(self.config.cache_ttl)
        self._cache.pop(cache_key, None)

    def _update_performance_metrics(self, metric_name: str, value: float) -> None:
        if metric_name not in self._performance_tracker:
            self._performance_tracker[metric_name] = []
        
        self._performance_tracker[metric_name].append(value)
        
        if len(self._performance_tracker[metric_name]) > 100:
            self._performance_tracker[metric_name] = self._performance_tracker[metric_name][-100:]
        
        avg_value = sum(self._performance_tracker[metric_name]) / len(self._performance_tracker[metric_name])
        self.metadata.performance_metrics[f"avg_{metric_name}"] = avg_value

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.metadata.agent_id,
            "name": self.metadata.name,
            "version": self.metadata.version,
            "state": self.metadata.state.value,
            "capabilities": self.metadata.capabilities,
            "performance_metrics": self.metadata.performance_metrics,
            "cache_size": len(self._cache),
            "uptime": (datetime.now() - self.metadata.created_at).total_seconds()
        }

    def get_health_check(self) -> Dict[str, Any]:
        return {
            "healthy": self.metadata.state != AgentState.ERROR,
            "state": self.metadata.state.value,
            "last_error": getattr(self, '_last_error', None),
            "performance": self.metadata.performance_metrics,
            "memory_usage": self._get_memory_usage()
        }

    def _get_memory_usage(self) -> float:
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0

    async def cleanup(self) -> None:
        self._cache.clear()
        self._performance_tracker.clear()
        logger.info(f"Agent {self.metadata.name} cleaned up successfully")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.create_task(self.cleanup())


class OpenBBDataScienceAgent(DataScienceAgent):
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config)
        self.openbb_client = None

    def _initialize_metadata(self) -> AgentMetadata:
        return AgentMetadata(
            agent_id="openbb-ds-agent",
            name="OpenBB Data Science Agent",
            version="1.0.0",
            capabilities=[
                "data_processing",
                "financial_analysis",
                "model_training",
                "prediction",
                "visualization"
            ],
            dependencies=[
                "pandas",
                "numpy",
                "scikit-learn",
                "openbb_ai"
            ]
        )

    async def _setup_openbb_client(self) -> None:
        try:
            from openbb_ai import get_widget_data
            self.openbb_client = get_widget_data
            logger.info("OpenBB client initialized successfully")
        except ImportError as e:
            raise ImportError(f"OpenBB client not available: {e}")

    async def process_data(self, data: Any) -> Dict[str, Any]:
        if not data:
            raise ValueError("Data cannot be empty")
        
        processed_data = {
            "status": "success",
            "data_type": type(data).__name__,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        if isinstance(data, dict):
            processed_data["record_count"] = len(data)
            processed_data["fields"] = list(data.keys())
        elif isinstance(data, list):
            processed_data["record_count"] = len(data)
            if data and isinstance(data[0], dict):
                processed_data["fields"] = list(data[0].keys())
        
        return processed_data

    async def train_model(self, training_data: Any) -> Dict[str, Any]:
        return {
            "status": "training_completed",
            "model_type": "base_model",
            "training_timestamp": datetime.now().isoformat(),
            "data_points": len(training_data) if hasattr(training_data, '__len__') else 0
        }

    async def predict(self, input_data: Any) -> Dict[str, Any]:
        return {
            "status": "prediction_completed",
            "prediction_timestamp": datetime.now().isoformat(),
            "input_type": type(input_data).__name__,
            "confidence": 0.85
        }