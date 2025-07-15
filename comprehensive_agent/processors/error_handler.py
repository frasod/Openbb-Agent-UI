from typing import Any, Callable, Dict, Optional
import asyncio
import logging
from functools import wraps
import random

logger = logging.getLogger(__name__)

class AgentError(Exception):
    """Base exception for agent errors"""
    pass

class DataProcessingError(AgentError):
    """Raised when data processing fails"""
    pass

class VisualizationError(AgentError):
    """Raised when visualization generation fails"""
    pass

class CitationError(AgentError):
    """Raised when citation generation fails"""
    pass

# Validation error type

class DataValidationError(AgentError):
    """Raised when data validation fails"""
    pass

class ErrorHandler:
    @staticmethod
    async def handle_data_error(error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        error_info = {
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context,
            "recoverable": True
        }
        
        if isinstance(error, (KeyError, ValueError)):
            error_info["suggestion"] = "Check data format and required fields"
        elif isinstance(error, ConnectionError):
            error_info["suggestion"] = "Verify network connectivity and service availability"
        else:
            error_info["recoverable"] = False
            
        logger.error(f"Data processing error: {error_info}")
        return error_info
    
    @staticmethod
    async def retry_with_backoff(
        func: Callable,
        *args,
        max_retries: int = 3,
        base_wait: float = 1.0,
        max_wait: float = 30.0,
        jitter: bool = True,
        **kwargs,
    ) -> Any:
        """Retry *func* with exponential back-off.

        Parameters
        ----------
        max_retries
            Total attempts before raising the final exception.
        base_wait
            Initial wait in seconds before the first retry.
        max_wait
            Upper bound on the back-off delay.
        jitter
            If *True*, add ±50 % random jitter to the computed delay to avoid
            thundering-herd problems.
        """

        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise

                wait_time = min(base_wait * 2 ** attempt, max_wait)
                if jitter:
                    wait_time = wait_time * random.uniform(0.5, 1.5)

                logger.warning(
                    "Attempt %s/%s failed: %s. Retrying in %.2fs",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
    
    @staticmethod
    async def log_error_with_context(error: Exception, widget_data: Any) -> None:
        context = {
            "widget_count": len(widget_data) if isinstance(widget_data, list) else 1,
            "data_type": type(widget_data).__name__,
            "has_items": hasattr(widget_data, 'items') if widget_data else False
        }
        
        logger.error(f"Error occurred with context: {context}", exc_info=error)
    
    @staticmethod
    async def graceful_degradation(failed_component: str) -> Dict[str, Any]:
        degradation_strategies = {
            "charts": {"fallback": "table", "message": "Chart generation failed, using table format"},
            "citations": {"fallback": "skip", "message": "Citations unavailable for this request"},
            "pdf": {"fallback": "text", "message": "PDF processing failed, using text extraction"},
        }
        
        strategy = degradation_strategies.get(failed_component, {
            "fallback": "skip",
            "message": f"Component {failed_component} unavailable"
        })
        
        logger.info(f"Graceful degradation: {strategy['message']}")
        return strategy

    # ---------------------------------------------------------------------
    # Helpers to surface errors to the OpenBB Workspace
    # ---------------------------------------------------------------------

    @staticmethod
    def build_reasoning_step_sse(
        message: str,
        *,
        event_type: str = "ERROR",
        details: dict[str, Any] | None = None,
    ):
        """Utility to build a StatusUpdateSSE for error surfacing."""

        try:
            from openbb_ai import reasoning_step  # imported lazily to avoid cycles

            return reasoning_step(
                event_type=event_type, message=message, details=details or {}
            )
        except Exception as exc:  # pragma: no cover – protect against missing dep
            logger.error("Failed to build reasoning_step SSE: %s", exc)
            return None

def error_boundary(error_type: type = AgentError, *, fallback: Any | None = None):
    """Decorator providing a soft boundary for *error_type*.

    If *error_type* (or subclass) is raised, it is logged and *fallback* is
    returned instead of propagating the exception. Other exception types still
    bubble up so they can be handled elsewhere.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except error_type as exc:
                logger.error("Error boundary caught %s: %s", error_type.__name__, exc)
                return fallback
            except Exception as exc:
                logger.error("Unexpected error in %s: %s", func.__name__, exc)
                raise

        return wrapper

    return decorator

async def safe_data_access(data: Any, key: str, default: Any = None) -> Any:
    """Safely access data with proper error handling"""
    try:
        if hasattr(data, key):
            return getattr(data, key)
        elif isinstance(data, dict):
            return data.get(key, default)
        else:
            return default
    except Exception:
        return default

async def validate_widget_data(widget_data: Any) -> bool:
    """Validate widget data structure"""
    if not widget_data:
        return False
    
    if isinstance(widget_data, list):
        return all(hasattr(item, 'items') or isinstance(item, dict) for item in widget_data)
    
    return hasattr(widget_data, 'items') or isinstance(widget_data, dict)