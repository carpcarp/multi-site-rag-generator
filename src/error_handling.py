import time
import logging
import asyncio
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import traceback
from anthropic import APIError, RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Types of errors in the RAG system"""
    VALIDATION_ERROR = "validation_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    API_ERROR = "api_error"
    CONNECTION_ERROR = "connection_error"
    TIMEOUT_ERROR = "timeout_error"
    RETRIEVAL_ERROR = "retrieval_error"
    PROCESSING_ERROR = "processing_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorInfo:
    """Error information for structured error handling"""
    error_type: ErrorType
    message: str
    original_exception: Exception
    timestamp: float
    retry_after: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

class RAGException(Exception):
    """Base exception for RAG system"""
    def __init__(self, error_info: ErrorInfo):
        self.error_info = error_info
        super().__init__(error_info.message)

class ValidationError(RAGException):
    """Validation error"""
    pass

class RetrievalError(RAGException):
    """Error during document retrieval"""
    pass

class ProcessingError(RAGException):
    """Error during document processing"""
    pass

class APIError(RAGException):
    """API-related error"""
    pass

class RetryConfig:
    """Configuration for retry logic"""
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True,
                 retryable_errors: List[Type[Exception]] = None):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_errors = retryable_errors or [
            ConnectionError,
            TimeoutError,
            RateLimitError,
            APIConnectionError
        ]

def retry_with_backoff(retry_config: RetryConfig = None):
    """Decorator for retry logic with exponential backoff"""
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except Exception as e:
                    last_exception = e
                    
                    # Check if error is retryable
                    if not any(isinstance(e, error_type) for error_type in retry_config.retryable_errors):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise
                    
                    # Don't retry on last attempt
                    if attempt == retry_config.max_attempts - 1:
                        break
                    
                    # Calculate delay
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    # Add jitter
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                    await asyncio.sleep(delay)
            
            logger.error(f"All {retry_config.max_attempts} attempts failed for {func.__name__}")
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_attempts):
                try:
                    return func(*args, **kwargs)
                        
                except Exception as e:
                    last_exception = e
                    
                    # Check if error is retryable
                    if not any(isinstance(e, error_type) for error_type in retry_config.retryable_errors):
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise
                    
                    # Don't retry on last attempt
                    if attempt == retry_config.max_attempts - 1:
                        break
                    
                    # Calculate delay
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    # Add jitter
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
            
            logger.error(f"All {retry_config.max_attempts} attempts failed for {func.__name__}")
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class ErrorHandler:
    """Centralized error handling for the RAG system"""
    
    def __init__(self):
        self.error_counts: Dict[ErrorType, int] = {}
        self.recent_errors: List[ErrorInfo] = []
        self.max_recent_errors = 100
    
    def handle_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Handle and categorize errors"""
        error_type = self._categorize_error(exception)
        
        # Create error info
        error_info = ErrorInfo(
            error_type=error_type,
            message=str(exception),
            original_exception=exception,
            timestamp=time.time(),
            context=context or {}
        )
        
        # Handle specific error types
        if error_type == ErrorType.RATE_LIMIT_ERROR:
            error_info.retry_after = self._extract_retry_after(exception)
        
        # Log error
        self._log_error(error_info)
        
        # Update statistics
        self._update_error_stats(error_info)
        
        return error_info
    
    def _categorize_error(self, exception: Exception) -> ErrorType:
        """Categorize exception into error type"""
        if isinstance(exception, (ValueError, TypeError)):
            return ErrorType.VALIDATION_ERROR
        elif isinstance(exception, RateLimitError):
            return ErrorType.RATE_LIMIT_ERROR
        elif isinstance(exception, APIConnectionError):
            return ErrorType.CONNECTION_ERROR
        elif isinstance(exception, APIError):
            return ErrorType.API_ERROR
        elif isinstance(exception, TimeoutError):
            return ErrorType.TIMEOUT_ERROR
        elif isinstance(exception, (ConnectionError, OSError)):
            return ErrorType.CONNECTION_ERROR
        elif "retrieval" in str(exception).lower():
            return ErrorType.RETRIEVAL_ERROR
        elif "processing" in str(exception).lower():
            return ErrorType.PROCESSING_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _extract_retry_after(self, exception: Exception) -> Optional[float]:
        """Extract retry-after time from rate limit errors"""
        try:
            if hasattr(exception, 'response') and exception.response:
                retry_after = exception.response.headers.get('Retry-After')
                if retry_after:
                    return float(retry_after)
        except:
            pass
        
        # Default retry after for rate limits
        return 60.0
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level"""
        log_message = f"Error [{error_info.error_type.value}]: {error_info.message}"
        
        if error_info.context:
            log_message += f" | Context: {error_info.context}"
        
        if error_info.error_type in [ErrorType.VALIDATION_ERROR, ErrorType.RATE_LIMIT_ERROR]:
            logger.warning(log_message)
        elif error_info.error_type == ErrorType.UNKNOWN_ERROR:
            logger.error(log_message)
            logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            logger.error(log_message)
    
    def _update_error_stats(self, error_info: ErrorInfo):
        """Update error statistics"""
        # Update counts
        self.error_counts[error_info.error_type] = self.error_counts.get(error_info.error_type, 0) + 1
        
        # Add to recent errors
        self.recent_errors.append(error_info)
        
        # Keep only recent errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        recent_window = 3600  # 1 hour
        current_time = time.time()
        
        recent_errors = [
            error for error in self.recent_errors
            if current_time - error.timestamp < recent_window
        ]
        
        recent_counts = {}
        for error in recent_errors:
            error_type = error.error_type.value
            recent_counts[error_type] = recent_counts.get(error_type, 0) + 1
        
        return {
            'total_errors': {error_type.value: count for error_type, count in self.error_counts.items()},
            'recent_errors': recent_counts,
            'recent_error_count': len(recent_errors),
            'error_rate': len(recent_errors) / (recent_window / 60) if recent_errors else 0  # errors per minute
        }

class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time < self.timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = 'HALF_OPEN'
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
        
        return wrapper
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

class GracefulDegradation:
    """Provide fallback responses when advanced features fail"""
    
    def __init__(self):
        self.fallback_responses = {
            'general': "I apologize, but I'm experiencing technical difficulties. Please try again later or contact support.",
            'validation': "I'm sorry, but your question couldn't be processed. Please rephrase your question and try again.",
            'rate_limit': "You've made too many requests recently. Please wait a moment and try again.",
            'retrieval': "I'm having trouble accessing the knowledge base right now. Please try again in a few moments.",
            'api': "I'm experiencing issues with the AI service. Please try again later."
        }
    
    def get_fallback_response(self, error_type: ErrorType, query: str = None) -> Dict[str, Any]:
        """Get appropriate fallback response"""
        if error_type == ErrorType.VALIDATION_ERROR:
            message = self.fallback_responses['validation']
        elif error_type == ErrorType.RATE_LIMIT_ERROR:
            message = self.fallback_responses['rate_limit']
        elif error_type == ErrorType.RETRIEVAL_ERROR:
            message = self.fallback_responses['retrieval']
        elif error_type in [ErrorType.API_ERROR, ErrorType.CONNECTION_ERROR]:
            message = self.fallback_responses['api']
        else:
            message = self.fallback_responses['general']
        
        return {
            'answer': message,
            'sources': [],
            'query': query or "Unknown query",
            'confidence': 0.0,
            'suggested_categories': [],
            'processing_time': 0.0,
            'fallback': True,
            'error_type': error_type.value
        }
    
    def get_basic_answer(self, query: str) -> Dict[str, Any]:
        """Get basic answer when advanced features fail"""
        # Simple keyword-based responses for common questions
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['contact', 'create contact']):
            answer = "To create a contact in HubSpot, go to Contacts > Contacts and click 'Create contact'. Fill in the contact information and save."
        elif any(word in query_lower for word in ['deal', 'pipeline']):
            answer = "HubSpot deals move through pipeline stages. You can manage deals in Sales > Deals and customize your pipeline in Settings > Objects > Deals."
        elif any(word in query_lower for word in ['automation', 'workflow']):
            answer = "HubSpot workflows automate your marketing and sales processes. Access them in Automation > Workflows to create automated sequences."
        else:
            answer = "I apologize, but I can only provide basic assistance right now. Please visit help.hubspot.com for detailed documentation."
        
        return {
            'answer': answer,
            'sources': [],
            'query': query,
            'confidence': 0.3,
            'suggested_categories': ['General'],
            'processing_time': 0.1,
            'fallback': True,
            'basic_response': True
        }