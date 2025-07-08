import re
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import bleach
from pydantic import BaseModel, validator, Field

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    """Rate limit information for a user"""
    requests: deque
    max_requests: int
    window: int
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        # Remove old requests outside the window
        while self.requests and self.requests[0] < now - self.window:
            self.requests.popleft()
        
        # Check if under limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        
        return False
    
    def time_until_reset(self) -> float:
        """Time until rate limit resets"""
        if not self.requests:
            return 0.0
        
        return max(0, self.requests[0] + self.window - time.time())

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 100, window: int = 3600):
        self.max_requests = max_requests
        self.window = window
        self.users: Dict[str, RateLimitInfo] = {}
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if user is allowed to make request"""
        if user_id not in self.users:
            self.users[user_id] = RateLimitInfo(
                requests=deque(),
                max_requests=self.max_requests,
                window=self.window
            )
        
        return self.users[user_id].is_allowed()
    
    def get_reset_time(self, user_id: str) -> float:
        """Get time until rate limit resets for user"""
        if user_id not in self.users:
            return 0.0
        
        return self.users[user_id].time_until_reset()
    
    def cleanup_old_users(self):
        """Remove inactive users to prevent memory leaks"""
        now = time.time()
        inactive_users = []
        
        for user_id, info in self.users.items():
            if not info.requests or info.requests[-1] < now - self.window * 2:
                inactive_users.append(user_id)
        
        for user_id in inactive_users:
            del self.users[user_id]

class InputValidator:
    """Validate and sanitize user inputs"""
    
    def __init__(self):
        self.max_query_length = 1000
        self.min_query_length = 3
        self.allowed_tags = []  # No HTML tags allowed
        self.blocked_patterns = [
            r'<script',
            r'javascript:',
            r'onload=',
            r'onclick=',
            r'eval\(',
            r'exec\(',
            r'system\(',
            r'os\.',
            r'import\s+os',
            r'__import__',
            r'subprocess',
        ]
        
    def validate_query(self, query: str) -> str:
        """Validate and sanitize query"""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Check length
        if len(query) < self.min_query_length:
            raise ValueError(f"Query must be at least {self.min_query_length} characters")
        
        if len(query) > self.max_query_length:
            raise ValueError(f"Query must be less than {self.max_query_length} characters")
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValueError("Query contains potentially harmful content")
        
        # Sanitize HTML
        clean_query = bleach.clean(query, tags=self.allowed_tags, strip=True)
        
        return clean_query
    
    def validate_category(self, category: str) -> Optional[str]:
        """Validate category filter"""
        if not category:
            return None
        
        valid_categories = {
            'Contacts', 'Companies', 'Deals', 'Service Hub', 'Reports',
            'Automation', 'Integrations', 'Getting Started', 'General'
        }
        
        category = category.strip()
        if category not in valid_categories:
            raise ValueError(f"Invalid category. Must be one of: {', '.join(valid_categories)}")
        
        return category
    
    def validate_max_context_length(self, length: int) -> int:
        """Validate max context length"""
        if not isinstance(length, int):
            raise ValueError("Max context length must be an integer")
        
        if length < 500:
            raise ValueError("Max context length must be at least 500")
        
        if length > 10000:
            raise ValueError("Max context length cannot exceed 10000")
        
        return length

class AuditLogger:
    """Audit logging for security and compliance"""
    
    def __init__(self, log_file: str = "logs/audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger("audit")
        
        # Create file handler if it doesn't exist
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_query(self, user_id: str, query: str, category: str = None, 
                  response_time: float = None, success: bool = True):
        """Log query details"""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        log_data = {
            'user_id': user_id,
            'query_hash': query_hash,
            'query_length': len(query),
            'category': category,
            'response_time': response_time,
            'success': success,
            'timestamp': time.time()
        }
        
        self.logger.info(f"QUERY: {log_data}")
    
    def log_error(self, user_id: str, error: str, query_hash: str = None):
        """Log error details"""
        log_data = {
            'user_id': user_id,
            'error': error,
            'query_hash': query_hash,
            'timestamp': time.time()
        }
        
        self.logger.error(f"ERROR: {log_data}")
    
    def log_security_event(self, user_id: str, event_type: str, details: Dict[str, Any]):
        """Log security-related events"""
        log_data = {
            'user_id': user_id,
            'event_type': event_type,
            'details': details,
            'timestamp': time.time()
        }
        
        self.logger.warning(f"SECURITY: {log_data}")

class SecurityManager:
    """Centralized security management"""
    
    def __init__(self, rate_limit_requests: int = 100, rate_limit_window: int = 3600):
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        self.validator = InputValidator()
        self.audit_logger = AuditLogger()
        
        # Track failed attempts
        self.failed_attempts: Dict[str, List[float]] = defaultdict(list)
        self.max_failed_attempts = 10
        self.failed_attempts_window = 300  # 5 minutes
    
    def validate_request(self, user_id: str, query: str, category: str = None,
                        max_context_length: int = 4000) -> Dict[str, Any]:
        """Validate entire request"""
        # Check rate limit
        if not self.rate_limiter.is_allowed(user_id):
            reset_time = self.rate_limiter.get_reset_time(user_id)
            self.audit_logger.log_security_event(
                user_id, "RATE_LIMIT_EXCEEDED", 
                {"reset_time": reset_time}
            )
            raise ValueError(f"Rate limit exceeded. Try again in {reset_time:.1f} seconds")
        
        # Check for too many failed attempts
        if self._too_many_failed_attempts(user_id):
            self.audit_logger.log_security_event(
                user_id, "TOO_MANY_FAILED_ATTEMPTS", 
                {"attempts": len(self.failed_attempts[user_id])}
            )
            raise ValueError("Too many failed attempts. Please try again later")
        
        # Validate inputs
        try:
            clean_query = self.validator.validate_query(query)
            clean_category = self.validator.validate_category(category)
            clean_max_length = self.validator.validate_max_context_length(max_context_length)
            
            return {
                'query': clean_query,
                'category': clean_category,
                'max_context_length': clean_max_length
            }
            
        except ValueError as e:
            self._record_failed_attempt(user_id)
            self.audit_logger.log_security_event(
                user_id, "VALIDATION_FAILED", 
                {"error": str(e), "original_query": query[:100]}
            )
            raise
    
    def _too_many_failed_attempts(self, user_id: str) -> bool:
        """Check if user has too many failed attempts"""
        if user_id not in self.failed_attempts:
            return False
        
        now = time.time()
        # Remove old failed attempts
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if now - attempt < self.failed_attempts_window
        ]
        
        return len(self.failed_attempts[user_id]) >= self.max_failed_attempts
    
    def _record_failed_attempt(self, user_id: str):
        """Record a failed attempt"""
        self.failed_attempts[user_id].append(time.time())
    
    def clear_failed_attempts(self, user_id: str):
        """Clear failed attempts for user (after successful request)"""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
    
    def cleanup(self):
        """Cleanup old data to prevent memory leaks"""
        self.rate_limiter.cleanup_old_users()
        
        # Clean up old failed attempts
        now = time.time()
        for user_id in list(self.failed_attempts.keys()):
            self.failed_attempts[user_id] = [
                attempt for attempt in self.failed_attempts[user_id]
                if now - attempt < self.failed_attempts_window
            ]
            
            if not self.failed_attempts[user_id]:
                del self.failed_attempts[user_id]

# Pydantic models for API validation
class QueryRequest(BaseModel):
    """Validated query request model"""
    question: str = Field(..., min_length=3, max_length=1000)
    category_filter: Optional[str] = Field(None, pattern=r'^(Contacts|Companies|Deals|Service Hub|Reports|Automation|Integrations|Getting Started|General)$')
    max_context_length: int = Field(4000, ge=500, le=10000)
    
    @validator('question')
    def validate_question(cls, v):
        """Validate question content"""
        validator = InputValidator()
        return validator.validate_query(v)

class QueryResponse(BaseModel):
    """API response model"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_categories: List[str]
    processing_time: float
    
class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    error_code: str
    timestamp: float
    request_id: Optional[str] = None