import time
import asyncio
import hashlib
import pickle
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from functools import wraps, lru_cache
from collections import OrderedDict
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class TTLCache:
    """Time-to-live cache implementation"""
    
    def __init__(self, maxsize: int = 1000, ttl: float = 3600):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.stats = CacheStats()
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Any:
        """Get item from cache"""
        with self._lock:
            current_time = time.time()
            
            # Check if key exists and is not expired
            if key in self.cache:
                if current_time - self.timestamps[key] <= self.ttl:
                    # Move to end (most recently used)
                    self.cache.move_to_end(key)
                    self.stats.hits += 1
                    return self.cache[key]
                else:
                    # Expired, remove
                    del self.cache[key]
                    del self.timestamps[key]
            
            self.stats.misses += 1
            return None
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        with self._lock:
            current_time = time.time()
            
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
            
            # Evict oldest items if at capacity
            while len(self.cache) >= self.maxsize:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
                self.stats.evictions += 1
            
            # Add new item
            self.cache[key] = value
            self.timestamps[key] = current_time
    
    def clear(self):
        """Clear all items from cache"""
        with self._lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def cleanup_expired(self):
        """Remove expired items"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key]
    
    def size(self) -> int:
        """Get current cache size"""
        with self._lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'size': len(self.cache),
                'maxsize': self.maxsize,
                'hits': self.stats.hits,
                'misses': self.stats.misses,
                'evictions': self.stats.evictions,
                'hit_rate': self.stats.hit_rate
            }

class QueryCache:
    """Specialized cache for RAG queries"""
    
    def __init__(self, maxsize: int = 500, ttl: float = 1800):  # 30 minutes TTL
        self.cache = TTLCache(maxsize, ttl)
        self.similarity_threshold = 0.8  # Cache hit threshold for similar queries
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for caching"""
        return query.lower().strip()
    
    def _get_query_hash(self, query: str, category: str = None, max_context: int = 4000) -> str:
        """Get cache key for query"""
        normalized = self._normalize_query(query)
        cache_key = f"{normalized}|{category or 'none'}|{max_context}"
        return hashlib.md5(cache_key.encode()).hexdigest()
    
    def get(self, query: str, category: str = None, max_context: int = 4000) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        cache_key = self._get_query_hash(query, category, max_context)
        return self.cache.get(cache_key)
    
    def set(self, query: str, response: Dict[str, Any], category: str = None, max_context: int = 4000):
        """Cache response"""
        cache_key = self._get_query_hash(query, category, max_context)
        
        # Store with timestamp for freshness tracking
        cached_response = {
            **response,
            'cached_at': time.time(),
            'original_query': query
        }
        
        self.cache.set(cache_key, cached_response)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache.get_stats()
        stats['type'] = 'query_cache'
        return stats

class EmbeddingCache:
    """Cache for embedding computations"""
    
    def __init__(self, maxsize: int = 10000, ttl: float = 7200):  # 2 hours TTL
        self.cache = TTLCache(maxsize, ttl)
    
    def _get_text_hash(self, text: str) -> str:
        """Get hash for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        cache_key = self._get_text_hash(text)
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            return np.array(cached)
        return None
    
    def set_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        cache_key = self._get_text_hash(text)
        # Convert to list for JSON serialization
        self.cache.set(cache_key, embedding.tolist())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache.get_stats()
        stats['type'] = 'embedding_cache'
        return stats

class BatchProcessor:
    """Batch processing for better performance"""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_embeddings_batch(self, texts: List[str], embedding_model) -> List[np.ndarray]:
        """Process embeddings in batches"""
        if len(texts) <= self.batch_size:
            return [embedding_model.encode(text) for text in texts]
        
        # Process in batches
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = embedding_model.encode(batch)
            results.extend(batch_embeddings)
        
        return results
    
    async def process_queries_async(self, queries: List[str], process_func: Callable) -> List[Any]:
        """Process multiple queries asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Submit tasks to thread pool
        futures = [
            loop.run_in_executor(self.executor, process_func, query)
            for query in queries
        ]
        
        # Wait for all to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)

class ConnectionPool:
    """Connection pool for database connections"""
    
    def __init__(self, max_connections: int = 10, connection_factory: Callable = None):
        self.max_connections = max_connections
        self.connection_factory = connection_factory
        self.pool: List[Any] = []
        self.in_use: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.Lock()
    
    def get_connection(self):
        """Get connection from pool"""
        with self._lock:
            if self.pool:
                conn = self.pool.pop()
                self.in_use.add(conn)
                return conn
            elif len(self.in_use) < self.max_connections and self.connection_factory:
                conn = self.connection_factory()
                self.in_use.add(conn)
                return conn
            else:
                raise Exception("No connections available")
    
    def return_connection(self, conn):
        """Return connection to pool"""
        with self._lock:
            if conn in self.in_use:
                self.pool.append(conn)
    
    def close_all(self):
        """Close all connections"""
        with self._lock:
            for conn in self.pool:
                if hasattr(conn, 'close'):
                    conn.close()
            self.pool.clear()

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'query_latency': [],
            'embedding_time': [],
            'retrieval_time': [],
            'generation_time': []
        }
        self.max_samples = 1000
        self._lock = threading.Lock()
    
    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric"""
        with self._lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            self.metrics[metric_name].append(value)
            
            # Keep only recent samples
            if len(self.metrics[metric_name]) > self.max_samples:
                self.metrics[metric_name] = self.metrics[metric_name][-self.max_samples:]
    
    def get_stats(self, metric_name: str = None) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            if metric_name:
                if metric_name not in self.metrics:
                    return {}
                
                values = self.metrics[metric_name]
                if not values:
                    return {'count': 0}
                
                return {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                # Return stats for all metrics
                stats = {}
                for name in self.metrics:
                    stats[name] = self.get_stats(name)
                return stats
    
    def clear_metrics(self):
        """Clear all metrics"""
        with self._lock:
            for metric_list in self.metrics.values():
                metric_list.clear()

def timed_operation(metric_name: str, monitor: PerformanceMonitor = None):
    """Decorator to time operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if monitor:
                    monitor.record_metric(metric_name, duration)
                logger.debug(f"{func.__name__} took {duration:.3f}s")
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if monitor:
                    monitor.record_metric(metric_name, duration)
                logger.debug(f"{func.__name__} took {duration:.3f}s")
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator

class OptimizedVectorStore:
    """Performance-optimized vector store wrapper"""
    
    def __init__(self, base_store, embedding_cache: EmbeddingCache = None):
        self.base_store = base_store
        self.embedding_cache = embedding_cache or EmbeddingCache()
        self.batch_processor = BatchProcessor()
        self.performance_monitor = PerformanceMonitor()
    
    @timed_operation('embedding_time')
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        # Check cache first
        cached = self.embedding_cache.get_embedding(text)
        if cached is not None:
            return cached
        
        # Generate embedding
        embedding = self.base_store.embedding_model.encode(text)
        
        # Cache result
        self.embedding_cache.set_embedding(text, embedding)
        
        return embedding
    
    @timed_operation('retrieval_time')
    def search_optimized(self, query: str, n_results: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Optimized search with performance monitoring"""
        return self.base_store.search(query, n_results, **kwargs)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'performance_metrics': self.performance_monitor.get_stats(),
            'embedding_cache': self.embedding_cache.get_stats()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.batch_processor.cleanup()

class MemoryProfiler:
    """Memory usage profiler"""
    
    def __init__(self):
        self.snapshots = []
        self.max_snapshots = 100
    
    def take_snapshot(self, label: str = None):
        """Take memory snapshot"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            snapshot = {
                'timestamp': time.time(),
                'label': label or f"snapshot_{len(self.snapshots)}",
                'rss': memory_info.rss,  # Resident Set Size
                'vms': memory_info.vms,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
            
            self.snapshots.append(snapshot)
            
            # Keep only recent snapshots
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
            
            return snapshot
            
        except ImportError:
            logger.warning("psutil not installed. Memory profiling unavailable.")
            return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        if not self.snapshots:
            return {}
        
        rss_values = [s['rss'] for s in self.snapshots]
        vms_values = [s['vms'] for s in self.snapshots]
        
        return {
            'snapshots_count': len(self.snapshots),
            'rss_current': rss_values[-1],
            'rss_peak': max(rss_values),
            'rss_average': np.mean(rss_values),
            'vms_current': vms_values[-1],
            'vms_peak': max(vms_values),
            'memory_growth': rss_values[-1] - rss_values[0] if len(rss_values) > 1 else 0
        }