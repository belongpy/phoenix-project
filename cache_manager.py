"""
Cache Manager Module - Phoenix Project

Implements intelligent caching to reduce API calls and improve performance.
Supports TTL-based expiration and memory management.
"""

import json
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger("phoenix.cache")

class CacheManager:
    """Manages cached data with TTL support to reduce API calls."""
    
    def __init__(self, max_memory_mb: int = 100):
        """
        Initialize the cache manager.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "api_calls_saved": 0
        }
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.lock = threading.Lock()
        
        # TTL configurations (in seconds)
        self.ttl_config = {
            "token_metadata": 86400,      # 24 hours
            "token_price": 300,           # 5 minutes  
            "price_history": 3600,        # 1 hour
            "wallet_analysis": 21600,     # 6 hours
            "wallet_stats": 3600,         # 1 hour
            "transaction": 86400,         # 24 hours
            "popular_tokens": 7200,       # 2 hours for frequently accessed tokens
            "dex_trades": 1800,          # 30 minutes
            "api_usage": 60              # 1 minute
        }
        
        # Track popular items for extended caching
        self.access_counts = {}
        self.popular_threshold = 10
        
        logger.info(f"Cache manager initialized with {max_memory_mb}MB limit")
    
    def _get_cache_key(self, category: str, identifier: str, params: Optional[Dict] = None) -> str:
        """Generate a unique cache key."""
        if params:
            # Sort params for consistent keys
            params_str = json.dumps(params, sort_keys=True)
            identifier = f"{identifier}:{params_str}"
        
        # Use hash for long identifiers
        if len(identifier) > 100:
            identifier = hashlib.md5(identifier.encode()).hexdigest()
        
        return f"{category}:{identifier}"
    
    def _is_popular(self, cache_key: str) -> bool:
        """Check if an item is frequently accessed."""
        return self.access_counts.get(cache_key, 0) >= self.popular_threshold
    
    def _get_ttl(self, category: str, cache_key: str) -> int:
        """Get TTL for a cache entry, with bonus for popular items."""
        base_ttl = self.ttl_config.get(category, 3600)
        
        # Popular items get 50% longer TTL
        if self._is_popular(cache_key):
            return int(base_ttl * 1.5)
        
        return base_ttl
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data in bytes."""
        try:
            return len(json.dumps(data).encode('utf-8'))
        except:
            return 1000  # Default estimate
    
    def _evict_oldest(self, required_space: int) -> None:
        """Evict oldest entries to make space."""
        with self.lock:
            # Sort by expiration time
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1]["expires_at"]
            )
            
            freed_space = 0
            for key, _ in sorted_items:
                if freed_space >= required_space:
                    break
                
                entry_size = self._estimate_size(self.cache[key]["data"])
                del self.cache[key]
                freed_space += entry_size
                self.cache_stats["evictions"] += 1
                
                logger.debug(f"Evicted cache entry: {key}")
    
    def get(self, category: str, identifier: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        Retrieve data from cache if available and not expired.
        
        Args:
            category: Cache category (e.g., "token_metadata", "wallet_analysis")
            identifier: Unique identifier (e.g., token address, wallet address)
            params: Optional parameters that affect the cache key
            
        Returns:
            Cached data if available and valid, None otherwise
        """
        cache_key = self._get_cache_key(category, identifier, params)
        
        with self.lock:
            # Track access for popularity
            self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
            
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check expiration
                if time.time() < entry["expires_at"]:
                    self.cache_stats["hits"] += 1
                    logger.debug(f"Cache hit: {cache_key}")
                    
                    # Update last accessed time
                    entry["last_accessed"] = time.time()
                    
                    return entry["data"]
                else:
                    # Expired entry
                    del self.cache[cache_key]
                    logger.debug(f"Cache expired: {cache_key}")
            
            self.cache_stats["misses"] += 1
            return None
    
    def set(self, category: str, identifier: str, data: Any, 
            params: Optional[Dict] = None, custom_ttl: Optional[int] = None) -> None:
        """
        Store data in cache with TTL.
        
        Args:
            category: Cache category
            identifier: Unique identifier
            data: Data to cache
            params: Optional parameters that affect the cache key
            custom_ttl: Optional custom TTL in seconds
        """
        cache_key = self._get_cache_key(category, identifier, params)
        
        # Estimate data size
        data_size = self._estimate_size(data)
        
        # Check if we need to evict entries
        current_size = sum(self._estimate_size(entry["data"]) for entry in self.cache.values())
        if current_size + data_size > self.max_memory_bytes:
            self._evict_oldest(data_size)
        
        # Determine TTL
        ttl = custom_ttl if custom_ttl is not None else self._get_ttl(category, cache_key)
        
        with self.lock:
            self.cache[cache_key] = {
                "data": data,
                "expires_at": time.time() + ttl,
                "created_at": time.time(),
                "last_accessed": time.time(),
                "category": category,
                "size_bytes": data_size
            }
            
            logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s, Size: {data_size} bytes)")
    
    def invalidate(self, category: Optional[str] = None, identifier: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            category: Optional category to invalidate (invalidates all if None)
            identifier: Optional specific identifier to invalidate
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        with self.lock:
            if category is None:
                # Clear all cache
                count = len(self.cache)
                self.cache.clear()
                self.access_counts.clear()
                logger.info("Cleared entire cache")
            elif identifier is None:
                # Clear category
                keys_to_delete = [k for k in self.cache.keys() if k.startswith(f"{category}:")]
                count = len(keys_to_delete)
                for key in keys_to_delete:
                    del self.cache[key]
                logger.info(f"Cleared {count} entries from category: {category}")
            else:
                # Clear specific entry
                cache_key = self._get_cache_key(category, identifier)
                if cache_key in self.cache:
                    del self.cache[cache_key]
                    count = 1
                    logger.debug(f"Invalidated cache: {cache_key}")
        
        return count
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns number of entries removed."""
        count = 0
        current_time = time.time()
        
        with self.lock:
            expired_keys = [
                k for k, v in self.cache.items() 
                if v["expires_at"] < current_time
            ]
            
            for key in expired_keys:
                del self.cache[key]
                count += 1
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired cache entries")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_entries = len(self.cache)
            total_size = sum(self._estimate_size(entry["data"]) for entry in self.cache.values())
            
            hit_rate = 0
            if self.cache_stats["hits"] + self.cache_stats["misses"] > 0:
                hit_rate = self.cache_stats["hits"] / (self.cache_stats["hits"] + self.cache_stats["misses"]) * 100
            
            # Category breakdown
            category_counts = {}
            for entry in self.cache.values():
                cat = entry["category"]
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Popular items
            popular_items = [
                k for k, v in self.access_counts.items() 
                if v >= self.popular_threshold
            ]
            
            return {
                "total_entries": total_entries,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "max_size_mb": self.max_memory_bytes / 1024 / 1024,
                "usage_percent": round(total_size / self.max_memory_bytes * 100, 2),
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "hit_rate_percent": round(hit_rate, 2),
                "evictions": self.cache_stats["evictions"],
                "api_calls_saved": self.cache_stats["hits"],
                "category_breakdown": category_counts,
                "popular_items_count": len(popular_items),
                "estimated_api_cost_saved": self.cache_stats["hits"] * 0.001  # Rough estimate
            }
    
    def should_refresh(self, category: str, identifier: str, 
                      params: Optional[Dict] = None, force_refresh_percent: float = 0.1) -> bool:
        """
        Determine if cached data should be refreshed even if not expired.
        Useful for critical data that benefits from occasional updates.
        
        Args:
            category: Cache category
            identifier: Unique identifier
            params: Optional parameters
            force_refresh_percent: Probability of forcing refresh (0.0-1.0)
            
        Returns:
            True if should refresh, False otherwise
        """
        import random
        
        # Always refresh if not in cache
        cache_key = self._get_cache_key(category, identifier, params)
        if cache_key not in self.cache:
            return True
        
        # Random refresh for non-critical data
        if category in ["token_metadata", "transaction"]:
            return random.random() < force_refresh_percent
        
        # More aggressive refresh for critical data
        if category in ["token_price", "wallet_stats"]:
            entry = self.cache[cache_key]
            age = time.time() - entry["created_at"]
            ttl = self._get_ttl(category, cache_key)
            
            # Refresh if more than 80% through TTL
            if age > ttl * 0.8:
                return True
        
        return False
    
    def preload_popular_tokens(self, token_addresses: List[str]) -> None:
        """
        Mark tokens as popular for extended caching.
        
        Args:
            token_addresses: List of frequently accessed token addresses
        """
        for address in token_addresses:
            # Mark various cache entries as popular
            for category in ["token_metadata", "token_price", "price_history"]:
                cache_key = self._get_cache_key(category, address)
                self.access_counts[cache_key] = self.popular_threshold
        
        logger.info(f"Marked {len(token_addresses)} tokens as popular for extended caching")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            stats = self.get_stats()
            logger.info(f"Cache manager shutting down. Final stats: {stats}")
        except:
            pass

# Global cache instance
_cache_instance = None

def get_cache_manager(max_memory_mb: int = 100) -> CacheManager:
    """Get or create the global cache manager instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager(max_memory_mb)
    return _cache_instance