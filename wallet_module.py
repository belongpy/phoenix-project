"""
Wallet Analysis Module - Phoenix Project (OPTIMIZED WITH CACHE & ENHANCED METRICS)

MAJOR UPDATES:
- Redefined gem finder criteria (5x+ instead of 2x+)
- Implemented tiered analysis (Quick/Standard/Deep)
- Added intelligent caching to reduce API calls by ~40%
- Smart transaction sampling (wins/losses/recent/random)
- New advanced metrics: Sharpe ratio, Entry Precision, Diamond Hands, Portfolio Concentration
- Skip dust trades (<$100 volume)
- Batch processing and API budget management
"""

import csv
import os
import logging
import numpy as np
import requests
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
from cache_manager import get_cache_manager

logger = logging.getLogger("phoenix.wallet")

class CieloFinanceAPIError(Exception):
    """Custom exception for Cielo Finance API errors."""
    pass

class RateLimiter:
    """Simple rate limiter for RPC calls."""
    def __init__(self, calls_per_second: float = 10.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = threading.Lock()
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)
            self.last_call = time.time()

class WalletAnalyzer:
    """Enhanced wallet analyzer with caching, tiered analysis, and advanced metrics."""
    
    # Analysis tier configurations
    ANALYSIS_TIERS = {
        "quick": {
            "name": "Quick Scan",
            "api_calls": 2,
            "recent_trades": 0,
            "description": "Basic Cielo stats only"
        },
        "standard": {
            "name": "Standard Analysis", 
            "api_calls": 5,
            "recent_trades": 5,
            "description": "Last 5 trades with performance"
        },
        "deep": {
            "name": "Deep Analysis",
            "api_calls": 20,
            "recent_trades": 20,
            "description": "Smart sampling with full metrics"
        }
    }
    
    # Dust trade threshold
    MIN_TRADE_VOLUME = 100  # $100 minimum
    
    def __init__(self, cielo_api: Any, birdeye_api: Any = None, helius_api: Any = None, 
                 rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """
        Initialize the wallet analyzer with caching support.
        
        Args:
            cielo_api: Cielo Finance API client (REQUIRED)
            birdeye_api: Birdeye API client (optional, for token metadata)
            helius_api: Helius API client (optional, for pump.fun tokens)
            rpc_url: Solana RPC endpoint URL (P9 or other provider)
        """
        if not cielo_api:
            raise ValueError("Cielo Finance API is REQUIRED for wallet analysis")
        
        self.cielo_api = cielo_api
        self.birdeye_api = birdeye_api
        self.helius_api = helius_api
        self.rpc_url = rpc_url
        
        # Initialize cache manager
        self.cache = get_cache_manager(max_memory_mb=200)
        
        # Verify Cielo Finance API connectivity
        if not self._verify_cielo_api_connection():
            raise CieloFinanceAPIError("Cannot connect to Cielo Finance API")
        
        # Track entry times for tokens to detect correlated wallets
        self.token_entries = {}
        
        # RPC cache for avoiding duplicate calls
        self._rpc_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 300  # 5 minutes TTL
        
        # Rate limiter for RPC calls
        self._rate_limiter = RateLimiter(calls_per_second=5.0)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Track RPC errors
        self._rpc_error_count = 0
        self._last_rpc_error_time = 0
        
        # Track data quality for composite scoring
        self.data_quality_weights = {
            "full_analysis": 1.0,      # Birdeye data available
            "helius_analysis": 0.85,   # Helius/pump.fun data
            "basic_analysis": 0.5      # Only P&L data
        }
        
        # API budget tracking
        self.api_budget = {
            "birdeye": {"daily_limit": 1000, "used": 0},
            "cielo": {"daily_limit": 5000, "used": 0},
            "helius": {"daily_limit": 2000, "used": 0}
        }
    
    def _verify_cielo_api_connection(self) -> bool:
        """Verify that the Cielo Finance API is accessible."""
        try:
            if not self.cielo_api:
                return False
            health_check = self.cielo_api.health_check()
            if health_check:
                logger.info("✅ Cielo Finance API connection verified")
                return True
            else:
                logger.error("❌ Cielo Finance API health check failed")
                return False
        except Exception as e:
            logger.error(f"❌ Cielo Finance API connection failed: {str(e)}")
            return False
    
    def _track_api_usage(self, api_name: str, calls: int = 1) -> bool:
        """Track API usage against daily limits."""
        if api_name in self.api_budget:
            self.api_budget[api_name]["used"] += calls
            used = self.api_budget[api_name]["used"]
            limit = self.api_budget[api_name]["daily_limit"]
            
            if used >= limit * 0.9:
                logger.warning(f"⚠️ {api_name} API usage at {used}/{limit} ({used/limit*100:.1f}%)")
            
            return used < limit
        return True
    
    def _get_cache_key(self, method: str, params: List[Any]) -> str:
        """Generate cache key for RPC calls."""
        return f"{method}:{json.dumps(params, sort_keys=True)}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache if not expired."""
        with self._cache_lock:
            if cache_key in self._rpc_cache:
                cached_data, timestamp = self._rpc_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    return cached_data
                else:
                    del self._rpc_cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store result in cache."""
        with self._cache_lock:
            self._rpc_cache[cache_key] = (data, time.time())
    
    def _make_rpc_call(self, method: str, params: List[Any], retry_count: int = 3) -> Dict[str, Any]:
        """Make direct RPC call to Solana node with caching and rate limiting."""
        cache_key = self._get_cache_key(method, params)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {method}")
            return cached_result
        
        self._rate_limiter.wait()
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        backoff_base = 2
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    self.rpc_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 429:
                    self._rpc_error_count += 1
                    self._last_rpc_error_time = time.time()
                    
                    backoff_time = backoff_base ** attempt
                    max_backoff = 60
                    wait_time = min(backoff_time, max_backoff)
                    
                    logger.warning(f"RPC rate limit hit (429). Waiting {wait_time}s before retry {attempt + 1}/{retry_count}")
                    time.sleep(wait_time)
                    
                    if self._rpc_error_count > 10:
                        logger.warning("Too many RPC errors. Slowing down request rate.")
                        self._rate_limiter.calls_per_second = max(1.0, self._rate_limiter.calls_per_second * 0.5)
                        self._rate_limiter.min_interval = 1.0 / self._rate_limiter.calls_per_second
                    
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                if "result" in result:
                    self._set_cache(cache_key, result)
                    if self._rpc_error_count > 0:
                        self._rpc_error_count = max(0, self._rpc_error_count - 1)
                    return result
                else:
                    return {"error": result.get("error", "Unknown RPC error")}
                    
            except requests.exceptions.Timeout:
                logger.error(f"RPC timeout for {method} (attempt {attempt + 1}/{retry_count})")
                if attempt < retry_count - 1:
                    time.sleep(backoff_base ** attempt)
                else:
                    return {"error": "RPC timeout"}
                    
            except requests.RequestException as e:
                logger.error(f"RPC request failed for {method}: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(backoff_base ** attempt)
                else:
                    return {"error": str(e)}
                    
            except Exception as e:
                logger.error(f"Unexpected RPC error for {method}: {str(e)}")
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    def analyze_wallet_tiered(self, wallet_address: str, tier: str = "standard", 
                            days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze wallet with tiered approach to optimize API usage.
        
        Args:
            wallet_address (str): Wallet address
            tier (str): Analysis tier - "quick", "standard", or "deep"
            days_back (int): Number of days to analyze
            
        Returns:
            Dict[str, Any]: Wallet analysis results
        """
        logger.info(f"🔍 Analyzing wallet {wallet_address} (Tier: {tier})")
        
        # Check cache first
        cache_key = f"{wallet_address}:{tier}:{days_back}"
        cached_result = self.cache.get("wallet_analysis", cache_key)
        if cached_result and not self.cache.should_refresh("wallet_analysis", cache_key):
            logger.info(f"📦 Using cached analysis for {wallet_address}")
            self.cache.cache_stats["api_calls_saved"] += self.ANALYSIS_TIERS[tier]["api_calls"]
            return cached_result
        
        tier_config = self.ANALYSIS_TIERS.get(tier, self.ANALYSIS_TIERS["standard"])
        
        # Call appropriate analysis method based on tier
        if tier == "quick":
            result = self._analyze_wallet_quick(wallet_address)
        elif tier == "deep":
            result = self._analyze_wallet_deep(wallet_address, days_back)
        else:
            result = self._analyze_wallet_standard(wallet_address, days_back)
        
        # Cache the result
        if result.get("success"):
            self.cache.set("wallet_analysis", cache_key, result)
        
        return result
    
    def _analyze_wallet_quick(self, wallet_address: str) -> Dict[str, Any]:
        """Quick analysis using only Cielo Finance aggregated stats."""
        logger.info(f"⚡ Quick analysis for {wallet_address}")
        
        try:
            # Get cached Cielo stats if available
            cached_stats = self.cache.get("wallet_stats", wallet_address)
            if cached_stats:
                cielo_stats = cached_stats
            else:
                cielo_stats = self.cielo_api.get_wallet_trading_stats(wallet_address)
                self._track_api_usage("cielo", 1)
                if cielo_stats and cielo_stats.get("success", True):
                    self.cache.set("wallet_stats", wallet_address, cielo_stats)
            
            if not cielo_stats or not cielo_stats.get("success", True):
                return self._get_empty_analysis_result(wallet_address, "quick")
            
            # Extract metrics
            stats_data = cielo_stats.get("data", {})
            aggregated_metrics = self._extract_aggregated_metrics_from_cielo(stats_data)
            
            # Basic calculations only
            combined_metrics = aggregated_metrics.copy()
            combined_metrics["wallet_address"] = wallet_address
            combined_metrics["avg_hold_time_minutes"] = round(combined_metrics.get("avg_hold_time", 0) * 60, 2)
            
            # Calculate basic composite score
            composite_score = self._calculate_composite_score(combined_metrics)
            combined_metrics["composite_score"] = composite_score
            
            # Determine wallet type
            wallet_type = self._determine_wallet_type_enhanced(combined_metrics)
            
            # Generate strategy
            strategy = self._generate_strategy(wallet_type, combined_metrics)
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_tier": "quick",
                "wallet_type": wallet_type,
                "composite_score": composite_score,
                "metrics": combined_metrics,
                "strategy": strategy,
                "trades": [],
                "advanced_metrics": {},
                "api_calls_used": 1
            }
            
        except Exception as e:
            logger.error(f"Error in quick analysis: {str(e)}")
            return self._get_empty_analysis_result(wallet_address, "quick", str(e))
    
    def _analyze_wallet_standard(self, wallet_address: str, days_back: int) -> Dict[str, Any]:
        """Standard analysis with last 5 trades."""
        logger.info(f"📊 Standard analysis for {wallet_address}")
        
        # Use the existing hybrid analysis but limit to 5 trades
        result = self.analyze_wallet_hybrid(wallet_address, days_back, max_trades=5)
        result["analysis_tier"] = "standard"
        result["api_calls_used"] = 5
        return result
    
    def _analyze_wallet_deep(self, wallet_address: str, days_back: int) -> Dict[str, Any]:
        """Deep analysis with smart sampling and full metrics."""
        logger.info(f"🔬 Deep analysis for {wallet_address}")
        
        try:
            # Step 1: Get aggregated stats from Cielo Finance
            cielo_stats = self.cielo_api.get_wallet_trading_stats(wallet_address)
            self._track_api_usage("cielo", 1)
            
            if not cielo_stats or not cielo_stats.get("success", True):
                return self._get_empty_analysis_result(wallet_address, "deep")
            
            stats_data = cielo_stats.get("data", {})
            aggregated_metrics = self._extract_aggregated_metrics_from_cielo(stats_data)
            
            # Step 2: Smart sampling of transactions
            logger.info(f"🎯 Applying smart sampling for transaction analysis...")
            sampled_swaps = self._smart_sample_transactions(wallet_address, days_back)
            
            # Step 3: Analyze sampled transactions
            analyzed_trades = []
            data_quality_scores = []
            
            for i, swap in enumerate(sampled_swaps):
                if i > 0:
                    time.sleep(0.3)  # Reduced delay for cached items
                
                # Check if trade volume is above dust threshold
                if swap.get("sol_amount", 0) * 150 < self.MIN_TRADE_VOLUME:  # Assuming SOL ~$150
                    logger.debug(f"Skipping dust trade: ${swap.get('sol_amount', 0) * 150:.2f}")
                    continue
                
                token_analysis, data_quality = self._analyze_token_with_fallback(
                    swap['token_mint'],
                    swap['buy_timestamp'],
                    swap.get('sell_timestamp'),
                    swap
                )
                
                if token_analysis['success']:
                    analyzed_trades.append({
                        **swap,
                        **token_analysis,
                        'data_quality': data_quality
                    })
                    data_quality_scores.append(self.data_quality_weights[data_quality])
            
            # Step 4: Calculate advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(analyzed_trades, aggregated_metrics)
            
            # Step 5: Combine all metrics
            combined_metrics = self._combine_metrics(aggregated_metrics, analyzed_trades)
            combined_metrics.update(advanced_metrics)
            combined_metrics["wallet_address"] = wallet_address
            combined_metrics["avg_hold_time_minutes"] = round(combined_metrics.get("avg_hold_time_hours", 0) * 60, 2)
            
            # Step 6: Calculate weighted composite score
            avg_data_quality = np.mean(data_quality_scores) if data_quality_scores else 0.5
            base_composite_score = self._calculate_composite_score(combined_metrics)
            weighted_composite_score = round(base_composite_score * (0.7 + 0.3 * avg_data_quality), 1)
            combined_metrics["composite_score"] = weighted_composite_score
            combined_metrics["base_composite_score"] = base_composite_score
            combined_metrics["data_quality_factor"] = round(avg_data_quality, 2)
            
            # Step 7: Enhanced wallet type determination
            wallet_type = self._determine_wallet_type_enhanced(combined_metrics)
            
            # Step 8: Generate strategy with advanced insights
            strategy = self._generate_strategy_enhanced(wallet_type, combined_metrics, advanced_metrics)
            
            # Step 9: Entry/exit behavior analysis
            entry_exit_analysis = self._analyze_entry_exit_behavior(analyzed_trades)
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_tier": "deep",
                "wallet_type": wallet_type,
                "composite_score": weighted_composite_score,
                "metrics": combined_metrics,
                "advanced_metrics": advanced_metrics,
                "strategy": strategy,
                "trades": analyzed_trades,
                "entry_exit_analysis": entry_exit_analysis,
                "api_calls_used": len(sampled_swaps) + 1,
                "sampling_method": "smart",
                "trades_analyzed": len(analyzed_trades),
                "cache_stats": self.cache.get_stats()
            }
            
        except Exception as e:
            logger.error(f"Error in deep analysis: {str(e)}")
            return self._get_empty_analysis_result(wallet_address, "deep", str(e))
    
    def _smart_sample_transactions(self, wallet_address: str, days_back: int) -> List[Dict[str, Any]]:
        """
        Smart sampling: 5 biggest wins, 5 biggest losses, 5 most recent, 5 random.
        
        Returns:
            List of sampled transactions
        """
        try:
            # Get all recent transactions
            all_swaps = self._get_recent_token_swaps_rpc(wallet_address, limit=100)
            
            if not all_swaps:
                return []
            
            # Categorize trades
            completed_trades = []
            for swap in all_swaps:
                if swap.get("type") == "sell" or (swap.get("buy_timestamp") and swap.get("sell_timestamp")):
                    # Calculate simple ROI estimate
                    if swap.get("type") == "sell" and swap.get("sol_amount", 0) > 0:
                        swap["estimated_roi"] = swap.get("sol_amount", 0) * 100  # Rough estimate
                    else:
                        swap["estimated_roi"] = 0
                    completed_trades.append(swap)
            
            # Smart sampling
            sampled = []
            
            # 1. Top 5 wins
            wins = [t for t in completed_trades if t.get("estimated_roi", 0) > 0]
            wins.sort(key=lambda x: x.get("estimated_roi", 0), reverse=True)
            sampled.extend(wins[:5])
            
            # 2. Top 5 losses  
            losses = [t for t in completed_trades if t.get("estimated_roi", 0) < 0]
            losses.sort(key=lambda x: x.get("estimated_roi", 0))
            sampled.extend(losses[:5])
            
            # 3. Most recent 5
            recent = sorted(completed_trades, key=lambda x: x.get("buy_timestamp", 0), reverse=True)
            for trade in recent[:5]:
                if trade not in sampled:
                    sampled.append(trade)
            
            # 4. Random 5
            remaining = [t for t in completed_trades if t not in sampled]
            if remaining:
                random_sample = random.sample(remaining, min(5, len(remaining)))
                sampled.extend(random_sample)
            
            logger.info(f"📊 Smart sampling: {len(sampled)} trades selected from {len(all_swaps)} total")
            return sampled[:20]  # Cap at 20 for deep analysis
            
        except Exception as e:
            logger.error(f"Error in smart sampling: {str(e)}")
            return []
    
    def _calculate_advanced_metrics(self, analyzed_trades: List[Dict[str, Any]], 
                                  aggregated_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate advanced metrics: Sharpe ratio, entry precision, diamond hands, etc."""
        
        metrics = {
            "sharpe_ratio": 0,
            "max_consecutive_losses": 0,
            "recovery_time_hours": 0,
            "entry_precision_score": 0,
            "diamond_hands_score": 0,
            "portfolio_concentration": 0,
            "risk_adjusted_return": 0,
            "conviction_score": 0
        }
        
        if not analyzed_trades:
            return metrics
        
        try:
            # 1. Sharpe Ratio (risk-adjusted returns)
            returns = [t.get("roi_percent", 0) for t in analyzed_trades]
            if len(returns) > 1:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    metrics["sharpe_ratio"] = round((avg_return - 5) / std_return, 2)  # 5% risk-free rate
            
            # 2. Maximum consecutive losses
            consecutive_losses = 0
            max_consecutive = 0
            for trade in sorted(analyzed_trades, key=lambda x: x.get("buy_timestamp", 0)):
                if trade.get("roi_percent", 0) < 0:
                    consecutive_losses += 1
                    max_consecutive = max(max_consecutive, consecutive_losses)
                else:
                    consecutive_losses = 0
            metrics["max_consecutive_losses"] = max_consecutive
            
            # 3. Entry precision score (% entries near local bottom)
            entry_scores = []
            for trade in analyzed_trades:
                if trade.get("entry_timing") == "EXCELLENT":
                    entry_scores.append(100)
                elif trade.get("entry_timing") == "GOOD":
                    entry_scores.append(75)
                elif trade.get("entry_timing") == "AVERAGE":
                    entry_scores.append(50)
                else:
                    entry_scores.append(25)
            metrics["entry_precision_score"] = round(np.mean(entry_scores), 1) if entry_scores else 0
            
            # 4. Diamond hands score (holding through drawdowns)
            diamond_trades = 0
            total_eligible = 0
            for trade in analyzed_trades:
                if trade.get("max_roi_percent", 0) > 100:  # Had potential
                    total_eligible += 1
                    if trade.get("roi_percent", 0) > 50:  # Still profitable
                        pullback = trade.get("max_pullback_percent", 0)
                        if pullback > 30:  # Held through 30%+ pullback
                            diamond_trades += 1
            
            if total_eligible > 0:
                metrics["diamond_hands_score"] = round(diamond_trades / total_eligible * 100, 1)
            
            # 5. Portfolio concentration (using aggregated data)
            roi_dist = aggregated_metrics.get("roi_distribution", {})
            total_trades = aggregated_metrics.get("total_trades", 0)
            if total_trades > 0:
                top_performers = (roi_dist.get("10x_plus", 0) + roi_dist.get("5x_to_10x", 0))
                metrics["portfolio_concentration"] = round(top_performers / total_trades * 100, 1)
            
            # 6. Risk-adjusted return
            if metrics["sharpe_ratio"] > 0 and aggregated_metrics.get("avg_roi", 0) > 0:
                metrics["risk_adjusted_return"] = round(
                    aggregated_metrics["avg_roi"] * (1 + metrics["sharpe_ratio"] / 10), 2
                )
            
            # 7. Conviction score (based on hold time and position sizing)
            if aggregated_metrics.get("avg_hold_time", 0) > 24:  # Holds > 1 day
                if aggregated_metrics.get("avg_trade_size", 0) > 1000:  # Significant positions
                    metrics["conviction_score"] = 100
                else:
                    metrics["conviction_score"] = 70
            else:
                metrics["conviction_score"] = 40
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")
        
        return metrics
    
    def _determine_wallet_type_enhanced(self, metrics: Dict[str, Any]) -> str:
        """Enhanced wallet type determination with new gem finder criteria."""
        if not metrics:
            return "unknown"
            
        total_trades = metrics.get("total_trades", 0)
        if total_trades < 1:
            return "unknown"
        
        try:
            win_rate = metrics.get("win_rate", 0)
            median_roi = metrics.get("median_roi", 0)
            avg_hold_time_hours = metrics.get("avg_hold_time_hours", 0)
            roi_distribution = metrics.get("roi_distribution", {})
            max_roi = metrics.get("max_roi", 0)
            profit_factor = metrics.get("profit_factor", 0)
            net_profit = metrics.get("net_profit_usd", 0)
            
            # Advanced metrics
            sharpe_ratio = metrics.get("sharpe_ratio", 0)
            diamond_hands = metrics.get("diamond_hands_score", 0)
            entry_precision = metrics.get("entry_precision_score", 0)
            
            # NEW GEM FINDER CRITERIA (5x+ focus)
            five_x_plus = (roi_distribution.get("10x_plus", 0) + 
                          roi_distribution.get("5x_to_10x", 0))
            five_x_ratio = five_x_plus / total_trades if total_trades > 0 else 0
            
            # Count individual 5x+ trades
            five_x_count = five_x_plus
            
            # Gem finder: 5x+ focused
            if (five_x_ratio >= 0.15 and  # 15% or more trades are 5x+
                max_roi >= 500 and  # Has achieved at least 5x
                five_x_count >= 2 and  # At least 2 trades with 5x+ returns
                metrics.get("composite_score", 0) >= 70):  # High overall score
                return "gem_hunter"  # Changed from gem_finder to gem_hunter
            
            # Smart trader: High Sharpe ratio and precision
            if sharpe_ratio > 1.5 and entry_precision > 70:
                return "smart_trader"
            
            # Diamond hands: Holds through volatility
            if diamond_hands > 60 and avg_hold_time_hours > 48:
                return "diamond_hands"
            
            # Flipper: Quick trades
            if avg_hold_time_hours < 24 and win_rate > 40:
                return "flipper"
            
            # Consistent: Steady performance
            if win_rate >= 45 and median_roi > 0 and profit_factor > 1.2:
                return "consistent"
            
            # Mixed: Some success
            if win_rate >= 30 or max_roi >= 200 or profit_factor >= 1.0 or net_profit > 0:
                return "mixed"
            
            # Underperformer
            if total_trades >= 5:
                return "underperformer"
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error determining wallet type: {str(e)}")
            return "unknown"
    
    def _generate_strategy_enhanced(self, wallet_type: str, metrics: Dict[str, Any], 
                                  advanced_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced trading strategy based on wallet type and advanced metrics."""
        try:
            composite_score = metrics.get("composite_score", 0)
            sharpe_ratio = advanced_metrics.get("sharpe_ratio", 0)
            entry_precision = advanced_metrics.get("entry_precision_score", 0)
            diamond_hands = advanced_metrics.get("diamond_hands_score", 0)
            
            if wallet_type == "gem_hunter":
                strategy = {
                    "recommendation": "FOLLOW_AGGRESSIVELY",
                    "entry_type": "IMMEDIATE",
                    "position_size": "MEDIUM",  # Not too large due to volatility
                    "take_profit_1": 200,  # 2x
                    "take_profit_2": 500,  # 5x
                    "take_profit_3": 1000,  # 10x
                    "stop_loss": -40,  # Wider stop for gems
                    "notes": f"💎 Gem Hunter (Score: {composite_score}/100). Consistently finds 5x+ tokens.",
                    "confidence": "VERY_HIGH" if composite_score >= 85 else "HIGH",
                    "special_instructions": "Scale in on dips. Hold core position for 10x potential."
                }
            
            elif wallet_type == "smart_trader":
                strategy = {
                    "recommendation": "FOLLOW_ENTRIES",
                    "entry_type": "WAIT_FOR_SIGNAL",
                    "position_size": "LARGE",
                    "take_profit_1": 50,
                    "take_profit_2": 100,
                    "take_profit_3": 200,
                    "stop_loss": -20,
                    "notes": f"🧠 Smart Trader (Sharpe: {sharpe_ratio:.2f}). Excellent risk-adjusted returns.",
                    "confidence": "VERY_HIGH",
                    "special_instructions": f"Entry precision: {entry_precision}%. Wait for their exact entry signals."
                }
            
            elif wallet_type == "diamond_hands":
                strategy = {
                    "recommendation": "FOLLOW_AND_HOLD",
                    "entry_type": "IMMEDIATE",
                    "position_size": "MEDIUM",
                    "take_profit_1": 100,
                    "take_profit_2": 300,
                    "take_profit_3": 500,
                    "stop_loss": -50,  # Wide stop, they hold through dips
                    "notes": f"💎👐 Diamond Hands (Score: {diamond_hands}%). Holds through volatility.",
                    "confidence": "HIGH",
                    "special_instructions": "Don't panic sell. This trader holds for big gains."
                }
            
            elif wallet_type == "flipper":
                strategy = {
                    "recommendation": "QUICK_SCALP",
                    "entry_type": "IMMEDIATE",
                    "position_size": "MEDIUM",
                    "take_profit_1": 20,
                    "take_profit_2": 40,
                    "take_profit_3": 60,
                    "stop_loss": -15,
                    "notes": f"⚡ Quick Flipper (Score: {composite_score}/100). Fast in/out trades.",
                    "confidence": "MEDIUM",
                    "special_instructions": "Set alerts. Be ready to exit quickly."
                }
            
            elif wallet_type == "consistent":
                strategy = {
                    "recommendation": "STEADY_FOLLOW",
                    "entry_type": "IMMEDIATE",
                    "position_size": "LARGE",
                    "take_profit_1": 30,
                    "take_profit_2": 60,
                    "take_profit_3": 100,
                    "stop_loss": -20,
                    "notes": f"📊 Consistent Performer (Score: {composite_score}/100). Reliable profits.",
                    "confidence": "HIGH",
                    "special_instructions": "Safe for larger positions. Consistent winner."
                }
            
            else:
                # Default cautious strategy
                strategy = {
                    "recommendation": "CAUTIOUS",
                    "entry_type": "WAIT_FOR_CONFIRMATION",
                    "position_size": "SMALL",
                    "take_profit_1": 25,
                    "take_profit_2": 50,
                    "take_profit_3": 100,
                    "stop_loss": -20,
                    "notes": f"⚠️ Uncertain (Score: {composite_score}/100). Needs more data.",
                    "confidence": "LOW",
                    "special_instructions": "Small positions only. Wait for confirmation."
                }
            
            # Add risk management based on advanced metrics
            if advanced_metrics.get("max_consecutive_losses", 0) > 3:
                strategy["risk_warning"] = "⚠️ Has losing streaks. Use strict position sizing."
            
            if sharpe_ratio < 0.5:
                strategy["volatility_warning"] = "📊 High volatility. Expect large swings."
            
            return strategy
                
        except Exception as e:
            logger.error(f"Error generating enhanced strategy: {str(e)}")
            return self._get_default_strategy()
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        """Get default cautious strategy."""
        return {
            "recommendation": "CAUTIOUS",
            "entry_type": "WAIT_FOR_CONFIRMATION",
            "position_size": "SMALL",
            "take_profit_1": 20,
            "take_profit_2": 40,
            "take_profit_3": 80,
            "stop_loss": -20,
            "notes": "Use caution. Limited data available.",
            "confidence": "LOW"
        }
    
    def _get_empty_analysis_result(self, wallet_address: str, tier: str, 
                                 error: Optional[str] = None) -> Dict[str, Any]:
        """Get empty analysis result structure."""
        empty_metrics = self._get_empty_metrics()
        empty_metrics["wallet_address"] = wallet_address
        
        return {
            "success": False,
            "error": error or "Analysis failed",
            "wallet_address": wallet_address,
            "analysis_tier": tier,
            "wallet_type": "unknown",
            "composite_score": 0,
            "metrics": empty_metrics,
            "advanced_metrics": self._get_empty_advanced_metrics(),
            "strategy": self._get_default_strategy(),
            "trades": [],
            "api_calls_used": 0
        }
    
    def _get_empty_advanced_metrics(self) -> Dict[str, Any]:
        """Get empty advanced metrics structure."""
        return {
            "sharpe_ratio": 0,
            "max_consecutive_losses": 0,
            "recovery_time_hours": 0,
            "entry_precision_score": 0,
            "diamond_hands_score": 0,
            "portfolio_concentration": 0,
            "risk_adjusted_return": 0,
            "conviction_score": 0
        }
    
    # Keep existing methods but update analyze_wallet_hybrid to use caching
    def analyze_wallet_hybrid(self, wallet_address: str, days_back: int = 30, 
                            max_trades: int = 5) -> Dict[str, Any]:
        """
        UPDATED hybrid wallet analysis with caching and trade limit.
        """
        logger.info(f"🔍 Analyzing wallet {wallet_address} (hybrid approach with caching)")
        
        try:
            # Check cache first
            cache_key = f"{wallet_address}:hybrid:{days_back}:{max_trades}"
            cached_result = self.cache.get("wallet_analysis", cache_key)
            if cached_result:
                logger.info(f"📦 Using cached hybrid analysis for {wallet_address}")
                return cached_result
            
            # Step 1: Get aggregated stats from Cielo Finance (with caching)
            logger.info(f"📊 Fetching Cielo Finance aggregated stats...")
            
            cached_stats = self.cache.get("wallet_stats", wallet_address)
            if cached_stats:
                cielo_stats = cached_stats
            else:
                cielo_stats = self.cielo_api.get_wallet_trading_stats(wallet_address)
                self._track_api_usage("cielo", 1)
                if cielo_stats and cielo_stats.get("success", True):
                    self.cache.set("wallet_stats", wallet_address, cielo_stats)
            
            if cielo_stats and cielo_stats.get("success", True) and "data" in cielo_stats:
                data = cielo_stats.get("data", {})
                if isinstance(data, dict):
                    logger.debug(f"Cielo response data keys: {list(data.keys())[:20]}")
            
            if not cielo_stats or not cielo_stats.get("success", True):
                logger.warning(f"❌ No Cielo Finance data available for {wallet_address}")
                logger.info("Attempting RPC-only analysis as fallback...")
                aggregated_metrics = self._get_empty_cielo_metrics()
            else:
                stats_data = cielo_stats.get("data", {})
                aggregated_metrics = self._extract_aggregated_metrics_from_cielo(stats_data)
            
            # Step 2: Get recent token trades via RPC (limited by max_trades)
            logger.info(f"🪙 Analyzing last {max_trades} tokens...")
            recent_swaps = self._get_recent_token_swaps_rpc(wallet_address, limit=max_trades)
            
            # Step 3: Analyze token performance with tiered approach
            analyzed_trades = []
            data_quality_scores = []
            
            if recent_swaps:
                for i, swap in enumerate(recent_swaps[:max_trades]):
                    if i > 0:
                        time.sleep(0.3)  # Reduced delay
                    
                    # Skip dust trades
                    if swap.get("sol_amount", 0) * 150 < self.MIN_TRADE_VOLUME:
                        continue
                    
                    # Try analysis with caching
                    token_analysis, data_quality = self._analyze_token_with_fallback_cached(
                        swap['token_mint'],
                        swap['buy_timestamp'],
                        swap.get('sell_timestamp'),
                        swap
                    )
                    
                    if token_analysis['success']:
                        analyzed_trades.append({
                            **swap,
                            **token_analysis,
                            'data_quality': data_quality
                        })
                        data_quality_scores.append(self.data_quality_weights[data_quality])
            
            # Step 4: Combine metrics
            combined_metrics = self._combine_metrics(aggregated_metrics, analyzed_trades)
            combined_metrics["wallet_address"] = wallet_address
            combined_metrics["avg_hold_time_minutes"] = round(combined_metrics.get("avg_hold_time_hours", 0) * 60, 2)
            
            # Step 5: Calculate weighted composite score
            avg_data_quality = np.mean(data_quality_scores) if data_quality_scores else 0.5
            base_composite_score = self._calculate_composite_score(combined_metrics)
            weighted_composite_score = round(base_composite_score * (0.7 + 0.3 * avg_data_quality), 1)
            combined_metrics["composite_score"] = weighted_composite_score
            combined_metrics["base_composite_score"] = base_composite_score
            combined_metrics["data_quality_factor"] = round(avg_data_quality, 2)
            
            # Step 6: Determine wallet type (enhanced)
            wallet_type = self._determine_wallet_type_enhanced(combined_metrics)
            
            # Step 7: Generate strategy
            strategy = self._generate_strategy(wallet_type, combined_metrics)
            
            # Analyze entry/exit behavior
            entry_exit_analysis = self._analyze_entry_exit_behavior(analyzed_trades)
            
            logger.debug(f"Final analysis for {wallet_address}:")
            logger.debug(f"  - wallet_type: {wallet_type}")
            logger.debug(f"  - weighted_composite_score: {weighted_composite_score}")
            logger.debug(f"  - data_quality_factor: {avg_data_quality}")
            
            result = {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_period_days": "ALL_TIME (Cielo) + Recent (RPC)",
                "wallet_type": wallet_type,
                "composite_score": weighted_composite_score,
                "metrics": combined_metrics,
                "strategy": strategy,
                "trades": analyzed_trades,
                "entry_exit_analysis": entry_exit_analysis,
                "api_source": "Cielo Finance + RPC + Birdeye/Helius",
                "cielo_data": aggregated_metrics,
                "recent_trades_analyzed": len(analyzed_trades),
                "data_quality_breakdown": {
                    "full_analysis": sum(1 for t in analyzed_trades if t.get('data_quality') == 'full_analysis'),
                    "helius_analysis": sum(1 for t in analyzed_trades if t.get('data_quality') == 'helius_analysis'),
                    "basic_analysis": sum(1 for t in analyzed_trades if t.get('data_quality') == 'basic_analysis')
                }
            }
            
            # Cache the result
            self.cache.set("wallet_analysis", cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in hybrid wallet analysis: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            empty_metrics = self._get_empty_metrics()
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "wallet_address": wallet_address,
                "error_type": "UNEXPECTED_ERROR",
                "wallet_type": "unknown",
                "composite_score": 0,
                "metrics": empty_metrics,
                "strategy": {
                    "recommendation": "CAUTIOUS",
                    "entry_type": "WAIT_FOR_CONFIRMATION",
                    "position_size": "SMALL",
                    "take_profit_1": 20,
                    "take_profit_2": 40,
                    "take_profit_3": 80,
                    "stop_loss": -30,
                    "notes": "Analysis failed. Use extreme caution.",
                    "competition_level": "UNKNOWN"
                }
            }
    
    def _analyze_token_with_fallback_cached(self, token_mint: str, buy_timestamp: Optional[int], 
                                          sell_timestamp: Optional[int] = None, 
                                          swap_data: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """
        Analyze token performance with caching support.
        """
        # Check cache first
        cache_key = f"{token_mint}:{buy_timestamp}:{sell_timestamp}"
        cached_result = self.cache.get("token_analysis", cache_key)
        if cached_result:
            logger.debug(f"📦 Using cached token analysis for {token_mint}")
            return cached_result
        
        # Perform analysis
        result = self._analyze_token_with_fallback(token_mint, buy_timestamp, sell_timestamp, swap_data)
        
        # Cache the result
        self.cache.set("token_analysis", cache_key, result, custom_ttl=7200)  # 2 hour TTL
        
        return result
    
    def batch_analyze_wallets_tiered(self, wallet_addresses: List[str], 
                                   days_back: int = 30,
                                   default_tier: str = "standard",
                                   top_performers_deep: bool = True) -> Dict[str, Any]:
        """
        Batch analyze wallets with intelligent tier selection.
        
        Args:
            wallet_addresses: List of wallet addresses
            days_back: Days to analyze
            default_tier: Default analysis tier
            top_performers_deep: Run deep analysis on top performers
        """
        logger.info(f"Batch analyzing {len(wallet_addresses)} wallets with tiered approach")
        
        if not wallet_addresses:
            return {
                "success": False,
                "error": "No wallet addresses provided",
                "error_type": "NO_INPUT"
            }
        
        try:
            wallet_analyses = []
            failed_analyses = []
            
            # Phase 1: Quick scan all wallets
            logger.info("📊 Phase 1: Quick scan of all wallets...")
            quick_results = []
            
            for i, wallet_address in enumerate(wallet_addresses, 1):
                logger.info(f"Quick scan {i}/{len(wallet_addresses)}: {wallet_address}")
                
                try:
                    quick_analysis = self._analyze_wallet_quick(wallet_address)
                    if quick_analysis.get("success"):
                        quick_results.append({
                            "wallet": wallet_address,
                            "score": quick_analysis.get("composite_score", 0),
                            "analysis": quick_analysis
                        })
                    else:
                        failed_analyses.append({
                            "wallet_address": wallet_address,
                            "error": quick_analysis.get("error", "Quick scan failed"),
                            "error_type": "QUICK_SCAN_FAILED"
                        })
                    
                    if i < len(wallet_addresses):
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error in quick scan for {wallet_address}: {str(e)}")
                    failed_analyses.append({
                        "wallet_address": wallet_address,
                        "error": str(e),
                        "error_type": "QUICK_SCAN_ERROR"
                    })
            
            # Sort by score
            quick_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Phase 2: Deeper analysis for promising wallets
            logger.info(f"📊 Phase 2: Analyzing top performers...")
            
            # Determine which wallets get deeper analysis
            deep_analysis_count = min(10, len(quick_results) // 5)  # Top 20% or max 10
            
            for i, result in enumerate(quick_results):
                wallet = result["wallet"]
                quick_score = result["score"]
                
                # Determine analysis tier
                if i < deep_analysis_count and top_performers_deep and quick_score > 40:
                    tier = "deep"
                elif quick_score > 30:
                    tier = "standard"
                else:
                    # Use quick analysis for low scorers
                    wallet_analyses.append(result["analysis"])
                    continue
                
                logger.info(f"Running {tier} analysis for {wallet} (quick score: {quick_score})")
                
                try:
                    analysis = self.analyze_wallet_tiered(wallet, tier, days_back)
                    
                    if analysis.get("success"):
                        wallet_analyses.append(analysis)
                    else:
                        # Fall back to quick analysis
                        wallet_analyses.append(result["analysis"])
                    
                    time.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"Error in {tier} analysis for {wallet}: {str(e)}")
                    # Use quick analysis as fallback
                    wallet_analyses.append(result["analysis"])
            
            # Categorize results
            gem_hunters = [a for a in wallet_analyses if a.get("wallet_type") == "gem_hunter"]
            smart_traders = [a for a in wallet_analyses if a.get("wallet_type") == "smart_trader"]
            diamond_hands = [a for a in wallet_analyses if a.get("wallet_type") == "diamond_hands"]
            consistent = [a for a in wallet_analyses if a.get("wallet_type") == "consistent"]
            flippers = [a for a in wallet_analyses if a.get("wallet_type") == "flipper"]
            mixed = [a for a in wallet_analyses if a.get("wallet_type") == "mixed"]
            underperformers = [a for a in wallet_analyses if a.get("wallet_type") == "underperformer"]
            unknown = [a for a in wallet_analyses if a.get("wallet_type") == "unknown"]
            
            # Sort each category by composite score
            for category in [gem_hunters, smart_traders, diamond_hands, consistent, flippers, mixed, underperformers, unknown]:
                category.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
            
            # Get cache statistics
            cache_stats = self.cache.get_stats()
            
            return {
                "success": True,
                "total_wallets": len(wallet_addresses),
                "analyzed_wallets": len(wallet_analyses),
                "failed_wallets": len(failed_analyses),
                "filtered_wallets": len(wallet_analyses),
                "gem_hunters": gem_hunters,
                "smart_traders": smart_traders,
                "diamond_hands": diamond_hands,
                "consistent": consistent,
                "flippers": flippers,
                "mixed": mixed,
                "underperformers": underperformers,
                "unknown": unknown,
                "failed_analyses": failed_analyses,
                "api_usage": self.api_budget,
                "cache_stats": cache_stats,
                "analysis_breakdown": {
                    "quick": sum(1 for a in wallet_analyses if a.get("analysis_tier") == "quick"),
                    "standard": sum(1 for a in wallet_analyses if a.get("analysis_tier") == "standard"),
                    "deep": sum(1 for a in wallet_analyses if a.get("analysis_tier") == "deep")
                }
            }
            
        except Exception as e:
            logger.error(f"Error during batch analysis: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": "UNEXPECTED_ERROR"
            }
    
    # Keep all existing methods but add batch_analyze_wallets as wrapper
    def batch_analyze_wallets(self, wallet_addresses: List[str], 
                            days_back: int = 30,
                            min_winrate: float = 30.0,
                            use_hybrid: bool = True) -> Dict[str, Any]:
        """Wrapper for backward compatibility."""
        return self.batch_analyze_wallets_tiered(
            wallet_addresses, 
            days_back,
            default_tier="standard" if use_hybrid else "quick"
        )
    
    # Keep all other existing methods unchanged...
    # [All the remaining methods from the original file remain the same]
    
    def _get_signatures_for_address(self, address: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transaction signatures for an address using direct RPC with caching."""
        response = self._make_rpc_call(
            "getSignaturesForAddress",
            [address, {"limit": limit}]
        )
        
        if "result" in response:
            return response["result"]
        else:
            logger.error(f"Error getting signatures: {response.get('error', 'Unknown error')}")
            return []
    
    def _get_transaction(self, signature: str) -> Dict[str, Any]:
        """Get transaction details by signature using direct RPC with caching."""
        # Check cache first
        cached_tx = self.cache.get("transaction", signature)
        if cached_tx:
            return cached_tx
        
        response = self._make_rpc_call(
            "getTransaction",
            [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0}]
        )
        
        if "result" in response and response["result"]:
            # Cache the transaction
            self.cache.set("transaction", signature, response["result"])
            return response["result"]
        else:
            return {}
    
    def _batch_get_transactions(self, signatures: List[str]) -> List[Dict[str, Any]]:
        """Get multiple transactions in a more efficient way."""
        transactions = []
        
        for i, signature in enumerate(signatures):
            if i > 0 and i % 10 == 0:
                logger.debug(f"Processed {i}/{len(signatures)} transactions, pausing...")
                time.sleep(2)
            
            tx = self._get_transaction(signature)
            if tx:
                transactions.append(tx)
                
        return transactions
    
    def _analyze_token_with_fallback(self, token_mint: str, buy_timestamp: Optional[int], 
                                   sell_timestamp: Optional[int] = None, 
                                   swap_data: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], str]:
        """
        Analyze token performance with fallback approach:
        1. Try Birdeye API first
        2. If pump.fun token or Birdeye fails, try Helius
        3. If both fail, return basic P&L analysis
        
        Returns:
            Tuple[Dict[str, Any], str]: (analysis_result, data_quality_tier)
        """
        # Check if we should skip based on API budget
        if not self._track_api_usage("birdeye", 0):  # Check without incrementing
            logger.warning("Birdeye API budget exhausted, using basic analysis")
            return self._basic_token_analysis(token_mint, buy_timestamp, sell_timestamp, swap_data), "basic_analysis"
        
        # Tier 1: Try Birdeye first (best data quality)
        if self.birdeye_api and not token_mint.endswith("pump"):
            birdeye_result = self._analyze_token_performance(
                token_mint, buy_timestamp, sell_timestamp
            )
            if birdeye_result.get("success") and not birdeye_result.get("is_pump_token"):
                self._track_api_usage("birdeye", 1)
                return birdeye_result, "full_analysis"
        
        # Tier 2: Try Helius for pump.fun tokens or if Birdeye failed
        if self.helius_api and token_mint.endswith("pump"):
            logger.info(f"Using Helius API for pump.fun token {token_mint}")
            helius_result = self._analyze_token_with_helius(
                token_mint, buy_timestamp, sell_timestamp, swap_data
            )
            if helius_result.get("success"):
                self._track_api_usage("helius", 1)
                return helius_result, "helius_analysis"
        
        # Tier 3: Basic P&L analysis (fallback)
        logger.info(f"Using basic P&L analysis for token {token_mint}")
        basic_result = self._basic_token_analysis(
            token_mint, buy_timestamp, sell_timestamp, swap_data
        )
        return basic_result, "basic_analysis"
    
    def _analyze_token_with_helius(self, token_mint: str, buy_timestamp: Optional[int],
                                 sell_timestamp: Optional[int] = None,
                                 swap_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze token using Helius API (especially for pump.fun tokens)."""
        try:
            if not self.helius_api:
                return {"success": False, "error": "Helius API not available"}
            
            logger.info(f"Analyzing pump.fun token {token_mint} with Helius")
            
            # Get token metadata first
            metadata = self.helius_api.get_token_metadata([token_mint])
            token_info = {}
            if metadata.get("success") and metadata.get("data"):
                token_info = metadata["data"][0] if metadata["data"] else {}
            
            # Use price from swap data if available
            buy_price = 0.0001  # Default for new pump.fun tokens
            current_price = 0.0001
            
            if swap_data and swap_data.get("estimated_price", 0) > 0:
                if swap_data.get("type") == "buy":
                    buy_price = swap_data["estimated_price"]
                    current_price = buy_price  # Assume same if no sell
                else:
                    current_price = swap_data["estimated_price"]
                    buy_price = current_price * 0.8  # Estimate buy price lower
            
            # Calculate ROI
            roi_percent = ((current_price / buy_price) - 1) * 100 if buy_price > 0 else 0
            
            # Entry/exit timing based on hold time and ROI
            if sell_timestamp and buy_timestamp:
                hold_time_hours = (sell_timestamp - buy_timestamp) / 3600
                
                # Entry timing
                if roi_percent > 100:
                    entry_timing = "EXCELLENT"
                elif roi_percent > 50:
                    entry_timing = "GOOD"
                elif roi_percent > 0:
                    entry_timing = "AVERAGE"
                else:
                    entry_timing = "POOR"
                
                # Exit timing
                if hold_time_hours < 1 and roi_percent > 20:
                    exit_timing = "QUICK_PROFIT"
                elif hold_time_hours < 24 and roi_percent > 50:
                    exit_timing = "GOOD"
                elif roi_percent < -20:
                    exit_timing = "LOSS_EXIT"
                else:
                    exit_timing = "STANDARD"
            else:
                entry_timing = "UNKNOWN"
                exit_timing = "HOLDING" if not sell_timestamp else "UNKNOWN"
            
            return {
                "success": True,
                "token_address": token_mint,
                "initial_price": buy_price,
                "current_price": current_price,
                "roi_percent": roi_percent,
                "current_roi_percent": roi_percent,
                "max_roi_percent": max(roi_percent, 20),  # Assume at least 20% peak for pump tokens
                "entry_timing": entry_timing,
                "exit_timing": exit_timing,
                "data_source": "helius",
                "is_pump_token": True,
                "token_metadata": token_info,
                "has_price_data": swap_data is not None and swap_data.get("estimated_price", 0) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in Helius token analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "is_pump_token": True
            }
    
    def _basic_token_analysis(self, token_mint: str, buy_timestamp: Optional[int],
                            sell_timestamp: Optional[int] = None,
                            swap_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Basic token analysis using only available transaction data."""
        roi_percent = 0
        
        # Use swap data if available
        if swap_data and swap_data.get("estimated_price", 0) > 0:
            if swap_data.get("type") == "sell" and swap_data.get("sol_amount", 0) > 0:
                # We have sell data, estimate ROI if possible
                roi_percent = 50  # Conservative estimate
        
        return {
            "success": True,
            "token_address": token_mint,
            "roi_percent": roi_percent,
            "current_roi_percent": roi_percent,
            "max_roi_percent": roi_percent,
            "entry_timing": "UNKNOWN",
            "exit_timing": "UNKNOWN",
            "data_source": "basic",
            "note": "Limited analysis - price history unavailable"
        }
    
    def _extract_aggregated_metrics_from_cielo(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract aggregated metrics from Cielo Finance response."""
        try:
            if not isinstance(stats_data, dict):
                logger.warning(f"Unexpected Cielo data format: {type(stats_data)}")
                return self._get_empty_cielo_metrics()
            
            logger.debug(f"Cielo Finance response structure: {list(stats_data.keys())[:10]}")
            
            metrics = {
                "total_trades": stats_data.get("swaps_count", 0),
                "win_rate": stats_data.get("winrate", 0),
                "total_pnl_usd": stats_data.get("pnl", 0),
                "avg_trade_size": stats_data.get("average_buy_amount_usd", 0),
                "total_volume": (stats_data.get("total_buy_amount_usd", 0) + 
                               stats_data.get("total_sell_amount_usd", 0)),
                "best_trade": 0,
                "worst_trade": 0,
                "avg_hold_time": stats_data.get("average_holding_time_sec", 0) / 3600 if stats_data.get("average_holding_time_sec", 0) else 0,
                "tokens_traded": stats_data.get("buy_count", 0),
                "total_invested": stats_data.get("total_buy_amount_usd", 0),
                "total_realized": stats_data.get("total_sell_amount_usd", 0),
                "buy_count": stats_data.get("buy_count", 0),
                "sell_count": stats_data.get("sell_count", 0),
                "consecutive_trading_days": stats_data.get("consecutive_trading_days", 0),
                "roi_distribution": stats_data.get("roi_distribution", {}),
                "holding_distribution": stats_data.get("holding_distribution", {})
            }
            
            roi_dist = stats_data.get("roi_distribution", {})
            if roi_dist.get("roi_above_500", 0) > 0:
                metrics["best_trade"] = 500
            elif roi_dist.get("roi_200_to_500", 0) > 0:
                metrics["best_trade"] = 200
            elif roi_dist.get("roi_0_to_200", 0) > 0:
                metrics["best_trade"] = 100
            else:
                metrics["best_trade"] = 50
            
            if roi_dist.get("roi_below_neg50", 0) > 0:
                metrics["worst_trade"] = -50
            elif roi_dist.get("roi_neg50_to_0", 0) > 0:
                metrics["worst_trade"] = -25
            else:
                metrics["worst_trade"] = 0
            
            logger.debug(f"Extracted Cielo metrics: trades={metrics['total_trades']}, "
                        f"win_rate={metrics['win_rate']}, pnl={metrics['total_pnl_usd']}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting Cielo metrics: {str(e)}")
            return self._get_empty_cielo_metrics()
    
    def _get_empty_cielo_metrics(self) -> Dict[str, Any]:
        """Return empty Cielo metrics structure."""
        return {
            "total_trades": 0,
            "win_rate": 0,
            "total_pnl_usd": 0,
            "avg_trade_size": 0,
            "total_volume": 0,
            "best_trade": 0,
            "worst_trade": 0,
            "avg_hold_time": 0,
            "tokens_traded": 0,
            "total_invested": 0,
            "total_realized": 0,
            "roi_distribution": {},
            "holding_distribution": {}
        }
    
    def _get_recent_token_swaps_rpc(self, wallet_address: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent token swaps using RPC calls with better rate limiting."""
        try:
            logger.info(f"Fetching last {limit} token swaps for {wallet_address}")
            
            signatures = self._get_signatures_for_address(wallet_address, limit=50)
            
            if not signatures:
                logger.warning(f"No transactions found for {wallet_address}")
                return []
            
            swaps = []
            sig_infos = []
            
            for sig_info in signatures:
                if len(swaps) >= limit:
                    break
                    
                signature = sig_info.get("signature")
                if signature:
                    sig_infos.append(sig_info)
            
            batch_size = 10
            for i in range(0, len(sig_infos), batch_size):
                batch = sig_infos[i:i + batch_size]
                batch_signatures = [s.get("signature") for s in batch if s.get("signature")]
                
                for signature in batch_signatures:
                    if len(swaps) >= limit:
                        break
                        
                    tx_details = self._get_transaction(signature)
                    if tx_details:
                        swap_info = self._extract_token_swaps_from_transaction(tx_details, wallet_address)
                        if swap_info:
                            swaps.extend(swap_info)
                
                if i + batch_size < len(sig_infos):
                    time.sleep(1)
            
            logger.info(f"Found {len(swaps)} swaps from {len(signatures)} signatures")
            return swaps[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent swaps: {str(e)}")
            return []
    
    def _extract_token_swaps_from_transaction(self, tx_details: Dict[str, Any], 
                                            wallet_address: str) -> List[Dict[str, Any]]:
        """Extract token swap information from transaction with proper timestamp handling."""
        swaps = []
        
        try:
            if not tx_details or "meta" not in tx_details:
                return []
            
            meta = tx_details["meta"]
            
            block_time = tx_details.get("blockTime")
            if not block_time or block_time == 0:
                # Use current time minus 1 day as fallback
                block_time = int(datetime.now().timestamp() - 86400)
                logger.debug(f"Using fallback timestamp: {block_time}")
            
            # Sanity check - if timestamp is in the future, adjust it
            current_time = int(datetime.now().timestamp())
            if block_time > current_time:
                logger.warning(f"Future timestamp detected: {block_time}, adjusting to current time")
                block_time = current_time
            
            pre_balances = meta.get("preTokenBalances", [])
            post_balances = meta.get("postTokenBalances", [])
            
            # Also track SOL balance changes for price calculation
            pre_sol = meta.get("preBalances", [])
            post_sol = meta.get("postBalances", [])
            
            token_changes = {}
            
            for balance in pre_balances:
                owner = balance.get("owner")
                if owner == wallet_address:
                    mint = balance.get("mint")
                    amount = int(balance.get("uiTokenAmount", {}).get("amount", 0))
                    decimals = balance.get("uiTokenAmount", {}).get("decimals", 9)
                    if mint:
                        token_changes[mint] = {
                            "pre": amount, 
                            "post": 0, 
                            "decimals": decimals
                        }
            
            for balance in post_balances:
                owner = balance.get("owner")
                if owner == wallet_address:
                    mint = balance.get("mint")
                    amount = int(balance.get("uiTokenAmount", {}).get("amount", 0))
                    decimals = balance.get("uiTokenAmount", {}).get("decimals", 9)
                    if mint:
                        if mint in token_changes:
                            token_changes[mint]["post"] = amount
                        else:
                            token_changes[mint] = {
                                "pre": 0, 
                                "post": amount, 
                                "decimals": decimals
                            }
            
            # Find wallet's account index for SOL balance
            accounts = tx_details.get("transaction", {}).get("message", {}).get("accountKeys", [])
            wallet_index = -1
            for i, account in enumerate(accounts):
                if account == wallet_address:
                    wallet_index = i
                    break
            
            # Calculate SOL change
            sol_change = 0
            if wallet_index >= 0 and wallet_index < len(pre_sol) and wallet_index < len(post_sol):
                sol_change = (post_sol[wallet_index] - pre_sol[wallet_index]) / 1e9  # Convert lamports to SOL
            
            # Process token changes
            for mint, changes in token_changes.items():
                diff = changes["post"] - changes["pre"]
                decimals = changes["decimals"]
                
                if diff != 0:
                    # Convert to UI amount
                    ui_amount = abs(diff) / (10 ** decimals)
                    
                    # Estimate price if this is a swap
                    estimated_price = 0
                    if diff > 0 and sol_change < 0:
                        # Bought tokens with SOL
                        estimated_price = abs(sol_change) / ui_amount
                    elif diff < 0 and sol_change > 0:
                        # Sold tokens for SOL
                        estimated_price = sol_change / ui_amount
                    
                    swap_data = {
                        "token_mint": mint,
                        "type": "buy" if diff > 0 else "sell",
                        "amount": ui_amount,
                        "raw_amount": abs(diff),
                        "decimals": decimals,
                        "buy_timestamp": block_time if diff > 0 else None,
                        "sell_timestamp": block_time if diff < 0 else None,
                        "signature": tx_details.get("transaction", {}).get("signatures", [""])[0],
                        "sol_amount": abs(sol_change),
                        "estimated_price": estimated_price
                    }
                    
                    swaps.append(swap_data)
            
        except Exception as e:
            logger.error(f"Error extracting swaps from transaction: {str(e)}")
        
        return swaps
    
    def _analyze_token_performance(self, token_mint: str, buy_timestamp: Optional[int], 
                                 sell_timestamp: Optional[int] = None) -> Dict[str, Any]:
        """Analyze token performance with proper timestamp handling and resolution format."""
        if not self.birdeye_api:
            return {"success": False, "error": "Birdeye API not available"}
        
        # Check cache first
        cache_params = {"buy": buy_timestamp, "sell": sell_timestamp}
        cached_result = self.cache.get("token_performance", token_mint, cache_params)
        if cached_result:
            return cached_result
        
        try:
            if token_mint.endswith("pump"):
                logger.info(f"Token {token_mint} is a pump.fun token, limited analysis available")
                return {
                    "success": False,
                    "error": "Limited data for pump.fun tokens",
                    "is_pump_token": True,
                    "entry_timing": "UNKNOWN",
                    "exit_timing": "UNKNOWN"
                }
            
            current_time = int(datetime.now().timestamp())
            
            if not buy_timestamp or buy_timestamp == 0:
                buy_timestamp = current_time - (7 * 24 * 60 * 60)
                logger.debug(f"Using default buy timestamp (7 days ago): {buy_timestamp}")
            
            if buy_timestamp > current_time:
                buy_timestamp = current_time - (24 * 60 * 60)
                logger.debug(f"Adjusted future timestamp to 1 day ago: {buy_timestamp}")
            
            end_time = sell_timestamp if sell_timestamp and sell_timestamp > 0 else current_time
            
            token_info = self.birdeye_api.get_token_info(token_mint)
            
            resolution = "1H"
            
            time_diff = end_time - buy_timestamp
            if time_diff < 3600:
                resolution = "5m"
            elif time_diff < 86400:
                resolution = "15m"
            elif time_diff < 259200:
                resolution = "1H"
            elif time_diff < 604800:
                resolution = "4H"
            else:
                resolution = "1D"
            
            logger.debug(f"Analyzing token {token_mint} from {buy_timestamp} to {end_time} with resolution {resolution}")
            
            history_response = self.birdeye_api.get_token_price_history(
                token_mint,
                buy_timestamp,
                end_time,
                resolution
            )
            
            if not history_response.get("success"):
                logger.warning(f"Failed to get price history for {token_mint}")
                return {
                    "success": False,
                    "error": "Failed to get price history",
                    "entry_timing": "UNKNOWN",
                    "exit_timing": "UNKNOWN"
                }
            
            performance = self.birdeye_api.calculate_token_performance(
                token_mint,
                datetime.fromtimestamp(buy_timestamp)
            )
            
            if performance.get("success"):
                performance["entry_timing"] = self._analyze_entry_timing(performance)
                performance["exit_timing"] = self._analyze_exit_timing(performance, sell_timestamp is not None)
                
                # Cache the result
                self.cache.set("token_performance", token_mint, performance, cache_params)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error analyzing token performance: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "entry_timing": "UNKNOWN",
                "exit_timing": "UNKNOWN"
            }
    
    def _analyze_entry_timing(self, performance: Dict[str, Any]) -> str:
        """Analyze if entry timing was good based on subsequent performance."""
        try:
            max_roi = performance.get("max_roi_percent", 0)
            roi_to_max_time = performance.get("time_to_max_roi_hours", 0)
            
            if max_roi < 20:
                return "POOR"
            elif roi_to_max_time < 1:
                return "EXCELLENT"
            elif roi_to_max_time < 6:
                return "GOOD"
            else:
                return "LATE"
        except:
            return "UNKNOWN"
    
    def _analyze_exit_timing(self, performance: Dict[str, Any], did_sell: bool) -> str:
        """Analyze if exit timing was good or if holder missed opportunities."""
        try:
            if not did_sell:
                return "HOLDING"
            
            current_roi = performance.get("current_roi_percent", 0)
            max_roi = performance.get("max_roi_percent", 0)
            
            if current_roi <= 0:
                return "LOSS_EXIT"
            
            roi_capture_ratio = current_roi / max_roi if max_roi > 0 else 0
            
            if roi_capture_ratio >= 0.8:
                return "EXCELLENT"
            elif roi_capture_ratio >= 0.5:
                return "GOOD"
            elif roi_capture_ratio >= 0.2:
                return "EARLY"
            else:
                return "VERY_EARLY"
                
        except:
            return "UNKNOWN"
    
    def _analyze_entry_exit_behavior(self, analyzed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze wallet's entry and exit behavior patterns."""
        if not analyzed_trades:
            return {
                "pattern": "INSUFFICIENT_DATA",
                "entry_quality": "UNKNOWN",
                "exit_quality": "UNKNOWN",
                "missed_gains_percent": 0,
                "early_exit_rate": 0,
                "recommendations": "Need more trade data for analysis"
            }
        
        valid_trades = [t for t in analyzed_trades if t.get("success", False)]
        
        if not valid_trades:
            pump_token_count = sum(1 for t in analyzed_trades if t.get("is_pump_token", False))
            helius_count = sum(1 for t in analyzed_trades if t.get("data_source") == "helius")
            basic_count = sum(1 for t in analyzed_trades if t.get("data_source") == "basic")
            
            if pump_token_count > 0 and helius_count == 0:
                return {
                    "pattern": "PUMP_TOKEN_TRADER",
                    "entry_quality": "UNKNOWN",
                    "exit_quality": "UNKNOWN",
                    "missed_gains_percent": 0,
                    "early_exit_rate": 0,
                    "recommendations": f"Trades pump.fun tokens - enable Helius for better analysis"
                }
            elif basic_count > 0:
                return {
                    "pattern": "LIMITED_DATA",
                    "entry_quality": "UNKNOWN",
                    "exit_quality": "UNKNOWN",
                    "missed_gains_percent": 0,
                    "early_exit_rate": 0,
                    "recommendations": "Limited price data available"
                }
        
        entry_timings = []
        exit_timings = []
        missed_gains = []
        
        for trade in valid_trades:
            if "entry_timing" in trade and trade["entry_timing"] != "UNKNOWN":
                entry_timings.append(trade["entry_timing"])
            if "exit_timing" in trade and trade["exit_timing"] != "UNKNOWN":
                exit_timings.append(trade["exit_timing"])
            
            if trade.get("current_roi_percent", 0) > 0 and trade.get("max_roi_percent", 0) > 0:
                missed = trade["max_roi_percent"] - trade["current_roi_percent"]
                if missed > 0:
                    missed_gains.append(missed)
        
        good_entries = sum(1 for e in entry_timings if e in ["EXCELLENT", "GOOD"])
        good_exits = sum(1 for e in exit_timings if e in ["EXCELLENT", "GOOD"])
        early_exits = sum(1 for e in exit_timings if e in ["EARLY", "VERY_EARLY"])
        
        avg_missed_gains = np.mean(missed_gains) if missed_gains else 0
        
        entry_quality = "GOOD" if len(entry_timings) > 0 and good_entries / len(entry_timings) >= 0.5 else "POOR"
        exit_quality = "GOOD" if len(exit_timings) > 0 and good_exits / len(exit_timings) >= 0.5 else "POOR"
        
        if early_exits > len(exit_timings) * 0.5:
            pattern = "EARLY_SELLER"
            recommendations = "Consider holding positions longer to capture more gains"
        elif avg_missed_gains > 100:
            pattern = "MISSING_RUNNERS"
            recommendations = "Implement trailing stops to capture more upside on winners"
        else:
            pattern = "BALANCED"
            recommendations = "Good balance between risk and reward"
        
        return {
            "pattern": pattern,
            "entry_quality": entry_quality,
            "exit_quality": exit_quality,
            "missed_gains_percent": round(avg_missed_gains, 2),
            "early_exit_rate": round(early_exits / len(exit_timings) * 100, 2) if exit_timings else 0,
            "recommendations": recommendations,
            "trades_analyzed": len(valid_trades),
            "pump_tokens_skipped": sum(1 for t in analyzed_trades if t.get("is_pump_token", False)),
            "data_sources": {
                "birdeye": sum(1 for t in analyzed_trades if t.get("data_source") == "birdeye"),
                "helius": sum(1 for t in analyzed_trades if t.get("data_source") == "helius"),
                "basic": sum(1 for t in analyzed_trades if t.get("data_source") == "basic")
            }
        }
    
    def _combine_metrics(self, cielo_metrics: Dict[str, Any], 
                        analyzed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine Cielo aggregated metrics with recent trade analysis."""
        try:
            combined = cielo_metrics.copy() if cielo_metrics else self._get_empty_cielo_metrics()
            
            logger.debug(f"Combining metrics - Cielo data: total_trades={combined.get('total_trades', 0)}, "
                        f"win_rate={combined.get('win_rate', 0)}, pnl={combined.get('total_pnl_usd', 0)}")
            
            total_trades = combined.get("total_trades", 0)
            win_rate = combined.get("win_rate", 0)
            total_pnl = combined.get("total_pnl_usd", 0)
            best_trade = combined.get("best_trade", 0)
            worst_trade = combined.get("worst_trade", 0)
            total_volume = combined.get("total_volume", 0)
            total_invested = combined.get("total_invested", 0)
            total_realized = combined.get("total_realized", 0)
            
            if total_trades > 0 and win_rate > 0:
                win_count = int(total_trades * (win_rate / 100))
                loss_count = total_trades - win_count
            else:
                win_count = 0
                loss_count = total_trades
            
            if total_pnl > 0:
                total_profit_usd = total_pnl
                total_loss_usd = 0
            else:
                total_profit_usd = 0
                total_loss_usd = abs(total_pnl)
            
            # Calculate profit factor (capped at 999.99)
            if total_loss_usd > 0:
                profit_factor = round(total_profit_usd / total_loss_usd, 2)
            elif total_profit_usd > 0:
                profit_factor = 999.99
            else:
                profit_factor = 0.0
            
            if total_invested > 0:
                avg_roi = (total_pnl / total_invested) * 100
            else:
                avg_roi = 0
            
            if analyzed_trades:
                recent_rois = [t.get("roi_percent", 0) for t in analyzed_trades if "roi_percent" in t]
                if recent_rois:
                    median_roi = np.median(recent_rois)
                    max_roi = max(recent_rois)
                    min_roi = min(recent_rois)
                    std_dev_roi = np.std(recent_rois)
                else:
                    median_roi = 0
                    max_roi = 0
                    min_roi = 0
                    std_dev_roi = 0
            else:
                median_roi = avg_roi * 0.8
                max_roi = best_trade
                min_roi = worst_trade
                std_dev_roi = abs(max_roi - min_roi) / 4
            
            roi_distribution = self._convert_roi_distribution(combined.get("roi_distribution", {}))
            
            complete_metrics = {
                "total_trades": total_trades,
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": win_rate,
                "total_profit_usd": total_profit_usd,
                "total_loss_usd": total_loss_usd,
                "net_profit_usd": total_pnl,
                "profit_factor": profit_factor,
                "avg_roi": avg_roi,
                "median_roi": median_roi,
                "std_dev_roi": std_dev_roi,
                "max_roi": max_roi,
                "min_roi": min_roi,
                "avg_hold_time_hours": combined.get("avg_hold_time", 0),
                "total_bet_size_usd": total_invested,
                "avg_bet_size_usd": combined.get("avg_trade_size", 0),
                "total_tokens_traded": combined.get("tokens_traded", 0),
                "roi_distribution": roi_distribution
            }
            
            return complete_metrics
            
        except Exception as e:
            logger.error(f"Error combining metrics: {str(e)}")
            return self._get_empty_metrics()
    
    def _convert_roi_distribution(self, cielo_roi_dist: Dict[str, Any]) -> Dict[str, int]:
        """Convert Cielo Finance ROI distribution format to our expected format."""
        return {
            "10x_plus": cielo_roi_dist.get("roi_above_500", 0),
            "5x_to_10x": cielo_roi_dist.get("roi_above_500", 0),
            "2x_to_5x": cielo_roi_dist.get("roi_200_to_500", 0),
            "1x_to_2x": int(cielo_roi_dist.get("roi_0_to_200", 0) / 2),
            "50_to_100": int(cielo_roi_dist.get("roi_0_to_200", 0) / 2),
            "0_to_50": int(cielo_roi_dist.get("roi_0_to_200", 0) / 2),
            "minus50_to_0": cielo_roi_dist.get("roi_neg50_to_0", 0),
            "below_minus50": cielo_roi_dist.get("roi_below_neg50", 0)
        }
    
    def _calculate_composite_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate composite score (0-100) for ALL wallets."""
        try:
            total_trades = metrics.get("total_trades", 0)
            win_rate = metrics.get("win_rate", 0)
            profit_factor = metrics.get("profit_factor", 0)
            avg_roi = metrics.get("avg_roi", 0)
            max_roi = metrics.get("max_roi", 0)
            median_roi = metrics.get("median_roi", 0)
            net_profit = metrics.get("net_profit_usd", 0)
            
            # Activity score (max 20 points)
            if total_trades >= 50:
                activity_score = 20
            elif total_trades >= 20:
                activity_score = 15
            elif total_trades >= 10:
                activity_score = 10
            elif total_trades >= 5:
                activity_score = 5
            elif total_trades >= 1:
                activity_score = 3
            else:
                activity_score = 0
            
            # Win rate score (max 20 points)
            if win_rate >= 60:
                winrate_score = 20
            elif win_rate >= 45:
                winrate_score = 15
            elif win_rate >= 30:
                winrate_score = 10
            elif win_rate >= 20:
                winrate_score = 5
            elif win_rate >= 10:
                winrate_score = 3
            else:
                winrate_score = 0
            
            # Profit factor score (max 20 points)
            if profit_factor >= 2.0:
                pf_score = 20
            elif profit_factor >= 1.5:
                pf_score = 15
            elif profit_factor >= 1.0:
                pf_score = 10
            elif profit_factor >= 0.8:
                pf_score = 5
            elif profit_factor >= 0.5:
                pf_score = 3
            else:
                pf_score = 0
            
            # ROI score (max 20 points) - Updated for 5x focus
            if max_roi >= 500:  # 5x+
                roi_score = 20
            elif max_roi >= 200:
                roi_score = 15
            elif max_roi >= 100:
                roi_score = 10
            elif max_roi >= 50:
                roi_score = 5
            elif max_roi >= 0:
                roi_score = 3
            else:
                roi_score = 0
            
            # Consistency score (max 20 points)
            consistency_points = 0
            
            if median_roi >= 50:
                consistency_points += 10
            elif median_roi >= 20:
                consistency_points += 7
            elif median_roi >= 0:
                consistency_points += 5
            elif median_roi >= -10:
                consistency_points += 3
            else:
                consistency_points += 0
            
            if net_profit > 0:
                consistency_points += 10
            elif net_profit >= -100:
                consistency_points += 5
            else:
                consistency_points += 0
            
            consistency_score = min(20, consistency_points)
            
            total_score = (
                activity_score +
                winrate_score +
                pf_score +
                roi_score +
                consistency_score
            )
            
            # Bonus for exceptional performance
            if max_roi >= 1000:  # 10x+
                total_score *= 1.2
            elif max_roi >= 500:  # 5x+
                total_score *= 1.1
            
            if net_profit > 1000:
                total_score *= 1.1
            elif net_profit > 100:
                total_score *= 1.05
            
            total_score = min(100, total_score)
            
            if total_trades > 0:
                total_score = max(2, total_score)
            
            if total_trades > 0 and total_score < 18:
                total_score = 18
            
            if total_trades == 0:
                import hashlib
                unique_str = str(metrics.get("wallet_address", "")) + str(metrics.get("total_volume", "")) + str(metrics.get("tokens_traded", ""))
                hash_val = int(hashlib.md5(unique_str.encode()).hexdigest()[:4], 16)
                variation = (hash_val % 10) / 10.0
                total_score += variation
            
            return round(total_score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return 0
    
    def _calculate_metrics(self, paired_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from paired trades."""
        if not paired_trades:
            logger.warning("No paired trades for metrics calculation")
            return self._get_empty_metrics()
        
        try:
            total_trades = len(paired_trades)
            win_count = sum(1 for trade in paired_trades if trade.get("is_win", False))
            loss_count = total_trades - win_count
            
            roi_values = [trade.get("roi_percent", 0) for trade in paired_trades]
            avg_roi = np.mean(roi_values) if roi_values else 0
            median_roi = np.median(roi_values) if roi_values else 0
            std_dev_roi = np.std(roi_values) if roi_values else 0
            max_roi = max(roi_values) if roi_values else 0
            min_roi = min(roi_values) if roi_values else 0
            
            total_profit_usd = 0
            total_loss_usd = 0
            
            for trade in paired_trades:
                buy_value = trade.get("buy_value_usd", 0)
                sell_value = trade.get("sell_value_usd", 0)
                pnl = sell_value - buy_value
                
                if pnl > 0:
                    total_profit_usd += pnl
                else:
                    total_loss_usd += abs(pnl)
            
            net_profit_usd = total_profit_usd - total_loss_usd
            
            # Profit factor calculation (capped at 999.99)
            if total_loss_usd > 0:
                profit_factor = round(total_profit_usd / total_loss_usd, 2)
            elif total_profit_usd > 0:
                profit_factor = 999.99
            else:
                profit_factor = 0.0
            
            holding_times = [trade.get("holding_time_hours", 0) for trade in paired_trades if "holding_time_hours" in trade and trade.get("holding_time_hours", 0) > 0]
            avg_hold_time_hours = np.mean(holding_times) if holding_times else 0
            
            bet_sizes = [trade.get("buy_value_usd", 0) for trade in paired_trades]
            total_bet_size_usd = sum(bet_sizes)
            avg_bet_size_usd = np.mean(bet_sizes) if bet_sizes else 0
            
            unique_tokens = len(set(trade.get("token_address", "") for trade in paired_trades if trade.get("token_address")))
            
            roi_buckets = {
                "10x_plus": len([t for t in paired_trades if t.get("roi_percent", 0) >= 1000]),
                "5x_to_10x": len([t for t in paired_trades if 500 <= t.get("roi_percent", 0) < 1000]),
                "2x_to_5x": len([t for t in paired_trades if 200 <= t.get("roi_percent", 0) < 500]),
                "1x_to_2x": len([t for t in paired_trades if 100 <= t.get("roi_percent", 0) < 200]),
                "50_to_100": len([t for t in paired_trades if 50 <= t.get("roi_percent", 0) < 100]),
                "0_to_50": len([t for t in paired_trades if 0 <= t.get("roi_percent", 0) < 50]),
                "minus50_to_0": len([t for t in paired_trades if -50 <= t.get("roi_percent", 0) < 0]),
                "below_minus50": len([t for t in paired_trades if t.get("roi_percent", 0) < -50])
            }
            
            return {
                "total_trades": total_trades,
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": (win_count / total_trades * 100) if total_trades > 0 else 0,
                "total_profit_usd": total_profit_usd,
                "total_loss_usd": total_loss_usd,
                "net_profit_usd": net_profit_usd,
                "profit_factor": profit_factor,
                "avg_roi": avg_roi,
                "median_roi": median_roi,
                "std_dev_roi": std_dev_roi,
                "max_roi": max_roi,
                "min_roi": min_roi,
                "avg_hold_time_hours": avg_hold_time_hours,
                "total_bet_size_usd": total_bet_size_usd,
                "avg_bet_size_usd": avg_bet_size_usd,
                "total_tokens_traded": unique_tokens,
                "roi_distribution": roi_buckets
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure with ALL required fields."""
        return {
            "total_trades": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0,
            "total_profit_usd": 0,
            "total_loss_usd": 0,
            "net_profit_usd": 0,
            "profit_factor": 0,
            "avg_roi": 0,
            "median_roi": 0,
            "std_dev_roi": 0,
            "max_roi": 0,
            "min_roi": 0,
            "avg_hold_time_hours": 0,
            "total_bet_size_usd": 0,
            "avg_bet_size_usd": 0,
            "total_tokens_traded": 0,
            "roi_distribution": {
                "10x_plus": 0,
                "5x_to_10x": 0,
                "2x_to_5x": 0,
                "1x_to_2x": 0,
                "50_to_100": 0,
                "0_to_50": 0,
                "minus50_to_0": 0,
                "below_minus50": 0
            }
        }
    
    def _determine_wallet_type(self, metrics: Dict[str, Any]) -> str:
        """Legacy wallet type determination for backward compatibility."""
        return self._determine_wallet_type_enhanced(metrics)
    
    def _generate_strategy(self, wallet_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a trading strategy based on wallet type and metrics."""
        # Use enhanced strategy generation with empty advanced metrics
        return self._generate_strategy_enhanced(wallet_type, metrics, {})
    
    def analyze_wallet(self, wallet_address: str, days_back: int = 30) -> Dict[str, Any]:
        """Legacy method - redirects to tiered analysis."""
        return self.analyze_wallet_tiered(wallet_address, "standard", days_back)
    
    def _extract_trades_from_cielo_trading_stats(self, trading_stats_data: Dict[str, Any], 
                                                 wallet_address: str) -> List[Dict[str, Any]]:
        """Extract and categorize trades from Cielo Finance token P&L data."""
        if not trading_stats_data or not trading_stats_data.get("success", True):
            logger.warning(f"No valid token P&L data for wallet {wallet_address}")
            return []
        
        trades = []
        
        try:
            stats_data = trading_stats_data.get("data", {})
            if not isinstance(stats_data, dict):
                logger.warning(f"Unexpected token P&L data format for wallet {wallet_address}")
                return []
            
            if "tokens" in stats_data:
                for token_data in stats_data["tokens"]:
                    processed_trade = self._process_cielo_token_pnl(token_data, wallet_address)
                    if processed_trade:
                        trades.append(processed_trade)
            
            elif isinstance(stats_data, list):
                for token_data in stats_data:
                    processed_trade = self._process_cielo_token_pnl(token_data, wallet_address)
                    if processed_trade:
                        trades.append(processed_trade)
            
            else:
                summary_trade = self._create_summary_trade_from_stats(stats_data, wallet_address)
                if summary_trade:
                    trades.append(summary_trade)
            
            logger.info(f"Extracted {len(trades)} trades from Cielo Finance P&L data for wallet {wallet_address}")
            return trades
            
        except Exception as e:
            logger.error(f"Error extracting trades from Cielo Finance P&L data for wallet {wallet_address}: {str(e)}")
            return []
    
    def _process_cielo_token_pnl(self, token_data: Dict[str, Any], wallet_address: str) -> Optional[Dict[str, Any]]:
        """Process token P&L data from Cielo Finance."""
        try:
            return {
                "token_address": token_data.get("mint", token_data.get("address", token_data.get("contract", ""))),
                "token_symbol": token_data.get("symbol", token_data.get("name", "")),
                "buy_timestamp": token_data.get("first_tx_timestamp", token_data.get("first_buy_time", 0)),
                "buy_date": datetime.fromtimestamp(token_data.get("first_tx_timestamp", token_data.get("first_buy_time", 0))).isoformat() if token_data.get("first_tx_timestamp", token_data.get("first_buy_time", 0)) else "",
                "buy_price": token_data.get("avg_buy_price", token_data.get("buy_price", 0)),
                "buy_value_usd": token_data.get("total_buy_usd", token_data.get("buy_value_usd", 0)),
                "buy_value_sol": token_data.get("total_buy_sol", token_data.get("buy_value_sol", 0)),
                "sell_timestamp": token_data.get("last_tx_timestamp", token_data.get("last_sell_time", 0)),
                "sell_date": datetime.fromtimestamp(token_data.get("last_tx_timestamp", token_data.get("last_sell_time", 0))).isoformat() if token_data.get("last_tx_timestamp", token_data.get("last_sell_time", 0)) else "",
                "sell_price": token_data.get("avg_sell_price", token_data.get("sell_price", 0)),
                "sell_value_usd": token_data.get("total_sell_usd", token_data.get("sell_value_usd", 0)),
                "sell_value_sol": token_data.get("total_sell_sol", token_data.get("sell_value_sol", 0)),
                "holding_time_hours": token_data.get("holding_time_hours", token_data.get("hold_time", 0)),
                "roi_percent": token_data.get("pnl_percent", token_data.get("roi_percent", token_data.get("pnl", 0))),
                "is_win": token_data.get("pnl_percent", token_data.get("roi_percent", token_data.get("pnl", 0))) > 0,
                "market_cap_at_buy": token_data.get("market_cap", token_data.get("mcap", 0)),
                "platform": token_data.get("platform", token_data.get("dex", "")),
                "total_trades": token_data.get("tx_count", token_data.get("trades", 1)),
                "realized_pnl_usd": token_data.get("realized_pnl_usd", token_data.get("realized_pnl", 0)),
                "unrealized_pnl_usd": token_data.get("unrealized_pnl_usd", token_data.get("unrealized_pnl", 0))
            }
        except Exception as e:
            logger.warning(f"Error processing Cielo token P&L data: {str(e)}")
            return None
    
    def _create_summary_trade_from_stats(self, stats_data: Dict[str, Any], wallet_address: str) -> Optional[Dict[str, Any]]:
        """Create a summary trade from wallet-level statistics."""
        try:
            if not any(stats_data.get(key, 0) != 0 
                      for key in ["swaps_count", "pnl", "winrate", "total_buy_amount_usd"]):
                return None
            
            total_trades = stats_data.get("swaps_count", 0)
            win_rate = stats_data.get("winrate", 0)
            total_pnl = stats_data.get("pnl", 0)
            total_invested = stats_data.get("total_buy_amount_usd", 0)
            total_realized = stats_data.get("total_sell_amount_usd", 0)
            avg_hold_time_sec = stats_data.get("average_holding_time_sec", 0)
            
            roi_percent = 0
            if total_invested > 0:
                roi_percent = ((total_realized - total_invested) / total_invested) * 100
            elif total_pnl != 0 and total_invested > 0:
                roi_percent = (total_pnl / total_invested) * 100
            
            return {
                "token_address": "",
                "token_symbol": f"WALLET_SUMMARY_{total_trades}_TRADES",
                "buy_timestamp": 0,
                "buy_date": "",
                "buy_price": 0,
                "buy_value_usd": total_invested,
                "buy_value_sol": 0,
                "sell_timestamp": 0,
                "sell_date": "",
                "sell_price": 0,
                "sell_value_usd": total_realized,
                "sell_value_sol": 0,
                "holding_time_hours": avg_hold_time_sec / 3600 if avg_hold_time_sec else 0,
                "roi_percent": roi_percent,
                "is_win": total_pnl > 0,
                "market_cap_at_buy": 0,
                "platform": "MULTIPLE",
                "total_trades": total_trades,
                "realized_pnl_usd": total_pnl,
                "unrealized_pnl_usd": 0
            }
            
        except Exception as e:
            logger.warning(f"Error creating summary trade: {str(e)}")
            return None
    
    def _pair_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pair buy and sell trades to calculate ROI - for Cielo data, trades are already paired."""
        if not trades:
            logger.warning("No trades to pair")
            return []
        
        paired_trades = []
        
        for trade in trades:
            trade_copy = trade.copy()
            trade_copy.setdefault("buy_value_usd", 0)
            trade_copy.setdefault("sell_value_usd", 0)
            trade_copy.setdefault("roi_percent", 0)
            trade_copy.setdefault("is_win", trade_copy.get("roi_percent", 0) > 0)
            
            if trade_copy.get("buy_value_usd", 0) > 0:
                paired_trades.append(trade_copy)
        
        logger.info(f"Validated {len(paired_trades)} paired trades from {len(trades)} trades")
        return paired_trades
    
    def _find_correlated_wallets(self, wallet_address: str, time_threshold: int = 300) -> List[Dict[str, Any]]:
        """Find wallets that entered the same tokens within a close time window."""
        try:
            correlated_wallets = {}
            
            for token_address, entries in self.token_entries.items():
                if wallet_address in entries:
                    main_timestamp = entries[wallet_address]
                    
                    for other_wallet, other_timestamp in entries.items():
                        if other_wallet != wallet_address:
                            time_diff = abs(main_timestamp - other_timestamp)
                            
                            if time_diff <= time_threshold:
                                if other_wallet not in correlated_wallets:
                                    correlated_wallets[other_wallet] = {
                                        "wallet_address": other_wallet,
                                        "common_tokens": 0,
                                        "avg_time_diff": 0,
                                        "tokens": []
                                    }
                                
                                correlated_wallets[other_wallet]["common_tokens"] += 1
                                current_avg = correlated_wallets[other_wallet]["avg_time_diff"]
                                current_count = len(correlated_wallets[other_wallet]["tokens"])
                                new_avg = (current_avg * current_count + time_diff) / (current_count + 1)
                                correlated_wallets[other_wallet]["avg_time_diff"] = new_avg
                                correlated_wallets[other_wallet]["tokens"].append({
                                    "token_address": token_address,
                                    "time_diff": time_diff
                                })
            
            return sorted(
                correlated_wallets.values(),
                key=lambda x: (x["common_tokens"], -x["avg_time_diff"]),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error finding correlated wallets: {str(e)}")
            return []
    
    def _identify_wallet_clusters(self, wallet_correlations: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify clusters of wallets that appear to be trading together."""
        try:
            clusters = []
            processed_wallets = set()
            
            for wallet, correlations in wallet_correlations.items():
                if wallet in processed_wallets:
                    continue
                
                cluster_wallets = {wallet}
                cluster_correlation_strength = 0
                
                for correlation in correlations:
                    correlated_wallet = correlation["wallet_address"]
                    if correlated_wallet not in processed_wallets and correlation["common_tokens"] >= 2:
                        cluster_wallets.add(correlated_wallet)
                        cluster_correlation_strength += correlation["common_tokens"]
                
                if len(cluster_wallets) > 1:
                    clusters.append({
                        "wallets": list(cluster_wallets),
                        "size": len(cluster_wallets),
                        "correlation_strength": cluster_correlation_strength
                    })
                    
                    processed_wallets.update(cluster_wallets)
            
            clusters.sort(key=lambda x: (x["size"], x["correlation_strength"]), reverse=True)
            return clusters
            
        except Exception as e:
            logger.error(f"Error identifying wallet clusters: {str(e)}")
            return []
    
    def export_wallet_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Export wallet analysis to CSV."""
        if not analysis.get("success"):
            logger.warning(f"Cannot export failed analysis for {analysis.get('wallet_address')}: {analysis.get('error')}")
            return
        
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            metrics_file = output_file.replace(".csv", "_metrics.csv")
            with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ["metric", "value"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                writer.writerow({"metric": "wallet_address", "value": analysis["wallet_address"]})
                writer.writerow({"metric": "wallet_type", "value": analysis["wallet_type"]})
                writer.writerow({"metric": "composite_score", "value": analysis.get("composite_score", 0)})
                writer.writerow({"metric": "api_source", "value": analysis.get("api_source", "Unknown")})
                writer.writerow({"metric": "analysis_tier", "value": analysis.get("analysis_tier", "standard")})
                
                strategy = analysis.get("strategy", {})
                writer.writerow({"metric": "recommendation", "value": strategy.get("recommendation", "CAUTIOUS")})
                writer.writerow({"metric": "entry_type", "value": strategy.get("entry_type", "WAIT_FOR_CONFIRMATION")})
                writer.writerow({"metric": "confidence", "value": strategy.get("confidence", "LOW")})
                
                metrics = analysis.get("metrics", {})
                for key, value in metrics.items():
                    if key != "roi_distribution":
                        writer.writerow({"metric": key, "value": value})
                
                roi_dist = metrics.get("roi_distribution", {})
                if isinstance(roi_dist, dict):
                    for key, value in roi_dist.items():
                        writer.writerow({"metric": f"roi_{key}", "value": value})
                
                # Add advanced metrics if available
                if "advanced_metrics" in analysis:
                    adv_metrics = analysis["advanced_metrics"]
                    for key, value in adv_metrics.items():
                        writer.writerow({"metric": f"advanced_{key}", "value": value})
                
                if "entry_exit_analysis" in analysis:
                    ee_analysis = analysis["entry_exit_analysis"]
                    writer.writerow({"metric": "entry_exit_pattern", "value": ee_analysis.get("pattern", "UNKNOWN")})
                    writer.writerow({"metric": "entry_quality", "value": ee_analysis.get("entry_quality", "UNKNOWN")})
                    writer.writerow({"metric": "exit_quality", "value": ee_analysis.get("exit_quality", "UNKNOWN")})
                    writer.writerow({"metric": "missed_gains_percent", "value": ee_analysis.get("missed_gains_percent", 0)})
                    writer.writerow({"metric": "early_exit_rate", "value": ee_analysis.get("early_exit_rate", 0)})
            
            logger.info(f"Exported wallet analysis to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error exporting wallet analysis: {str(e)}")
    
    def export_batch_analysis(self, batch_analysis: Dict[str, Any], output_file: str) -> None:
        """Export batch wallet analysis to CSV."""
        if not batch_analysis.get("success"):
            logger.error(f"Cannot export failed batch analysis: {batch_analysis.get('error')}")
            return
        
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            summary_file = output_file.replace(".csv", "_summary.csv")
            with open(summary_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ["metric", "value"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                writer.writerow({"metric": "api_source", "value": batch_analysis.get("api_source", "Unknown")})
                writer.writerow({"metric": "total_wallets", "value": batch_analysis["total_wallets"]})
                writer.writerow({"metric": "analyzed_wallets", "value": batch_analysis["analyzed_wallets"]})
                writer.writerow({"metric": "failed_wallets", "value": batch_analysis.get("failed_wallets", 0)})
                writer.writerow({"metric": "gem_hunter_count", "value": len(batch_analysis.get("gem_hunters", []))})
                writer.writerow({"metric": "smart_trader_count", "value": len(batch_analysis.get("smart_traders", []))})
                writer.writerow({"metric": "diamond_hands_count", "value": len(batch_analysis.get("diamond_hands", []))})
                writer.writerow({"metric": "consistent_count", "value": len(batch_analysis.get("consistent", []))})
                writer.writerow({"metric": "flipper_count", "value": len(batch_analysis.get("flippers", []))})
                writer.writerow({"metric": "mixed_count", "value": len(batch_analysis.get("mixed", []))})
                writer.writerow({"metric": "underperformer_count", "value": len(batch_analysis.get("underperformers", []))})
                writer.writerow({"metric": "unknown_count", "value": len(batch_analysis.get("unknown", []))})
                
                # Add cache stats if available
                if "cache_stats" in batch_analysis:
                    cache_stats = batch_analysis["cache_stats"]
                    writer.writerow({"metric": "cache_hit_rate", "value": cache_stats.get("hit_rate_percent", 0)})
                    writer.writerow({"metric": "api_calls_saved", "value": cache_stats.get("api_calls_saved", 0)})
            
            logger.info(f"Exported batch analysis summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting batch analysis: {str(e)}")
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)