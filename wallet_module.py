"""
Wallet Analysis Module - Phoenix Project (3-TIER ANALYSIS EDITION)

MAJOR UPDATES:
- 3-tier analysis system: Initial (5), Standard (10), Deep (20) tokens
- Stricter criteria to reduce API calls (~60% initial, ~30% standard, ~10% deep)
- Elite wallet detection for 5x+ gem hunters
- Improved initial scoring with net profit consideration
"""

import csv
import os
import logging
import numpy as np
import requests
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache

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
    """Class for analyzing wallets for copy trading using Cielo Finance API + RPC + Helius."""
    
    # 3-Tier analysis constants
    INITIAL_SCAN_TOKENS = 5
    STANDARD_SCAN_TOKENS = 10
    DEEP_SCAN_TOKENS = 20
    
    def __init__(self, cielo_api: Any, birdeye_api: Any = None, helius_api: Any = None, 
                 rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """
        Initialize the wallet analyzer.
        
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
        
        # Market cap buckets for memecoin analysis
        self.market_cap_buckets = {
            "ultra_low": (5000, 50000),      # $5K - $50K
            "low": (50000, 500000),          # $50K - $500K
            "mid": (500000, 5000000),        # $500K - $5M
            "high": (5000000, float('inf'))  # $5M+
        }
        
        # Track API calls for reporting
        self.api_call_stats = {
            "cielo": 0,
            "birdeye": 0,
            "helius": 0,
            "rpc": 0
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
        self.api_call_stats["rpc"] += 1
        
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
                    
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error (attempt {attempt + 1}/{retry_count}): {str(e)}")
                if attempt < retry_count - 1:
                    wait_time = min(60, 2 ** attempt)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    return {"error": f"Connection error after {retry_count} attempts: {str(e)}"}
                    
            except requests.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{retry_count}): {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(min(60, 2 ** attempt))
                else:
                    return {"error": str(e)}
                    
            except Exception as e:
                logger.error(f"Unexpected RPC error for {method}: {str(e)}")
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
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
        response = self._make_rpc_call(
            "getTransaction",
            [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0}]
        )
        
        if "result" in response:
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
    
    def _determine_scan_tier(self, metrics: Dict[str, Any], initial_score: float) -> Tuple[int, str]:
        """Determine scan depth based on wallet performance metrics."""
        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)
        total_trades = metrics.get("total_trades", 0)
        net_profit = metrics.get("net_profit_usd", 0)
        
        # Check for DEEP tier (elite performers)
        if self._is_elite_wallet(metrics, initial_score):
            return self.DEEP_SCAN_TOKENS, "DEEP"
        
        # Check for STANDARD tier (decent performers)
        elif self._is_standard_wallet(metrics, initial_score):
            return self.STANDARD_SCAN_TOKENS, "STANDARD"
        
        # Default to INITIAL tier
        else:
            return self.INITIAL_SCAN_TOKENS, "INITIAL"
    
    def _is_elite_wallet(self, metrics: Dict[str, Any], initial_score: float) -> bool:
        """Check if wallet qualifies for deep analysis."""
        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)
        total_trades = metrics.get("total_trades", 0)
        net_profit = metrics.get("net_profit_usd", 0)
        
        # Score-based qualification
        if initial_score >= 70:
            return True
        
        # High performance combo
        if win_rate >= 65 and profit_factor >= 4.0:
            return True
        
        # High profit with good win rate
        if net_profit >= 10000 and win_rate >= 60:
            return True
        
        # Exceptional gem hunter
        roi_dist = metrics.get("roi_distribution", {})
        trades_5x_plus = roi_dist.get("roi_above_500", 0)
        if total_trades >= 30 and trades_5x_plus >= total_trades * 0.1:  # 10%+ are 5x
            return True
        
        # Check for exceptional ROI distribution
        high_roi_trades = roi_dist.get("roi_above_500", 0) + roi_dist.get("roi_200_to_500", 0)
        if total_trades >= 20 and high_roi_trades >= total_trades * 0.2:  # 20%+ are 2x+
            return True
        
        return False
    
    def _is_standard_wallet(self, metrics: Dict[str, Any], initial_score: float) -> bool:
        """Check if wallet qualifies for standard analysis."""
        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)
        total_trades = metrics.get("total_trades", 0)
        net_profit = metrics.get("net_profit_usd", 0)
        
        # Must meet ALL criteria for standard tier
        return (
            win_rate >= 55 and win_rate < 65 and
            profit_factor >= 2.0 and profit_factor < 4.0 and
            total_trades >= 20 and
            net_profit >= 1000 and
            initial_score >= 50 and initial_score < 70
        )
    
    def _calculate_initial_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate initial score with stricter thresholds."""
        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)
        total_trades = metrics.get("total_trades", 0)
        net_profit = metrics.get("net_profit_usd", 0)
        
        score = 0
        
        # Win rate scoring (0-25 points)
        if win_rate >= 70:
            score += 25
        elif win_rate >= 65:
            score += 20
        elif win_rate >= 60:
            score += 15
        elif win_rate >= 55:
            score += 10
        elif win_rate >= 50:
            score += 5
        
        # Profit factor scoring (0-25 points)
        if profit_factor >= 5.0:
            score += 25
        elif profit_factor >= 4.0:
            score += 20
        elif profit_factor >= 3.0:
            score += 15
        elif profit_factor >= 2.0:
            score += 10
        elif profit_factor >= 1.5:
            score += 5
        
        # Trade count scoring (0-20 points)
        if total_trades >= 100:
            score += 20
        elif total_trades >= 50:
            score += 15
        elif total_trades >= 30:
            score += 10
        elif total_trades >= 20:
            score += 5
        
        # Net profit scoring (0-20 points)
        if net_profit >= 20000:
            score += 20
        elif net_profit >= 10000:
            score += 15
        elif net_profit >= 5000:
            score += 10
        elif net_profit >= 2000:
            score += 5
        
        # Bonus points for exceptional combinations (0-10 points)
        if win_rate >= 60 and profit_factor >= 3.0 and net_profit >= 5000:
            score += 10
        elif win_rate >= 55 and profit_factor >= 2.5 and net_profit >= 2000:
            score += 5
        
        return min(score, 100)  # Cap at 100
    
    def analyze_wallet_hybrid(self, wallet_address: str, days_back: int = 30) -> Dict[str, Any]:
        """
        UPDATED hybrid wallet analysis with 3-tier scanning system.
        
        Args:
            wallet_address (str): Wallet address
            days_back (int): Number of days to analyze (for RPC calls)
            
        Returns:
            Dict[str, Any]: Wallet analysis results
        """
        logger.info(f"🔍 Analyzing wallet {wallet_address} (3-tier memecoin analysis)")
        
        try:
            # Step 1: Get aggregated stats from Cielo Finance
            logger.info(f"📊 Fetching Cielo Finance aggregated stats...")
            self.api_call_stats["cielo"] += 1
            cielo_stats = self.cielo_api.get_wallet_trading_stats(wallet_address)
            
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
            
            # Step 2: Calculate initial score and determine scan tier
            initial_score = self._calculate_initial_score(aggregated_metrics)
            scan_limit, tier = self._determine_scan_tier(aggregated_metrics, initial_score)
            
            logger.info(f"📊 Wallet tier: {tier} - Scanning {scan_limit} recent trades")
            logger.info(f"   Score: {initial_score}/100")
            logger.info(f"   Win Rate: {aggregated_metrics.get('win_rate', 0):.1f}%")
            logger.info(f"   Profit Factor: {aggregated_metrics.get('profit_factor', 0):.2f}")
            logger.info(f"   Net Profit: ${aggregated_metrics.get('net_profit_usd', 0):.2f}")
            
            # Step 3: Get recent token swaps based on tier
            recent_swaps = self._get_recent_token_swaps_rpc(wallet_address, limit=scan_limit)
            
            # Step 4: Analyze token performance with market cap data
            analyzed_trades = []
            
            if recent_swaps:
                for i, swap in enumerate(recent_swaps):
                    if i > 0:
                        time.sleep(0.5)
                    
                    # Enhanced analysis with market cap
                    token_analysis = self._analyze_token_with_market_cap(
                        swap['token_mint'],
                        swap['buy_timestamp'],
                        swap.get('sell_timestamp'),
                        swap
                    )
                    
                    if token_analysis['success']:
                        analyzed_trades.append({
                            **swap,
                            **token_analysis
                        })
            
            # Step 5: Calculate enhanced metrics
            enhanced_metrics = self._calculate_enhanced_memecoin_metrics(
                aggregated_metrics, 
                analyzed_trades
            )
            
            # Step 6: Calculate composite score with distribution factored in
            composite_score = self._calculate_memecoin_composite_score_with_distribution(enhanced_metrics)
            enhanced_metrics["composite_score"] = composite_score
            
            # Step 7: Determine wallet type based on hold patterns (5x+ for gem hunters)
            wallet_type = self._determine_memecoin_wallet_type_5x(enhanced_metrics)
            
            # Step 8: Generate memecoin-specific strategy
            strategy = self._generate_memecoin_strategy(wallet_type, enhanced_metrics)
            
            # Step 9: Analyze entry/exit behavior with proper calculations
            entry_exit_analysis = self._analyze_memecoin_entry_exit(analyzed_trades, enhanced_metrics)
            
            # Step 10: Detect bundle and copytrader activity
            bundle_analysis = self._detect_bundle_activity(analyzed_trades, recent_swaps)
            
            # Log API call statistics
            logger.info(f"📊 API Calls - Cielo: {self.api_call_stats['cielo']}, "
                       f"Birdeye: {self.api_call_stats['birdeye']}, "
                       f"Helius: {self.api_call_stats['helius']}, "
                       f"RPC: {self.api_call_stats['rpc']}")
            
            logger.debug(f"Final analysis for {wallet_address}:")
            logger.debug(f"  - wallet_type: {wallet_type}")
            logger.debug(f"  - composite_score: {composite_score}")
            logger.debug(f"  - gem_rate_5x_plus: {enhanced_metrics.get('gem_rate_5x_plus', 0):.1f}%")
            logger.debug(f"  - tokens_scanned: {len(analyzed_trades)}/{scan_limit}")
            logger.debug(f"  - analysis_tier: {tier}")
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_period_days": "ALL_TIME (Cielo) + Recent (RPC)",
                "wallet_type": wallet_type,
                "composite_score": composite_score,
                "metrics": enhanced_metrics,
                "strategy": strategy,
                "trades": analyzed_trades,
                "entry_exit_analysis": entry_exit_analysis,
                "bundle_analysis": bundle_analysis,
                "api_source": "Cielo Finance + RPC + Birdeye/Helius",
                "cielo_data": aggregated_metrics,
                "recent_trades_analyzed": len(analyzed_trades),
                "analysis_tier": tier,
                "tokens_scanned": len(analyzed_trades),
                "api_calls": self.api_call_stats.copy()
            }
            
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
                    "recommendation": "DO_NOT_COPY",
                    "avg_first_take_profit": 0,
                    "confidence": "VERY_LOW",
                    "notes": "Analysis failed",
                    "filter_market_cap_min": 0,
                    "filter_market_cap_max": 0
                }
            }
    
    def _analyze_token_with_market_cap(self, token_mint: str, buy_timestamp: Optional[int], 
                                     sell_timestamp: Optional[int] = None, 
                                     swap_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze token with market cap data."""
        try:
            # Get token info including market cap
            token_info = {}
            market_cap_at_buy = 0
            
            if self.birdeye_api and not token_mint.endswith("pump"):
                self.api_call_stats["birdeye"] += 1
                token_result = self.birdeye_api.get_token_info(token_mint)
                if token_result.get("success") and token_result.get("data"):
                    token_info = token_result["data"]
                    market_cap_at_buy = token_info.get("mc", 0) or token_info.get("marketCap", 0)
            elif self.helius_api and token_mint.endswith("pump"):
                self.api_call_stats["helius"] += 1
                # For pump.fun tokens, estimate market cap
                metadata = self.helius_api.get_token_metadata([token_mint])
                if metadata.get("success") and metadata.get("data"):
                    token_info = metadata["data"][0] if metadata["data"] else {}
                    # Pump.fun tokens often start at ~$5-10K market cap
                    market_cap_at_buy = 10000  # Default for pump tokens
            
            # Get price history for performance analysis
            price_history = self._get_price_history_with_fallback(
                token_mint, buy_timestamp, sell_timestamp, swap_data
            )
            
            # Calculate performance metrics
            roi_percent = 0
            max_roi_percent = 0
            current_roi_percent = 0
            entry_timing = "UNKNOWN"
            exit_timing = "UNKNOWN"
            
            if price_history and "initial_price" in price_history:
                initial_price = price_history["initial_price"]
                current_price = price_history.get("current_price", initial_price)
                max_price = price_history.get("max_price", current_price)
                
                if initial_price > 0:
                    roi_percent = ((current_price / initial_price) - 1) * 100
                    max_roi_percent = ((max_price / initial_price) - 1) * 100
                    current_roi_percent = roi_percent
                    
                    # Analyze entry timing (5x+ focus)
                    if max_roi_percent >= 400:  # 5x+
                        entry_timing = "EXCELLENT"
                    elif max_roi_percent >= 200:  # 3x+
                        entry_timing = "GOOD"
                    elif max_roi_percent >= 100:  # 2x+
                        entry_timing = "AVERAGE"
                    else:
                        entry_timing = "POOR"
                    
                    # Analyze exit timing if sold
                    if sell_timestamp:
                        roi_at_exit = roi_percent
                        capture_ratio = roi_at_exit / max_roi_percent if max_roi_percent > 0 else 0
                        
                        if capture_ratio >= 0.8:
                            exit_timing = "EXCELLENT"
                        elif capture_ratio >= 0.6:
                            exit_timing = "GOOD"
                        elif capture_ratio >= 0.4:
                            exit_timing = "AVERAGE"
                        else:
                            exit_timing = "POOR"
                    else:
                        exit_timing = "HOLDING"
            
            # Calculate hold time
            hold_time_seconds = 0
            hold_time_minutes = 0
            if buy_timestamp and sell_timestamp:
                hold_time_seconds = sell_timestamp - buy_timestamp
                hold_time_minutes = hold_time_seconds / 60
            
            return {
                "success": True,
                "token_address": token_mint,
                "market_cap_at_buy": market_cap_at_buy,
                "roi_percent": roi_percent,
                "current_roi_percent": current_roi_percent,
                "max_roi_percent": max_roi_percent,
                "entry_timing": entry_timing,
                "exit_timing": exit_timing,
                "hold_time_seconds": hold_time_seconds,
                "hold_time_minutes": hold_time_minutes,
                "price_data": price_history
            }
            
        except Exception as e:
            logger.error(f"Error analyzing token {token_mint}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "token_address": token_mint
            }
    
    def _get_price_history_with_fallback(self, token_mint: str, buy_timestamp: Optional[int],
                                       sell_timestamp: Optional[int], 
                                       swap_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get price history with multiple fallback options."""
        try:
            # Try Birdeye first
            if self.birdeye_api and not token_mint.endswith("pump"):
                self.api_call_stats["birdeye"] += 1
                current_time = int(datetime.now().timestamp())
                end_time = sell_timestamp if sell_timestamp else current_time
                
                history = self.birdeye_api.get_token_price_history(
                    token_mint,
                    buy_timestamp,
                    end_time,
                    "15m"
                )
                
                if history.get("success") and history.get("data", {}).get("items"):
                    prices = history["data"]["items"]
                    return {
                        "initial_price": prices[0].get("value", 0),
                        "current_price": prices[-1].get("value", 0),
                        "max_price": max(p.get("value", 0) for p in prices),
                        "min_price": min(p.get("value", 0) for p in prices),
                        "price_points": len(prices)
                    }
            
            # Fallback to swap data
            if swap_data and swap_data.get("estimated_price", 0) > 0:
                return {
                    "initial_price": swap_data["estimated_price"],
                    "current_price": swap_data["estimated_price"] * 1.5,  # Estimate
                    "max_price": swap_data["estimated_price"] * 2,  # Conservative estimate
                    "min_price": swap_data["estimated_price"] * 0.8,
                    "price_points": 1
                }
            
            # Default fallback
            return {
                "initial_price": 0.0001,
                "current_price": 0.0001,
                "max_price": 0.0001,
                "min_price": 0.0001,
                "price_points": 0
            }
            
        except Exception as e:
            logger.error(f"Error getting price history: {str(e)}")
            return {}
    
    def _calculate_enhanced_memecoin_metrics(self, cielo_metrics: Dict[str, Any], 
                                           analyzed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate enhanced metrics specific to memecoin trading."""
        try:
            # Start with base metrics
            metrics = cielo_metrics.copy() if cielo_metrics else self._get_empty_cielo_metrics()
            
            # Calculate distribution that sums to 100%
            total_trades = metrics.get("total_trades", 0)
            if total_trades > 0:
                distribution = self._calculate_proper_distribution(metrics, analyzed_trades)
                metrics.update(distribution)
            
            # Calculate gem rate for 5x+ trades (not 2x)
            gem_5x_count = 0
            gem_2x_count = 0
            for trade in analyzed_trades:
                max_roi = trade.get("max_roi_percent", 0)
                if max_roi >= 400:  # 5x (500%)
                    gem_5x_count += 1
                if max_roi >= 100:  # 2x
                    gem_2x_count += 1
            
            gem_rate_5x = (gem_5x_count / len(analyzed_trades) * 100) if analyzed_trades else 0
            gem_rate_2x = (gem_2x_count / len(analyzed_trades) * 100) if analyzed_trades else 0
            
            metrics["gem_rate_5x_plus"] = round(gem_rate_5x, 2)
            metrics["gem_rate_2x_plus"] = round(gem_rate_2x, 2)  # Keep for backward compatibility
            
            # Calculate average hold times (ONLY MINUTES, NO SECONDS)
            hold_times_minutes = []
            for trade in analyzed_trades:
                if trade.get("hold_time_minutes", 0) > 0:
                    hold_times_minutes.append(trade["hold_time_minutes"])
            
            avg_hold_minutes = np.mean(hold_times_minutes) if hold_times_minutes else 0
            
            # Also check if Cielo provided hold time in seconds
            if avg_hold_minutes == 0 and metrics.get("avg_hold_time", 0) > 0:
                # Convert hours to minutes
                avg_hold_minutes = metrics.get("avg_hold_time", 0) * 60
            elif avg_hold_minutes == 0 and "average_holding_time_sec" in cielo_metrics:
                # Convert seconds to minutes
                avg_hold_minutes = cielo_metrics.get("average_holding_time_sec", 0) / 60
            
            metrics["avg_hold_time_minutes"] = round(avg_hold_minutes, 2)
            metrics["avg_hold_time_hours"] = round(avg_hold_minutes / 60, 2)
            # Remove seconds - we don't need them
            metrics.pop("avg_hold_time_seconds", None)
            
            # Calculate market cap metrics
            market_caps = [t.get("market_cap_at_buy", 0) for t in analyzed_trades if t.get("market_cap_at_buy", 0) > 0]
            if market_caps:
                metrics["avg_buy_market_cap_usd"] = round(np.mean(market_caps), 2)
                metrics["median_buy_market_cap_usd"] = round(np.median(market_caps), 2)
            else:
                metrics["avg_buy_market_cap_usd"] = 0
                metrics["median_buy_market_cap_usd"] = 0
            
            # Calculate proper avg_first_take_profit_percent
            first_profits = []
            for trade in analyzed_trades:
                if trade.get("roi_percent", 0) > 0:
                    # This is their exit ROI, which is their first (and often only) take profit
                    first_profits.append(trade["roi_percent"])
            
            if first_profits:
                metrics["avg_first_take_profit_percent"] = round(np.mean(first_profits), 1)
                metrics["median_first_take_profit_percent"] = round(np.median(first_profits), 1)
            else:
                metrics["avg_first_take_profit_percent"] = 0
                metrics["median_first_take_profit_percent"] = 0
            
            # Calculate average buy amount
            buy_amounts = [t.get("sol_amount", 0) * 150 for t in analyzed_trades]  # Estimate USD
            if buy_amounts:
                metrics["avg_buy_amount_usd"] = round(np.mean(buy_amounts), 2)
            
            # Fix median_roi if missing
            if "median_roi" not in metrics or metrics["median_roi"] == 0:
                roi_values = [t.get("roi_percent", 0) for t in analyzed_trades]
                if roi_values:
                    metrics["median_roi"] = round(np.median(roi_values), 2)
            
            # Ensure all required fields are present
            required_fields = [
                "total_trades", "win_rate", "profit_factor", "net_profit_usd",
                "avg_roi", "median_roi", "max_roi", "avg_hold_time_minutes",
                "total_tokens_traded"
            ]
            
            for field in required_fields:
                if field not in metrics:
                    metrics[field] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {str(e)}")
            return cielo_metrics or self._get_empty_metrics()
    
    def _calculate_proper_distribution(self, metrics: Dict[str, Any], 
                                     analyzed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate ROI distribution that sums to 100%."""
        # Use existing distribution if available
        roi_dist = metrics.get("roi_distribution", {})
        
        # Define buckets
        buckets = {
            "distribution_500_plus_%": 0,      # 500%+ (5x+)
            "distribution_200_500_%": 0,       # 200-500% (2x-5x)
            "distribution_0_200_%": 0,         # 0-200% (profitable <2x)
            "distribution_neg50_0_%": 0,       # -50% to 0%
            "distribution_below_neg50_%": 0    # Below -50%
        }
        
        total_trades = metrics.get("total_trades", 0)
        
        if total_trades > 0:
            # Try to use Cielo distribution first
            if roi_dist:
                total_in_dist = sum([
                    roi_dist.get("roi_above_500", 0),
                    roi_dist.get("roi_200_to_500", 0),
                    roi_dist.get("roi_0_to_200", 0),
                    roi_dist.get("roi_neg50_to_0", 0),
                    roi_dist.get("roi_below_neg50", 0)
                ])
                
                if total_in_dist > 0:
                    # Convert to percentages that sum to 100%
                    buckets["distribution_500_plus_%"] = round(roi_dist.get("roi_above_500", 0) / total_trades * 100, 1)
                    buckets["distribution_200_500_%"] = round(roi_dist.get("roi_200_to_500", 0) / total_trades * 100, 1)
                    buckets["distribution_0_200_%"] = round(roi_dist.get("roi_0_to_200", 0) / total_trades * 100, 1)
                    buckets["distribution_neg50_0_%"] = round(roi_dist.get("roi_neg50_to_0", 0) / total_trades * 100, 1)
                    buckets["distribution_below_neg50_%"] = round(roi_dist.get("roi_below_neg50", 0) / total_trades * 100, 1)
            
            # If no distribution or doesn't sum properly, calculate from analyzed trades
            total_percent = sum(buckets.values())
            if abs(total_percent - 100) > 1:  # Allow 1% tolerance
                # Recalculate from analyzed trades
                for trade in analyzed_trades:
                    roi = trade.get("roi_percent", 0)
                    if roi >= 500:
                        buckets["distribution_500_plus_%"] += 1
                    elif roi >= 200:
                        buckets["distribution_200_500_%"] += 1
                    elif roi >= 0:
                        buckets["distribution_0_200_%"] += 1
                    elif roi >= -50:
                        buckets["distribution_neg50_0_%"] += 1
                    else:
                        buckets["distribution_below_neg50_%"] += 1
                
                # Convert counts to percentages
                trade_count = sum(buckets.values())
                if trade_count > 0:
                    for key in buckets:
                        buckets[key] = round(buckets[key] / trade_count * 100, 1)
                
                # Ensure it sums to 100%
                total = sum(buckets.values())
                if total > 0 and total != 100:
                    # Adjust the largest bucket
                    largest_bucket = max(buckets, key=buckets.get)
                    buckets[largest_bucket] += (100 - total)
        
        return buckets
    
    def _calculate_memecoin_composite_score_with_distribution(self, metrics: Dict[str, Any]) -> float:
        """Calculate composite score optimized for memecoin trading WITH distribution factored in."""
        try:
            total_trades = metrics.get("total_trades", 0)
            win_rate = metrics.get("win_rate", 0)
            profit_factor = metrics.get("profit_factor", 0)
            gem_rate_5x = metrics.get("gem_rate_5x_plus", 0)
            net_profit = metrics.get("net_profit_usd", 0)
            avg_first_tp = metrics.get("avg_first_take_profit_percent", 0)
            
            # Activity score (max 10 points) - reduced importance
            if total_trades >= 50:
                activity_score = 10
            elif total_trades >= 20:
                activity_score = 8
            elif total_trades >= 10:
                activity_score = 5
            elif total_trades >= 5:
                activity_score = 3
            else:
                activity_score = 0
            
            # Win rate score (max 15 points)
            if win_rate >= 60:
                winrate_score = 15
            elif win_rate >= 45:
                winrate_score = 10
            elif win_rate >= 30:
                winrate_score = 5
            else:
                winrate_score = 0
            
            # Gem finding score for 5x+ (max 25 points) - critical for memecoins
            if gem_rate_5x >= 30:  # 30%+ trades hit 5x
                gem_score = 25
            elif gem_rate_5x >= 20:
                gem_score = 20
            elif gem_rate_5x >= 10:
                gem_score = 15
            elif gem_rate_5x >= 5:
                gem_score = 10
            elif gem_rate_5x >= 2:
                gem_score = 5
            else:
                gem_score = 0
            
            # Profit factor score (max 15 points)
            if profit_factor >= 2.0:
                pf_score = 15
            elif profit_factor >= 1.5:
                pf_score = 10
            elif profit_factor >= 1.0:
                pf_score = 5
            else:
                pf_score = 0
            
            # Net profit score (max 15 points)
            if net_profit >= 10000:
                profit_score = 15
            elif net_profit >= 5000:
                profit_score = 10
            elif net_profit >= 1000:
                profit_score = 5
            elif net_profit >= 0:
                profit_score = 2
            else:
                profit_score = 0
            
            # NEW: Distribution quality score (max 20 points)
            # Favor wallets with high % in 500%+ bucket
            dist_500_plus = metrics.get("distribution_500_plus_%", 0)
            dist_200_500 = metrics.get("distribution_200_500_%", 0)
            dist_below_neg50 = metrics.get("distribution_below_neg50_%", 0)
            
            dist_score = 0
            if dist_500_plus >= 10:  # 10%+ trades are 5x+
                dist_score += 10
            elif dist_500_plus >= 5:
                dist_score += 5
            
            if dist_200_500 >= 10:  # Good 2x-5x rate
                dist_score += 5
            
            if dist_below_neg50 <= 10:  # Low catastrophic loss rate
                dist_score += 5
            elif dist_below_neg50 <= 20:
                dist_score += 2
            
            # Calculate total
            total_score = (
                activity_score +
                winrate_score +
                gem_score +
                pf_score +
                profit_score +
                dist_score
            )
            
            # Apply multipliers for exceptional performance
            if gem_rate_5x >= 20 and win_rate >= 50:
                total_score *= 1.3  # Excellent 5x gem finder
            elif avg_first_tp >= 100 and avg_first_tp <= 200:
                total_score *= 1.1  # Good profit taking for 2x-3x
            
            # Cap at 100
            total_score = min(100, total_score)
            
            # Ensure minimum score if they have trades
            if total_trades > 0:
                total_score = max(10, total_score)
            
            return round(total_score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return 0
    
    def _determine_memecoin_wallet_type_5x(self, metrics: Dict[str, Any]) -> str:
        """Determine wallet type with 5x+ criteria for gem hunters."""
        total_trades = metrics.get("total_trades", 0)
        if total_trades < 1:
            return "unknown"
        
        try:
            avg_hold_minutes = metrics.get("avg_hold_time_minutes", 0)
            gem_rate_5x = metrics.get("gem_rate_5x_plus", 0)
            win_rate = metrics.get("win_rate", 0)
            avg_first_tp = metrics.get("avg_first_take_profit_percent", 0)
            
            # Sniper: Buys at launch, exits within 1 minute
            if avg_hold_minutes > 0 and avg_hold_minutes <= 1:
                return "sniper"
            
            # Flipper: Quick trades, 1-10 minutes
            elif avg_hold_minutes > 1 and avg_hold_minutes <= 10:
                return "flipper"
            
            # Scalper: 10-60 minutes, takes 20-50% profits
            elif avg_hold_minutes > 10 and avg_hold_minutes <= 60:
                if avg_first_tp >= 20 and avg_first_tp <= 50:
                    return "scalper"
                else:
                    return "flipper"  # Default to flipper if profit pattern doesn't match
            
            # Gem Hunter: Holds for 5x+ gains (not 2x)
            elif gem_rate_5x >= 15:  # 15%+ of trades reach 5x
                return "gem_hunter"
            
            # Swing Trader: Holds 1-24 hours
            elif avg_hold_minutes > 60 and avg_hold_minutes <= 1440:
                return "swing_trader"
            
            # Position Trader: Holds 24+ hours
            elif avg_hold_minutes > 1440:
                return "position_trader"
            
            # Default based on performance
            elif win_rate >= 45:
                return "consistent"
            else:
                return "mixed"
                
        except Exception as e:
            logger.error(f"Error determining wallet type: {str(e)}")
            return "unknown"
    
    def _generate_memecoin_strategy(self, wallet_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate memecoin-specific trading strategy."""
        try:
            composite_score = metrics.get("composite_score", 0)
            avg_first_tp = metrics.get("avg_first_take_profit_percent", 0)
            avg_market_cap = metrics.get("avg_buy_market_cap_usd", 0)
            gem_rate_5x = metrics.get("gem_rate_5x_plus", 0)
            win_rate = metrics.get("win_rate", 0)
            
            # Determine market cap filter range
            if avg_market_cap > 0:
                # Set range based on their typical buying pattern
                if avg_market_cap < 50000:
                    filter_min = 5000
                    filter_max = 100000
                elif avg_market_cap < 500000:
                    filter_min = 50000
                    filter_max = 1000000
                elif avg_market_cap < 5000000:
                    filter_min = 250000
                    filter_max = 10000000
                else:
                    filter_min = 1000000
                    filter_max = 50000000
            else:
                # Default range for unknown
                filter_min = 50000
                filter_max = 5000000
            
            # Base strategy on wallet type
            if wallet_type == "sniper":
                strategy = {
                    "recommendation": "COPY_SNIPER",
                    "avg_first_take_profit": avg_first_tp if avg_first_tp > 0 else 50,
                    "confidence": "HIGH" if composite_score >= 60 else "MEDIUM",
                    "notes": f"Sniper with {gem_rate_5x:.1f}% 5x rate. Quick entries/exits.",
                    "filter_market_cap_min": filter_min,
                    "filter_market_cap_max": filter_max,
                    "suggested_slippage": 25,  # High slippage for sniping
                    "suggested_gas": "high"
                }
            
            elif wallet_type == "flipper":
                strategy = {
                    "recommendation": "COPY_FLIPPER",
                    "avg_first_take_profit": avg_first_tp if avg_first_tp > 0 else 30,
                    "confidence": "HIGH" if composite_score >= 60 else "MEDIUM",
                    "notes": f"Flipper with {win_rate:.1f}% win rate. Exit at {avg_first_tp:.1f}%.",
                    "filter_market_cap_min": filter_min,
                    "filter_market_cap_max": filter_max,
                    "suggested_slippage": 20,
                    "suggested_gas": "high"
                }
            
            elif wallet_type == "scalper":
                strategy = {
                    "recommendation": "COPY_SCALPER",
                    "avg_first_take_profit": avg_first_tp if avg_first_tp > 0 else 35,
                    "confidence": "HIGH" if composite_score >= 60 else "MEDIUM",
                    "notes": f"Scalper taking {avg_first_tp:.1f}% profits. Consistent wins.",
                    "filter_market_cap_min": filter_min,
                    "filter_market_cap_max": filter_max,
                    "suggested_slippage": 15,
                    "suggested_gas": "medium"
                }
            
            elif wallet_type == "gem_hunter":
                strategy = {
                    "recommendation": "COPY_GEM_HUNTER",
                    "avg_first_take_profit": 400,  # Hold for 5x minimum
                    "confidence": "VERY_HIGH" if composite_score >= 70 else "HIGH",
                    "notes": f"5x+ Gem hunter with {gem_rate_5x:.1f}% success finding 5x+. Hold for moonshots.",
                    "filter_market_cap_min": filter_min,
                    "filter_market_cap_max": filter_max,
                    "suggested_slippage": 15,
                    "suggested_gas": "medium"
                }
            
            elif wallet_type in ["swing_trader", "position_trader"]:
                strategy = {
                    "recommendation": "COPY_POSITION",
                    "avg_first_take_profit": avg_first_tp if avg_first_tp > 0 else 100,
                    "confidence": "HIGH" if composite_score >= 60 else "MEDIUM",
                    "notes": f"Position trader. Consider holding for bigger gains.",
                    "filter_market_cap_min": filter_min,
                    "filter_market_cap_max": filter_max,
                    "suggested_slippage": 10,
                    "suggested_gas": "low"
                }
            
            else:
                strategy = {
                    "recommendation": "CAUTIOUS" if composite_score < 40 else "COPY_SELECTIVE",
                    "avg_first_take_profit": avg_first_tp if avg_first_tp > 0 else 30,
                    "confidence": "LOW" if composite_score < 40 else "MEDIUM",
                    "notes": f"Mixed results. Score: {composite_score:.1f}/100.",
                    "filter_market_cap_min": filter_min,
                    "filter_market_cap_max": filter_max,
                    "suggested_slippage": 15,
                    "suggested_gas": "medium"
                }
            
            # Adjust confidence based on score
            if composite_score >= 80:
                strategy["confidence"] = "VERY_HIGH"
            elif composite_score >= 60:
                strategy["confidence"] = "HIGH"
            elif composite_score >= 40:
                strategy["confidence"] = "MEDIUM"
            elif composite_score >= 20:
                strategy["confidence"] = "LOW"
            else:
                strategy["confidence"] = "VERY_LOW"
            
            # Add warning for low activity
            if metrics.get("total_trades", 0) < 10:
                strategy["notes"] += " ⚠️ Low trade count - less reliable."
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return {
                "recommendation": "DO_NOT_COPY",
                "avg_first_take_profit": 0,
                "confidence": "VERY_LOW",
                "notes": "Error during strategy generation",
                "filter_market_cap_min": 0,
                "filter_market_cap_max": 0
            }
    
    def _analyze_memecoin_entry_exit(self, analyzed_trades: List[Dict[str, Any]], 
                                   metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entry/exit behavior with proper quality assessment."""
        if not analyzed_trades:
            return {
                "pattern": "INSUFFICIENT_DATA",
                "entry_quality": "UNKNOWN",
                "exit_quality": "UNKNOWN",
                "missed_gains_percent": 0,
                "early_exit_rate": 0,
                "avg_exit_roi": 0,
                "hold_pattern": "UNKNOWN"
            }
        
        try:
            # Analyze entry quality
            entry_scores = []
            exit_scores = []
            missed_gains_list = []
            early_exits = 0
            exit_rois = []
            
            for trade in analyzed_trades:
                # Entry quality based on subsequent performance
                entry_timing = trade.get("entry_timing", "UNKNOWN")
                if entry_timing == "EXCELLENT":
                    entry_scores.append(100)
                elif entry_timing == "GOOD":
                    entry_scores.append(75)
                elif entry_timing == "AVERAGE":
                    entry_scores.append(50)
                elif entry_timing == "POOR":
                    entry_scores.append(25)
                
                # Exit quality and missed gains
                exit_timing = trade.get("exit_timing", "UNKNOWN")
                max_roi = trade.get("max_roi_percent", 0)
                exit_roi = trade.get("roi_percent", 0)
                
                if exit_timing != "HOLDING" and max_roi > 0:
                    # Calculate missed gains
                    missed = max_roi - exit_roi
                    if missed > 0:
                        missed_gains_list.append(missed)
                    
                    # Track exit ROIs
                    exit_rois.append(exit_roi)
                    
                    # Check for early exit (sold before reaching potential)
                    if exit_roi < 50 and max_roi > 100:
                        early_exits += 1
                    
                    # Exit quality scoring
                    if exit_timing == "EXCELLENT":
                        exit_scores.append(100)
                    elif exit_timing == "GOOD":
                        exit_scores.append(75)
                    elif exit_timing == "AVERAGE":
                        exit_scores.append(50)
                    elif exit_timing == "POOR":
                        exit_scores.append(25)
            
            # Calculate averages
            avg_entry_score = np.mean(entry_scores) if entry_scores else 50
            avg_exit_score = np.mean(exit_scores) if exit_scores else 50
            avg_missed_gains = np.mean(missed_gains_list) if missed_gains_list else 0
            avg_exit_roi = np.mean(exit_rois) if exit_rois else 0
            
            # Determine quality labels
            entry_quality = "GOOD" if avg_entry_score >= 60 else "POOR"
            exit_quality = "GOOD" if avg_exit_score >= 60 else "POOR"
            
            # Adjust based on actual performance
            composite_score = metrics.get("composite_score", 0)
            if composite_score >= 70:
                # High performing wallet should have at least average quality
                if entry_quality == "POOR":
                    entry_quality = "AVERAGE"
                if exit_quality == "POOR" and avg_missed_gains < 50:
                    exit_quality = "AVERAGE"
            
            # Calculate early exit rate
            trades_with_exit = len([t for t in analyzed_trades if t.get("exit_timing") != "HOLDING"])
            early_exit_rate = (early_exits / trades_with_exit * 100) if trades_with_exit > 0 else 0
            
            # Determine pattern
            if avg_missed_gains > 200:
                pattern = "LEAVES_MONEY"  # Exits way too early
            elif early_exit_rate > 50:
                pattern = "EARLY_SELLER"
            elif avg_exit_roi > 100:
                pattern = "DIAMOND_HANDS"  # Holds for big gains
            elif avg_exit_roi > 50:
                pattern = "BALANCED"
            else:
                pattern = "QUICK_PROFITS"
            
            # Determine hold pattern
            avg_hold_minutes = metrics.get("avg_hold_time_minutes", 0)
            if avg_hold_minutes < 1:
                hold_pattern = "ULTRA_FAST"
            elif avg_hold_minutes < 10:
                hold_pattern = "FAST"
            elif avg_hold_minutes < 60:
                hold_pattern = "MEDIUM"
            elif avg_hold_minutes < 1440:
                hold_pattern = "LONG"
            else:
                hold_pattern = "VERY_LONG"
            
            return {
                "pattern": pattern,
                "entry_quality": entry_quality,
                "exit_quality": exit_quality,
                "missed_gains_percent": round(avg_missed_gains, 1),
                "early_exit_rate": round(early_exit_rate, 1),
                "avg_exit_roi": round(avg_exit_roi, 1),
                "hold_pattern": hold_pattern,
                "trades_analyzed": len(analyzed_trades),
                "avg_entry_score": round(avg_entry_score, 1),
                "avg_exit_score": round(avg_exit_score, 1)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing entry/exit: {str(e)}")
            return {
                "pattern": "ERROR",
                "entry_quality": "UNKNOWN",
                "exit_quality": "UNKNOWN",
                "missed_gains_percent": 0,
                "early_exit_rate": 0,
                "avg_exit_roi": 0,
                "hold_pattern": "UNKNOWN"
            }
    
    def _detect_bundle_activity(self, analyzed_trades: List[Dict[str, Any]], 
                               recent_swaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect potential bundle and copytrader activity."""
        try:
            # Check for bundle indicators
            bundle_indicators = 0
            
            # 1. Check for consistent buy amounts (bundles often use fixed amounts)
            buy_amounts = [s.get("sol_amount", 0) for s in recent_swaps if s.get("type") == "buy"]
            if buy_amounts:
                amount_variance = np.std(buy_amounts) / np.mean(buy_amounts) if np.mean(buy_amounts) > 0 else 1
                if amount_variance < 0.1:  # Very consistent amounts
                    bundle_indicators += 1
            
            # 2. Check for rapid succession trades (multiple buys within seconds)
            buy_timestamps = sorted([s.get("buy_timestamp", 0) for s in recent_swaps if s.get("buy_timestamp")])
            rapid_buys = 0
            for i in range(1, len(buy_timestamps)):
                if buy_timestamps[i] - buy_timestamps[i-1] < 5:  # Within 5 seconds
                    rapid_buys += 1
            
            if rapid_buys >= 3:
                bundle_indicators += 1
            
            # 3. Check for similar sell patterns (dumps at similar times)
            sell_timestamps = sorted([s.get("sell_timestamp", 0) for s in recent_swaps if s.get("sell_timestamp")])
            rapid_sells = 0
            for i in range(1, len(sell_timestamps)):
                if sell_timestamps[i] - sell_timestamps[i-1] < 10:  # Within 10 seconds
                    rapid_sells += 1
            
            if rapid_sells >= 3:
                bundle_indicators += 1
            
            # Determine if likely bundler
            is_likely_bundler = bundle_indicators >= 2
            
            # Estimate copytraders (this would need on-chain analysis for accuracy)
            # For now, we'll use heuristics
            avg_profit = np.mean([t.get("roi_percent", 0) for t in analyzed_trades])
            if avg_profit > 50 and len(analyzed_trades) > 20:
                estimated_copytraders = "10+"  # Successful traders usually have followers
            elif avg_profit > 20 and len(analyzed_trades) > 10:
                estimated_copytraders = "5-10"
            else:
                estimated_copytraders = "0-5"
            
            return {
                "is_likely_bundler": is_likely_bundler,
                "bundle_indicators": bundle_indicators,
                "rapid_succession_buys": rapid_buys,
                "rapid_succession_sells": rapid_sells,
                "buy_amount_consistency": "HIGH" if amount_variance < 0.1 else "LOW",
                "estimated_copytraders": estimated_copytraders,
                "warning": "⚠️ Possible bundler - verify on-chain" if is_likely_bundler else ""
            }
            
        except Exception as e:
            logger.error(f"Error detecting bundle activity: {str(e)}")
            return {
                "is_likely_bundler": False,
                "bundle_indicators": 0,
                "estimated_copytraders": "UNKNOWN"
            }
    
    def _extract_aggregated_metrics_from_cielo(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract aggregated metrics from Cielo Finance response."""
        try:
            if not isinstance(stats_data, dict):
                logger.warning(f"Unexpected Cielo data format: {type(stats_data)}")
                return self._get_empty_cielo_metrics()
            
            logger.debug(f"Cielo Finance response structure: {list(stats_data.keys())[:10]}")
            
            # Extract basic metrics
            total_trades = stats_data.get("swaps_count", 0)
            win_rate = stats_data.get("winrate", 0)
            total_pnl_usd = stats_data.get("pnl", 0)
            total_buy_amount_usd = stats_data.get("total_buy_amount_usd", 0)
            total_sell_amount_usd = stats_data.get("total_sell_amount_usd", 0)
            
            # Extract hold time (convert seconds to minutes)
            avg_hold_time_minutes = 0
            if "average_holding_time_sec" in stats_data:
                avg_hold_time_minutes = stats_data.get("average_holding_time_sec", 0) / 60
            
            # Calculate profit factor
            # Get profit and loss data
            total_profits = 0
            total_losses = 0
            
            # Try to get from roi distribution
            roi_dist = stats_data.get("roi_distribution", {})
            if roi_dist:
                # Positive ROI trades contribute to profits
                profitable_trades = (
                    roi_dist.get("roi_above_500", 0) +
                    roi_dist.get("roi_200_to_500", 0) +
                    roi_dist.get("roi_0_to_200", 0)
                )
                # Negative ROI trades contribute to losses
                loss_trades = (
                    roi_dist.get("roi_neg50_to_0", 0) +
                    roi_dist.get("roi_below_neg50", 0)
                )
                
                # Estimate based on PnL and trade counts
                if total_pnl_usd > 0 and profitable_trades > 0:
                    avg_profit_per_trade = total_pnl_usd / profitable_trades
                    total_profits = avg_profit_per_trade * profitable_trades
                elif total_sell_amount_usd > total_buy_amount_usd:
                    total_profits = total_sell_amount_usd - total_buy_amount_usd
                
                if loss_trades > 0:
                    # Assume average loss is proportional to total trades
                    avg_loss_estimate = abs(total_pnl_usd) / total_trades if total_trades > 0 else 100
                    total_losses = avg_loss_estimate * loss_trades
            
            # Alternative calculation if no roi distribution
            if total_profits == 0 and total_losses == 0:
                if total_pnl_usd > 0:
                    total_profits = total_pnl_usd
                    total_losses = 1  # Avoid division by zero
                else:
                    total_profits = 1
                    total_losses = abs(total_pnl_usd) if total_pnl_usd < 0 else 1
            
            # Calculate profit factor
            profit_factor = total_profits / total_losses if total_losses > 0 else total_profits
            
            # Calculate average ROI
            avg_roi = 0
            if total_buy_amount_usd > 0:
                avg_roi = (total_pnl_usd / total_buy_amount_usd) * 100
            
            metrics = {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": round(profit_factor, 2),
                "net_profit_usd": total_pnl_usd,
                "total_pnl_usd": total_pnl_usd,
                "avg_trade_size": stats_data.get("average_buy_amount_usd", 0),
                "total_volume": total_buy_amount_usd + total_sell_amount_usd,
                "best_trade": 0,
                "worst_trade": 0,
                "avg_hold_time": avg_hold_time_minutes / 60,  # Convert to hours
                "avg_hold_time_minutes": round(avg_hold_time_minutes, 2),
                "tokens_traded": stats_data.get("buy_count", 0),
                "total_tokens_traded": stats_data.get("buy_count", 0),
                "total_invested": total_buy_amount_usd,
                "total_realized": total_sell_amount_usd,
                "buy_count": stats_data.get("buy_count", 0),
                "sell_count": stats_data.get("sell_count", 0),
                "consecutive_trading_days": stats_data.get("consecutive_trading_days", 0),
                "roi_distribution": roi_dist,
                "holding_distribution": stats_data.get("holding_distribution", {}),
                "avg_roi": round(avg_roi, 2),
                "max_roi": 0,
                "median_roi": 0
            }
            
            # Estimate best/worst trades from ROI distribution
            if roi_dist.get("roi_above_500", 0) > 0:
                metrics["best_trade"] = 500
                metrics["max_roi"] = 500
            elif roi_dist.get("roi_200_to_500", 0) > 0:
                metrics["best_trade"] = 200
                metrics["max_roi"] = 200
            elif roi_dist.get("roi_0_to_200", 0) > 0:
                metrics["best_trade"] = 100
                metrics["max_roi"] = 100
            else:
                metrics["best_trade"] = 50
                metrics["max_roi"] = 50
            
            if roi_dist.get("roi_below_neg50", 0) > 0:
                metrics["worst_trade"] = -50
            elif roi_dist.get("roi_neg50_to_0", 0) > 0:
                metrics["worst_trade"] = -25
            else:
                metrics["worst_trade"] = 0
            
            logger.debug(f"Extracted Cielo metrics: trades={metrics['total_trades']}, "
                        f"win_rate={metrics['win_rate']}, pnl={metrics['total_pnl_usd']}, "
                        f"profit_factor={metrics['profit_factor']}, "
                        f"avg_hold_time_minutes={metrics['avg_hold_time_minutes']}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting Cielo metrics: {str(e)}")
            return self._get_empty_cielo_metrics()
    
    def _get_empty_cielo_metrics(self) -> Dict[str, Any]:
        """Return empty Cielo metrics structure."""
        return {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "net_profit_usd": 0,
            "total_pnl_usd": 0,
            "avg_trade_size": 0,
            "total_volume": 0,
            "best_trade": 0,
            "worst_trade": 0,
            "avg_hold_time": 0,
            "avg_hold_time_minutes": 0,
            "tokens_traded": 0,
            "total_tokens_traded": 0,
            "total_invested": 0,
            "total_realized": 0,
            "roi_distribution": {},
            "holding_distribution": {},
            "avg_roi": 0,
            "max_roi": 0,
            "median_roi": 0
        }
    
    def _get_recent_token_swaps_rpc(self, wallet_address: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent token swaps using RPC calls with better rate limiting."""
        try:
            logger.info(f"Fetching last {limit} token swaps for {wallet_address}")
            
            signatures = self._get_signatures_for_address(wallet_address, limit=100)  # Get more to filter
            
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
            "avg_hold_time_minutes": 0,
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
            },
            "distribution_500_plus_%": 0,
            "distribution_200_500_%": 0,
            "distribution_0_200_%": 0,
            "distribution_neg50_0_%": 0,
            "distribution_below_neg50_%": 0,
            "gem_rate_2x_plus": 0,
            "gem_rate_5x_plus": 0,
            "avg_buy_market_cap_usd": 0,
            "avg_buy_amount_usd": 0,
            "avg_first_take_profit_percent": 0
        }
    
    def batch_analyze_wallets(self, wallet_addresses: List[str], 
                            days_back: int = 30,
                            min_winrate: float = 30.0,
                            use_hybrid: bool = True) -> Dict[str, Any]:
        """Batch analyze multiple wallets with 3-tier scanning."""
        logger.info(f"Batch analyzing {len(wallet_addresses)} wallets with 3-tier memecoin analysis")
        
        if not wallet_addresses:
            return {
                "success": False,
                "error": "No wallet addresses provided",
                "error_type": "NO_INPUT"
            }
        
        try:
            wallet_analyses = []
            failed_analyses = []
            
            # Reset API call statistics
            self.api_call_stats = {
                "cielo": 0,
                "birdeye": 0,
                "helius": 0,
                "rpc": 0
            }
            
            # Track tier distribution
            tier_counts = {"INITIAL": 0, "STANDARD": 0, "DEEP": 0}
            
            for i, wallet_address in enumerate(wallet_addresses, 1):
                logger.info(f"Analyzing wallet {i}/{len(wallet_addresses)}: {wallet_address}")
                
                try:
                    if use_hybrid:
                        analysis = self.analyze_wallet_hybrid(wallet_address, days_back)
                    else:
                        analysis = self.analyze_wallet(wallet_address, days_back)
                    
                    if "metrics" in analysis:
                        wallet_analyses.append(analysis)
                        score = analysis.get("composite_score", analysis.get("metrics", {}).get("composite_score", 0))
                        tier = analysis.get("analysis_tier", "INITIAL")
                        tier_counts[tier] = tier_counts.get(tier, 0) + 1
                        logger.info(f"  └─ Score: {score}/100, Type: {analysis.get('wallet_type', 'unknown')}, Tier: {tier}")
                    else:
                        failed_analyses.append({
                            "wallet_address": wallet_address,
                            "error": analysis.get("error", "No metrics available"),
                            "error_type": analysis.get("error_type", "NO_METRICS")
                        })
                    
                    if i < len(wallet_addresses):
                        time.sleep(2)
                        
                except Exception as e:
                    logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
                    failed_analyses.append({
                        "wallet_address": wallet_address,
                        "error": str(e),
                        "error_type": "ANALYSIS_ERROR"
                    })
            
            if not wallet_analyses:
                return {
                    "success": False,
                    "error": "No wallets could be successfully analyzed",
                    "failed_analyses": failed_analyses,
                    "error_type": "ALL_FAILED"
                }
            
            # Categorize wallets by memecoin types
            snipers = [a for a in wallet_analyses if a.get("wallet_type") == "sniper"]
            flippers = [a for a in wallet_analyses if a.get("wallet_type") == "flipper"]
            scalpers = [a for a in wallet_analyses if a.get("wallet_type") == "scalper"]
            gem_hunters = [a for a in wallet_analyses if a.get("wallet_type") == "gem_hunter"]
            swing_traders = [a for a in wallet_analyses if a.get("wallet_type") == "swing_trader"]
            position_traders = [a for a in wallet_analyses if a.get("wallet_type") == "position_trader"]
            consistent = [a for a in wallet_analyses if a.get("wallet_type") == "consistent"]
            mixed = [a for a in wallet_analyses if a.get("wallet_type") == "mixed"]
            unknown = [a for a in wallet_analyses if a.get("wallet_type") == "unknown"]
            
            # Sort each category by composite score
            for category in [snipers, flippers, scalpers, gem_hunters, swing_traders, 
                           position_traders, consistent, mixed, unknown]:
                category.sort(key=lambda x: x.get("composite_score", x.get("metrics", {}).get("composite_score", 0)), reverse=True)
            
            # Calculate tier distribution percentages
            total_analyzed = len(wallet_analyses)
            tier_percentages = {
                tier: (count / total_analyzed * 100) if total_analyzed > 0 else 0
                for tier, count in tier_counts.items()
            }
            
            # Log final API call statistics
            logger.info(f"📊 FINAL API CALL STATISTICS:")
            logger.info(f"   Cielo: {self.api_call_stats['cielo']} calls")
            logger.info(f"   Birdeye: {self.api_call_stats['birdeye']} calls")
            logger.info(f"   Helius: {self.api_call_stats['helius']} calls")
            logger.info(f"   RPC: {self.api_call_stats['rpc']} calls")
            logger.info(f"   Total API calls: {sum(self.api_call_stats.values())}")
            logger.info(f"📊 TIER DISTRIBUTION:")
            logger.info(f"   INITIAL (5 tokens): {tier_counts['INITIAL']} wallets ({tier_percentages['INITIAL']:.1f}%)")
            logger.info(f"   STANDARD (10 tokens): {tier_counts['STANDARD']} wallets ({tier_percentages['STANDARD']:.1f}%)")
            logger.info(f"   DEEP (20 tokens): {tier_counts['DEEP']} wallets ({tier_percentages['DEEP']:.1f}%)")
            
            return {
                "success": True,
                "total_wallets": len(wallet_addresses),
                "analyzed_wallets": len(wallet_analyses),
                "failed_wallets": len(failed_analyses),
                "filtered_wallets": len(wallet_analyses),
                "snipers": snipers,
                "flippers": flippers,
                "scalpers": scalpers,
                "gem_hunters": gem_hunters,
                "swing_traders": swing_traders,
                "position_traders": position_traders,
                "consistent": consistent,
                "mixed": mixed,
                "unknown": unknown,
                "failed_analyses": failed_analyses,
                "api_source": "Hybrid (Cielo + RPC + Birdeye/Helius)" if use_hybrid else "Cielo Finance + RPC",
                "api_calls": self.api_call_stats.copy(),
                "tier_distribution": tier_counts,
                "tier_percentages": tier_percentages
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
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)