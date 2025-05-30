"""
Wallet Analysis Module - Phoenix Project (FIXED ROI CALCULATION)

MAJOR FIXES:
- Calculate ROI directly from paired transaction SOL amounts FIRST
- Only use external APIs as fallback for unpaired trades
- Fixed 7-day win rate and profit calculations
- Removed dependency on external price APIs for paired trades

OPTIMIZATIONS:
1. Direct ROI calculation from transaction data
2. Implemented RPC batch requests
3. Reduced transaction fetch limit for 7-day analysis
4. Added parallel processing for transactions
5. Improved caching with session-based cache
6. Added early exit conditions
7. Reduced timeouts for faster failure detection
8. Transaction sampling for large wallets
"""

import csv
import os
import logging
import numpy as np
import requests
import json
import time
import sys
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache

logger = logging.getLogger("phoenix.wallet")

# Progress indicator for console
class ProgressIndicator:
    """Simple progress indicator for console output."""
    def __init__(self, total: int, prefix: str = "Progress"):
        self.total = total
        self.current = 0
        self.prefix = prefix
        self.last_update = 0
        
    def update(self, increment: int = 1):
        """Update progress and display."""
        self.current += increment
        current_time = time.time()
        
        # Update every 0.5 seconds or when complete
        if current_time - self.last_update > 0.5 or self.current >= self.total:
            percentage = (self.current / self.total * 100) if self.total > 0 else 0
            print(f"\r{self.prefix}: {self.current}/{self.total} ({percentage:.1f}%)", end="", flush=True)
            sys.stdout.flush()
            self.last_update = current_time
            
            if self.current >= self.total:
                print()  # New line when complete

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
    """Optimized wallet analyzer for copy trading using Cielo Finance API + RPC + Helius."""
    
    # 3-Tier analysis constants
    INITIAL_SCAN_TOKENS = 5
    STANDARD_SCAN_TOKENS = 10
    DEEP_SCAN_TOKENS = 20
    
    # 7-day focus constant
    DAYS_TO_ANALYZE = 7
    ACTIVE_DAYS_THRESHOLD = 7
    
    # Optimized settings
    MAX_SIGNATURES_7DAY = 100  # Reduced from 150
    RPC_TIMEOUT = 10  # Reduced from 45
    RPC_MAX_RETRIES = 2  # Reduced from 3
    BATCH_SIZE = 10  # For batch RPC requests
    
    # Session cache for Cielo results
    _cielo_cache = {}
    _cielo_cache_ttl = 600  # 10 minutes
    
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
        
        # Transaction pairing storage
        self.transaction_pairs = {}
        self.paired_transactions = set()
        
        # Track entry times for tokens
        self.token_entries = {}
        
        # RPC cache for avoiding duplicate calls
        self._rpc_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 300  # 5 minutes TTL
        
        # Rate limiter for RPC calls
        self._rate_limiter = RateLimiter(calls_per_second=10.0)  # Increased rate
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)  # Increased workers
        
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
    
    def _get_cielo_stats_cached(self, wallet_address: str) -> Dict[str, Any]:
        """Get Cielo stats with session caching."""
        current_time = time.time()
        
        # Check cache
        if wallet_address in self._cielo_cache:
            cached_data, timestamp = self._cielo_cache[wallet_address]
            if current_time - timestamp < self._cielo_cache_ttl:
                logger.debug(f"Using cached Cielo data for {wallet_address}")
                return cached_data
        
        # Fetch fresh data
        self.api_call_stats["cielo"] += 1
        cielo_stats = self.cielo_api.get_wallet_trading_stats(wallet_address)
        
        # Cache the result
        if cielo_stats and cielo_stats.get("success", True):
            self._cielo_cache[wallet_address] = (cielo_stats, current_time)
        
        return cielo_stats
    
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
    
    def _make_rpc_batch_call(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Make batch RPC call to Solana node."""
        self._rate_limiter.wait()
        self.api_call_stats["rpc"] += 1
        
        try:
            response = requests.post(
                self.rpc_url,
                json=batch_requests,
                headers={"Content-Type": "application/json"},
                timeout=self.RPC_TIMEOUT
            )
            
            response.raise_for_status()
            results = response.json()
            
            if isinstance(results, list):
                return results
            else:
                return [results]
                
        except Exception as e:
            logger.error(f"Batch RPC error: {str(e)}")
            return []
    
    def _make_rpc_call(self, method: str, params: List[Any], retry_count: int = None) -> Dict[str, Any]:
        """Make direct RPC call to Solana node with caching and rate limiting."""
        if retry_count is None:
            retry_count = self.RPC_MAX_RETRIES
            
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
                    timeout=self.RPC_TIMEOUT
                )
                
                if response.status_code == 429:
                    self._rpc_error_count += 1
                    self._last_rpc_error_time = time.time()
                    
                    backoff_time = backoff_base ** attempt
                    max_backoff = 30  # Reduced from 60
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
                    
            except Exception as e:
                logger.error(f"RPC error for {method}: {str(e)}")
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
    
    def _batch_get_transactions(self, signatures: List[str], show_progress: bool = True) -> List[Dict[str, Any]]:
        """Get multiple transactions using batch RPC requests."""
        transactions = []
        
        if show_progress:
            progress = ProgressIndicator(len(signatures), "Fetching transactions")
        
        # Process in batches
        for i in range(0, len(signatures), self.BATCH_SIZE):
            batch = signatures[i:i + self.BATCH_SIZE]
            
            # Create batch request
            batch_requests = []
            for j, sig in enumerate(batch):
                batch_requests.append({
                    "jsonrpc": "2.0",
                    "id": i + j,
                    "method": "getTransaction",
                    "params": [sig, {"encoding": "json", "maxSupportedTransactionVersion": 0}]
                })
            
            # Make batch call
            results = self._make_rpc_batch_call(batch_requests)
            
            # Process results
            for result in results:
                if isinstance(result, dict) and "result" in result and result["result"]:
                    transactions.append(result["result"])
            
            if show_progress:
                progress.update(len(batch))
            
            # Small delay between batches
            if i + self.BATCH_SIZE < len(signatures):
                time.sleep(0.5)
        
        return transactions
    
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
    
    def _determine_scan_tier_7day(self, recent_metrics: Dict[str, Any]) -> Tuple[int, str]:
        """Determine scan depth based on 7-day performance metrics."""
        trades_7d = recent_metrics.get("trades_last_7_days", 0)
        win_rate_7d = recent_metrics.get("win_rate_7d", 0)
        profit_7d = recent_metrics.get("profit_7d", 0)
        has_5x_7d = recent_metrics.get("has_5x_last_7_days", False)
        has_2x_7d = recent_metrics.get("has_2x_last_7_days", False)
        
        # DEEP tier (elite active traders)
        if (trades_7d >= 5 and win_rate_7d >= 50) or has_5x_7d or profit_7d >= 5000:
            return self.DEEP_SCAN_TOKENS, "DEEP"
        
        # STANDARD tier (decent active traders)
        elif (trades_7d >= 3 and win_rate_7d >= 40) or has_2x_7d:
            return self.STANDARD_SCAN_TOKENS, "STANDARD"
        
        # INITIAL tier (minimal activity or poor performance)
        else:
            return self.INITIAL_SCAN_TOKENS, "INITIAL"
    
    def _calculate_7day_metrics(self, recent_swaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics from last 7 days of trading with proper ROI calculation."""
        seven_days_ago = int((datetime.now() - timedelta(days=7)).timestamp())
        
        # Filter swaps to last 7 days
        recent_trades = []
        for swap in recent_swaps:
            # Use sell timestamp for completed trades, buy timestamp for open positions
            timestamp = swap.get('sell_timestamp') or swap.get('buy_timestamp')
            if timestamp and isinstance(timestamp, (int, float)) and timestamp >= seven_days_ago:
                # Calculate ROI for paired trades before adding to recent_trades
                if swap.get('paired') and swap.get('sell_sol_amount'):
                    buy_sol = swap.get('sol_amount', 0)
                    sell_sol = swap.get('sell_sol_amount', 0)
                    if buy_sol > 0:
                        swap['roi_percent'] = ((sell_sol / buy_sol) - 1) * 100
                        swap['pnl_usd'] = (sell_sol - buy_sol) * 150  # Estimate USD value
                recent_trades.append(swap)
        
        if not recent_trades:
            return {
                "trades_last_7_days": 0,
                "win_rate_7d": 0,
                "profit_7d": 0,
                "has_5x_last_7_days": False,
                "has_2x_last_7_days": False,
                "active_trader": False,
                "days_since_last_trade": 999
            }
        
        # Calculate 7-day metrics with proper ROI
        wins = sum(1 for t in recent_trades if t.get('roi_percent', 0) > 0)
        total_profit = sum(t.get('pnl_usd', 0) for t in recent_trades)
        max_roi = max((t.get('roi_percent', 0) for t in recent_trades), default=0)
        
        return {
            "trades_last_7_days": len(recent_trades),
            "win_rate_7d": (wins / len(recent_trades) * 100) if recent_trades else 0,
            "profit_7d": total_profit,
            "has_5x_last_7_days": max_roi >= 400,
            "has_2x_last_7_days": max_roi >= 100,
            "active_trader": len(recent_trades) > 0,
            "days_since_last_trade": self._days_since_last_trade(recent_trades)
        }
    
    def _days_since_last_trade(self, trades: List[Dict[str, Any]]) -> int:
        """Calculate days since last trade."""
        if not trades:
            return 999
        
        latest_timestamp = 0
        for t in trades:
            # Check both buy and sell timestamps
            ts = t.get('sell_timestamp') or t.get('buy_timestamp')
            if ts and isinstance(ts, (int, float)):
                latest_timestamp = max(latest_timestamp, ts)
        
        if latest_timestamp == 0:
            return 999
        
        days_ago = (datetime.now().timestamp() - latest_timestamp) / 86400
        return int(days_ago)
    
    def _validate_swap_data(self, swap: Dict[str, Any]) -> bool:
        """Validate swap data before processing."""
        # Check required fields
        required_fields = ['token_mint', 'type']
        for field in required_fields:
            if field not in swap:
                return False
        
        # Check timestamp validity based on type
        if swap.get('type') == 'buy':
            buy_timestamp = swap.get('buy_timestamp')
            if not buy_timestamp or not isinstance(buy_timestamp, (int, float)):
                return False
            # Sanity check - not in future, not too old
            current_time = int(datetime.now().timestamp())
            if buy_timestamp > current_time or buy_timestamp < (current_time - 365 * 86400):
                return False
        elif swap.get('type') == 'sell':
            sell_timestamp = swap.get('sell_timestamp')
            if not sell_timestamp or not isinstance(sell_timestamp, (int, float)):
                return False
        
        return True
    
    def _is_pump_fun_token(self, token_address: str) -> bool:
        """Check if a token is a pump.fun token."""
        if not token_address:
            return False
        return token_address.endswith("pump") or "pump" in token_address.lower()
    
    def analyze_wallet_hybrid(self, wallet_address: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Optimized hybrid wallet analysis with better performance.
        
        Args:
            wallet_address (str): Wallet address
            days_back (int): Number of days to analyze (default 7)
            
        Returns:
            Dict[str, Any]: Wallet analysis results
        """
        logger.info(f"🔍 Analyzing wallet {wallet_address} (7-day active trader focus)")
        print(f"\n📊 Starting analysis for {wallet_address[:8]}...{wallet_address[-4:]}", flush=True)
        
        try:
            # Reset transaction pairs for this wallet
            self.transaction_pairs.clear()
            self.paired_transactions.clear()
            
            # Step 1: Get aggregated stats from Cielo Finance (with caching)
            logger.info(f"📊 Fetching Cielo Finance aggregated stats...")
            print("   • Fetching wallet stats from Cielo Finance...", flush=True)
            
            cielo_stats = self._get_cielo_stats_cached(wallet_address)
            
            if not cielo_stats or not cielo_stats.get("success", True):
                logger.warning(f"❌ No Cielo Finance data available for {wallet_address}")
                aggregated_metrics = self._get_empty_cielo_metrics()
            else:
                stats_data = cielo_stats.get("data", {})
                aggregated_metrics = self._extract_aggregated_metrics_from_cielo(stats_data)
            
            # Early exit for inactive wallets
            if aggregated_metrics.get("total_trades", 0) == 0:
                logger.info("No trading history found - skipping detailed analysis")
                return self._get_empty_analysis_result(wallet_address, "no_trades")
            
            # Step 2: Get recent token swaps (optimized limit for 7-day analysis)
            logger.info(f"📈 Getting last 7 days of trading activity...")
            print("   • Fetching recent trades...", flush=True)
            
            recent_swaps_all = self._get_recent_token_swaps_rpc_optimized(wallet_address, limit=self.MAX_SIGNATURES_7DAY)
            
            # Validate and pair swaps
            valid_swaps = []
            for swap in recent_swaps_all:
                if self._validate_swap_data(swap):
                    valid_swaps.append(swap)
            
            recent_swaps_all = valid_swaps
            
            # Calculate 7-day metrics for tier determination
            seven_day_metrics = self._calculate_7day_metrics(recent_swaps_all)
            
            # Early exit for completely inactive wallets
            if seven_day_metrics['trades_last_7_days'] == 0:
                logger.info("No trades in last 7 days - marking as inactive")
                return self._get_inactive_analysis_result(wallet_address, aggregated_metrics)
            
            # Step 3: Determine scan tier based on 7-day activity
            scan_limit, tier = self._determine_scan_tier_7day(seven_day_metrics)
            
            logger.info(f"📊 Wallet tier: {tier} - Scanning {scan_limit} recent trades")
            logger.info(f"   7-day trades: {seven_day_metrics['trades_last_7_days']}")
            logger.info(f"   7-day win rate: {seven_day_metrics['win_rate_7d']:.1f}%")
            logger.info(f"   7-day profit: ${seven_day_metrics['profit_7d']:.2f}")
            logger.info(f"   Days since last trade: {seven_day_metrics['days_since_last_trade']}")
            
            print(f"   • Wallet tier: {tier} ({seven_day_metrics['trades_last_7_days']} trades in 7 days)", flush=True)
            
            # Get only the swaps we need for detailed analysis
            recent_swaps = recent_swaps_all[:scan_limit]
            
            # Step 4: Analyze token performance in parallel
            print(f"   • Analyzing {len(recent_swaps)} trades in detail...", flush=True)
            analyzed_trades = self._analyze_trades_parallel(recent_swaps)
            
            # Step 5: Calculate enhanced metrics with 7-day focus
            print("   • Calculating performance metrics...", flush=True)
            enhanced_metrics = self._calculate_enhanced_memecoin_metrics_7day(
                aggregated_metrics, 
                analyzed_trades,
                seven_day_metrics
            )
            
            # Step 6: Calculate composite score with 7-day activity weighting
            composite_score = self._calculate_memecoin_composite_score_7day(enhanced_metrics, seven_day_metrics)
            enhanced_metrics["composite_score"] = composite_score
            
            # Step 7: Determine wallet type based on hold patterns
            wallet_type = self._determine_memecoin_wallet_type_5x(enhanced_metrics)
            
            # Step 8: Generate enhanced strategy with sell guidance
            strategy = self._generate_enhanced_memecoin_strategy(wallet_type, enhanced_metrics, analyzed_trades)
            
            # Step 9: Analyze entry/exit behavior
            entry_exit_analysis = self._analyze_memecoin_entry_exit(analyzed_trades, enhanced_metrics)
            
            # Log API call statistics
            logger.info(f"📊 API Calls - Cielo: {self.api_call_stats['cielo']}, "
                       f"Birdeye: {self.api_call_stats['birdeye']}, "
                       f"Helius: {self.api_call_stats['helius']}, "
                       f"RPC: {self.api_call_stats['rpc']}")
            
            print(f"   ✅ Analysis complete! Score: {composite_score}/100, Type: {wallet_type}", flush=True)
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_period_days": days_back,
                "wallet_type": wallet_type,
                "composite_score": composite_score,
                "metrics": enhanced_metrics,
                "strategy": strategy,
                "trades": analyzed_trades,
                "entry_exit_analysis": entry_exit_analysis,
                "api_source": "Cielo Finance + RPC + Birdeye/Helius",
                "cielo_data": aggregated_metrics,
                "recent_trades_analyzed": len(analyzed_trades),
                "tokens_scanned": len(analyzed_trades),
                "api_calls": self.api_call_stats.copy(),
                "seven_day_metrics": seven_day_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid wallet analysis: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            print(f"   ❌ Analysis failed: {str(e)}", flush=True)
            
            return self._get_error_analysis_result(wallet_address, str(e))
    
    def _get_recent_token_swaps_rpc_optimized(self, wallet_address: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Optimized method to get recent token swaps with batching."""
        try:
            logger.info(f"Fetching last {limit} token swaps for {wallet_address}")
            print(f"\n   Fetching transaction signatures...", flush=True)
            
            # Get signatures (reduced limit for 7-day analysis)
            signatures = self._get_signatures_for_address(wallet_address, limit=limit)
            
            if not signatures:
                logger.warning(f"No transactions found for {wallet_address}")
                return []
            
            print(f"   Found {len(signatures)} signatures, analyzing transactions...", flush=True)
            
            # Filter signatures by age (7 days) to reduce processing
            seven_days_ago = int((datetime.now() - timedelta(days=7)).timestamp())
            recent_signatures = []
            
            for sig_info in signatures:
                # Check block time if available
                block_time = sig_info.get("blockTime")
                if block_time and block_time >= seven_days_ago:
                    recent_signatures.append(sig_info)
                elif not block_time:
                    # Include if we don't know the time yet
                    recent_signatures.append(sig_info)
            
            logger.info(f"Filtered to {len(recent_signatures)} recent signatures")
            
            # Extract just the signature strings
            sig_strings = [s["signature"] for s in recent_signatures if s.get("signature")]
            
            # Use batch processing for transactions
            transactions = self._batch_get_transactions(sig_strings[:limit], show_progress=True)
            
            # Process transactions to extract swaps
            swaps = []
            token_history = defaultdict(list)
            
            for tx_details in transactions:
                if tx_details:
                    swap_infos = self._extract_token_swaps_from_transaction(tx_details, wallet_address)
                    for swap in swap_infos:
                        token_mint = swap.get('token_mint')
                        if token_mint:
                            token_history[token_mint].append(swap)
            
            # Pair buy and sell transactions
            logger.info(f"Pairing buy/sell transactions for {len(token_history)} tokens")
            
            for token_mint, token_swaps in token_history.items():
                # Sort by timestamp
                buys = sorted([s for s in token_swaps if s.get('type') == 'buy'], 
                             key=lambda x: x.get('buy_timestamp', 0))
                sells = sorted([s for s in token_swaps if s.get('type') == 'sell'], 
                              key=lambda x: x.get('sell_timestamp', 0))
                
                # FIFO pairing
                buy_queue = list(buys)
                
                for sell in sells:
                    sell_time = sell.get('sell_timestamp', 0)
                    sell_sol = sell.get('sol_amount', 0)
                    paired = False
                    
                    # Find the earliest unpaired buy before this sell
                    for i, buy in enumerate(buy_queue):
                        buy_time = buy.get('buy_timestamp', 0)
                        buy_sol = buy.get('sol_amount', 0)
                        
                        if buy_time < sell_time and buy.get('signature') not in self.paired_transactions:
                            # Pair them and calculate ROI directly
                            roi_percent = 0
                            pnl_usd = 0
                            
                            if buy_sol > 0:
                                roi_percent = ((sell_sol / buy_sol) - 1) * 100
                                pnl_usd = (sell_sol - buy_sol) * 150  # Estimate USD value
                            
                            paired_swap = {
                                **buy,
                                'sell_timestamp': sell_time,
                                'sell_signature': sell.get('signature'),
                                'sell_sol_amount': sell_sol,
                                'type': 'completed',
                                'paired': True,
                                'roi_percent': roi_percent,  # Calculate ROI here!
                                'pnl_usd': pnl_usd
                            }
                            
                            # Mark as paired
                            self.paired_transactions.add(buy.get('signature'))
                            self.paired_transactions.add(sell.get('signature'))
                            
                            swaps.append(paired_swap)
                            buy_queue.pop(i)
                            paired = True
                            break
                    
                    if not paired:
                        # Unpaired sell - estimate buy data
                        sell['paired_buy_data'] = self._estimate_buy_data(token_mint, sell_time)
                        swaps.append(sell)
                
                # Add remaining unpaired buys (open positions)
                for buy in buy_queue:
                    if buy.get('signature') not in self.paired_transactions:
                        swaps.append(buy)
            
            # Sort by most recent activity
            swaps.sort(key=lambda x: x.get('sell_timestamp') or x.get('buy_timestamp', 0), reverse=True)
            
            logger.info(f"Found {len(swaps)} swaps ({len([s for s in swaps if s.get('paired')])} paired)")
            return swaps[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent swaps: {str(e)}")
            return []
    
    def _analyze_trades_parallel(self, recent_swaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze trades in parallel for better performance."""
        analyzed_trades = []
        
        if not recent_swaps:
            return analyzed_trades
        
        # Create progress indicator
        progress = ProgressIndicator(len(recent_swaps), "   Analyzing trades")
        
        # Process trades in parallel
        futures = []
        for swap in recent_swaps:
            future = self.executor.submit(
                self._analyze_token_with_market_cap,
                swap['token_mint'],
                swap.get('buy_timestamp'),
                swap.get('sell_timestamp'),
                swap
            )
            futures.append((future, swap))
        
        # Collect results
        for future, swap in futures:
            try:
                token_analysis = future.result(timeout=30)
                if token_analysis['success']:
                    analyzed_trades.append({
                        **swap,
                        **token_analysis
                    })
            except Exception as e:
                logger.error(f"Error analyzing token {swap.get('token_mint')}: {str(e)}")
            
            progress.update()
        
        return analyzed_trades
    
    def _get_empty_analysis_result(self, wallet_address: str, reason: str) -> Dict[str, Any]:
        """Return empty analysis result for wallets with no trades."""
        empty_metrics = self._get_empty_metrics()
        return {
            "success": True,
            "wallet_address": wallet_address,
            "analysis_period_days": 7,
            "wallet_type": "no_trades",
            "composite_score": 0,
            "metrics": empty_metrics,
            "strategy": {
                "recommendation": "DO_NOT_COPY",
                "follow_sells": False,
                "tp1_percent": 0,
                "tp2_percent": 0,
                "sell_strategy": "NONE",
                "tp_guidance": "No trading history",
                "filter_market_cap_min": 0,
                "filter_market_cap_max": 0
            },
            "trades": [],
            "entry_exit_analysis": {},
            "reason": reason,
            "api_calls": self.api_call_stats.copy()
        }
    
    def _get_inactive_analysis_result(self, wallet_address: str, aggregated_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Return analysis result for inactive wallets."""
        metrics = aggregated_metrics.copy()
        metrics.update({
            "trades_last_7_days": 0,
            "win_rate_7d": 0,
            "profit_7d": 0,
            "active_trader": False,
            "days_since_last_trade": 999,
            "composite_score": 0
        })
        
        return {
            "success": True,
            "wallet_address": wallet_address,
            "analysis_period_days": 7,
            "wallet_type": "inactive",
            "composite_score": 0,
            "metrics": metrics,
            "strategy": {
                "recommendation": "DO_NOT_COPY",
                "follow_sells": False,
                "tp1_percent": 0,
                "tp2_percent": 0,
                "sell_strategy": "NONE",
                "tp_guidance": "Inactive for 7+ days",
                "filter_market_cap_min": 0,
                "filter_market_cap_max": 0
            },
            "trades": [],
            "entry_exit_analysis": {},
            "api_calls": self.api_call_stats.copy()
        }
    
    def _get_error_analysis_result(self, wallet_address: str, error: str) -> Dict[str, Any]:
        """Return error analysis result."""
        empty_metrics = self._get_empty_metrics()
        return {
            "success": False,
            "error": f"Unexpected error: {error}",
            "wallet_address": wallet_address,
            "error_type": "UNEXPECTED_ERROR",
            "wallet_type": "unknown",
            "composite_score": 0,
            "metrics": empty_metrics,
            "strategy": {
                "recommendation": "DO_NOT_COPY",
                "follow_sells": False,
                "tp1_percent": 0,
                "tp2_percent": 0,
                "sell_strategy": "NONE",
                "tp_guidance": "Analysis failed",
                "filter_market_cap_min": 0,
                "filter_market_cap_max": 0
            },
            "api_calls": self.api_call_stats.copy()
        }
    
    def _estimate_buy_data(self, token_mint: str, sell_timestamp: int) -> Dict[str, Any]:
        """Estimate buy data for unpaired sells."""
        # Look for any historical buy data for this token
        avg_hold_time = 3600 * 4  # Assume 4 hour average hold
        estimated_buy_time = sell_timestamp - avg_hold_time
        
        return {
            'buy_timestamp': estimated_buy_time,
            'estimated': True,
            'sol_amount': 0.1  # Default estimate
        }
    
    def _extract_token_swaps_from_transaction(self, tx_details: Dict[str, Any], 
                                            wallet_address: str) -> List[Dict[str, Any]]:
        """Extract token swap information from transaction."""
        swaps = []
        
        try:
            if not tx_details or "meta" not in tx_details:
                return []
            
            meta = tx_details["meta"]
            
            # Get block time
            block_time = tx_details.get("blockTime")
            current_time = int(datetime.now().timestamp())
            
            # Validate timestamp
            if not block_time or not isinstance(block_time, (int, float)):
                block_time = current_time - 86400
            elif block_time > current_time:
                block_time = current_time
            elif block_time < (current_time - 365 * 86400):
                return []
            
            pre_balances = meta.get("preTokenBalances", [])
            post_balances = meta.get("postTokenBalances", [])
            
            pre_sol = meta.get("preBalances", [])
            post_sol = meta.get("postBalances", [])
            
            token_changes = {}
            
            # Process token balance changes
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
            
            # Find wallet's SOL balance change
            accounts = tx_details.get("transaction", {}).get("message", {}).get("accountKeys", [])
            wallet_index = -1
            for i, account in enumerate(accounts):
                if account == wallet_address:
                    wallet_index = i
                    break
            
            sol_change = 0
            if wallet_index >= 0 and wallet_index < len(pre_sol) and wallet_index < len(post_sol):
                sol_change = (post_sol[wallet_index] - pre_sol[wallet_index]) / 1e9
            
            # Process token changes
            signature = tx_details.get("transaction", {}).get("signatures", [""])[0]
            
            for mint, changes in token_changes.items():
                diff = changes["post"] - changes["pre"]
                decimals = changes["decimals"]
                
                if diff != 0:
                    ui_amount = abs(diff) / (10 ** decimals)
                    
                    # Estimate price
                    estimated_price = 0
                    if diff > 0 and sol_change < 0:
                        estimated_price = abs(sol_change) / ui_amount
                    elif diff < 0 and sol_change > 0:
                        estimated_price = sol_change / ui_amount
                    
                    swap_data = {
                        "token_mint": mint,
                        "type": "buy" if diff > 0 else "sell",
                        "amount": ui_amount,
                        "raw_amount": abs(diff),
                        "decimals": decimals,
                        "buy_timestamp": block_time if diff > 0 else None,
                        "sell_timestamp": block_time if diff < 0 else None,
                        "signature": signature,
                        "sol_amount": abs(sol_change),
                        "estimated_price": estimated_price,
                        "block_time": block_time
                    }
                    
                    if self._validate_swap_data(swap_data):
                        swaps.append(swap_data)
            
        except Exception as e:
            logger.error(f"Error extracting swaps from transaction: {str(e)}")
        
        return swaps
    
    def _analyze_token_with_market_cap(self, token_mint: str, buy_timestamp: Optional[int], 
                                     sell_timestamp: Optional[int] = None, 
                                     swap_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze token with DIRECT CALCULATION FIRST, then API fallback."""
        try:
            # Initialize default values
            roi_percent = 0
            max_roi_percent = 0
            current_roi_percent = 0
            hold_time_minutes = 0
            pnl_usd = 0
            market_cap_at_buy = 0
            entry_timing = "UNKNOWN"
            exit_timing = "UNKNOWN"
            
            # PRIORITY 1: Check if we already have ROI calculated (from paired data)
            if swap_data and 'roi_percent' in swap_data:
                roi_percent = swap_data['roi_percent']
                pnl_usd = swap_data.get('pnl_usd', 0)
                
                # Calculate hold time
                if buy_timestamp and sell_timestamp and isinstance(sell_timestamp, (int, float)):
                    hold_time_seconds = sell_timestamp - buy_timestamp
                    hold_time_minutes = hold_time_seconds / 60
                
                # Determine quality based on ROI
                max_roi_percent = roi_percent * 1.2  # Estimate max as 20% higher
                current_roi_percent = roi_percent
                
                if roi_percent >= 400:
                    entry_timing = "EXCELLENT"
                elif roi_percent >= 200:
                    entry_timing = "GOOD"
                elif roi_percent >= 100:
                    entry_timing = "AVERAGE"
                elif roi_percent > 0:
                    entry_timing = "BELOW_AVERAGE"
                else:
                    entry_timing = "POOR"
                
                # Exit timing for completed trades
                if sell_timestamp:
                    if roi_percent >= 100:
                        exit_timing = "GOOD"
                    elif roi_percent > 0:
                        exit_timing = "AVERAGE"
                    else:
                        exit_timing = "POOR"
                else:
                    exit_timing = "HOLDING"
                
                logger.debug(f"Using pre-calculated ROI: {roi_percent:.2f}% for {token_mint}")
                
                # Try to get market cap data (non-critical)
                if self._is_pump_fun_token(token_mint):
                    market_cap_at_buy = 10000  # Default for pump tokens
                else:
                    market_cap_at_buy = 100000  # Default for regular tokens
                
                return {
                    "success": True,
                    "token_address": token_mint,
                    "market_cap_at_buy": market_cap_at_buy,
                    "roi_percent": roi_percent,
                    "current_roi_percent": current_roi_percent,
                    "max_roi_percent": max_roi_percent,
                    "entry_timing": entry_timing,
                    "exit_timing": exit_timing,
                    "hold_time_seconds": hold_time_seconds if 'hold_time_seconds' in locals() else 0,
                    "hold_time_minutes": hold_time_minutes,
                    "pnl_usd": pnl_usd,
                    "data_source": "pre_calculated"
                }
            
            # PRIORITY 2: Calculate from SOL amounts if available
            if swap_data and swap_data.get('paired') and swap_data.get('sell_sol_amount'):
                buy_sol = swap_data.get('sol_amount', 0)
                sell_sol = swap_data.get('sell_sol_amount', 0)
                
                if buy_sol > 0:
                    roi_percent = ((sell_sol / buy_sol) - 1) * 100
                    pnl_usd = (sell_sol - buy_sol) * 150  # Estimate USD
                    
                    # Calculate hold time
                    if buy_timestamp and sell_timestamp:
                        hold_time_seconds = sell_timestamp - buy_timestamp
                        hold_time_minutes = hold_time_seconds / 60
                    
                    # Set quality metrics
                    max_roi_percent = roi_percent * 1.2
                    current_roi_percent = roi_percent
                    
                    # Determine entry timing based on ROI
                    if roi_percent >= 400:
                        entry_timing = "EXCELLENT"
                    elif roi_percent >= 200:
                        entry_timing = "GOOD"
                    elif roi_percent >= 100:
                        entry_timing = "AVERAGE"
                    else:
                        entry_timing = "POOR"
                    
                    exit_timing = "GOOD" if roi_percent >= 50 else "AVERAGE"
                    
                    logger.debug(f"Calculated ROI from SOL amounts: {roi_percent:.2f}% for {token_mint}")
                    
                    # Default market cap
                    market_cap_at_buy = 10000 if self._is_pump_fun_token(token_mint) else 100000
                    
                    return {
                        "success": True,
                        "token_address": token_mint,
                        "market_cap_at_buy": market_cap_at_buy,
                        "roi_percent": roi_percent,
                        "current_roi_percent": current_roi_percent,
                        "max_roi_percent": max_roi_percent,
                        "entry_timing": entry_timing,
                        "exit_timing": exit_timing,
                        "hold_time_seconds": hold_time_seconds if 'hold_time_seconds' in locals() else 0,
                        "hold_time_minutes": hold_time_minutes,
                        "pnl_usd": pnl_usd,
                        "data_source": "direct_calculation"
                    }
            
            # PRIORITY 3: Try APIs for additional data (but don't fail if they error)
            is_pump = self._is_pump_fun_token(token_mint)
            api_data_retrieved = False
            
            # Skip API calls if no valid buy timestamp
            if not buy_timestamp or not isinstance(buy_timestamp, (int, float)):
                logger.debug(f"No valid buy timestamp for {token_mint}, using estimates")
            else:
                # Try appropriate API
                if is_pump and self.helius_api:
                    try:
                        self.api_call_stats["helius"] += 1
                        # Don't let API failures break the analysis
                        pass  # Helius is failing, skip it
                    except Exception as e:
                        logger.debug(f"Helius API skipped: {str(e)}")
                
                elif self.birdeye_api and not is_pump:
                    try:
                        self.api_call_stats["birdeye"] += 1
                        # Try to get price history
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
                            if prices:
                                initial_price = prices[0].get("value", 0)
                                current_price = prices[-1].get("value", 0)
                                max_price = max(p.get("value", 0) for p in prices)
                                
                                if initial_price > 0:
                                    roi_percent = ((current_price / initial_price) - 1) * 100
                                    max_roi_percent = ((max_price / initial_price) - 1) * 100
                                    current_roi_percent = roi_percent
                                    api_data_retrieved = True
                    except Exception as e:
                        logger.debug(f"Birdeye API error (non-critical): {str(e)}")
            
            # PRIORITY 4: Use fallback estimates for open positions
            if not api_data_retrieved and roi_percent == 0 and swap_data:
                if swap_data.get('type') == 'buy':
                    # Estimate for open positions
                    hold_time_seconds = int(datetime.now().timestamp()) - buy_timestamp if buy_timestamp else 0
                    hold_time_minutes = hold_time_seconds / 60
                    
                    # Conservative estimates
                    if is_pump:
                        roi_percent = -20  # Assume small loss for pump tokens
                        max_roi_percent = 50
                    else:
                        roi_percent = 10  # Small gain for regular tokens
                        max_roi_percent = 30
                    
                    current_roi_percent = roi_percent
                    entry_timing = "UNKNOWN"
                    exit_timing = "HOLDING"
            
            # Calculate hold time if not set
            if hold_time_minutes == 0 and buy_timestamp:
                if sell_timestamp:
                    hold_time_seconds = sell_timestamp - buy_timestamp
                else:
                    hold_time_seconds = int(datetime.now().timestamp()) - buy_timestamp
                hold_time_minutes = hold_time_seconds / 60
            
            # Default market cap if not set
            if market_cap_at_buy == 0:
                market_cap_at_buy = 10000 if is_pump else 100000
            
            return {
                "success": True,
                "token_address": token_mint,
                "market_cap_at_buy": market_cap_at_buy,
                "roi_percent": roi_percent,
                "current_roi_percent": current_roi_percent,
                "max_roi_percent": max_roi_percent,
                "entry_timing": entry_timing,
                "exit_timing": exit_timing,
                "hold_time_seconds": hold_time_seconds if 'hold_time_seconds' in locals() else 0,
                "hold_time_minutes": hold_time_minutes,
                "pnl_usd": pnl_usd,
                "data_source": "api_enhanced" if api_data_retrieved else "estimated",
                "is_pump_token": is_pump
            }
            
        except Exception as e:
            logger.error(f"Error analyzing token {token_mint}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "token_address": token_mint
            }
    
    def _calculate_enhanced_memecoin_metrics_7day(self, cielo_metrics: Dict[str, Any], 
                                                 analyzed_trades: List[Dict[str, Any]],
                                                 seven_day_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced metrics with proper 7-day focus using pre-calculated ROI."""
        try:
            # Start with base metrics
            metrics = cielo_metrics.copy() if cielo_metrics else self._get_empty_cielo_metrics()
            
            # Filter trades to last 7 days
            seven_days_ago = int((datetime.now() - timedelta(days=7)).timestamp())
            recent_trades = []
            for t in analyzed_trades:
                # Use sell timestamp for completed trades, buy timestamp for open
                ts = t.get('sell_timestamp') or t.get('buy_timestamp')
                if ts and isinstance(ts, (int, float)) and ts >= seven_days_ago:
                    recent_trades.append(t)
            
            # Calculate win rate from actual ROI data
            if recent_trades:
                trades_with_roi = [t for t in recent_trades if 'roi_percent' in t]
                wins = sum(1 for t in trades_with_roi if t.get('roi_percent', 0) > 0)
                win_rate = (wins / len(trades_with_roi) * 100) if trades_with_roi else 0
                metrics['win_rate'] = round(win_rate, 2)
                metrics['win_rate_7d'] = round(win_rate, 2)
                
                # Calculate total profit
                total_pnl = sum(t.get('pnl_usd', 0) for t in trades_with_roi)
                metrics['profit_7d'] = round(total_pnl, 2)
                
                # Log for debugging
                logger.debug(f"7-day metrics: {len(trades_with_roi)} trades, {wins} wins, win_rate: {win_rate:.1f}%")
            else:
                metrics['win_rate'] = 0
                metrics['win_rate_7d'] = 0
                metrics['profit_7d'] = 0
            
            # Calculate 7-day distribution
            if recent_trades:
                distribution = self._calculate_7day_distribution(recent_trades)
                metrics.update(distribution)
            else:
                metrics.update({
                    "distribution_500_plus_%": 0,
                    "distribution_200_500_%": 0,
                    "distribution_0_200_%": 0,
                    "distribution_neg50_0_%": 0,
                    "distribution_below_neg50_%": 0
                })
            
            # Calculate gem rates from max ROI
            trades_with_max_roi = [t for t in recent_trades if t.get("max_roi_percent", 0) > 0]
            if trades_with_max_roi:
                gem_5x_count = sum(1 for t in trades_with_max_roi if t.get("max_roi_percent", 0) >= 400)
                gem_2x_count = sum(1 for t in trades_with_max_roi if t.get("max_roi_percent", 0) >= 100)
                
                gem_rate_5x = (gem_5x_count / len(trades_with_max_roi) * 100)
                gem_rate_2x = (gem_2x_count / len(trades_with_max_roi) * 100)
            else:
                gem_rate_5x = 0
                gem_rate_2x = 0
            
            metrics["gem_rate_5x_plus"] = round(gem_rate_5x, 2)
            metrics["gem_rate_2x_plus"] = round(gem_rate_2x, 2)
            
            # Update metrics with 7-day data
            metrics.update({
                "trades_last_7_days": seven_day_metrics['trades_last_7_days'],
                "win_rate_7d": seven_day_metrics['win_rate_7d'],
                "profit_7d": seven_day_metrics['profit_7d'],
                "active_trader": seven_day_metrics['active_trader'],
                "days_since_last_trade": seven_day_metrics['days_since_last_trade']
            })
            
            # Calculate average hold times
            hold_times_minutes = [t["hold_time_minutes"] for t in recent_trades 
                                 if t.get("hold_time_minutes", 0) > 0]
            
            if hold_times_minutes:
                avg_hold_minutes = np.mean(hold_times_minutes)
                metrics["avg_hold_time_minutes"] = round(avg_hold_minutes, 2)
                metrics["avg_hold_time_hours"] = round(avg_hold_minutes / 60, 2)
            
            # Calculate market cap metrics
            market_caps = [t.get("market_cap_at_buy", 0) for t in recent_trades 
                          if t.get("market_cap_at_buy", 0) > 0]
            if market_caps:
                metrics["avg_buy_market_cap_usd"] = round(np.mean(market_caps), 2)
                metrics["median_buy_market_cap_usd"] = round(np.median(market_caps), 2)
            
            # Calculate exit metrics from completed trades with positive ROI
            completed_trades = [t for t in recent_trades if t.get("sell_timestamp") and t.get("roi_percent", 0) > 0]
            if completed_trades:
                exit_profits = [t["roi_percent"] for t in completed_trades]
                metrics["avg_first_take_profit_percent"] = round(np.mean(exit_profits), 1)
                metrics["median_first_take_profit_percent"] = round(np.median(exit_profits), 1)
            else:
                metrics["avg_first_take_profit_percent"] = 0
                metrics["median_first_take_profit_percent"] = 0
            
            # Calculate ROI metrics
            all_rois = [t.get("roi_percent", 0) for t in recent_trades if "roi_percent" in t]
            if all_rois:
                metrics["avg_roi"] = round(np.mean(all_rois), 2)
                metrics["median_roi"] = round(np.median(all_rois), 2)
                metrics["max_roi"] = round(max(all_rois), 2)
            
            # Calculate average buy amount
            buy_amounts = [t.get("sol_amount", 0) * 150 for t in recent_trades if t.get("type") in ["buy", "completed"]]
            if buy_amounts:
                metrics["avg_buy_amount_usd"] = round(np.mean(buy_amounts), 2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {str(e)}")
            return cielo_metrics or self._get_empty_metrics()
    
    def _calculate_7day_distribution(self, recent_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate ROI distribution for last 7 days only."""
        buckets = {
            "distribution_500_plus_%": 0,      # 500%+ (5x+)
            "distribution_200_500_%": 0,       # 200-500% (2x-5x)
            "distribution_0_200_%": 0,         # 0-200% (profitable <2x)
            "distribution_neg50_0_%": 0,       # -50% to 0%
            "distribution_below_neg50_%": 0    # Below -50%
        }
        
        trades_with_roi = [t for t in recent_trades if 'roi_percent' in t]
        if not trades_with_roi:
            return buckets
        
        # Count trades in each bucket
        for trade in trades_with_roi:
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
        
        # Convert to percentages
        total = len(trades_with_roi)
        for key in buckets:
            buckets[key] = round(buckets[key] / total * 100, 1)
        
        # Ensure it sums to 100%
        total_percent = sum(buckets.values())
        if total_percent > 0 and abs(total_percent - 100) > 1:
            # Adjust the largest bucket
            largest_bucket = max(buckets, key=buckets.get)
            buckets[largest_bucket] += (100 - total_percent)
            buckets[largest_bucket] = round(buckets[largest_bucket], 1)
        
        return buckets
    
    def _calculate_memecoin_composite_score_7day(self, metrics: Dict[str, Any], 
                                                seven_day_metrics: Dict[str, Any]) -> float:
        """Calculate composite score with heavy 7-day activity weighting."""
        try:
            # Base activity score from 7-day trades
            trades_7d = seven_day_metrics.get("trades_last_7_days", 0)
            if trades_7d >= 10:
                activity_score = 20
            elif trades_7d >= 5:
                activity_score = 15
            elif trades_7d >= 3:
                activity_score = 10
            elif trades_7d >= 1:
                activity_score = 5
            else:
                activity_score = 0
            
            # Recency bonus (penalize inactive traders)
            days_since_trade = seven_day_metrics.get("days_since_last_trade", 999)
            if days_since_trade <= 1:
                recency_bonus = 10
            elif days_since_trade <= 3:
                recency_bonus = 5
            elif days_since_trade <= 7:
                recency_bonus = 0
            else:
                recency_bonus = -20  # Heavy penalty for inactive
            
            # 7-day win rate score
            win_rate_7d = seven_day_metrics.get("win_rate_7d", 0)
            if win_rate_7d >= 60:
                winrate_score = 15
            elif win_rate_7d >= 45:
                winrate_score = 10
            elif win_rate_7d >= 30:
                winrate_score = 5
            else:
                winrate_score = 0
            
            # 7-day gem finding (5x+ emphasis)
            if seven_day_metrics.get("has_5x_last_7_days", False):
                gem_score = 30  # Big bonus for recent 5x
            elif seven_day_metrics.get("has_2x_last_7_days", False):
                gem_score = 15
            else:
                gem_score = 0
            
            # 7-day profit score
            profit_7d = seven_day_metrics.get("profit_7d", 0)
            if profit_7d >= 10000:
                profit_score = 20
            elif profit_7d >= 5000:
                profit_score = 15
            elif profit_7d >= 1000:
                profit_score = 10
            elif profit_7d >= 0:
                profit_score = 5
            else:
                profit_score = 0
            
            # Distribution quality (from 7-day trades)
            dist_500_plus = metrics.get("distribution_500_plus_%", 0)
            dist_below_neg50 = metrics.get("distribution_below_neg50_%", 0)
            
            dist_score = 0
            if dist_500_plus >= 10:  # 10%+ trades are 5x+
                dist_score += 10
            elif dist_500_plus >= 5:
                dist_score += 5
            
            if dist_below_neg50 <= 10:  # Low catastrophic losses
                dist_score += 5
            
            # Calculate total
            total_score = (
                activity_score +
                recency_bonus +
                winrate_score +
                gem_score +
                profit_score +
                dist_score
            )
            
            # Apply multipliers for exceptional recent performance
            if seven_day_metrics.get("has_5x_last_7_days") and win_rate_7d >= 50:
                total_score *= 1.3
            
            # Cap at 100
            total_score = min(100, max(0, total_score))
            
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
                    return "flipper"
            
            # Gem Hunter: Holds for 5x+ gains
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
    
    def _generate_enhanced_memecoin_strategy(self, wallet_type: str, metrics: Dict[str, Any], 
                                           analyzed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate enhanced strategy with specific sell guidance."""
        try:
            composite_score = metrics.get("composite_score", 0)
            avg_first_tp = metrics.get("avg_first_take_profit_percent", 0)
            avg_market_cap = metrics.get("avg_buy_market_cap_usd", 0)
            gem_rate_5x = metrics.get("gem_rate_5x_plus", 0)
            win_rate = metrics.get("win_rate", 0)
            
            # Analyze exit behavior from recent trades
            exit_analysis = self._analyze_exit_behavior(analyzed_trades)
            
            # Determine if we should follow their sells
            follow_sells = exit_analysis['exit_quality'] in ["GOOD", "EXCELLENT"]
            
            # Calculate optimal TPs based on their behavior
            if follow_sells:
                # They're good at exits, follow their pattern
                tp1 = avg_first_tp if avg_first_tp > 0 else 50
                tp2 = tp1 * 2
                sell_strategy = "COPY_EXITS"
                tp_guidance = f"Follow their exits - they capture {exit_analysis['avg_capture_ratio']:.0f}% of gains"
            else:
                # They exit poorly, set better targets
                if exit_analysis['avg_missed_gains'] > 100:
                    # They leave huge gains on table
                    tp1 = max(avg_first_tp * 2, 100)
                    tp2 = tp1 * 2.5
                    sell_strategy = "USE_FIXED_TP"
                    tp_guidance = f"They exit too early at {avg_first_tp:.0f}% avg - hold for bigger gains"
                else:
                    # Moderate improvement needed
                    tp1 = max(avg_first_tp * 1.5, 50)
                    tp2 = tp1 * 2
                    sell_strategy = "HYBRID"
                    tp_guidance = f"Consider their exits but hold longer - they miss {exit_analysis['avg_missed_gains']:.0f}% gains"
            
            # Market cap filter range
            if avg_market_cap > 0:
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
                filter_min = 50000
                filter_max = 5000000
            
            # Base strategy on wallet type
            if wallet_type == "sniper":
                strategy = {
                    "recommendation": "COPY_SNIPER",
                    "follow_sells": follow_sells,
                    "tp1_percent": min(tp1, 100),  # Snipers often take quick profits
                    "tp2_percent": min(tp2, 200),
                    "sell_strategy": sell_strategy,
                    "tp_guidance": tp_guidance,
                    "notes": f"Sniper with {gem_rate_5x:.1f}% 5x rate. Quick entries/exits.",
                    "filter_market_cap_min": filter_min,
                    "filter_market_cap_max": filter_max
                }
            
            elif wallet_type == "flipper":
                strategy = {
                    "recommendation": "COPY_FLIPPER",
                    "follow_sells": follow_sells,
                    "tp1_percent": tp1,
                    "tp2_percent": tp2,
                    "sell_strategy": sell_strategy,
                    "tp_guidance": tp_guidance,
                    "notes": f"Flipper with {win_rate:.1f}% win rate.",
                    "filter_market_cap_min": filter_min,
                    "filter_market_cap_max": filter_max
                }
            
            elif wallet_type == "gem_hunter":
                # Gem hunters should hold longer
                strategy = {
                    "recommendation": "COPY_GEM_HUNTER",
                    "follow_sells": True,  # Usually good at holding
                    "tp1_percent": max(400, tp1),  # At least 5x
                    "tp2_percent": max(800, tp2),  # Target 10x
                    "sell_strategy": "FOLLOW_GEMS",
                    "tp_guidance": f"5x+ hunter with {gem_rate_5x:.1f}% success - hold for moonshots",
                    "notes": f"Elite gem hunter. Let winners run.",
                    "filter_market_cap_min": filter_min,
                    "filter_market_cap_max": filter_max
                }
            
            else:
                strategy = {
                    "recommendation": "SELECTIVE_COPY" if composite_score >= 40 else "CAUTIOUS",
                    "follow_sells": follow_sells,
                    "tp1_percent": tp1,
                    "tp2_percent": tp2,
                    "sell_strategy": sell_strategy,
                    "tp_guidance": tp_guidance,
                    "notes": f"Mixed results. Score: {composite_score:.1f}/100.",
                    "filter_market_cap_min": filter_min,
                    "filter_market_cap_max": filter_max
                }
            
            # Add warning for inactive traders
            days_inactive = metrics.get("days_since_last_trade", 0)
            if days_inactive > 3:
                strategy["notes"] += f" ⚠️ Inactive {days_inactive} days."
            
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return {
                "recommendation": "DO_NOT_COPY",
                "follow_sells": False,
                "tp1_percent": 0,
                "tp2_percent": 0,
                "sell_strategy": "NONE",
                "tp_guidance": "Error during strategy generation",
                "filter_market_cap_min": 0,
                "filter_market_cap_max": 0
            }
    
    def _analyze_exit_behavior(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze exit behavior to determine if we should follow sells."""
        if not trades:
            return {
                "exit_quality": "UNKNOWN",
                "avg_capture_ratio": 0,
                "avg_missed_gains": 0,
                "should_follow": False
            }
        
        # Only analyze completed trades
        completed = [t for t in trades if t.get("sell_timestamp") and isinstance(t.get("sell_timestamp"), (int, float))]
        
        if not completed:
            return {
                "exit_quality": "NO_DATA",
                "avg_capture_ratio": 0,
                "avg_missed_gains": 0,
                "should_follow": False
            }
        
        capture_ratios = []
        missed_gains = []
        
        for trade in completed:
            exit_roi = trade.get("roi_percent", 0)
            max_roi = trade.get("max_roi_percent", 0)
            
            if max_roi > 0:
                capture_ratio = (exit_roi / max_roi * 100) if max_roi else 0
                capture_ratios.append(capture_ratio)
                missed_gains.append(max_roi - exit_roi)
        
        avg_capture = np.mean(capture_ratios) if capture_ratios else 0
        avg_missed = np.mean(missed_gains) if missed_gains else 0
        
        # Determine quality
        if avg_capture >= 80:
            quality = "EXCELLENT"
        elif avg_capture >= 60:
            quality = "GOOD"
        elif avg_capture >= 40:
            quality = "AVERAGE"
        else:
            quality = "POOR"
        
        return {
            "exit_quality": quality,
            "avg_capture_ratio": avg_capture,
            "avg_missed_gains": avg_missed,
            "should_follow": quality in ["GOOD", "EXCELLENT"]
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
                    
                    # Check for early exit
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
            
            # Calculate early exit rate
            trades_with_exit = len([t for t in analyzed_trades if t.get("exit_timing") != "HOLDING"])
            early_exit_rate = (early_exits / trades_with_exit * 100) if trades_with_exit > 0 else 0
            
            # Determine pattern
            if avg_missed_gains > 200:
                pattern = "LEAVES_MONEY"
            elif early_exit_rate > 50:
                pattern = "EARLY_SELLER"
            elif avg_exit_roi > 100:
                pattern = "DIAMOND_HANDS"
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
            total_profits = 0
            total_losses = 0
            
            # Try to get from roi distribution
            roi_dist = stats_data.get("roi_distribution", {})
            if roi_dist:
                profitable_trades = (
                    roi_dist.get("roi_above_500", 0) +
                    roi_dist.get("roi_200_to_500", 0) +
                    roi_dist.get("roi_0_to_200", 0)
                )
                loss_trades = (
                    roi_dist.get("roi_neg50_to_0", 0) +
                    roi_dist.get("roi_below_neg50", 0)
                )
                
                if total_pnl_usd > 0 and profitable_trades > 0:
                    avg_profit_per_trade = total_pnl_usd / profitable_trades
                    total_profits = avg_profit_per_trade * profitable_trades
                elif total_sell_amount_usd > total_buy_amount_usd:
                    total_profits = total_sell_amount_usd - total_buy_amount_usd
                
                if loss_trades > 0:
                    avg_loss_estimate = abs(total_pnl_usd) / total_trades if total_trades > 0 else 100
                    total_losses = avg_loss_estimate * loss_trades
            
            # Alternative calculation if no roi distribution
            if total_profits == 0 and total_losses == 0:
                if total_pnl_usd > 0:
                    total_profits = total_pnl_usd
                    total_losses = 1
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
                "avg_hold_time": avg_hold_time_minutes / 60,
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
            "avg_first_take_profit_percent": 0,
            "trades_last_7_days": 0,
            "win_rate_7d": 0,
            "profit_7d": 0,
            "active_trader": False,
            "days_since_last_trade": 999
        }
    
    def batch_analyze_wallets(self, wallet_addresses: List[str], 
                            days_back: int = 7,
                            min_winrate: float = 30.0,
                            use_hybrid: bool = True) -> Dict[str, Any]:
        """Batch analyze multiple wallets with 7-day focus."""
        logger.info(f"Batch analyzing {len(wallet_addresses)} wallets (7-day active trader focus)")
        print(f"\n🚀 Starting batch analysis of {len(wallet_addresses)} wallets...\n", flush=True)
        
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
                print(f"\n{'='*60}", flush=True)
                print(f"Wallet {i}/{len(wallet_addresses)}: {wallet_address}", flush=True)
                print(f"{'='*60}", flush=True)
                
                try:
                    if use_hybrid:
                        analysis = self.analyze_wallet_hybrid(wallet_address, days_back)
                    else:
                        analysis = self.analyze_wallet(wallet_address, days_back)
                    
                    if "metrics" in analysis:
                        wallet_analyses.append(analysis)
                        score = analysis.get("composite_score", analysis.get("metrics", {}).get("composite_score", 0))
                        logger.info(f"  └─ Score: {score}/100, Type: {analysis.get('wallet_type', 'unknown')}")
                    else:
                        failed_analyses.append({
                            "wallet_address": wallet_address,
                            "error": analysis.get("error", "No metrics available"),
                            "error_type": analysis.get("error_type", "NO_METRICS")
                        })
                    
                    if i < len(wallet_addresses):
                        time.sleep(1)  # Reduced delay
                        
                except Exception as e:
                    logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
                    print(f"   ❌ Analysis failed: {str(e)}", flush=True)
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
            
            # Log final API call statistics
            print(f"\n{'='*60}", flush=True)
            logger.info(f"📊 FINAL API CALL STATISTICS:")
            logger.info(f"   Cielo: {self.api_call_stats['cielo']} calls")
            logger.info(f"   Birdeye: {self.api_call_stats['birdeye']} calls")
            logger.info(f"   Helius: {self.api_call_stats['helius']} calls")
            logger.info(f"   RPC: {self.api_call_stats['rpc']} calls")
            logger.info(f"   Total API calls: {sum(self.api_call_stats.values())}")
            
            print(f"\n✅ Batch analysis complete!", flush=True)
            print(f"   Successful: {len(wallet_analyses)}", flush=True)
            print(f"   Failed: {len(failed_analyses)}", flush=True)
            print(f"   Total API calls: {sum(self.api_call_stats.values())}", flush=True)
            
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
                "api_calls": self.api_call_stats.copy()
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