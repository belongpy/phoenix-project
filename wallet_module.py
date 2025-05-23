"""
Wallet Analysis Module - Phoenix Project (COMPLETE WITH HELIUS INTEGRATION)

UPDATES:
- Integrated Helius API for pump.fun token analysis
- Fallback analysis tiers (Birdeye -> Helius -> Basic)
- Weighted composite scoring based on data quality
- Fixed profit factor capping at 999.99
- Hold time in minutes
- Improved entry/exit analysis
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
        
        # Track data quality for composite scoring
        self.data_quality_weights = {
            "full_analysis": 1.0,      # Birdeye data available
            "helius_analysis": 0.85,   # Helius/pump.fun data
            "basic_analysis": 0.5      # Only P&L data
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
    
    def analyze_wallet_hybrid(self, wallet_address: str, days_back: int = 30) -> Dict[str, Any]:
        """
        UPDATED hybrid wallet analysis using Cielo Finance for aggregated stats and RPC for recent transactions.
        Now with Helius integration for pump.fun tokens.
        
        Args:
            wallet_address (str): Wallet address
            days_back (int): Number of days to analyze (for RPC calls)
            
        Returns:
            Dict[str, Any]: Wallet analysis results
        """
        logger.info(f"🔍 Analyzing wallet {wallet_address} (hybrid approach with Helius)")
        
        try:
            # Step 1: Get aggregated stats from Cielo Finance
            logger.info(f"📊 Fetching Cielo Finance aggregated stats...")
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
            
            # Step 2: Get recent token trades via RPC for detailed analysis
            logger.info(f"🪙 Analyzing last 5 tokens...")
            recent_swaps = self._get_recent_token_swaps_rpc(wallet_address, limit=5)
            
            # Step 3: Analyze token performance with tiered approach
            analyzed_trades = []
            data_quality_scores = []
            
            if recent_swaps:
                for i, swap in enumerate(recent_swaps[:5]):
                    if i > 0:
                        time.sleep(0.5)
                    
                    # Try analysis in order: Birdeye -> Helius -> Basic
                    token_analysis, data_quality = self._analyze_token_with_fallback(
                        swap['token_mint'],
                        swap['buy_timestamp'],
                        swap.get('sell_timestamp')
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
            
            # Step 6: Determine wallet type
            wallet_type = self._determine_wallet_type(combined_metrics)
            
            # Step 7: Generate strategy
            strategy = self._generate_strategy(wallet_type, combined_metrics)
            
            # Analyze entry/exit behavior
            entry_exit_analysis = self._analyze_entry_exit_behavior(analyzed_trades)
            
            logger.debug(f"Final analysis for {wallet_address}:")
            logger.debug(f"  - wallet_type: {wallet_type}")
            logger.debug(f"  - weighted_composite_score: {weighted_composite_score}")
            logger.debug(f"  - data_quality_factor: {avg_data_quality}")
            
            return {
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
    
    def _analyze_token_with_fallback(self, token_mint: str, buy_timestamp: Optional[int], 
                                   sell_timestamp: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
        """
        Analyze token performance with fallback approach:
        1. Try Birdeye API first
        2. If pump.fun token or Birdeye fails, try Helius
        3. If both fail, return basic P&L analysis
        
        Returns:
            Tuple[Dict[str, Any], str]: (analysis_result, data_quality_tier)
        """
        # Tier 1: Try Birdeye first (best data quality)
        if self.birdeye_api and not token_mint.endswith("pump"):
            birdeye_result = self._analyze_token_performance(
                token_mint, buy_timestamp, sell_timestamp
            )
            if birdeye_result.get("success") and not birdeye_result.get("is_pump_token"):
                return birdeye_result, "full_analysis"
        
        # Tier 2: Try Helius for pump.fun tokens or if Birdeye failed
        if self.helius_api:
            logger.info(f"Using Helius API for token {token_mint}")
            helius_result = self._analyze_token_with_helius(
                token_mint, buy_timestamp, sell_timestamp
            )
            if helius_result.get("success"):
                return helius_result, "helius_analysis"
        
        # Tier 3: Basic P&L analysis (fallback)
        logger.info(f"Using basic P&L analysis for token {token_mint}")
        basic_result = self._basic_token_analysis(
            token_mint, buy_timestamp, sell_timestamp
        )
        return basic_result, "basic_analysis"
    
    def _analyze_token_with_helius(self, token_mint: str, buy_timestamp: Optional[int],
                                 sell_timestamp: Optional[int] = None) -> Dict[str, Any]:
        """Analyze token using Helius API (especially for pump.fun tokens)."""
        try:
            if not self.helius_api:
                return {"success": False, "error": "Helius API not available"}
            
            # Get price at buy time
            buy_price_data = self.helius_api.get_pump_fun_token_price(
                token_mint, buy_timestamp
            )
            
            if not buy_price_data.get("success"):
                return {
                    "success": False,
                    "error": "Could not get buy price from Helius",
                    "is_pump_token": True
                }
            
            buy_price = buy_price_data.get("data", {}).get("price", 0)
            
            # Get current or sell price
            if sell_timestamp:
                sell_price_data = self.helius_api.get_pump_fun_token_price(
                    token_mint, sell_timestamp
                )
                current_price = sell_price_data.get("data", {}).get("price", buy_price)
            else:
                current_price_data = self.helius_api.get_pump_fun_token_price(token_mint)
                current_price = current_price_data.get("data", {}).get("price", buy_price)
            
            # Calculate performance metrics
            roi_percent = ((current_price / buy_price) - 1) * 100 if buy_price > 0 else 0
            
            # Get token metadata
            metadata = self.helius_api.get_token_metadata([token_mint])
            token_info = {}
            if metadata.get("success") and metadata.get("data"):
                token_info = metadata["data"][0] if metadata["data"] else {}
            
            # Analyze entry/exit timing
            entry_timing = "GOOD" if roi_percent > 50 else "POOR" if roi_percent < -20 else "AVERAGE"
            exit_timing = "HOLDING" if not sell_timestamp else (
                "GOOD" if roi_percent > 30 else "EARLY" if roi_percent > 0 else "LOSS_EXIT"
            )
            
            return {
                "success": True,
                "token_address": token_mint,
                "initial_price": buy_price,
                "current_price": current_price,
                "roi_percent": roi_percent,
                "current_roi_percent": roi_percent,
                "max_roi_percent": max(roi_percent, 0),  # Conservative estimate
                "entry_timing": entry_timing,
                "exit_timing": exit_timing,
                "data_source": "helius",
                "is_pump_token": True,
                "token_metadata": token_info
            }
            
        except Exception as e:
            logger.error(f"Error in Helius token analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "is_pump_token": True
            }
    
    def _basic_token_analysis(self, token_mint: str, buy_timestamp: Optional[int],
                            sell_timestamp: Optional[int] = None) -> Dict[str, Any]:
        """Basic token analysis using only available transaction data."""
        return {
            "success": True,
            "token_address": token_mint,
            "roi_percent": 0,
            "current_roi_percent": 0,
            "max_roi_percent": 0,
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
                block_time = int((datetime.now() - timedelta(days=1)).timestamp())
                logger.debug(f"Using fallback timestamp: {block_time}")
            
            pre_balances = meta.get("preTokenBalances", [])
            post_balances = meta.get("postTokenBalances", [])
            
            token_changes = {}
            
            for balance in pre_balances:
                owner = balance.get("owner")
                if owner == wallet_address:
                    mint = balance.get("mint")
                    amount = int(balance.get("uiTokenAmount", {}).get("amount", 0))
                    if mint:
                        token_changes[mint] = {"pre": amount, "post": 0}
            
            for balance in post_balances:
                owner = balance.get("owner")
                if owner == wallet_address:
                    mint = balance.get("mint")
                    amount = int(balance.get("uiTokenAmount", {}).get("amount", 0))
                    if mint:
                        if mint in token_changes:
                            token_changes[mint]["post"] = amount
                        else:
                            token_changes[mint] = {"pre": 0, "post": amount}
            
            for mint, changes in token_changes.items():
                diff = changes["post"] - changes["pre"]
                
                if diff > 0:
                    swaps.append({
                        "token_mint": mint,
                        "type": "buy",
                        "amount": diff,
                        "buy_timestamp": block_time,
                        "sell_timestamp": None,
                        "signature": tx_details.get("transaction", {}).get("signatures", [""])[0]
                    })
                elif diff < 0:
                    swaps.append({
                        "token_mint": mint,
                        "type": "sell",
                        "amount": abs(diff),
                        "buy_timestamp": None,
                        "sell_timestamp": block_time,
                        "signature": tx_details.get("transaction", {}).get("signatures", [""])[0]
                    })
            
        except Exception as e:
            logger.error(f"Error extracting swaps from transaction: {str(e)}")
        
        return swaps
    
    def _analyze_token_performance(self, token_mint: str, buy_timestamp: Optional[int], 
                                 sell_timestamp: Optional[int] = None) -> Dict[str, Any]:
        """Analyze token performance with proper timestamp handling and resolution format."""
        if not self.birdeye_api:
            return {"success": False, "error": "Birdeye API not available"}
        
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
            
            # ROI score (max 20 points)
            if max_roi >= 500:
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
            
            if max_roi >= 1000:
                total_score *= 1.2
            elif max_roi >= 500:
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
        """Determine the wallet type based on metrics."""
        if not metrics:
            return "unknown"
            
        total_trades = metrics.get("total_trades", 0)
        if total_trades < 1:
            return "unknown"
        
        try:
            win_rate = metrics.get("win_rate", 0)
            median_roi = metrics.get("median_roi", 0)
            std_dev_roi = metrics.get("std_dev_roi", 0)
            avg_hold_time_hours = metrics.get("avg_hold_time_hours", 0)
            roi_distribution = metrics.get("roi_distribution", {})
            max_roi = metrics.get("max_roi", 0)
            profit_factor = metrics.get("profit_factor", 0)
            net_profit = metrics.get("net_profit_usd", 0)
            
            big_win_count = (roi_distribution.get("10x_plus", 0) + 
                           roi_distribution.get("5x_to_10x", 0) + 
                           roi_distribution.get("2x_to_5x", 0))
            big_win_ratio = big_win_count / total_trades if total_trades > 0 else 0
            
            if big_win_ratio >= 0.10 and max_roi >= 200:
                return "gem_finder"
            
            if avg_hold_time_hours < 24 and win_rate > 40:
                return "flipper"
            
            if win_rate >= 35 and median_roi > -10:
                return "consistent"
            
            if win_rate >= 30 or max_roi >= 100 or profit_factor >= 1.0 or net_profit > 0:
                return "mixed"
            
            if total_trades >= 5:
                return "underperformer"
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error determining wallet type: {str(e)}")
            return "unknown"
    
    def _generate_strategy(self, wallet_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a trading strategy based on wallet type and metrics."""
        try:
            if not metrics:
                metrics = {}
            
            composite_score = metrics.get("composite_score", 0)
            
            if wallet_type == "gem_finder":
                max_roi = metrics.get("max_roi", 0)
                if max_roi >= 500:
                    strategy = {
                        "recommendation": "HOLD_MOON",
                        "entry_type": "IMMEDIATE",
                        "position_size": "SMALL",
                        "take_profit_1": 100,
                        "take_profit_2": 200,
                        "take_profit_3": 500,
                        "stop_loss": -30,
                        "notes": f"Gem finder (Score: {composite_score}/100). Hold for moonshots."
                    }
                else:
                    strategy = {
                        "recommendation": "SCALP_AND_HOLD",
                        "entry_type": "IMMEDIATE",
                        "position_size": "SMALL",
                        "take_profit_1": 50,
                        "take_profit_2": 100,
                        "take_profit_3": 300,
                        "stop_loss": -30,
                        "notes": f"Potential gems (Score: {composite_score}/100). Take partials, hold rest."
                    }
            
            elif wallet_type == "consistent":
                win_rate = metrics.get("win_rate", 0)
                median_roi = metrics.get("median_roi", 0)
                if win_rate > 50 and median_roi > 20:
                    strategy = {
                        "recommendation": "FOLLOW_CLOSELY",
                        "entry_type": "IMMEDIATE",
                        "position_size": "MEDIUM",
                        "take_profit_1": 30,
                        "take_profit_2": 60,
                        "take_profit_3": 100,
                        "stop_loss": -20,
                        "notes": f"Consistent performer (Score: {composite_score}/100). Reliable signals."
                    }
                else:
                    strategy = {
                        "recommendation": "SCALP",
                        "entry_type": "IMMEDIATE",
                        "position_size": "MEDIUM",
                        "take_profit_1": 20,
                        "take_profit_2": 40,
                        "take_profit_3": 80,
                        "stop_loss": -20,
                        "notes": f"Steady trader (Score: {composite_score}/100). Quick profits."
                    }
            
            elif wallet_type == "flipper":
                strategy = {
                    "recommendation": "QUICK_SCALP",
                    "entry_type": "IMMEDIATE",
                    "position_size": "MEDIUM",
                    "take_profit_1": 15,
                    "take_profit_2": 30,
                    "take_profit_3": 50,
                    "stop_loss": -15,
                    "notes": f"Quick flipper (Score: {composite_score}/100). In and out fast."
                }
            
            elif wallet_type == "mixed":
                strategy = {
                    "recommendation": "SELECTIVE",
                    "entry_type": "WAIT_CONFIRMATION",
                    "position_size": "SMALL",
                    "take_profit_1": 25,
                    "take_profit_2": 50,
                    "take_profit_3": 100,
                    "stop_loss": -25,
                    "notes": f"Mixed results (Score: {composite_score}/100). Be selective."
                }
            
            elif wallet_type == "underperformer":
                strategy = {
                    "recommendation": "CAUTIOUS",
                    "entry_type": "WAIT_FOR_CONFIRMATION",
                    "position_size": "VERY_SMALL",
                    "take_profit_1": 20,
                    "take_profit_2": 40,
                    "take_profit_3": 80,
                    "stop_loss": -20,
                    "notes": f"Underperformer (Score: {composite_score}/100). High risk."
                }
            
            else:
                strategy = {
                    "recommendation": "CAUTIOUS",
                    "entry_type": "WAIT_FOR_CONFIRMATION",
                    "position_size": "VERY_SMALL",
                    "take_profit_1": 20,
                    "take_profit_2": 40,
                    "take_profit_3": 80,
                    "stop_loss": -20,
                    "notes": f"Low activity (Score: {composite_score}/100). Insufficient data."
                }
            
            if composite_score >= 80:
                strategy["confidence"] = "VERY_HIGH"
                strategy["position_size"] = "LARGE"
            elif composite_score >= 60:
                strategy["confidence"] = "HIGH"
            elif composite_score >= 40:
                strategy["confidence"] = "MEDIUM"
            elif composite_score >= 20:
                strategy["confidence"] = "LOW"
            else:
                strategy["confidence"] = "VERY_LOW"
                strategy["position_size"] = "VERY_SMALL"
            
            return strategy
                
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return {
                "recommendation": "CAUTIOUS",
                "entry_type": "WAIT_FOR_CONFIRMATION",
                "position_size": "SMALL",
                "take_profit_1": 20,
                "take_profit_2": 40,
                "take_profit_3": 80,
                "stop_loss": -20,
                "notes": "Error during strategy generation. Using cautious default.",
                "confidence": "LOW"
            }
    
    def analyze_wallet(self, wallet_address: str, days_back: int = 30) -> Dict[str, Any]:
        """Analyze a wallet for copy trading with Cielo Finance API."""
        logger.info(f"Analyzing wallet {wallet_address} using Cielo Finance API")
        
        try:
            token_pnl_data = self.cielo_api.get_wallet_trading_stats(wallet_address)
            
            if not token_pnl_data or not token_pnl_data.get("success", True):
                logger.warning(f"No token P&L data available for wallet {wallet_address}")
                
                empty_metrics = self._get_empty_metrics()
                return {
                    "success": True,
                    "wallet_address": wallet_address,
                    "analysis_period_days": "ALL_TIME",
                    "wallet_type": "unknown",
                    "composite_score": 0,
                    "metrics": empty_metrics,
                    "strategy": self._generate_strategy("unknown", empty_metrics),
                    "trades": [],
                    "correlated_wallets": [],
                    "api_source": "Cielo Finance + RPC",
                    "note": "No trading data available"
                }
            
            trades = self._extract_trades_from_cielo_trading_stats(token_pnl_data, wallet_address)
            paired_trades = self._pair_trades(trades)
            
            if not paired_trades:
                empty_metrics = self._get_empty_metrics()
                return {
                    "success": True,
                    "wallet_address": wallet_address,
                    "analysis_period_days": "ALL_TIME",
                    "wallet_type": "unknown", 
                    "composite_score": 0,
                    "metrics": empty_metrics,
                    "strategy": self._generate_strategy("unknown", empty_metrics),
                    "trades": [],
                    "api_source": "Cielo Finance + RPC",
                    "note": "No complete trades found"
                }
            
            metrics = self._calculate_metrics(paired_trades)
            
            metrics["avg_hold_time_minutes"] = round(metrics.get("avg_hold_time_hours", 0) * 60, 2)
            
            composite_score = self._calculate_composite_score(metrics)
            metrics["composite_score"] = composite_score
            
            wallet_type = self._determine_wallet_type(metrics)
            logger.info(f"Wallet type: {wallet_type}, Score: {composite_score}/100")
            
            strategy = self._generate_strategy(wallet_type, metrics)
            
            correlated_wallets = self._find_correlated_wallets(wallet_address)
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_period_days": "ALL_TIME",
                "wallet_type": wallet_type,
                "composite_score": composite_score,
                "metrics": metrics,
                "strategy": strategy,
                "trades": paired_trades,
                "correlated_wallets": correlated_wallets,
                "api_source": "Cielo Finance + RPC"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
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
                    "stop_loss": -20,
                    "notes": "Analysis failed. Use extreme caution.",
                    "confidence": "VERY_LOW"
                }
            }
    
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
    
    def batch_analyze_wallets(self, wallet_addresses: List[str], 
                            days_back: int = 30,
                            min_winrate: float = 30.0,
                            use_hybrid: bool = True) -> Dict[str, Any]:
        """Batch analyze multiple wallets with Cielo Finance API."""
        logger.info(f"Batch analyzing {len(wallet_addresses)} wallets using {'hybrid' if use_hybrid else 'Cielo-only'} approach")
        
        if not wallet_addresses:
            return {
                "success": False,
                "error": "No wallet addresses provided",
                "error_type": "NO_INPUT"
            }
        
        try:
            wallet_analyses = []
            failed_analyses = []
            
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
                        logger.info(f"  └─ Score: {score}/100, Type: {analysis.get('wallet_type', 'unknown')}")
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
            
            gem_finders = [a for a in wallet_analyses if a.get("wallet_type") == "gem_finder"]
            consistent = [a for a in wallet_analyses if a.get("wallet_type") == "consistent"]
            flippers = [a for a in wallet_analyses if a.get("wallet_type") == "flipper"]
            mixed = [a for a in wallet_analyses if a.get("wallet_type") == "mixed"]
            underperformers = [a for a in wallet_analyses if a.get("wallet_type") == "underperformer"]
            unknown = [a for a in wallet_analyses if a.get("wallet_type") == "unknown"]
            
            for category in [gem_finders, consistent, flippers, mixed, underperformers, unknown]:
                category.sort(key=lambda x: x.get("composite_score", x.get("metrics", {}).get("composite_score", 0)), reverse=True)
            
            wallet_clusters = self._identify_wallet_clusters(
                {a["wallet_address"]: a.get("correlated_wallets", []) for a in wallet_analyses}
            )
            
            return {
                "success": True,
                "total_wallets": len(wallet_addresses),
                "analyzed_wallets": len(wallet_analyses),
                "failed_wallets": len(failed_analyses),
                "filtered_wallets": len(wallet_analyses),
                "gem_finders": gem_finders,
                "consistent": consistent,
                "flippers": flippers,
                "mixed": mixed,
                "underperformers": underperformers,
                "unknown": unknown,
                "wallet_correlations": {a["wallet_address"]: a.get("correlated_wallets", []) for a in wallet_analyses},
                "wallet_clusters": wallet_clusters,
                "failed_analyses": failed_analyses,
                "api_source": "Hybrid (Cielo + RPC + Birdeye/Helius)" if use_hybrid else "Cielo Finance + RPC"
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
                writer.writerow({"metric": "gem_finder_count", "value": len(batch_analysis.get("gem_finders", []))})
                writer.writerow({"metric": "consistent_count", "value": len(batch_analysis.get("consistent", []))})
                writer.writerow({"metric": "flipper_count", "value": len(batch_analysis.get("flippers", []))})
                writer.writerow({"metric": "mixed_count", "value": len(batch_analysis.get("mixed", []))})
                writer.writerow({"metric": "underperformer_count", "value": len(batch_analysis.get("underperformers", []))})
                writer.writerow({"metric": "unknown_count", "value": len(batch_analysis.get("unknown", []))})
            
            logger.info(f"Exported batch analysis summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting batch analysis: {str(e)}")
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)