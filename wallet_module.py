"""
Wallet Analysis Module - Phoenix Project (FIXED VERSION WITH RATE LIMITING)

FIXES:
- Added RPC rate limiting to prevent 429 errors
- Implemented exponential backoff for retries
- Reduced concurrent workers to prevent overwhelming RPC
- Added request throttling between RPC calls
- Better error handling for rate limit errors
- Request batching where possible
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
    """Class for analyzing wallets for copy trading using Cielo Finance API + RPC."""
    
    def __init__(self, cielo_api: Any, birdeye_api: Any = None, rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """
        Initialize the wallet analyzer.
        
        Args:
            cielo_api: Cielo Finance API client (REQUIRED)
            birdeye_api: Birdeye API client (optional, for token metadata only)
            rpc_url: Solana RPC endpoint URL (P9 or other provider)
        """
        if not cielo_api:
            raise ValueError("Cielo Finance API is REQUIRED for wallet analysis")
        
        self.cielo_api = cielo_api
        self.birdeye_api = birdeye_api  # Optional, for token metadata only
        self.rpc_url = rpc_url
        
        # Verify Cielo Finance API connectivity
        if not self._verify_cielo_api_connection():
            raise CieloFinanceAPIError("Cannot connect to Cielo Finance API")
        
        # Track entry times for tokens to detect correlated wallets
        self.token_entries = {}  # token_address -> {wallet_address -> timestamp}
        
        # RPC cache for avoiding duplicate calls
        self._rpc_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 300  # 5 minutes TTL
        
        # Rate limiter for RPC calls (reduced to 5 calls per second)
        self._rate_limiter = RateLimiter(calls_per_second=5.0)
        
        # Thread pool for parallel processing (reduced workers)
        self.executor = ThreadPoolExecutor(max_workers=3)  # Reduced from 10 to 3
        
        # Track RPC errors
        self._rpc_error_count = 0
        self._last_rpc_error_time = 0
    
    def _verify_cielo_api_connection(self) -> bool:
        """
        Verify that the Cielo Finance API is accessible.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            if not self.cielo_api:
                return False
            # Try a simple API call to verify connectivity
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
                    # Remove expired entry
                    del self._rpc_cache[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store result in cache."""
        with self._cache_lock:
            self._rpc_cache[cache_key] = (data, time.time())
    
    def _make_rpc_call(self, method: str, params: List[Any], retry_count: int = 3) -> Dict[str, Any]:
        """
        Make direct RPC call to Solana node with caching and rate limiting.
        
        Args:
            method (str): RPC method name
            params (List[Any]): Method parameters
            retry_count (int): Number of retries on failure
            
        Returns:
            Dict[str, Any]: RPC response
        """
        # Check cache first
        cache_key = self._get_cache_key(method, params)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {method}")
            return cached_result
        
        # Apply rate limiting
        self._rate_limiter.wait()
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        # Exponential backoff for retries
        backoff_base = 2
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    self.rpc_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                # Handle rate limiting specifically
                if response.status_code == 429:
                    self._rpc_error_count += 1
                    self._last_rpc_error_time = time.time()
                    
                    # Calculate backoff time
                    backoff_time = backoff_base ** attempt
                    max_backoff = 60  # Max 60 seconds
                    wait_time = min(backoff_time, max_backoff)
                    
                    logger.warning(f"RPC rate limit hit (429). Waiting {wait_time}s before retry {attempt + 1}/{retry_count}")
                    time.sleep(wait_time)
                    
                    # If we're getting too many errors, slow down globally
                    if self._rpc_error_count > 10:
                        logger.warning("Too many RPC errors. Slowing down request rate.")
                        self._rate_limiter.calls_per_second = max(1.0, self._rate_limiter.calls_per_second * 0.5)
                        self._rate_limiter.min_interval = 1.0 / self._rate_limiter.calls_per_second
                    
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                # Cache successful responses
                if "result" in result:
                    self._set_cache(cache_key, result)
                    # Reset error count on success
                    if self._rpc_error_count > 0:
                        self._rpc_error_count = max(0, self._rpc_error_count - 1)
                    return result
                else:
                    # RPC error response
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
            # Don't log every single transaction error to reduce noise
            return {}
    
    def _batch_get_transactions(self, signatures: List[str]) -> List[Dict[str, Any]]:
        """
        Get multiple transactions in a more efficient way.
        Still makes individual calls but with better rate limiting.
        """
        transactions = []
        
        for i, signature in enumerate(signatures):
            # Add extra delay every 10 transactions
            if i > 0 and i % 10 == 0:
                logger.debug(f"Processed {i}/{len(signatures)} transactions, pausing...")
                time.sleep(2)  # 2 second pause every 10 transactions
            
            tx = self._get_transaction(signature)
            if tx:
                transactions.append(tx)
                
        return transactions
    
    def analyze_wallet_hybrid(self, wallet_address: str, days_back: int = 30) -> Dict[str, Any]:
        """
        UPDATED hybrid wallet analysis using Cielo Finance for aggregated stats and RPC for recent transactions.
        Now with better rate limiting and error handling.
        
        Args:
            wallet_address (str): Wallet address
            days_back (int): Number of days to analyze (for RPC calls)
            
        Returns:
            Dict[str, Any]: Wallet analysis results
        """
        logger.info(f"🔍 Analyzing wallet {wallet_address} (hybrid approach)")
        
        try:
            # Step 1: Get aggregated stats from Cielo Finance
            logger.info(f"📊 Fetching Cielo Finance aggregated stats...")
            cielo_stats = self.cielo_api.get_wallet_trading_stats(wallet_address)
            
            # Log the raw response structure
            if cielo_stats and cielo_stats.get("success", True) and "data" in cielo_stats:
                data = cielo_stats.get("data", {})
                if isinstance(data, dict):
                    logger.debug(f"Cielo response data keys: {list(data.keys())[:20]}")
                    # Log a sample of the data
                    for key in list(data.keys())[:5]:
                        value = data[key]
                        if isinstance(value, (int, float, str, bool)):
                            logger.debug(f"  {key}: {value}")
                else:
                    logger.debug(f"Cielo response data type: {type(data)}")
            
            if not cielo_stats or not cielo_stats.get("success", True):
                logger.warning(f"❌ No Cielo Finance data available for {wallet_address}")
                logger.info("Attempting RPC-only analysis as fallback...")
                aggregated_metrics = self._get_empty_cielo_metrics()
            else:
                # Extract aggregated metrics from Cielo
                stats_data = cielo_stats.get("data", {})
                aggregated_metrics = self._extract_aggregated_metrics_from_cielo(stats_data)
            
            # Step 2: Get recent token trades via RPC for detailed analysis (with rate limiting)
            logger.info(f"🪙 Analyzing last 5 tokens...")
            recent_swaps = self._get_recent_token_swaps_rpc(wallet_address, limit=5)
            
            # Step 3: Analyze token performance for recent trades (if Birdeye available)
            analyzed_trades = []
            if self.birdeye_api and recent_swaps:
                for i, swap in enumerate(recent_swaps[:5]):  # Analyze up to 5 most recent
                    # Add delay between Birdeye API calls
                    if i > 0:
                        time.sleep(0.5)
                        
                    token_analysis = self._analyze_token_performance(
                        swap['token_mint'],
                        swap['buy_timestamp'],
                        swap.get('sell_timestamp')
                    )
                    if token_analysis['success']:
                        analyzed_trades.append({
                            **swap,
                            **token_analysis
                        })
            
            # Step 4: Combine Cielo aggregated data with recent trade analysis
            combined_metrics = self._combine_metrics(aggregated_metrics, analyzed_trades)
            combined_metrics["wallet_address"] = wallet_address  # Add for score variation
            
            # Step 5: Calculate composite score for ALL wallets
            composite_score = self._calculate_composite_score(combined_metrics)
            combined_metrics["composite_score"] = composite_score
            
            # Step 6: Determine wallet type with looser thresholds
            wallet_type = self._determine_wallet_type(combined_metrics)
            
            # Step 7: Generate strategy
            strategy = self._generate_strategy(wallet_type, combined_metrics)
            
            # Analyze entry/exit behavior from recent trades
            entry_exit_analysis = self._analyze_entry_exit_behavior(analyzed_trades)
            
            logger.debug(f"Final analysis structure for {wallet_address}:")
            logger.debug(f"  - wallet_type: {wallet_type}")
            logger.debug(f"  - composite_score: {composite_score}")
            logger.debug(f"  - metrics keys: {list(combined_metrics.keys())}")
            logger.debug(f"  - total_trades: {combined_metrics.get('total_trades', 'MISSING')}")
            logger.debug(f"  - win_rate: {combined_metrics.get('win_rate', 'MISSING')}")
            logger.debug(f"  - net_profit_usd: {combined_metrics.get('net_profit_usd', 'MISSING')}")
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_period_days": "ALL_TIME (Cielo) + Recent (RPC)",
                "wallet_type": wallet_type,
                "composite_score": composite_score,
                "metrics": combined_metrics,
                "strategy": strategy,
                "trades": analyzed_trades,
                "entry_exit_analysis": entry_exit_analysis,
                "api_source": "Cielo Finance + RPC + Birdeye",
                "cielo_data": aggregated_metrics,
                "recent_trades_analyzed": len(analyzed_trades)
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid wallet analysis: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return a valid structure even on error
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
    
    def _extract_aggregated_metrics_from_cielo(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract aggregated metrics from Cielo Finance response - FIXED."""
        try:
            # Check if stats_data has the expected structure
            if not isinstance(stats_data, dict):
                logger.warning(f"Unexpected Cielo data format: {type(stats_data)}")
                return self._get_empty_cielo_metrics()
            
            # Log the actual structure we receive
            logger.debug(f"Cielo Finance response structure: {list(stats_data.keys())[:10]}")
            
            # Log sample data to understand the format
            for key in list(stats_data.keys())[:5]:
                value = stats_data[key]
                if isinstance(value, (int, float, str)):
                    logger.debug(f"  {key}: {value}")
                elif isinstance(value, dict):
                    logger.debug(f"  {key}: dict with keys {list(value.keys())[:5]}")
                elif isinstance(value, list):
                    logger.debug(f"  {key}: list with {len(value)} items")
            
            # Handle different possible response structures from Cielo Finance
            # The trading-stats endpoint should return wallet-level statistics
            
            # First, check if the data might be the stats directly (not nested)
            # Common fields from Cielo Finance trading-stats endpoint
            direct_fields = [
                "totalTrades", "winRate", "totalPnl", "avgTradeSize", "totalVolume",
                "bestTrade", "worstTrade", "avgHoldTime", "tokensTraded",
                "totalInvested", "totalRealized", "profitFactor", "sharpeRatio"
            ]
            
            # Check if we have direct fields at root level
            has_direct_fields = any(field in stats_data for field in direct_fields)
            
            if has_direct_fields:
                logger.debug("Found direct fields in Cielo response")
                # Extract directly from root level
                metrics = {
                    "total_trades": self._safe_get_numeric(stats_data, [
                        "totalTrades", "total_trades", "transaction_count", 
                        "transactionCount", "trades_count", "tradesCount", "txCount"
                    ], 0),
                    
                    "win_rate": self._safe_get_numeric(stats_data, [
                        "winRate", "win_rate", "win_percentage", "winPercentage", 
                        "winning_percentage", "winningPercentage", "profitRate"
                    ], 0),
                    
                    "total_pnl_usd": self._safe_get_numeric(stats_data, [
                        "totalPnl", "total_pnl_usd", "totalPnlUsd", "total_pnl",
                        "realized_pnl", "realizedPnl", "pnl", "netProfit"
                    ], 0),
                    
                    "avg_trade_size": self._safe_get_numeric(stats_data, [
                        "avgTradeSize", "avg_trade_size", "average_trade_size",
                        "averageTradeSize", "avg_investment", "avgInvestment"
                    ], 0),
                    
                    "total_volume": self._safe_get_numeric(stats_data, [
                        "totalVolume", "total_volume", "volume", "total_traded",
                        "totalTraded", "tradingVolume"
                    ], 0),
                    
                    "best_trade": self._safe_get_numeric(stats_data, [
                        "bestTrade", "best_trade", "max_profit", "maxProfit",
                        "highest_profit", "highestProfit", "largestWin"
                    ], 0),
                    
                    "worst_trade": self._safe_get_numeric(stats_data, [
                        "worstTrade", "worst_trade", "max_loss", "maxLoss",
                        "largest_loss", "largestLoss", "biggestLoss"
                    ], 0),
                    
                    "avg_hold_time": self._safe_get_numeric(stats_data, [
                        "avgHoldTime", "avg_hold_time", "average_hold_time",
                        "averageHoldTime", "avg_holding_time", "avgHoldingTime"
                    ], 0),
                    
                    "tokens_traded": self._safe_get_numeric(stats_data, [
                        "tokensTraded", "tokens_traded", "unique_tokens",
                        "uniqueTokens", "token_count", "tokenCount"
                    ], 0),
                    
                    # Additional fields that might be useful
                    "total_invested": self._safe_get_numeric(stats_data, [
                        "totalInvested", "total_invested", "totalBuy", "total_buy"
                    ], 0),
                    
                    "total_realized": self._safe_get_numeric(stats_data, [
                        "totalRealized", "total_realized", "totalSell", "total_sell"
                    ], 0)
                }
            else:
                # Try extracting with multiple fallback keys for nested structures
                metrics = {
                    "total_trades": self._safe_get_numeric(stats_data, [
                        "total_trades", "totalTrades", "transaction_count", 
                        "transactionCount", "trades_count", "tradesCount", "txCount"
                    ], 0),
                    
                    "win_rate": self._safe_get_numeric(stats_data, [
                        "win_rate", "winRate", "win_percentage", "winPercentage", 
                        "winning_percentage", "winningPercentage", "profitRate"
                    ], 0),
                    
                    "total_pnl_usd": self._safe_get_numeric(stats_data, [
                        "total_pnl_usd", "totalPnlUsd", "total_pnl", "totalPnl",
                        "realized_pnl", "realizedPnl", "pnl", "netProfit"
                    ], 0),
                    
                    "avg_trade_size": self._safe_get_numeric(stats_data, [
                        "avg_trade_size", "avgTradeSize", "average_trade_size",
                        "averageTradeSize", "avg_investment", "avgInvestment"
                    ], 0),
                    
                    "total_volume": self._safe_get_numeric(stats_data, [
                        "total_volume", "totalVolume", "volume", "total_traded",
                        "totalTraded", "tradingVolume"
                    ], 0),
                    
                    "best_trade": self._safe_get_numeric(stats_data, [
                        "best_trade", "bestTrade", "max_profit", "maxProfit",
                        "highest_profit", "highestProfit", "largestWin"
                    ], 0),
                    
                    "worst_trade": self._safe_get_numeric(stats_data, [
                        "worst_trade", "worstTrade", "max_loss", "maxLoss",
                        "largest_loss", "largestLoss", "biggestLoss"
                    ], 0),
                    
                    "avg_hold_time": self._safe_get_numeric(stats_data, [
                        "avg_hold_time", "avgHoldTime", "average_hold_time",
                        "averageHoldTime", "avg_holding_time", "avgHoldingTime"
                    ], 0),
                    
                    "tokens_traded": self._safe_get_numeric(stats_data, [
                        "tokens_traded", "tokensTraded", "unique_tokens",
                        "uniqueTokens", "token_count", "tokenCount"
                    ], 0),
                    
                    # Additional fields that might be useful
                    "total_invested": self._safe_get_numeric(stats_data, [
                        "totalInvested", "total_invested", "totalBuy", "total_buy"
                    ], 0),
                    
                    "total_realized": self._safe_get_numeric(stats_data, [
                        "totalRealized", "total_realized", "totalSell", "total_sell"
                    ], 0)
                }
            
            # If we got mostly zeros, try to extract from nested structures
            if all(v == 0 for k, v in metrics.items() if k != "worst_trade"):
                logger.info("Trying to extract from nested data structures...")
                
                # Check for nested data or summary fields
                if "summary" in stats_data:
                    return self._extract_aggregated_metrics_from_cielo(stats_data["summary"])
                elif "stats" in stats_data:
                    return self._extract_aggregated_metrics_from_cielo(stats_data["stats"])
                elif "trading" in stats_data:
                    return self._extract_aggregated_metrics_from_cielo(stats_data["trading"])
                elif "performance" in stats_data:
                    return self._extract_aggregated_metrics_from_cielo(stats_data["performance"])
            
            # Log what we extracted
            logger.debug(f"Extracted Cielo metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting Cielo metrics: {str(e)}")
            return self._get_empty_cielo_metrics()
    
    def _safe_get_numeric(self, data: Dict[str, Any], keys: List[str], default: float = 0) -> float:
        """Safely get a numeric value from dict trying multiple keys."""
        for key in keys:
            if key in data:
                value = data[key]
                try:
                    # Handle percentage values that might be strings like "45.2%"
                    if isinstance(value, str) and value.endswith('%'):
                        return float(value.rstrip('%'))
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    continue
        return default
    
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
            "total_realized": 0
        }
    
    def _get_recent_token_swaps_rpc(self, wallet_address: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent token swaps using RPC calls with better rate limiting."""
        try:
            logger.info(f"Fetching last {limit} token swaps for {wallet_address}")
            
            # Get recent signatures (with rate limiting applied)
            signatures = self._get_signatures_for_address(wallet_address, limit=50)  # Reduced from 100
            
            if not signatures:
                logger.warning(f"No transactions found for {wallet_address}")
                return []
            
            swaps = []
            sig_infos = []
            
            # Collect signatures first
            for sig_info in signatures:
                if len(swaps) >= limit:
                    break
                    
                signature = sig_info.get("signature")
                if signature:
                    sig_infos.append(sig_info)
            
            # Get transactions in smaller batches
            batch_size = 10
            for i in range(0, len(sig_infos), batch_size):
                batch = sig_infos[i:i + batch_size]
                batch_signatures = [s.get("signature") for s in batch if s.get("signature")]
                
                # Process batch with delay
                for signature in batch_signatures:
                    if len(swaps) >= limit:
                        break
                        
                    tx_details = self._get_transaction(signature)
                    if tx_details:
                        # Extract swap info
                        swap_info = self._extract_token_swaps_from_transaction(tx_details, wallet_address)
                        if swap_info:
                            swaps.extend(swap_info)
                
                # Add delay between batches
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
            
            # Get block time with proper fallback
            block_time = tx_details.get("blockTime")
            if not block_time or block_time == 0:
                # Use current time minus 1 day as fallback
                block_time = int((datetime.now() - timedelta(days=1)).timestamp())
                logger.debug(f"Using fallback timestamp: {block_time}")
            
            # Get token balances
            pre_balances = meta.get("preTokenBalances", [])
            post_balances = meta.get("postTokenBalances", [])
            
            # Track token changes
            token_changes = {}
            
            # Process pre-balances
            for balance in pre_balances:
                owner = balance.get("owner")
                if owner == wallet_address:
                    mint = balance.get("mint")
                    amount = int(balance.get("uiTokenAmount", {}).get("amount", 0))
                    if mint:
                        token_changes[mint] = {"pre": amount, "post": 0}
            
            # Process post-balances
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
            
            # Identify swaps (one token decreases, another increases)
            for mint, changes in token_changes.items():
                diff = changes["post"] - changes["pre"]
                
                # Token was bought (balance increased)
                if diff > 0:
                    swaps.append({
                        "token_mint": mint,
                        "type": "buy",
                        "amount": diff,
                        "buy_timestamp": block_time,
                        "sell_timestamp": None,
                        "signature": tx_details.get("transaction", {}).get("signatures", [""])[0]
                    })
                # Token was sold (balance decreased)
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
            # Ensure we have valid timestamps
            current_time = int(datetime.now().timestamp())
            
            # Handle missing or invalid buy timestamp
            if not buy_timestamp or buy_timestamp == 0:
                # Default to 7 days ago if no timestamp
                buy_timestamp = current_time - (7 * 24 * 60 * 60)
                logger.debug(f"Using default buy timestamp (7 days ago): {buy_timestamp}")
            
            # Ensure buy_timestamp is not in the future
            if buy_timestamp > current_time:
                buy_timestamp = current_time - (24 * 60 * 60)  # 1 day ago
                logger.debug(f"Adjusted future timestamp to 1 day ago: {buy_timestamp}")
            
            # Use sell timestamp or current time
            end_time = sell_timestamp if sell_timestamp and sell_timestamp > 0 else current_time
            
            # Get token info
            token_info = self.birdeye_api.get_token_info(token_mint)
            
            # Use proper resolution format
            resolution = "1H"  # Default
            
            # Calculate time difference to choose appropriate resolution
            time_diff = end_time - buy_timestamp
            if time_diff < 3600:  # Less than 1 hour
                resolution = "5m"
            elif time_diff < 86400:  # Less than 1 day
                resolution = "15m"
            elif time_diff < 259200:  # Less than 3 days
                resolution = "1H"
            elif time_diff < 604800:  # Less than 1 week
                resolution = "4H"
            else:
                resolution = "1D"
            
            logger.debug(f"Analyzing token {token_mint} from {buy_timestamp} to {end_time} with resolution {resolution}")
            
            # Get price history
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
                    "error": "Failed to get price history"
                }
            
            # Calculate performance metrics
            performance = self.birdeye_api.calculate_token_performance(
                token_mint,
                datetime.fromtimestamp(buy_timestamp)
            )
            
            # Add entry/exit timing analysis
            if performance.get("success"):
                performance["entry_timing"] = self._analyze_entry_timing(performance)
                performance["exit_timing"] = self._analyze_exit_timing(performance, sell_timestamp is not None)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error analyzing token performance: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_entry_timing(self, performance: Dict[str, Any]) -> str:
        """Analyze if entry timing was good based on subsequent performance."""
        try:
            max_roi = performance.get("max_roi_percent", 0)
            roi_to_max_time = performance.get("time_to_max_roi_hours", 0)
            
            if max_roi < 20:
                return "POOR"  # Token didn't perform well
            elif roi_to_max_time < 1:
                return "EXCELLENT"  # Caught the pump early
            elif roi_to_max_time < 6:
                return "GOOD"  # Good entry timing
            else:
                return "LATE"  # Late entry
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
            
            # Check if sold too early or too late
            roi_capture_ratio = current_roi / max_roi if max_roi > 0 else 0
            
            if roi_capture_ratio >= 0.8:
                return "EXCELLENT"  # Captured most of the gains
            elif roi_capture_ratio >= 0.5:
                return "GOOD"  # Captured decent portion
            elif roi_capture_ratio >= 0.2:
                return "EARLY"  # Sold too early, missed gains
            else:
                return "VERY_EARLY"  # Sold way too early
                
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
                "recommendations": "Need more trade data for analysis"
            }
        
        entry_timings = []
        exit_timings = []
        missed_gains = []
        
        for trade in analyzed_trades:
            if "entry_timing" in trade:
                entry_timings.append(trade["entry_timing"])
            if "exit_timing" in trade:
                exit_timings.append(trade["exit_timing"])
            
            # Calculate missed gains
            if trade.get("current_roi_percent", 0) > 0 and trade.get("max_roi_percent", 0) > 0:
                missed = trade["max_roi_percent"] - trade["current_roi_percent"]
                if missed > 0:
                    missed_gains.append(missed)
        
        # Analyze patterns
        good_entries = sum(1 for e in entry_timings if e in ["EXCELLENT", "GOOD"])
        good_exits = sum(1 for e in exit_timings if e in ["EXCELLENT", "GOOD"])
        early_exits = sum(1 for e in exit_timings if e in ["EARLY", "VERY_EARLY"])
        
        avg_missed_gains = np.mean(missed_gains) if missed_gains else 0
        
        # Determine patterns
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
            "trades_analyzed": len(analyzed_trades)
        }
    
    def _combine_metrics(self, cielo_metrics: Dict[str, Any], 
                        analyzed_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine Cielo aggregated metrics with recent trade analysis - FIXED."""
        try:
            # Start with Cielo metrics or empty metrics
            combined = cielo_metrics.copy() if cielo_metrics else self._get_empty_cielo_metrics()
            
            # Log what we received
            logger.debug(f"Combining metrics - Cielo data: total_trades={combined.get('total_trades', 0)}, "
                        f"win_rate={combined.get('win_rate', 0)}, pnl={combined.get('total_pnl_usd', 0)}")
            
            # Calculate derived metrics from Cielo data
            total_trades = combined.get("total_trades", 0)
            win_rate = combined.get("win_rate", 0)
            total_pnl = combined.get("total_pnl_usd", 0)
            best_trade = combined.get("best_trade", 0)
            worst_trade = combined.get("worst_trade", 0)
            total_volume = combined.get("total_volume", 0)
            total_invested = combined.get("total_invested", 0)
            total_realized = combined.get("total_realized", 0)
            
            # If we have total_invested and total_realized, calculate PnL from them
            if total_invested > 0 and total_realized > 0 and total_pnl == 0:
                total_pnl = total_realized - total_invested
                combined["total_pnl_usd"] = total_pnl
            
            # Calculate win/loss counts
            if total_trades > 0 and win_rate > 0:
                win_count = int(total_trades * (win_rate / 100))
                loss_count = total_trades - win_count
            else:
                win_count = 0
                loss_count = total_trades
            
            # Calculate profit/loss breakdown
            if win_count > 0 and best_trade > 0:
                total_profit_usd = best_trade * win_count * 0.5  # Estimate
            else:
                total_profit_usd = max(0, total_pnl)
            
            if loss_count > 0 and worst_trade < 0:
                total_loss_usd = abs(worst_trade * loss_count * 0.5)  # Estimate
            else:
                total_loss_usd = max(0, -total_pnl) if total_pnl < 0 else 0
            
            # If we don't have good profit/loss data, derive from PnL
            if total_profit_usd == 0 and total_loss_usd == 0 and total_pnl != 0:
                if total_pnl > 0:
                    total_profit_usd = total_pnl
                    total_loss_usd = 0
                else:
                    total_profit_usd = 0
                    total_loss_usd = abs(total_pnl)
            
            # Calculate profit factor
            if total_loss_usd > 0:
                profit_factor = total_profit_usd / total_loss_usd
            elif total_profit_usd > 0:
                profit_factor = float('inf')
            else:
                profit_factor = 0
            
            # Calculate average ROI
            if total_volume > 0:
                avg_roi = (total_pnl / total_volume) * 100
            elif total_invested > 0 and total_pnl != 0:
                avg_roi = (total_pnl / total_invested) * 100
            else:
                avg_roi = 0
            
            # Add metrics from analyzed trades if available
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
                # Use estimates based on Cielo data
                median_roi = avg_roi * 0.8  # Conservative estimate
                if best_trade > 0 and total_volume > 0:
                    max_roi = (best_trade / (total_volume / total_trades)) * 100 if total_trades > 0 else 0
                else:
                    max_roi = 100 if win_rate > 50 else 50
                
                if worst_trade < 0 and total_volume > 0:
                    min_roi = (worst_trade / (total_volume / total_trades)) * 100 if total_trades > 0 else 0
                else:
                    min_roi = -50
                
                std_dev_roi = abs(max_roi - min_roi) / 4  # Rough estimate
            
            # Build complete metrics dictionary
            complete_metrics = {
                # Basic counts
                "total_trades": total_trades,
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": win_rate,
                
                # Profit/Loss metrics
                "total_profit_usd": total_profit_usd,
                "total_loss_usd": total_loss_usd,
                "net_profit_usd": total_pnl,  # This is the key field that was missing!
                "profit_factor": profit_factor,
                
                # ROI metrics
                "avg_roi": avg_roi,
                "median_roi": median_roi,
                "std_dev_roi": std_dev_roi,
                "max_roi": max_roi,
                "min_roi": min_roi,
                
                # Other metrics
                "avg_hold_time_hours": combined.get("avg_hold_time", 0),
                "total_bet_size_usd": total_volume,
                "avg_bet_size_usd": combined.get("avg_trade_size", 0),
                "total_tokens_traded": combined.get("tokens_traded", 0),
                
                # ROI distribution (estimate if no detailed data)
                "roi_distribution": self._estimate_roi_distribution(
                    total_trades, win_rate, max_roi, analyzed_trades
                )
            }
            
            return complete_metrics
            
        except Exception as e:
            logger.error(f"Error combining metrics: {str(e)}")
            # Return safe default metrics with all required fields
            return self._get_empty_metrics()
    
    def _estimate_roi_distribution(self, total_trades: int, win_rate: float, 
                                  max_roi: float, analyzed_trades: List[Dict[str, Any]]) -> Dict[str, int]:
        """Estimate ROI distribution from available data."""
        if analyzed_trades:
            # Calculate from actual trades
            roi_buckets = {
                "10x_plus": 0,
                "5x_to_10x": 0,
                "2x_to_5x": 0,
                "1x_to_2x": 0,
                "50_to_100": 0,
                "0_to_50": 0,
                "minus50_to_0": 0,
                "below_minus50": 0
            }
            
            for trade in analyzed_trades:
                roi = trade.get("roi_percent", 0)
                if roi >= 1000:
                    roi_buckets["10x_plus"] += 1
                elif roi >= 500:
                    roi_buckets["5x_to_10x"] += 1
                elif roi >= 200:
                    roi_buckets["2x_to_5x"] += 1
                elif roi >= 100:
                    roi_buckets["1x_to_2x"] += 1
                elif roi >= 50:
                    roi_buckets["50_to_100"] += 1
                elif roi >= 0:
                    roi_buckets["0_to_50"] += 1
                elif roi >= -50:
                    roi_buckets["minus50_to_0"] += 1
                else:
                    roi_buckets["below_minus50"] += 1
            
            return roi_buckets
        else:
            # Estimate distribution based on win rate and max ROI
            win_count = int(total_trades * (win_rate / 100))
            loss_count = total_trades - win_count
            
            # Rough distribution estimate
            roi_buckets = {
                "10x_plus": max(0, int(win_count * 0.01)) if max_roi >= 1000 else 0,
                "5x_to_10x": max(0, int(win_count * 0.02)) if max_roi >= 500 else 0,
                "2x_to_5x": max(0, int(win_count * 0.05)) if max_roi >= 200 else 0,
                "1x_to_2x": max(0, int(win_count * 0.1)) if max_roi >= 100 else 0,
                "50_to_100": max(0, int(win_count * 0.2)),
                "0_to_50": max(0, win_count - sum([
                    max(0, int(win_count * 0.01)) if max_roi >= 1000 else 0,
                    max(0, int(win_count * 0.02)) if max_roi >= 500 else 0,
                    max(0, int(win_count * 0.05)) if max_roi >= 200 else 0,
                    max(0, int(win_count * 0.1)) if max_roi >= 100 else 0,
                    max(0, int(win_count * 0.2))
                ])),
                "minus50_to_0": max(0, int(loss_count * 0.7)),
                "below_minus50": max(0, loss_count - int(loss_count * 0.7))
            }
            
            return roi_buckets
    
    def _calculate_composite_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate composite score (0-100) for ALL wallets.
        Adjusted for Solana memecoins with looser criteria.
        """
        try:
            # Get metrics with safe defaults
            total_trades = metrics.get("total_trades", 0)
            win_rate = metrics.get("win_rate", 0)
            profit_factor = metrics.get("profit_factor", 0)
            avg_roi = metrics.get("avg_roi", 0)
            max_roi = metrics.get("max_roi", 0)
            median_roi = metrics.get("median_roi", 0)
            net_profit = metrics.get("net_profit_usd", 0)
            
            # Base score components (adjusted for memecoins)
            
            # 1. Activity score (max 20 points)
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
            
            # 2. Win rate score (max 20 points) - LOOSENED
            if win_rate >= 60:
                winrate_score = 20
            elif win_rate >= 45:
                winrate_score = 15
            elif win_rate >= 30:  # Lowered threshold
                winrate_score = 10
            elif win_rate >= 20:
                winrate_score = 5
            elif win_rate >= 10:
                winrate_score = 3
            else:
                winrate_score = 0
            
            # 3. Profit factor score (max 20 points) - LOOSENED
            if profit_factor >= 2.0:
                pf_score = 20
            elif profit_factor >= 1.5:
                pf_score = 15
            elif profit_factor >= 1.0:  # Lowered threshold
                pf_score = 10
            elif profit_factor >= 0.8:
                pf_score = 5
            elif profit_factor >= 0.5:
                pf_score = 3
            else:
                pf_score = 0
            
            # 4. ROI score (max 20 points) - Focus on max potential
            if max_roi >= 500:  # 5x or more
                roi_score = 20
            elif max_roi >= 200:  # 2x or more
                roi_score = 15
            elif max_roi >= 100:  # 1x or more
                roi_score = 10
            elif max_roi >= 50:
                roi_score = 5
            elif max_roi >= 0:
                roi_score = 3
            else:
                roi_score = 0
            
            # 5. Consistency score (max 20 points) - Based on median ROI and profit
            consistency_points = 0
            
            # Median ROI component
            if median_roi >= 50:
                consistency_points += 10
            elif median_roi >= 20:
                consistency_points += 7
            elif median_roi >= 0:  # Break even or better
                consistency_points += 5
            elif median_roi >= -10:  # Small losses acceptable
                consistency_points += 3
            else:
                consistency_points += 0
            
            # Net profit component
            if net_profit > 0:
                consistency_points += 10
            elif net_profit >= -100:  # Small loss acceptable
                consistency_points += 5
            else:
                consistency_points += 0
            
            consistency_score = min(20, consistency_points)
            
            # Calculate total score
            total_score = (
                activity_score +
                winrate_score +
                pf_score +
                roi_score +
                consistency_score
            )
            
            # Apply bonus multipliers for exceptional performance
            if max_roi >= 1000:  # 10x achieved
                total_score *= 1.2
            elif max_roi >= 500:  # 5x achieved
                total_score *= 1.1
            
            # Additional bonus for profitable traders
            if net_profit > 1000:
                total_score *= 1.1
            elif net_profit > 100:
                total_score *= 1.05
            
            # Cap at 100
            total_score = min(100, total_score)
            
            # Ensure minimum score for active wallets
            if total_trades > 0:
                total_score = max(2, total_score)
            
            # Give base score of 18 for wallets with some data but poor performance
            if total_trades > 0 and total_score < 18:
                total_score = 18
            
            # Add small random variation to avoid identical scores
            # This helps differentiate wallets with similar metrics
            if total_trades == 0:
                # For wallets with no trades, add small variation based on available data
                import hashlib
                # Use any available unique identifier
                unique_str = str(metrics.get("wallet_address", "")) + str(metrics.get("total_volume", "")) + str(metrics.get("tokens_traded", ""))
                hash_val = int(hashlib.md5(unique_str.encode()).hexdigest()[:4], 16)
                variation = (hash_val % 10) / 10.0  # 0.0 to 0.9
                total_score += variation
            
            return round(total_score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return 0
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure with ALL required fields."""
        return {
            "total_trades": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0,
            "total_profit_usd": 0,
            "total_loss_usd": 0,
            "net_profit_usd": 0,  # CRITICAL: This field was missing!
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
        """
        Determine the wallet type based on metrics.
        LOOSENED thresholds for Solana memecoins.
        """
        # Ensure metrics exist
        if not metrics:
            return "unknown"
            
        total_trades = metrics.get("total_trades", 0)
        if total_trades < 1:  # Changed from 3
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
            
            # Calculate key indicators
            big_win_count = (roi_distribution.get("10x_plus", 0) + 
                           roi_distribution.get("5x_to_10x", 0) + 
                           roi_distribution.get("2x_to_5x", 0))
            big_win_ratio = big_win_count / total_trades if total_trades > 0 else 0
            
            # LOOSENED THRESHOLDS
            # Gem finder: Lower requirements
            if big_win_ratio >= 0.10 and max_roi >= 200:  # Was 0.15 and 300
                return "gem_finder"
            
            # Flipper: More lenient on hold time and win rate
            if avg_hold_time_hours < 24 and win_rate > 40:  # Was < 12 and > 50
                return "flipper"
            
            # Consistent: Lower bar for consistency
            if win_rate >= 35 and median_roi > -10:  # Was >= 45 and > 0
                return "consistent"
            
            # Mixed: If has some positive traits
            if win_rate >= 30 or max_roi >= 100 or profit_factor >= 1.0 or net_profit > 0:
                return "mixed"
            
            # Underperformer: Active but poor results
            if total_trades >= 5:
                return "underperformer"
            
            # Unknown: Low activity
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error determining wallet type: {str(e)}")
            return "unknown"
    
    def _generate_strategy(self, wallet_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading strategy based on wallet type and metrics.
        """
        try:
            # Ensure metrics exist
            if not metrics:
                metrics = {}
            
            # Get composite score
            composite_score = metrics.get("composite_score", 0)
            
            # Base strategies by wallet type
            if wallet_type == "gem_finder":
                max_roi = metrics.get("max_roi", 0)
                if max_roi >= 500:  # 5x+ potential
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
            
            else:  # unknown
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
            
            # Adjust based on composite score
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
        """
        Analyze a wallet for copy trading with Cielo Finance API.
        
        Args:
            wallet_address (str): Wallet address
            days_back (int): Number of days to analyze (not used for Cielo Finance API)
            
        Returns:
            Dict[str, Any]: Wallet analysis results
        """
        logger.info(f"Analyzing wallet {wallet_address} using Cielo Finance API")
        
        try:
            # Get wallet token P&L data from Cielo Finance API
            token_pnl_data = self.cielo_api.get_wallet_trading_stats(wallet_address)
            
            if not token_pnl_data or not token_pnl_data.get("success", True):
                logger.warning(f"No token P&L data available for wallet {wallet_address}")
                
                # Return valid structure with empty metrics instead of error
                empty_metrics = self._get_empty_metrics()
                return {
                    "success": True,  # Mark as success but with empty data
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
            
            # Extract trades from Cielo Finance token P&L data
            trades = self._extract_trades_from_cielo_trading_stats(token_pnl_data, wallet_address)
            paired_trades = self._pair_trades(trades)
            
            if not paired_trades:
                # Return with empty metrics rather than error
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
            
            # Calculate metrics
            metrics = self._calculate_metrics(paired_trades)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(metrics)
            metrics["composite_score"] = composite_score
            
            # Determine wallet type
            wallet_type = self._determine_wallet_type(metrics)
            logger.info(f"Wallet type: {wallet_type}, Score: {composite_score}/100")
            
            # Generate strategy
            strategy = self._generate_strategy(wallet_type, metrics)
            
            # Find correlated wallets
            correlated_wallets = self._find_correlated_wallets(wallet_address)
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_period_days": "ALL_TIME",  # Cielo Finance provides all-time data
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
            
            # Return a valid structure even on error
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
        """
        Extract and categorize trades from Cielo Finance token P&L data - FIXED.
        """
        if not trading_stats_data or not trading_stats_data.get("success", True):
            logger.warning(f"No valid token P&L data for wallet {wallet_address}")
            return []
        
        trades = []
        
        try:
            # Extract data from Cielo Finance response
            stats_data = trading_stats_data.get("data", {})
            if not isinstance(stats_data, dict):
                logger.warning(f"Unexpected token P&L data format for wallet {wallet_address}")
                return []
            
            # Try different possible structures
            # 1. Check for token list directly in data
            if "tokens" in stats_data:
                for token_data in stats_data["tokens"]:
                    processed_trade = self._process_cielo_token_pnl(token_data, wallet_address)
                    if processed_trade:
                        trades.append(processed_trade)
            
            # 2. Check for token array at root
            elif isinstance(stats_data, list):
                for token_data in stats_data:
                    processed_trade = self._process_cielo_token_pnl(token_data, wallet_address)
                    if processed_trade:
                        trades.append(processed_trade)
            
            # 3. Create a single aggregated trade from overall stats
            else:
                # Try to create a summary trade from wallet-level stats
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
            # Create a trade record from Cielo Finance token P&L data
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
            # Only create summary if we have meaningful data
            if not any(self._safe_get_numeric(stats_data, [key], 0) != 0 
                      for key in ["total_trades", "totalTrades", "transaction_count", 
                                  "transactionCount", "win_rate", "winRate"]):
                return None
            
            # Extract key metrics
            total_trades = self._safe_get_numeric(stats_data, [
                "total_trades", "totalTrades", "transaction_count", "transactionCount"
            ], 0)
            
            win_rate = self._safe_get_numeric(stats_data, [
                "win_rate", "winRate", "win_percentage", "winPercentage"
            ], 0)
            
            total_pnl = self._safe_get_numeric(stats_data, [
                "total_pnl_usd", "totalPnlUsd", "total_pnl", "totalPnl", "pnl"
            ], 0)
            
            total_invested = self._safe_get_numeric(stats_data, [
                "total_invested", "totalInvested", "total_buy", "totalBuy"
            ], 0)
            
            total_realized = self._safe_get_numeric(stats_data, [
                "total_realized", "totalRealized", "total_sell", "totalSell"
            ], 0)
            
            # Calculate ROI
            roi_percent = 0
            if total_invested > 0:
                roi_percent = ((total_realized - total_invested) / total_invested) * 100
            elif total_pnl != 0 and total_invested > 0:
                roi_percent = (total_pnl / total_invested) * 100
            
            # Create summary trade
            return {
                "token_address": "",
                "token_symbol": f"WALLET_SUMMARY_{total_trades}_TRADES",
                "buy_timestamp": stats_data.get("first_trade_time", stats_data.get("start_time", 0)),
                "buy_date": "",
                "buy_price": 0,
                "buy_value_usd": total_invested,
                "buy_value_sol": 0,
                "sell_timestamp": stats_data.get("last_trade_time", stats_data.get("end_time", 0)),
                "sell_date": "",
                "sell_price": 0,
                "sell_value_usd": total_realized,
                "sell_value_sol": 0,
                "holding_time_hours": self._safe_get_numeric(stats_data, ["avg_hold_time", "avgHoldTime"], 0),
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
        
        # Cielo Finance API should already provide paired trades with P&L calculated
        # So we just validate and clean the data
        paired_trades = []
        
        for trade in trades:
            # Ensure required fields exist with safe defaults
            trade_copy = trade.copy()
            trade_copy.setdefault("buy_value_usd", 0)
            trade_copy.setdefault("sell_value_usd", 0)
            trade_copy.setdefault("roi_percent", 0)
            trade_copy.setdefault("is_win", trade_copy.get("roi_percent", 0) > 0)
            
            # Only include trades with actual buy value
            if trade_copy.get("buy_value_usd", 0) > 0:
                paired_trades.append(trade_copy)
        
        logger.info(f"Validated {len(paired_trades)} paired trades from {len(trades)} trades")
        return paired_trades
    
    def _calculate_metrics(self, paired_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from paired trades - FIXED."""
        if not paired_trades:
            logger.warning("No paired trades for metrics calculation")
            return self._get_empty_metrics()
        
        try:
            # Basic counts
            total_trades = len(paired_trades)
            win_count = sum(1 for trade in paired_trades if trade.get("is_win", False))
            loss_count = total_trades - win_count
            
            # ROI statistics
            roi_values = [trade.get("roi_percent", 0) for trade in paired_trades]
            avg_roi = np.mean(roi_values) if roi_values else 0
            median_roi = np.median(roi_values) if roi_values else 0
            std_dev_roi = np.std(roi_values) if roi_values else 0
            max_roi = max(roi_values) if roi_values else 0
            min_roi = min(roi_values) if roi_values else 0
            
            # Profit/loss calculations
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
            profit_factor = total_profit_usd / total_loss_usd if total_loss_usd > 0 else float('inf') if total_profit_usd > 0 else 0
            
            # Holding time
            holding_times = [trade.get("holding_time_hours", 0) for trade in paired_trades if "holding_time_hours" in trade and trade.get("holding_time_hours", 0) > 0]
            avg_hold_time_hours = np.mean(holding_times) if holding_times else 0
            
            # Bet size
            bet_sizes = [trade.get("buy_value_usd", 0) for trade in paired_trades]
            total_bet_size_usd = sum(bet_sizes)
            avg_bet_size_usd = np.mean(bet_sizes) if bet_sizes else 0
            
            # Unique tokens
            unique_tokens = len(set(trade.get("token_address", "") for trade in paired_trades if trade.get("token_address")))
            
            # ROI distribution buckets
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
        """
        Batch analyze multiple wallets with Cielo Finance API.
        Added parallel processing with better rate limiting.
        Lowered thresholds.
        
        Args:
            wallet_addresses (List[str]): List of wallet addresses
            days_back (int): Number of days to analyze (not used for Cielo Finance)
            min_winrate (float): Minimum win rate percentage (lowered to 30%)
            use_hybrid (bool): Whether to use hybrid analysis (Cielo + RPC + Birdeye)
            
        Returns:
            Dict[str, Any]: Categorized wallet analyses
        """
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
            
            # Process wallets sequentially to avoid overwhelming RPC
            for i, wallet_address in enumerate(wallet_addresses, 1):
                logger.info(f"Analyzing wallet {i}/{len(wallet_addresses)}: {wallet_address}")
                
                try:
                    if use_hybrid:
                        analysis = self.analyze_wallet_hybrid(wallet_address, days_back)
                    else:
                        analysis = self.analyze_wallet(wallet_address, days_back)
                    
                    # Always add to analyses if we have metrics
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
                    
                    # Add delay between wallets to respect rate limits
                    if i < len(wallet_addresses):
                        time.sleep(2)  # 2 second delay between wallets
                        
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
            
            # NO FILTERING - Show all wallets regardless of win rate or profit factor
            # Just categorize them
            
            # Categorize wallets
            gem_finders = [a for a in wallet_analyses if a.get("wallet_type") == "gem_finder"]
            consistent = [a for a in wallet_analyses if a.get("wallet_type") == "consistent"]
            flippers = [a for a in wallet_analyses if a.get("wallet_type") == "flipper"]
            mixed = [a for a in wallet_analyses if a.get("wallet_type") == "mixed"]
            underperformers = [a for a in wallet_analyses if a.get("wallet_type") == "underperformer"]
            unknown = [a for a in wallet_analyses if a.get("wallet_type") == "unknown"]
            
            # Sort each category by composite score
            for category in [gem_finders, consistent, flippers, mixed, underperformers, unknown]:
                category.sort(key=lambda x: x.get("composite_score", x.get("metrics", {}).get("composite_score", 0)), reverse=True)
            
            # Find wallet clusters
            wallet_clusters = self._identify_wallet_clusters(
                {a["wallet_address"]: a.get("correlated_wallets", []) for a in wallet_analyses}
            )
            
            return {
                "success": True,
                "total_wallets": len(wallet_addresses),
                "analyzed_wallets": len(wallet_analyses),
                "failed_wallets": len(failed_analyses),
                "filtered_wallets": len(wallet_analyses),  # No filtering, so same as analyzed
                "gem_finders": gem_finders,
                "consistent": consistent,
                "flippers": flippers,
                "mixed": mixed,
                "underperformers": underperformers,
                "unknown": unknown,
                "wallet_correlations": {a["wallet_address"]: a.get("correlated_wallets", []) for a in wallet_analyses},
                "wallet_clusters": wallet_clusters,
                "failed_analyses": failed_analyses,
                "api_source": "Hybrid (Cielo + RPC + Birdeye)" if use_hybrid else "Cielo Finance + RPC"
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
    
    # Keep all existing export methods
    def export_wallet_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Export wallet analysis to CSV."""
        if not analysis.get("success"):
            logger.warning(f"Cannot export failed analysis for {analysis.get('wallet_address')}: {analysis.get('error')}")
            return
        
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Export main metrics
            metrics_file = output_file.replace(".csv", "_metrics.csv")
            with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ["metric", "value"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Basic info
                writer.writerow({"metric": "wallet_address", "value": analysis["wallet_address"]})
                writer.writerow({"metric": "wallet_type", "value": analysis["wallet_type"]})
                writer.writerow({"metric": "composite_score", "value": analysis.get("composite_score", 0)})
                writer.writerow({"metric": "api_source", "value": analysis.get("api_source", "Unknown")})
                
                # Strategy info
                strategy = analysis.get("strategy", {})
                writer.writerow({"metric": "recommendation", "value": strategy.get("recommendation", "CAUTIOUS")})
                writer.writerow({"metric": "entry_type", "value": strategy.get("entry_type", "WAIT_FOR_CONFIRMATION")})
                writer.writerow({"metric": "confidence", "value": strategy.get("confidence", "LOW")})
                
                # Performance metrics
                metrics = analysis.get("metrics", {})
                for key, value in metrics.items():
                    if key != "roi_distribution":
                        writer.writerow({"metric": key, "value": value})
                
                # ROI distribution
                roi_dist = metrics.get("roi_distribution", {})
                if isinstance(roi_dist, dict):
                    for key, value in roi_dist.items():
                        writer.writerow({"metric": f"roi_{key}", "value": value})
                
                # Entry/exit analysis
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
            
            # Summary file
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