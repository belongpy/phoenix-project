"""
Wallet Analysis Module - Phoenix Project (UPDATED)

CHANGES:
- Removed contested wallet analysis completely
- Increased token swap analysis from 3 to 5 tokens
- Lowered win rate threshold to 30%
- Lowered min profit factor requirements
- Always categorize wallets (no pre-filtering)
- Show all wallets
- Added parallel processing for wallet analysis
- Calculate composite score for all wallets
- Loosened thresholds for Solana memecoins:
  - Gem Finder: big_win_ratio >= 0.10 AND max_roi >= 200
  - Consistent: win_rate >= 35 AND median_roi > -10
  - Flipper: avg_hold_time < 24 hours AND win_rate > 40
- Added RPC result caching
"""

import csv
import os
import logging
import numpy as np
import requests
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import time

logger = logging.getLogger("phoenix.wallet")

class CieloFinanceAPIError(Exception):
    """Custom exception for Cielo Finance API errors."""
    pass

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
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=10)
    
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
    
    def _make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """
        Make direct RPC call to Solana node with caching.
        
        Args:
            method (str): RPC method name
            params (List[Any]): Method parameters
            
        Returns:
            Dict[str, Any]: RPC response
        """
        # Check cache first
        cache_key = self._get_cache_key(method, params)
        cached_result = self._get_from_cache(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for {method}")
            return cached_result
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            response = requests.post(
                self.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache successful responses
            if "result" in result:
                self._set_cache(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"RPC call failed for {method}: {str(e)}")
            return {"error": str(e)}
    
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
            logger.error(f"Error getting transaction: {response.get('error', 'Unknown error')}")
            return {}
    
    def analyze_wallet_hybrid(self, wallet_address: str, days_back: int = 30) -> Dict[str, Any]:
        """
        UPDATED hybrid wallet analysis using Cielo Finance for aggregated stats and RPC for recent transactions.
        REMOVED contested analysis.
        
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
            
            if not cielo_stats or not cielo_stats.get("success", True):
                logger.warning(f"❌ No Cielo Finance data available for {wallet_address}")
                
                # Try to continue with RPC-only analysis instead of returning error
                logger.info("Attempting RPC-only analysis as fallback...")
                aggregated_metrics = self._get_empty_cielo_metrics()
            else:
                # Extract aggregated metrics from Cielo
                stats_data = cielo_stats.get("data", {})
                aggregated_metrics = self._extract_aggregated_metrics_from_cielo(stats_data)
            
            # Step 2: Get recent token trades via RPC for detailed analysis (INCREASED TO 5)
            logger.info(f"🪙 Analyzing last 5 tokens...")
            recent_swaps = self._get_recent_token_swaps_rpc(wallet_address, limit=5)
            
            # Step 3: Analyze token performance for recent trades (if Birdeye available)
            analyzed_trades = []
            if self.birdeye_api and recent_swaps:
                for swap in recent_swaps[:5]:  # Analyze up to 5 most recent
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
            
            # Ensure metrics have all required fields
            if not combined_metrics:
                combined_metrics = self._get_empty_metrics()
            else:
                # Ensure profit_factor exists
                combined_metrics.setdefault("profit_factor", 0)
                combined_metrics.setdefault("win_rate", 0)
                combined_metrics.setdefault("median_roi", 0)
                combined_metrics.setdefault("max_roi", 0)
            
            # Step 5: Calculate composite score for ALL wallets
            composite_score = self._calculate_composite_score(combined_metrics)
            combined_metrics["composite_score"] = composite_score
            
            # Step 6: Determine wallet type with looser thresholds
            wallet_type = self._determine_wallet_type(combined_metrics)
            
            # Step 7: Generate strategy (REMOVED contested analysis parameter)
            strategy = self._generate_strategy(wallet_type, combined_metrics)
            
            # Analyze entry/exit behavior from recent trades
            entry_exit_analysis = self._analyze_entry_exit_behavior(analyzed_trades)
            
            # Log the final structure to ensure it's complete
            logger.debug(f"Final analysis structure for {wallet_address}:")
            logger.debug(f"  - wallet_type: {wallet_type}")
            logger.debug(f"  - composite_score: {composite_score}")
            logger.debug(f"  - metrics keys: {list(combined_metrics.keys())}")
            logger.debug(f"  - profit_factor: {combined_metrics.get('profit_factor', 'MISSING')}")
            logger.debug(f"  - win_rate: {combined_metrics.get('win_rate', 'MISSING')}")
            
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
        """Extract aggregated metrics from Cielo Finance response."""
        try:
            # Handle different response formats from Cielo
            if isinstance(stats_data, dict):
                metrics = {
                    "total_trades": stats_data.get("total_trades", stats_data.get("transaction_count", 0)),
                    "win_rate": stats_data.get("win_rate", stats_data.get("winning_percentage", 0)),
                    "total_pnl_usd": stats_data.get("total_pnl", stats_data.get("realized_pnl", 0)),
                    "avg_trade_size": stats_data.get("avg_trade_size", stats_data.get("avg_investment", 0)),
                    "total_volume": stats_data.get("total_volume", stats_data.get("total_traded", 0)),
                    "best_trade": stats_data.get("best_trade", stats_data.get("max_profit", 0)),
                    "worst_trade": stats_data.get("worst_trade", stats_data.get("max_loss", 0)),
                    "avg_hold_time": stats_data.get("avg_hold_time", 0),
                    "tokens_traded": stats_data.get("tokens_traded", stats_data.get("unique_tokens", 0))
                }
                
                # Ensure all required metrics exist
                return metrics
            
            # Return empty metrics if data format is unexpected
            return self._get_empty_cielo_metrics()
            
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
            "tokens_traded": 0
        }
    
    def _get_recent_token_swaps_rpc(self, wallet_address: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent token swaps using RPC calls (INCREASED TO 5)."""
        try:
            logger.info(f"Fetching last {limit} token swaps for {wallet_address}")
            
            # Get recent signatures
            signatures = self._get_signatures_for_address(wallet_address, limit=100)
            
            if not signatures:
                logger.warning(f"No transactions found for {wallet_address}")
                return []
            
            swaps = []
            for sig_info in signatures:
                if len(swaps) >= limit:
                    break
                
                signature = sig_info.get("signature")
                if not signature:
                    continue
                
                # Get transaction details
                tx_details = self._get_transaction(signature)
                if not tx_details:
                    continue
                
                # Extract swap info
                swap_info = self._extract_token_swaps_from_transaction(tx_details, wallet_address)
                if swap_info:
                    swaps.extend(swap_info)
            
            logger.info(f"Found {len(swaps)} swaps from {len(signatures)} transactions")
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
            
            # Use proper resolution format (uppercase)
            resolution = "1H"  # Changed from "1h" to "1H"
            
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
        """Combine Cielo aggregated metrics with recent trade analysis."""
        try:
            # Start with Cielo metrics or empty metrics
            combined = cielo_metrics.copy() if cielo_metrics else self._get_empty_cielo_metrics()
            
            # Ensure essential fields exist
            combined.setdefault("total_trades", 0)
            combined.setdefault("win_rate", 0)
            combined.setdefault("total_volume", 0)
            combined.setdefault("best_trade", 0)
            combined.setdefault("worst_trade", 0)
            
            # Add recent trade metrics if available
            if analyzed_trades:
                recent_rois = [t.get("roi_percent", 0) for t in analyzed_trades if "roi_percent" in t]
                if recent_rois:
                    combined["recent_avg_roi"] = np.mean(recent_rois)
                    combined["recent_max_roi"] = max(recent_rois)
                    combined["recent_trades_analyzed"] = len(analyzed_trades)
            
            # Calculate additional metrics
            if combined.get("total_trades", 0) > 0:
                # Safe division for profit factor
                best_trade = combined.get("best_trade", 0)
                worst_trade = combined.get("worst_trade", 0)
                if worst_trade != 0:
                    combined["profit_factor"] = abs(best_trade / worst_trade)
                else:
                    combined["profit_factor"] = float('inf') if best_trade > 0 else 0
                
                # Safe division for avg ROI
                total_pnl = combined.get("total_pnl_usd", 0)
                total_volume = combined.get("total_volume", 0)
                if total_volume > 0:
                    combined["avg_roi"] = (total_pnl / total_volume) * 100
                else:
                    combined["avg_roi"] = 0
                    
                # Calculate median ROI if we have trade data
                if analyzed_trades:
                    rois = [t.get("roi_percent", 0) for t in analyzed_trades if "roi_percent" in t]
                    combined["median_roi"] = np.median(rois) if rois else 0
                else:
                    combined["median_roi"] = 0
                
                # Calculate max ROI
                if analyzed_trades:
                    max_rois = [t.get("max_roi_percent", 0) for t in analyzed_trades if "max_roi_percent" in t]
                    combined["max_roi"] = max(max_rois) if max_rois else 0
                else:
                    combined["max_roi"] = 0
            else:
                # Ensure these metrics exist even if no trades
                combined["profit_factor"] = 0
                combined["avg_roi"] = 0
                combined["median_roi"] = 0
                combined["max_roi"] = 0
            
            # Add hold time in hours
            if combined.get("avg_hold_time", 0) > 0:
                combined["avg_hold_time_hours"] = combined["avg_hold_time"]
            else:
                combined["avg_hold_time_hours"] = 0
            
            # Add total tokens traded
            combined["total_tokens_traded"] = combined.get("tokens_traded", 0)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining metrics: {str(e)}")
            # Return safe default metrics
            safe_metrics = self._get_empty_cielo_metrics()
            safe_metrics["profit_factor"] = 0
            safe_metrics["avg_roi"] = 0
            safe_metrics["median_roi"] = 0
            safe_metrics["max_roi"] = 0
            safe_metrics["avg_hold_time_hours"] = 0
            safe_metrics["total_tokens_traded"] = 0
            return safe_metrics
    
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
            else:
                activity_score = 2  # Give some points even for low activity
            
            # 2. Win rate score (max 20 points) - LOOSENED
            if win_rate >= 60:
                winrate_score = 20
            elif win_rate >= 45:
                winrate_score = 15
            elif win_rate >= 30:  # Lowered threshold
                winrate_score = 10
            elif win_rate >= 20:
                winrate_score = 5
            else:
                winrate_score = 2
            
            # 3. Profit factor score (max 20 points) - LOOSENED
            if profit_factor >= 2.0:
                pf_score = 20
            elif profit_factor >= 1.5:
                pf_score = 15
            elif profit_factor >= 1.0:  # Lowered threshold
                pf_score = 10
            elif profit_factor >= 0.8:
                pf_score = 5
            else:
                pf_score = 2
            
            # 4. ROI score (max 20 points) - Focus on max potential
            if max_roi >= 500:  # 5x or more
                roi_score = 20
            elif max_roi >= 200:  # 2x or more
                roi_score = 15
            elif max_roi >= 100:  # 1x or more
                roi_score = 10
            elif max_roi >= 50:
                roi_score = 5
            else:
                roi_score = 2
            
            # 5. Consistency score (max 20 points) - Based on median ROI
            if median_roi >= 50:
                consistency_score = 20
            elif median_roi >= 20:
                consistency_score = 15
            elif median_roi >= 0:  # Break even or better
                consistency_score = 10
            elif median_roi >= -10:  # Small losses acceptable
                consistency_score = 5
            else:
                consistency_score = 2
            
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
            
            # Cap at 100
            total_score = min(100, total_score)
            
            # Ensure minimum score of 1 for active wallets
            if total_trades > 0:
                total_score = max(1, total_score)
            
            return round(total_score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return 0
    
    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
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
        """
        Determine the wallet type based on metrics.
        LOOSENED thresholds for Solana memecoins.
        """
        # Ensure metrics exist
        if not metrics:
            return "unknown"
            
        total_trades = metrics.get("total_trades", 0)
        if total_trades < 3:  # Lowered from 5
            return "unknown"
        
        try:
            win_rate = metrics.get("win_rate", 0)
            median_roi = metrics.get("median_roi", 0)
            std_dev_roi = metrics.get("std_dev_roi", 0)
            avg_hold_time_hours = metrics.get("avg_hold_time_hours", 0)
            roi_distribution = metrics.get("roi_distribution", {})
            max_roi = metrics.get("max_roi", 0)
            
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
            
            # If doesn't fit categories but has some positive traits
            if win_rate >= 30 or max_roi >= 100 or profit_factor >= 1.0:
                return "mixed"
            
            return "underperformer"
            
        except Exception as e:
            logger.error(f"Error determining wallet type: {str(e)}")
            return "unknown"
    
    def _generate_strategy(self, wallet_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading strategy based on wallet type and metrics.
        REMOVED contested analysis parameter.
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
            
            else:  # unknown or underperformer
                strategy = {
                    "recommendation": "CAUTIOUS",
                    "entry_type": "WAIT_FOR_CONFIRMATION",
                    "position_size": "VERY_SMALL",
                    "take_profit_1": 20,
                    "take_profit_2": 40,
                    "take_profit_3": 80,
                    "stop_loss": -20,
                    "notes": f"Low confidence (Score: {composite_score}/100). Proceed with caution."
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
        REMOVED contested analysis.
        
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
    
    def _extract_trades_from_cielo_trading_stats(self, trading_stats_data: Dict[str, Any], wallet_address: str) -> List[Dict[str, Any]]:
        """
        Extract and categorize trades from Cielo Finance token P&L data.
        Updated to handle the correct /pnl/tokens endpoint response format.
        
        Args:
            trading_stats_data (Dict[str, Any]): Cielo Finance /pnl/tokens response
            wallet_address (str): The wallet address being analyzed
            
        Returns:
            List[Dict[str, Any]]: Categorized trades
        """
        if not trading_stats_data or not trading_stats_data.get("success", True):
            logger.warning(f"No valid token P&L data for wallet {wallet_address}")
            return []
        
        trades = []
        
        try:
            # Extract data from Cielo Finance /pnl/tokens format
            stats_data = trading_stats_data.get("data", {})
            if not isinstance(stats_data, dict):
                logger.warning(f"Unexpected token P&L data format for wallet {wallet_address}")
                return []
            
            # The /pnl/tokens endpoint returns token-level P&L data
            # Check for different possible data structures from Cielo Finance
            
            if "tokens" in stats_data:
                # Process tokens with P&L data
                for token_data in stats_data["tokens"]:
                    processed_trade = self._process_cielo_token_pnl(token_data, wallet_address)
                    if processed_trade:
                        trades.append(processed_trade)
            
            elif isinstance(stats_data, list):
                # Data might be returned as a direct list of tokens
                for token_data in stats_data:
                    processed_trade = self._process_cielo_token_pnl(token_data, wallet_address)
                    if processed_trade:
                        trades.append(processed_trade)
            
            elif "pnl" in stats_data or "totalPnl" in stats_data:
                # Process aggregated P&L data
                processed_trade = self._process_cielo_aggregated_pnl(stats_data, wallet_address)
                if processed_trade:
                    trades.append(processed_trade)
            
            else:
                # Try to extract from root level data
                logger.info(f"Attempting to extract from root level P&L data for wallet {wallet_address}")
                processed_trade = self._process_cielo_root_pnl_data(stats_data, wallet_address)
                if processed_trade:
                    trades.append(processed_trade)
            
            logger.info(f"Extracted {len(trades)} trades from Cielo Finance P&L data for wallet {wallet_address}")
            return trades
            
        except Exception as e:
            logger.error(f"Error extracting trades from Cielo Finance P&L data for wallet {wallet_address}: {str(e)}")
            return []
    
    def _process_cielo_token_pnl(self, token_data: Dict[str, Any], wallet_address: str) -> Optional[Dict[str, Any]]:
        """Process token P&L data from Cielo Finance /pnl/tokens endpoint."""
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
    
    def _process_cielo_aggregated_pnl(self, pnl_data: Dict[str, Any], wallet_address: str) -> Optional[Dict[str, Any]]:
        """Process aggregated P&L data from Cielo Finance."""
        try:
            # Create an aggregate trade record from P&L data
            return {
                "token_address": "",
                "token_symbol": "AGGREGATE",
                "buy_timestamp": pnl_data.get("first_tx_time", 0),
                "buy_date": datetime.fromtimestamp(pnl_data.get("first_tx_time", 0)).isoformat() if pnl_data.get("first_tx_time") else "",
                "buy_price": 0,
                "buy_value_usd": pnl_data.get("total_invested_usd", pnl_data.get("total_buy_usd", 0)),
                "buy_value_sol": pnl_data.get("total_invested_sol", pnl_data.get("total_buy_sol", 0)),
                "sell_timestamp": pnl_data.get("last_tx_time", 0),
                "sell_date": datetime.fromtimestamp(pnl_data.get("last_tx_time", 0)).isoformat() if pnl_data.get("last_tx_time") else "",
                "sell_price": 0,
                "sell_value_usd": pnl_data.get("total_realized_usd", pnl_data.get("total_sell_usd", 0)),
                "sell_value_sol": pnl_data.get("total_realized_sol", pnl_data.get("total_sell_sol", 0)),
                "holding_time_hours": pnl_data.get("avg_holding_time", 0),
                "roi_percent": pnl_data.get("total_pnl_percent", pnl_data.get("pnl_percent", 0)),
                "is_win": pnl_data.get("total_pnl_percent", pnl_data.get("pnl_percent", 0)) > 0,
                "market_cap_at_buy": 0,
                "platform": "MULTIPLE",
                "total_trades": pnl_data.get("total_trades", pnl_data.get("tx_count", 0)),
                "realized_pnl_usd": pnl_data.get("realized_pnl_usd", 0),
                "unrealized_pnl_usd": pnl_data.get("unrealized_pnl_usd", 0)
            }
        except Exception as e:
            logger.warning(f"Error processing Cielo aggregated P&L data: {str(e)}")
            return None
    
    def _process_cielo_root_pnl_data(self, stats_data: Dict[str, Any], wallet_address: str) -> Optional[Dict[str, Any]]:
        """Process root level Cielo Finance P&L data."""
        try:
            # Create a summary trade from overall P&L stats
            return {
                "token_address": "",
                "token_symbol": "WALLET_SUMMARY",
                "buy_timestamp": stats_data.get("first_trade_time", stats_data.get("start_time", 0)),
                "buy_date": datetime.fromtimestamp(stats_data.get("first_trade_time", stats_data.get("start_time", 0))).isoformat() if stats_data.get("first_trade_time", stats_data.get("start_time")) else "",
                "buy_price": 0,
                "buy_value_usd": stats_data.get("total_invested", stats_data.get("total_buy", 0)),
                "buy_value_sol": stats_data.get("total_invested_sol", 0),
                "sell_timestamp": stats_data.get("last_trade_time", stats_data.get("end_time", 0)),
                "sell_date": datetime.fromtimestamp(stats_data.get("last_trade_time", stats_data.get("end_time", 0))).isoformat() if stats_data.get("last_trade_time", stats_data.get("end_time")) else "",
                "sell_price": 0,
                "sell_value_usd": stats_data.get("total_realized", stats_data.get("total_sell", 0)),
                "sell_value_sol": stats_data.get("total_realized_sol", 0),
                "holding_time_hours": stats_data.get("avg_hold_time", 0),
                "roi_percent": stats_data.get("pnl_percentage", stats_data.get("total_pnl_percent", 0)),
                "is_win": stats_data.get("pnl_percentage", stats_data.get("total_pnl_percent", 0)) > 0,
                "market_cap_at_buy": 0,
                "platform": "MULTIPLE",
                "total_trades": stats_data.get("transaction_count", stats_data.get("total_transactions", 0)),
                "realized_pnl_usd": stats_data.get("realized_pnl", 0),
                "unrealized_pnl_usd": stats_data.get("unrealized_pnl", 0)
            }
        except Exception as e:
            logger.warning(f"Error processing Cielo root P&L data: {str(e)}")
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
        """Calculate performance metrics from paired trades."""
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
            
            # Profit/loss
            winning_trades = [t for t in paired_trades if t.get("is_win", False)]
            losing_trades = [t for t in paired_trades if not t.get("is_win", False)]
            
            total_profit_usd = sum((t.get("sell_value_usd", 0) - t.get("buy_value_usd", 0)) for t in winning_trades)
            total_loss_usd = abs(sum((t.get("sell_value_usd", 0) - t.get("buy_value_usd", 0)) for t in losing_trades))
            net_profit_usd = total_profit_usd - total_loss_usd
            profit_factor = total_profit_usd / total_loss_usd if total_loss_usd > 0 else float('inf')
            
            # Holding time
            holding_times = [trade.get("holding_time_hours", 0) for trade in paired_trades if "holding_time_hours" in trade]
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
                            min_winrate: float = 30.0,  # LOWERED from 45.0
                            use_hybrid: bool = True) -> Dict[str, Any]:
        """
        Batch analyze multiple wallets with Cielo Finance API.
        REMOVED contested analysis parameter.
        Added parallel processing.
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
            
            # Use parallel processing for wallet analysis
            futures = []
            
            for wallet_address in wallet_addresses:
                if use_hybrid:
                    future = self.executor.submit(self.analyze_wallet_hybrid, wallet_address, days_back)
                else:
                    future = self.executor.submit(self.analyze_wallet, wallet_address, days_back)
                futures.append((wallet_address, future))
            
            # Collect results
            for i, (wallet_address, future) in enumerate(futures, 1):
                try:
                    logger.info(f"Collecting results for wallet {i}/{len(wallet_addresses)}: {wallet_address}")
                    analysis = future.result(timeout=60)  # 60 second timeout per wallet
                    
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