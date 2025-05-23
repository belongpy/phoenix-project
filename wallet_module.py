"""
Wallet Analysis Module - Phoenix Project (FIXED)

FIXED ISSUES:
- Birdeye API resolution format (1H not 1h)
- Proper timestamp extraction from transactions
- Better error handling for missing timestamps
- Fixed price history resolution parameters
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
    
    def _make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """
        Make direct RPC call to Solana node.
        
        Args:
            method (str): RPC method name
            params (List[Any]): Method parameters
            
        Returns:
            Dict[str, Any]: RPC response
        """
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
            return response.json()
        except Exception as e:
            logger.error(f"RPC call failed for {method}: {str(e)}")
            return {"error": str(e)}
    
    def _get_signatures_for_address(self, address: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get transaction signatures for an address using direct RPC."""
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
        """Get transaction details by signature using direct RPC."""
        response = self._make_rpc_call(
            "getTransaction",
            [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0}]
        )
        
        if "result" in response:
            return response["result"]
        else:
            logger.error(f"Error getting transaction: {response.get('error', 'Unknown error')}")
            return {}
    
    def analyze_wallet_hybrid(self, wallet_address: str, days_back: int = 30, 
                            include_contested: bool = True) -> Dict[str, Any]:
        """
        FIXED hybrid wallet analysis using Cielo Finance for aggregated stats and RPC for recent transactions.
        
        Args:
            wallet_address (str): Wallet address
            days_back (int): Number of days to analyze (for RPC calls)
            include_contested (bool): Whether to include contested analysis
            
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
            
            # Step 2: Get recent token trades via RPC for detailed analysis
            logger.info(f"🪙 Analyzing last 3 tokens...")
            recent_swaps = self._get_recent_token_swaps_rpc(wallet_address, limit=3)
            
            # Step 3: Analyze token performance for recent trades (if Birdeye available)
            analyzed_trades = []
            if self.birdeye_api and recent_swaps:
                for swap in recent_swaps[:3]:  # Limit to 3 most recent
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
            
            # Step 5: Determine wallet type
            wallet_type = self._determine_wallet_type(combined_metrics)
            
            # Step 6: Contested analysis (if requested)
            contested_analysis = {}
            if include_contested:
                logger.info("🎯 Analyzing contested level...")
                contested_analysis = self.analyze_contested_level(wallet_address)
            
            # Step 7: Generate strategy
            strategy = self._generate_strategy(wallet_type, combined_metrics, contested_analysis)
            
            # Log the final structure to ensure it's complete
            logger.debug(f"Final analysis structure for {wallet_address}:")
            logger.debug(f"  - wallet_type: {wallet_type}")
            logger.debug(f"  - metrics keys: {list(combined_metrics.keys())}")
            logger.debug(f"  - profit_factor: {combined_metrics.get('profit_factor', 'MISSING')}")
            logger.debug(f"  - win_rate: {combined_metrics.get('win_rate', 'MISSING')}")
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_period_days": "ALL_TIME (Cielo) + Recent (RPC)",
                "wallet_type": wallet_type,
                "metrics": combined_metrics,
                "strategy": strategy,
                "trades": analyzed_trades,
                "contested_analysis": contested_analysis,
                "api_source": "Cielo Finance + RPC + Birdeye",
                "cielo_data": aggregated_metrics,
                "recent_trades_analyzed": len(analyzed_trades)
            }
            
        except Exception as e:
            logger.error(f"Error in hybrid wallet analysis: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return a valid structure even on error
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "wallet_address": wallet_address,
                "error_type": "UNEXPECTED_ERROR",
                "wallet_type": "unknown",
                "metrics": self._get_empty_metrics(),
                "strategy": {
                    "recommendation": "CAUTIOUS",
                    "entry_type": "WAIT_FOR_CONFIRMATION",
                    "position_size": "SMALL",
                    "take_profit_1": 20,
                    "take_profit_2": 40,
                    "take_profit_3": 80,
                    "stop_loss": -20,
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
    
    def _get_recent_token_swaps_rpc(self, wallet_address: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get recent token swaps using RPC calls."""
        try:
            logger.info(f"Fetching last {limit} token swaps for {wallet_address}")
            
            # Get recent signatures
            signatures = self._get_signatures_for_address(wallet_address, limit=50)
            
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
        """FIXED: Extract token swap information from transaction with proper timestamp handling."""
        swaps = []
        
        try:
            if not tx_details or "meta" not in tx_details:
                return []
            
            meta = tx_details["meta"]
            
            # FIXED: Get block time with proper fallback
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
                        "buy_timestamp": block_time,  # Use the fixed timestamp
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
                        "sell_timestamp": block_time,  # Use the fixed timestamp
                        "signature": tx_details.get("transaction", {}).get("signatures", [""])[0]
                    })
            
        except Exception as e:
            logger.error(f"Error extracting swaps from transaction: {str(e)}")
        
        return swaps
    
    def _analyze_token_performance(self, token_mint: str, buy_timestamp: Optional[int], 
                                 sell_timestamp: Optional[int] = None) -> Dict[str, Any]:
        """FIXED: Analyze token performance with proper timestamp handling and resolution format."""
        if not self.birdeye_api:
            return {"success": False, "error": "Birdeye API not available"}
        
        try:
            # FIXED: Ensure we have valid timestamps
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
            
            # FIXED: Use proper resolution format (uppercase)
            # Valid resolutions: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 8H, 12H, 1D, 3D, 1W, 1M
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
                resolution  # Now using proper format
            )
            
            if not history_response.get("success"):
                logger.warning(f"Failed to get price history for {token_mint}")
                return {
                    "success": False,
                    "error": "Failed to get price history"
                }
            
            # Calculate performance metrics
            return self.birdeye_api.calculate_token_performance(
                token_mint,
                datetime.fromtimestamp(buy_timestamp)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing token performance: {str(e)}")
            return {
                "success": False,
                "error": str(e)
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
            else:
                # Ensure these metrics exist even if no trades
                combined["profit_factor"] = 0
                combined["avg_roi"] = 0
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining metrics: {str(e)}")
            # Return safe default metrics
            safe_metrics = self._get_empty_cielo_metrics()
            safe_metrics["profit_factor"] = 0
            safe_metrics["avg_roi"] = 0
            return safe_metrics
    
    def analyze_contested_level(self, wallet_address: str, max_transactions: int = 50) -> Dict[str, Any]:
        """
        Analyze how contested a wallet is (how many copy traders follow it).
        
        Args:
            wallet_address (str): Target wallet address
            max_transactions (int): Maximum recent transactions to analyze
            
        Returns:
            Dict[str, Any]: Contested analysis results
        """
        logger.info(f"Analyzing contested level for wallet {wallet_address}")
        
        try:
            # Get recent transactions for the wallet
            signatures = self._get_signatures_for_address(wallet_address, max_transactions)
            
            if not signatures:
                return {
                    "success": False,
                    "error": "No transactions found for wallet",
                    "contested_level": 0,
                    "copy_traders": []
                }
            
            copy_traders = {}
            token_interactions = {}
            
            # Analyze each transaction
            for sig_data in signatures[:max_transactions]:
                signature = sig_data.get("signature")
                block_time = sig_data.get("blockTime")
                slot = sig_data.get("slot")
                
                if not signature or not slot:
                    continue
                
                # Get full transaction details
                tx_details = self._get_transaction(signature)
                if not tx_details:
                    continue
                
                # Extract token interactions from the transaction
                token_mints = self._extract_token_mints_from_transaction(tx_details)
                
                for token_mint in token_mints:
                    if token_mint not in token_interactions:
                        token_interactions[token_mint] = []
                    
                    token_interactions[token_mint].append({
                        "wallet": wallet_address,
                        "signature": signature,
                        "slot": slot,
                        "block_time": block_time,
                        "is_target": True
                    })
                    
                    # Find other wallets that interacted with the same token around the same time
                    similar_traders = self._find_similar_token_interactions(token_mint, slot, 5)  # 5 block threshold
                    
                    for trader in similar_traders:
                        trader_wallet = trader.get("wallet")
                        if trader_wallet and trader_wallet != wallet_address:
                            if trader_wallet not in copy_traders:
                                copy_traders[trader_wallet] = {
                                    "wallet_address": trader_wallet,
                                    "common_tokens": set(),
                                    "interactions": [],
                                    "avg_block_delay": [],
                                    "copy_confidence": 0
                                }
                            
                            copy_traders[trader_wallet]["common_tokens"].add(token_mint)
                            copy_traders[trader_wallet]["interactions"].append({
                                "token_mint": token_mint,
                                "target_slot": slot,
                                "copy_slot": trader.get("slot"),
                                "block_delay": abs(trader.get("slot", 0) - slot),
                                "target_signature": signature,
                                "copy_signature": trader.get("signature")
                            })
                            copy_traders[trader_wallet]["avg_block_delay"].append(
                                abs(trader.get("slot", 0) - slot)
                            )
            
            # Calculate copy trader statistics
            analyzed_copy_traders = []
            for wallet, data in copy_traders.items():
                common_token_count = len(data["common_tokens"])
                avg_delay = np.mean(data["avg_block_delay"]) if data["avg_block_delay"] else 0
                interaction_count = len(data["interactions"])
                
                # Calculate copy confidence score (0-100)
                confidence = min(100, (common_token_count * 20) + 
                               (interaction_count * 10) + 
                               (max(0, 50 - avg_delay * 2)))
                
                if confidence >= 30 or common_token_count >= 2:  # Minimum threshold
                    analyzed_copy_traders.append({
                        "wallet_address": wallet,
                        "common_tokens": common_token_count,
                        "total_interactions": interaction_count,
                        "avg_block_delay": round(avg_delay, 2),
                        "copy_confidence": round(confidence, 2),
                        "interactions": data["interactions"]
                    })
            
            # Sort by copy confidence
            analyzed_copy_traders.sort(key=lambda x: x["copy_confidence"], reverse=True)
            
            # Calculate overall contested level
            if not analyzed_copy_traders:
                contested_level = 0
                classification = "NOT_CONTESTED"
            else:
                high_confidence_count = len([t for t in analyzed_copy_traders if t["copy_confidence"] >= 60])
                medium_confidence_count = len([t for t in analyzed_copy_traders if 30 <= t["copy_confidence"] < 60])
                
                contested_level = min(100, (high_confidence_count * 25) + (medium_confidence_count * 10))
                
                if contested_level >= 75:
                    classification = "HIGHLY_CONTESTED"
                elif contested_level >= 50:
                    classification = "MODERATELY_CONTESTED"
                elif contested_level >= 25:
                    classification = "LIGHTLY_CONTESTED"
                else:
                    classification = "NOT_CONTESTED"
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "contested_level": contested_level,
                "classification": classification,
                "total_copy_traders": len(analyzed_copy_traders),
                "high_confidence_copy_traders": len([t for t in analyzed_copy_traders if t["copy_confidence"] >= 60]),
                "copy_traders": analyzed_copy_traders[:10],  # Limit to top 10
                "analysis_summary": {
                    "transactions_analyzed": len(signatures),
                    "unique_tokens_found": len(token_interactions)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing contested level: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "contested_level": 0,
                "copy_traders": []
            }
    
    def _extract_token_mints_from_transaction(self, tx_details: Dict[str, Any]) -> List[str]:
        """Extract token mint addresses from transaction details."""
        token_mints = set()
        
        try:
            if "meta" in tx_details and "postTokenBalances" in tx_details["meta"]:
                for balance in tx_details["meta"]["postTokenBalances"]:
                    if "mint" in balance:
                        token_mints.add(balance["mint"])
            
            if "meta" in tx_details and "preTokenBalances" in tx_details["meta"]:
                for balance in tx_details["meta"]["preTokenBalances"]:
                    if "mint" in balance:
                        token_mints.add(balance["mint"])
        except Exception as e:
            logger.debug(f"Error extracting token mints: {str(e)}")
        
        return list(token_mints)
    
    def _find_similar_token_interactions(self, token_mint: str, target_slot: int, block_threshold: int = 5) -> List[Dict[str, Any]]:
        """Find other wallets that interacted with the same token around the same time."""
        similar_interactions = []
        
        try:
            # Get recent signatures for this token
            token_signatures = self._get_signatures_for_address(token_mint, 30)
            
            for sig_data in token_signatures:
                slot = sig_data.get("slot")
                signature = sig_data.get("signature")
                
                if not slot or not signature:
                    continue
                
                # Check if this interaction is within the block threshold  
                if abs(slot - target_slot) <= block_threshold:
                    # Get transaction details to find the wallet
                    tx_details = self._get_transaction(signature)
                    if tx_details and "transaction" in tx_details:
                        wallets = self._extract_wallet_addresses_from_transaction(tx_details)
                        
                        for wallet in wallets:
                            similar_interactions.append({
                                "wallet": wallet,
                                "signature": signature,
                                "slot": slot,
                                "block_time": sig_data.get("blockTime"),
                                "token_mint": token_mint
                            })
        except Exception as e:
            logger.debug(f"Error finding similar interactions: {str(e)}")
        
        return similar_interactions
    
    def _extract_wallet_addresses_from_transaction(self, tx_details: Dict[str, Any]) -> List[str]:
        """Extract wallet addresses from transaction."""
        wallets = set()
        
        try:
            if "transaction" in tx_details and "message" in tx_details["transaction"]:
                account_keys = tx_details["transaction"]["message"].get("accountKeys", [])
                for account in account_keys:
                    if isinstance(account, str) and len(account) >= 32:
                        wallets.add(account)
        except Exception as e:
            logger.debug(f"Error extracting wallet addresses: {str(e)}")
        
        return list(wallets)
    
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
        """Determine the wallet type based on metrics."""
        # Ensure metrics exist
        if not metrics:
            return "unknown"
            
        total_trades = metrics.get("total_trades", 0)
        if total_trades < 5:
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
            
            # Gem finder: High ROI potential
            if big_win_ratio >= 0.15 and max_roi >= 300:
                return "gem_finder"
            
            # Flipper: Quick trades, lower hold time
            if avg_hold_time_hours < 12 and win_rate > 50:
                return "flipper"
            
            # Consistent: Good win rate, stable returns
            if win_rate >= 45 and median_roi > 0 and std_dev_roi < 100:
                return "consistent"
            
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error determining wallet type: {str(e)}")
            return "unknown"
    
    def _generate_strategy(self, wallet_type: str, metrics: Dict[str, Any], contested_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a trading strategy based on wallet type, metrics, and contested level."""
        try:
            # Ensure metrics exist
            if not metrics:
                metrics = {}
            
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
                        "notes": "Massive moonshot finder. Take 30% at TP1, hold rest for major gains."
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
                        "notes": "Potential moonshot finder. Take partial profits and trail the rest."
                    }
            
            elif wallet_type == "consistent":
                win_rate = metrics.get("win_rate", 0)
                median_roi = metrics.get("median_roi", 0)
                if win_rate > 65 and median_roi > 50:
                    strategy = {
                        "recommendation": "SCALP_AND_HOLD",
                        "entry_type": "IMMEDIATE",
                        "position_size": "MEDIUM",
                        "take_profit_1": 40,
                        "take_profit_2": 80,
                        "take_profit_3": 150,
                        "stop_loss": -25,
                        "notes": "High-performance consistent wallet. Follow with confidence."
                    }
                else:
                    strategy = {
                        "recommendation": "SCALP",
                        "entry_type": "IMMEDIATE",
                        "position_size": "MEDIUM",
                        "take_profit_1": 30,
                        "take_profit_2": 50,
                        "take_profit_3": 100,
                        "stop_loss": -25,
                        "notes": "Consistent returns. Follow with disciplined exits."
                    }
            
            elif wallet_type == "flipper":
                strategy = {
                    "recommendation": "SCALP",
                    "entry_type": "IMMEDIATE",
                    "position_size": "MEDIUM",
                    "take_profit_1": 15,
                    "take_profit_2": 30,
                    "take_profit_3": 60,
                    "stop_loss": -15,
                    "notes": "Quick flipper. Enter and exit fast."
                }
            
            else:  # unknown
                strategy = {
                    "recommendation": "CAUTIOUS",
                    "entry_type": "WAIT_FOR_CONFIRMATION",
                    "position_size": "SMALL",
                    "take_profit_1": 20,
                    "take_profit_2": 40,
                    "take_profit_3": 80,
                    "stop_loss": -20,
                    "notes": "Limited data. Use caution."
                }
            
            # Adjust strategy based on contested level
            if contested_analysis and contested_analysis.get("success"):
                contested_level = contested_analysis.get("contested_level", 0)
                classification = contested_analysis.get("classification", "NOT_CONTESTED")
                
                if contested_level >= 75:
                    strategy["entry_type"] = "VERY_FAST"
                    strategy["notes"] += f" ⚠️ HIGHLY CONTESTED ({contested_level}%) - Many copy traders detected!"
                    strategy["competition_level"] = "HIGH"
                elif contested_level >= 50:
                    strategy["entry_type"] = "FAST"
                    strategy["notes"] += f" ⚠️ MODERATELY CONTESTED ({contested_level}%) - Some copy traders detected."
                    strategy["competition_level"] = "MEDIUM"
                elif contested_level >= 25:
                    strategy["notes"] += f" ℹ️ LIGHTLY CONTESTED ({contested_level}%) - Few copy traders detected."
                    strategy["competition_level"] = "LOW"
                else:
                    strategy["notes"] += f" ✅ NOT CONTESTED ({contested_level}%) - No significant competition."
                    strategy["competition_level"] = "NONE"
                
                strategy["contested_metrics"] = {
                    "contested_level": contested_level,
                    "classification": classification,
                    "copy_trader_count": contested_analysis.get("total_copy_traders", 0),
                    "high_confidence_followers": contested_analysis.get("high_confidence_copy_traders", 0)
                }
            
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
                "competition_level": "UNKNOWN"
            }
    
    def analyze_wallet(self, wallet_address: str, days_back: int = 30, include_contested: bool = True) -> Dict[str, Any]:
        """
        Analyze a wallet for copy trading with Cielo Finance API and optional contested analysis.
        
        Args:
            wallet_address (str): Wallet address
            days_back (int): Number of days to analyze (not used for Cielo Finance API)
            include_contested (bool): Whether to include contested wallet analysis
            
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
                return {
                    "success": True,  # Mark as success but with empty data
                    "wallet_address": wallet_address,
                    "analysis_period_days": "ALL_TIME",
                    "wallet_type": "unknown",
                    "metrics": self._get_empty_metrics(),
                    "strategy": self._generate_strategy("unknown", {}),
                    "trades": [],
                    "correlated_wallets": [],
                    "contested_analysis": {},
                    "api_source": "Cielo Finance + RPC",
                    "note": "No trading data available"
                }
            
            # Extract trades from Cielo Finance token P&L data
            trades = self._extract_trades_from_cielo_trading_stats(token_pnl_data, wallet_address)
            paired_trades = self._pair_trades(trades)
            
            if not paired_trades:
                return {
                    "success": False,
                    "error": "No complete trades found in Cielo Finance data",
                    "wallet_address": wallet_address,
                    "error_type": "NO_PAIRED_TRADES"
                }
            
            # Calculate metrics
            metrics = self._calculate_metrics(paired_trades)
            
            # Determine wallet type
            wallet_type = self._determine_wallet_type(metrics)
            logger.info(f"Wallet type: {wallet_type}")
            
            # Contested analysis
            contested_analysis = {}
            if include_contested:
                logger.info("Performing contested wallet analysis...")
                contested_analysis = self.analyze_contested_level(wallet_address)
            
            # Generate strategy (including contested adjustments)
            strategy = self._generate_strategy(wallet_type, metrics, contested_analysis)
            
            # Find correlated wallets
            correlated_wallets = self._find_correlated_wallets(wallet_address)
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_period_days": "ALL_TIME",  # Cielo Finance provides all-time data
                "wallet_type": wallet_type,
                "metrics": metrics,
                "strategy": strategy,
                "trades": paired_trades,
                "correlated_wallets": correlated_wallets,
                "contested_analysis": contested_analysis,
                "api_source": "Cielo Finance + RPC"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return a valid structure even on error
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "wallet_address": wallet_address,
                "error_type": "UNEXPECTED_ERROR",
                "wallet_type": "unknown",
                "metrics": self._get_empty_metrics(),
                "strategy": {
                    "recommendation": "CAUTIOUS",
                    "entry_type": "WAIT_FOR_CONFIRMATION",
                    "position_size": "SMALL",
                    "take_profit_1": 20,
                    "take_profit_2": 40,
                    "take_profit_3": 80,
                    "stop_loss": -20,
                    "notes": "Analysis failed. Use extreme caution.",
                    "competition_level": "UNKNOWN"
                }
            }
    
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
                            min_winrate: float = 45.0,
                            include_contested: bool = True,
                            use_hybrid: bool = True) -> Dict[str, Any]:
        """
        Batch analyze multiple wallets with Cielo Finance API and contested analysis.
        
        Args:
            wallet_addresses (List[str]): List of wallet addresses
            days_back (int): Number of days to analyze (not used for Cielo Finance)
            min_winrate (float): Minimum win rate percentage
            include_contested (bool): Whether to include contested analysis
            use_hybrid (bool): Whether to use hybrid analysis (Cielo + RPC + Birdeye)
            
        Returns:
            Dict[str, Any]: Categorized wallet analyses
        """
        logger.info(f"Batch analyzing {len(wallet_addresses)} wallets using {'hybrid' if use_hybrid else 'Cielo-only'} approach (contested: {include_contested})")
        
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
                        analysis = self.analyze_wallet_hybrid(wallet_address, days_back, include_contested)
                    else:
                        analysis = self.analyze_wallet(wallet_address, days_back, include_contested)
                    
                    if analysis.get("success"):
                        wallet_analyses.append(analysis)
                        # Log contested level if available
                        if include_contested and "contested_analysis" in analysis:
                            contested = analysis["contested_analysis"]
                            if contested.get("success"):
                                logger.info(f"  └─ Contested Level: {contested.get('contested_level', 0)}% ({contested.get('classification', 'UNKNOWN')})")
                    else:
                        # Add failed analyses with safe defaults if they have metrics
                        if "metrics" in analysis:
                            wallet_analyses.append(analysis)
                        failed_analyses.append({
                            "wallet_address": wallet_address,
                            "error": analysis.get("error", "Unknown error"),
                            "error_type": analysis.get("error_type", "UNKNOWN")
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
            
            # Filter by minimum win rate or profit factor
            filtered_analyses = [
                analysis for analysis in wallet_analyses
                if analysis.get("metrics", {}).get("win_rate", 0) >= min_winrate or
                analysis.get("metrics", {}).get("profit_factor", 0) > 1.5
            ]
            
            # Categorize wallets
            gem_finders = [a for a in filtered_analyses if a.get("wallet_type") == "gem_finder"]
            consistent = [a for a in filtered_analyses if a.get("wallet_type") == "consistent"]
            flippers = [a for a in filtered_analyses if a.get("wallet_type") == "flipper"]
            others = [a for a in filtered_analyses if a.get("wallet_type") == "unknown" or not a.get("wallet_type")]
            
            # Sort each category by performance metrics
            gem_finders.sort(key=lambda x: x.get("metrics", {}).get("max_roi", 0), reverse=True)
            consistent.sort(key=lambda x: x.get("metrics", {}).get("median_roi", 0), reverse=True)
            flippers.sort(key=lambda x: x.get("metrics", {}).get("win_rate", 0), reverse=True)
            
            # Find wallet clusters
            wallet_clusters = self._identify_wallet_clusters(
                {a["wallet_address"]: a.get("correlated_wallets", []) for a in wallet_analyses}
            )
            
            return {
                "success": True,
                "total_wallets": len(wallet_addresses),
                "analyzed_wallets": len(wallet_analyses),
                "failed_wallets": len(failed_analyses),
                "filtered_wallets": len(filtered_analyses),
                "gem_finders": gem_finders,
                "consistent": consistent,
                "flippers": flippers,
                "others": others,
                "wallet_correlations": {a["wallet_address"]: a.get("correlated_wallets", []) for a in wallet_analyses},
                "wallet_clusters": wallet_clusters,
                "failed_analyses": failed_analyses,
                "api_source": "Hybrid (Cielo + RPC + Birdeye)" if use_hybrid else "Cielo Finance + RPC",
                "contested_analysis_included": include_contested
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
        """Export wallet analysis to CSV with contested analysis included."""
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
                writer.writerow({"metric": "api_source", "value": analysis.get("api_source", "Unknown")})
                
                # Strategy info
                strategy = analysis.get("strategy", {})
                writer.writerow({"metric": "recommendation", "value": strategy.get("recommendation", "CAUTIOUS")})
                writer.writerow({"metric": "entry_type", "value": strategy.get("entry_type", "WAIT_FOR_CONFIRMATION")})
                writer.writerow({"metric": "competition_level", "value": strategy.get("competition_level", "UNKNOWN")})
                
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
                
                # Contested analysis
                if "contested_analysis" in analysis and analysis["contested_analysis"].get("success"):
                    contested = analysis["contested_analysis"]
                    writer.writerow({"metric": "contested_level", "value": contested.get("contested_level", 0)})
                    writer.writerow({"metric": "contested_classification", "value": contested.get("classification", "UNKNOWN")})
                    writer.writerow({"metric": "total_copy_traders", "value": contested.get("total_copy_traders", 0)})
                    writer.writerow({"metric": "high_confidence_copy_traders", "value": contested.get("high_confidence_copy_traders", 0)})
            
            logger.info(f"Exported enhanced wallet analysis to {metrics_file}")
            
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
                writer.writerow({"metric": "contested_analysis_included", "value": batch_analysis.get("contested_analysis_included", False)})
                writer.writerow({"metric": "total_wallets", "value": batch_analysis["total_wallets"]})
                writer.writerow({"metric": "analyzed_wallets", "value": batch_analysis["analyzed_wallets"]})
                writer.writerow({"metric": "failed_wallets", "value": batch_analysis.get("failed_wallets", 0)})
                writer.writerow({"metric": "gem_finder_count", "value": len(batch_analysis["gem_finders"])})
                writer.writerow({"metric": "consistent_count", "value": len(batch_analysis["consistent"])})
                writer.writerow({"metric": "flipper_count", "value": len(batch_analysis["flippers"])})
            
            logger.info(f"Exported batch analysis summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting batch analysis: {str(e)}")