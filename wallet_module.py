"""
Wallet Analysis Module - Phoenix Project (HYBRID APPROACH WITH RPC CONFIG)

This version combines:
1. Cielo Finance aggregated stats for overall wallet performance
2. On-chain RPC analysis of the last 3 tokens traded
3. Birdeye API for token price data (if available)
4. Support for custom RPC providers (QuickNode, Helius, etc.)

This provides both high-level metrics and specific recent token insights.
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
import base64

logger = logging.getLogger("phoenix.wallet")

class CieloFinanceAPIError(Exception):
    """Custom exception for Cielo Finance API errors."""
    pass

class WalletAnalyzer:
    """Hybrid wallet analyzer combining Cielo Finance stats with on-chain token analysis."""
    
    def __init__(self, cielo_api: Any, birdeye_api: Any = None, rpc_url: str = None):
        """
        Initialize the wallet analyzer.
        
        Args:
            cielo_api: Cielo Finance API client (REQUIRED for overall stats)
            birdeye_api: Birdeye API client (optional, for token prices)
            rpc_url: Solana RPC endpoint URL (defaults to public RPC if not provided)
        """
        if not cielo_api:
            raise ValueError("Cielo Finance API is REQUIRED for wallet analysis")
        
        self.cielo_api = cielo_api
        self.birdeye_api = birdeye_api
        
        # Use provided RPC URL or default to public endpoint
        if rpc_url:
            self.rpc_url = rpc_url
            logger.info(f"Using custom RPC endpoint: {rpc_url[:50]}...")
        else:
            # Try to load from config file
            self.rpc_url = self._load_rpc_from_config()
            if not self.rpc_url:
                self.rpc_url = "https://api.mainnet-beta.solana.com"
                logger.warning("Using default public RPC. Consider using QuickNode/Helius for better performance!")
        
        # Token program IDs
        self.TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        self.TOKEN_2022_PROGRAM_ID = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
        
        # Known DEX program IDs
        self.DEX_PROGRAMS = {
            "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4": "Jupiter",
            "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "Orca",
            "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium",
            "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": "Raydium CPMM",
            "EewxydAPCCVuNEyrVN68PuSYdQ7wKn27V9Gjeoi8dy3S": "Pump.fun"
        }
        
        # Verify API connections
        if not self._verify_connections():
            raise CieloFinanceAPIError("Failed to verify API connections")
    
    def _load_rpc_from_config(self) -> Optional[str]:
        """Try to load RPC URL from config file."""
        try:
            config_path = os.path.expanduser("~/.phoenix_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    rpc_url = config.get("solana_rpc_url")
                    if rpc_url and rpc_url != "https://api.mainnet-beta.solana.com":
                        logger.info(f"Loaded RPC URL from config: {rpc_url[:50]}...")
                        return rpc_url
        except Exception as e:
            logger.debug(f"Could not load RPC from config: {str(e)}")
        return None
    
    def _verify_connections(self) -> bool:
        """Verify API connections."""
        try:
            # Check Cielo Finance
            if self.cielo_api and self.cielo_api.health_check():
                logger.info("✅ Cielo Finance API connected")
            else:
                logger.error("❌ Cielo Finance API connection failed")
                return False
            
            # Check RPC
            rpc_test = self._make_rpc_call("getHealth", [])
            if "result" in rpc_test:
                logger.info(f"✅ Solana RPC connected ({self._get_rpc_provider_name()})")
            else:
                logger.warning("⚠️ Solana RPC connection issues - recent token analysis may fail")
            
            # Birdeye is optional
            if self.birdeye_api:
                logger.info("✅ Birdeye API available for token prices")
            else:
                logger.info("ℹ️ Birdeye API not configured (token prices unavailable)")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection verification failed: {str(e)}")
            return False
    
    def _get_rpc_provider_name(self) -> str:
        """Identify RPC provider from URL."""
        if "quicknode" in self.rpc_url.lower():
            return "QuickNode"
        elif "helius" in self.rpc_url.lower():
            return "Helius"
        elif "alchemy" in self.rpc_url.lower():
            return "Alchemy"
        elif "ankr" in self.rpc_url.lower():
            return "Ankr"
        elif "mainnet-beta" in self.rpc_url:
            return "Solana Public RPC"
        else:
            return "Custom RPC"
    
    def _make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """Make direct RPC call to Solana node."""
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
            
            # Check for RPC errors
            if "error" in result:
                logger.error(f"RPC error: {result['error']}")
                # Check for rate limit errors
                if "429" in str(result.get("error", {})):
                    logger.error("⚠️ RPC rate limit hit! Consider upgrading your QuickNode plan or using a different provider.")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error(f"RPC timeout for {method} - RPC may be overloaded")
            return {"error": "timeout"}
        except Exception as e:
            logger.error(f"RPC call failed for {method}: {str(e)}")
            return {"error": str(e)}
    
    def _get_recent_signatures(self, wallet_address: str, limit: int = 100) -> List[str]:
        """Get recent transaction signatures for a wallet."""
        response = self._make_rpc_call(
            "getSignaturesForAddress",
            [wallet_address, {"limit": limit}]
        )
        
        if "result" in response:
            return [sig["signature"] for sig in response["result"]]
        return []
    
    def _get_transaction(self, signature: str) -> Dict[str, Any]:
        """Get full transaction details."""
        response = self._make_rpc_call(
            "getTransaction",
            [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0}]
        )
        
        if "result" in response:
            return response["result"]
        return {}
    
    def _extract_token_swaps_from_transaction(self, tx_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract token swap information from a transaction."""
        swaps = []
        
        try:
            if not tx_data or "meta" not in tx_data:
                return swaps
            
            meta = tx_data["meta"]
            
            # Check if this is a successful transaction
            if meta.get("err") is not None:
                return swaps
            
            # Get pre and post token balances
            pre_balances = {
                bal["accountIndex"]: {
                    "mint": bal["mint"],
                    "amount": int(bal["uiTokenAmount"]["amount"]),
                    "decimals": bal["uiTokenAmount"]["decimals"]
                }
                for bal in meta.get("preTokenBalances", [])
            }
            
            post_balances = {
                bal["accountIndex"]: {
                    "mint": bal["mint"],
                    "amount": int(bal["uiTokenAmount"]["amount"]),
                    "decimals": bal["uiTokenAmount"]["decimals"]
                }
                for bal in meta.get("postTokenBalances", [])
            }
            
            # Find tokens that changed balance
            all_indices = set(pre_balances.keys()) | set(post_balances.keys())
            
            tokens_changed = []
            for idx in all_indices:
                pre = pre_balances.get(idx, {"amount": 0, "mint": None})
                post = post_balances.get(idx, {"amount": 0, "mint": None})
                
                if pre["mint"] == post["mint"] and pre["mint"] is not None:
                    delta = post["amount"] - pre["amount"]
                    if delta != 0:
                        tokens_changed.append({
                            "mint": pre["mint"],
                            "delta": delta,
                            "decimals": pre.get("decimals", post.get("decimals", 0)),
                            "is_buy": delta > 0
                        })
            
            # Identify buy/sell pairs (simplified)
            buys = [t for t in tokens_changed if t["is_buy"]]
            sells = [t for t in tokens_changed if not t["is_buy"]]
            
            # Create swap records
            for buy in buys:
                swap = {
                    "token_bought": buy["mint"],
                    "amount_bought": abs(buy["delta"]),
                    "decimals_bought": buy["decimals"],
                    "token_sold": sells[0]["mint"] if sells else "SOL",
                    "amount_sold": abs(sells[0]["delta"]) if sells else 0,
                    "decimals_sold": sells[0]["decimals"] if sells else 9,
                    "signature": tx_data.get("transaction", {}).get("signatures", [""])[0],
                    "timestamp": meta.get("blockTime", 0),
                    "slot": meta.get("slot", 0)
                }
                swaps.append(swap)
            
            return swaps
            
        except Exception as e:
            logger.debug(f"Error extracting swaps: {str(e)}")
            return swaps
    
    def _get_last_n_token_swaps(self, wallet_address: str, n: int = 3) -> List[Dict[str, Any]]:
        """Get the last N token swaps for a wallet."""
        logger.info(f"Fetching last {n} token swaps for {wallet_address}")
        
        try:
            signatures = self._get_recent_signatures(wallet_address, limit=50)
            
            if not signatures:
                logger.warning(f"No recent transactions found for {wallet_address}")
                return []
            
            swaps = []
            for sig in signatures:
                if len(swaps) >= n:
                    break
                
                tx_data = self._get_transaction(sig)
                if tx_data:
                    tx_swaps = self._extract_token_swaps_from_transaction(tx_data)
                    swaps.extend(tx_swaps)
            
            logger.info(f"Found {len(swaps)} swaps from {len(signatures)} transactions")
            return swaps[:n]
            
        except Exception as e:
            logger.error(f"Error getting recent swaps: {str(e)}")
            return []
    
    def _analyze_token_performance(self, token_mint: str, buy_timestamp: int) -> Dict[str, Any]:
        """Analyze token performance since purchase using Birdeye API."""
        if not self.birdeye_api:
            return {
                "success": False,
                "error": "Birdeye API not configured",
                "current_roi": 0,
                "max_roi": 0
            }
        
        try:
            # Get current price
            current_price_response = self.birdeye_api.get_token_price(token_mint)
            if not current_price_response.get("success"):
                return {"success": False, "error": "Could not get current price"}
            
            current_price = current_price_response.get("data", {}).get("value", 0)
            
            # Get historical price at buy time
            history_response = self.birdeye_api.get_token_price_history(
                token_mint,
                buy_timestamp,
                int(datetime.now().timestamp()),
                "1h"
            )
            
            if not history_response.get("success"):
                return {"success": False, "error": "Could not get price history"}
            
            prices = history_response.get("data", {}).get("items", [])
            if not prices:
                return {"success": False, "error": "No price data available"}
            
            # Get price at buy time (first price point)
            buy_price = prices[0].get("value", 0)
            if buy_price == 0:
                return {"success": False, "error": "Invalid buy price"}
            
            # Calculate metrics
            current_roi = ((current_price / buy_price) - 1) * 100
            max_price = max(p.get("value", 0) for p in prices)
            max_roi = ((max_price / buy_price) - 1) * 100
            
            # Get token info
            token_info = self.birdeye_api.get_token_info(token_mint)
            
            return {
                "success": True,
                "token_mint": token_mint,
                "token_symbol": token_info.get("data", {}).get("symbol", "UNKNOWN"),
                "token_name": token_info.get("data", {}).get("name", "Unknown Token"),
                "buy_price": buy_price,
                "current_price": current_price,
                "max_price": max_price,
                "current_roi": round(current_roi, 2),
                "max_roi": round(max_roi, 2),
                "is_profitable": current_roi > 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing token {token_mint}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def analyze_contested_level(self, wallet_address: str, max_transactions: int = 30) -> Dict[str, Any]:
        """
        Analyze contested level based on wallet performance and activity.
        """
        logger.info(f"Analyzing contested level for wallet {wallet_address}")
        
        # Get wallet stats from Cielo
        stats_response = self.cielo_api.get_wallet_trading_stats(wallet_address)
        
        if not stats_response.get("success", False) and stats_response.get("status") != "ok":
            return {
                "success": False,
                "error": "Could not retrieve wallet stats",
                "contested_level": 0,
                "classification": "UNKNOWN"
            }
        
        data = stats_response.get("data", {})
        
        # Calculate contested score based on performance
        win_rate = data.get("winrate", 0)
        total_pnl = data.get("pnl", 0)
        swap_count = data.get("swaps_count", 0)
        consecutive_days = data.get("consecutive_trading_days", 0)
        
        contested_score = 0
        
        # High win rate attracts copiers
        if win_rate >= 85:
            contested_score += 35
        elif win_rate >= 75:
            contested_score += 25
        elif win_rate >= 65:
            contested_score += 15
        
        # High P&L attracts attention
        if total_pnl >= 50000:
            contested_score += 30
        elif total_pnl >= 20000:
            contested_score += 20
        elif total_pnl >= 5000:
            contested_score += 10
        
        # Active traders are more visible
        if swap_count >= 200:
            contested_score += 20
        elif swap_count >= 100:
            contested_score += 15
        elif swap_count >= 50:
            contested_score += 10
        
        # Consistent activity
        if consecutive_days >= 30:
            contested_score += 15
        elif consecutive_days >= 14:
            contested_score += 10
        elif consecutive_days >= 7:
            contested_score += 5
        
        # Determine classification
        if contested_score >= 70:
            classification = "HIGHLY_CONTESTED"
        elif contested_score >= 45:
            classification = "MODERATELY_CONTESTED"
        elif contested_score >= 25:
            classification = "LIGHTLY_CONTESTED"
        else:
            classification = "NOT_CONTESTED"
        
        return {
            "success": True,
            "wallet_address": wallet_address,
            "contested_level": min(contested_score, 100),
            "classification": classification,
            "factors": {
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "swap_count": swap_count,
                "consecutive_days": consecutive_days
            },
            "recommendation": self._get_contested_recommendation(classification)
        }
    
    def _get_contested_recommendation(self, classification: str) -> str:
        """Get recommendation based on contested level."""
        recommendations = {
            "HIGHLY_CONTESTED": "⚠️ Very competitive! Use market orders and be extremely fast.",
            "MODERATELY_CONTESTED": "⚡ Some competition. Be quick with entries.",
            "LIGHTLY_CONTESTED": "✅ Low competition. Normal entry speed should work.",
            "NOT_CONTESTED": "🎯 No significant competition. Take your time with entries.",
            "UNKNOWN": "❓ Unable to determine competition level."
        }
        return recommendations.get(classification, recommendations["UNKNOWN"])
    
    def _calculate_metrics_from_aggregated(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Cielo aggregated data to metrics format."""
        roi_dist = data.get("roi_distribution", {})
        
        # Calculate total trades from ROI distribution
        total_trades = sum(roi_dist.values()) if roi_dist else data.get("swaps_count", 0)
        
        # Estimate wins/losses
        wins = (
            roi_dist.get("roi_above_500", 0) +
            roi_dist.get("roi_200_to_500", 0) +
            roi_dist.get("roi_0_to_200", 0)
        )
        losses = (
            roi_dist.get("roi_neg50_to_0", 0) +
            roi_dist.get("roi_below_neg50", 0)
        )
        
        return {
            "total_trades": total_trades,
            "win_count": wins,
            "loss_count": losses,
            "win_rate": data.get("winrate", 0),
            "total_profit_usd": max(0, data.get("pnl", 0)),
            "total_loss_usd": abs(min(0, data.get("pnl", 0))),
            "net_profit_usd": data.get("pnl", 0),
            "profit_factor": data.get("pnl", 0) / 1000 if data.get("pnl", 0) > 0 else 0,
            "avg_roi": data.get("pnl", 0) / total_trades if total_trades > 0 else 0,
            "median_roi": 0,  # Not available in aggregated
            "std_dev_roi": 0,  # Not available in aggregated
            "max_roi": 500 if roi_dist.get("roi_above_500", 0) > 0 else 200,
            "min_roi": -50 if roi_dist.get("roi_below_neg50", 0) > 0 else 0,
            "avg_hold_time_hours": data.get("average_holding_time_sec", 0) / 3600,
            "total_bet_size_usd": data.get("total_buy_amount_usd", 0),
            "avg_bet_size_usd": data.get("average_buy_amount_usd", 0),
            "total_tokens_traded": data.get("swaps_count", 0),
            "roi_distribution": roi_dist,
            "consecutive_trading_days": data.get("consecutive_trading_days", 0)
        }
    
    def _determine_wallet_type(self, metrics: Dict[str, Any]) -> str:
        """Determine wallet type based on metrics."""
        win_rate = metrics["win_rate"]
        avg_hold_time = metrics["avg_hold_time_hours"]
        roi_dist = metrics["roi_distribution"]
        
        # Calculate high ROI percentage
        high_roi_trades = (
            roi_dist.get("roi_above_500", 0) + 
            roi_dist.get("roi_200_to_500", 0)
        )
        total_roi_trades = sum(roi_dist.values()) if roi_dist else 1
        high_roi_percent = (high_roi_trades / total_roi_trades * 100) if total_roi_trades > 0 else 0
        
        # Determine type
        if high_roi_percent >= 30 and win_rate >= 60:
            return "gem_finder"
        elif avg_hold_time < 6 and win_rate >= 80:
            return "scalper"
        elif win_rate >= 75 and metrics["net_profit_usd"] > 0:
            return "consistent"
        elif avg_hold_time < 12:
            return "flipper"
        else:
            return "mixed"
    
    def _generate_strategy(self, wallet_type: str, metrics: Dict[str, Any], 
                         contested_analysis: Dict[str, Any] = None,
                         recent_tokens: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive trading strategy."""
        base_strategies = {
            "gem_finder": {
                "recommendation": "FOLLOW_SELECTIVE",
                "entry_type": "IMMEDIATE",
                "position_size": "SMALL",
                "take_profit_1": 100,
                "take_profit_2": 300,
                "take_profit_3": 500,
                "stop_loss": -30,
                "notes": "High ROI trader. Follow with small positions for potential moonshots."
            },
            "scalper": {
                "recommendation": "QUICK_SCALP",
                "entry_type": "IMMEDIATE",
                "position_size": "MEDIUM",
                "take_profit_1": 15,
                "take_profit_2": 30,
                "take_profit_3": 50,
                "stop_loss": -10,
                "notes": "Ultra-fast scalper. Enter and exit quickly."
            },
            "consistent": {
                "recommendation": "FOLLOW_CONFIDENT",
                "entry_type": "IMMEDIATE",
                "position_size": "MEDIUM",
                "take_profit_1": 50,
                "take_profit_2": 100,
                "take_profit_3": 200,
                "stop_loss": -25,
                "notes": "Consistent winner. Good for steady profits."
            },
            "flipper": {
                "recommendation": "QUICK_FLIP",
                "entry_type": "FAST",
                "position_size": "SMALL",
                "take_profit_1": 20,
                "take_profit_2": 40,
                "take_profit_3": 60,
                "stop_loss": -15,
                "notes": "Quick flipper. Don't hold too long."
            },
            "mixed": {
                "recommendation": "CAUTIOUS",
                "entry_type": "WAIT_CONFIRMATION",
                "position_size": "SMALL",
                "take_profit_1": 25,
                "take_profit_2": 50,
                "take_profit_3": 100,
                "stop_loss": -20,
                "notes": "Mixed results. Trade with caution."
            }
        }
        
        strategy = base_strategies.get(wallet_type, base_strategies["mixed"]).copy()
        
        # Add win rate confidence
        win_rate = metrics["win_rate"]
        if win_rate >= 85:
            strategy["confidence_level"] = "VERY_HIGH"
        elif win_rate >= 75:
            strategy["confidence_level"] = "HIGH"
        elif win_rate >= 65:
            strategy["confidence_level"] = "MEDIUM"
        else:
            strategy["confidence_level"] = "LOW"
        
        # Adjust for contested level
        if contested_analysis and contested_analysis.get("success"):
            classification = contested_analysis.get("classification", "UNKNOWN")
            if classification == "HIGHLY_CONTESTED":
                strategy["entry_type"] = "VERY_FAST"
                strategy["notes"] += " ⚠️ HIGHLY CONTESTED - Be extremely fast!"
            elif classification == "MODERATELY_CONTESTED":
                strategy["entry_type"] = "FAST"
                strategy["notes"] += " ⚡ Moderately contested - Be quick!"
        
        # Add recent token insights
        if recent_tokens:
            profitable_tokens = [t for t in recent_tokens if t.get("is_profitable", False)]
            if profitable_tokens:
                avg_recent_roi = np.mean([t.get("current_roi", 0) for t in profitable_tokens])
                strategy["recent_performance"] = f"Recent tokens: {len(profitable_tokens)}/3 profitable, avg ROI: {avg_recent_roi:.1f}%"
        
        return strategy
    
    def analyze_wallet(self, wallet_address: str, days_back: int = 30, include_contested: bool = True) -> Dict[str, Any]:
        """
        Comprehensive wallet analysis using hybrid approach.
        
        Combines:
        1. Cielo Finance aggregated stats
        2. Last 3 token analysis via RPC + Birdeye
        3. Contested level analysis
        """
        logger.info(f"🔍 Analyzing wallet {wallet_address} (hybrid approach)")
        
        try:
            # Step 1: Get aggregated stats from Cielo Finance
            logger.info("📊 Fetching Cielo Finance aggregated stats...")
            stats_response = self.cielo_api.get_wallet_trading_stats(wallet_address)
            
            if not stats_response.get("success", False) and stats_response.get("status") != "ok":
                return {
                    "success": False,
                    "error": f"Cielo Finance API error: {stats_response.get('error', 'Unknown')}",
                    "wallet_address": wallet_address
                }
            
            cielo_data = stats_response.get("data", {})
            
            # Skip wallets with no trading data
            if cielo_data.get("swaps_count", 0) == 0:
                return {
                    "success": False,
                    "error": "No trading data available",
                    "wallet_address": wallet_address
                }
            
            # Step 2: Calculate metrics from aggregated data
            metrics = self._calculate_metrics_from_aggregated(cielo_data)
            
            # Step 3: Get last 3 token swaps
            logger.info("🪙 Analyzing last 3 tokens...")
            recent_swaps = self._get_last_n_token_swaps(wallet_address, n=3)
            
            recent_tokens_analysis = []
            for swap in recent_swaps:
                if self.birdeye_api:
                    token_analysis = self._analyze_token_performance(
                        swap.get("token_bought"),
                        swap.get("timestamp", 0)
                    )
                else:
                    # Without Birdeye, just record the swap
                    token_analysis = {
                        "success": False,
                        "token_mint": swap.get("token_bought"),
                        "error": "Birdeye API not configured for price analysis"
                    }
                token_analysis["swap_info"] = swap
                recent_tokens_analysis.append(token_analysis)
            
            # Step 4: Determine wallet type
            wallet_type = self._determine_wallet_type(metrics)
            
            # Step 5: Contested analysis
            contested_analysis = {}
            if include_contested:
                logger.info("🎯 Analyzing contested level...")
                contested_analysis = self.analyze_contested_level(wallet_address)
            
            # Step 6: Generate strategy
            strategy = self._generate_strategy(
                wallet_type, 
                metrics, 
                contested_analysis,
                recent_tokens_analysis
            )
            
            # Compile final analysis
            return {
                "success": True,
                "wallet_address": wallet_address,
                "wallet_type": wallet_type,
                "metrics": metrics,
                "strategy": strategy,
                "recent_tokens": recent_tokens_analysis,
                "contested_analysis": contested_analysis,
                "cielo_raw_data": cielo_data,
                "analysis_type": "hybrid",
                "rpc_provider": self._get_rpc_provider_name(),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "wallet_address": wallet_address
            }
    
    def batch_analyze_wallets(self, wallet_addresses: List[str], 
                            days_back: int = 30,
                            min_winrate: float = 45.0,
                            include_contested: bool = True) -> Dict[str, Any]:
        """Batch analyze multiple wallets."""
        logger.info(f"📊 Batch analyzing {len(wallet_addresses)} wallets...")
        logger.info(f"🌐 Using RPC: {self._get_rpc_provider_name()}")
        
        results = {
            "success": True,
            "total_wallets": len(wallet_addresses),
            "analyzed_wallets": 0,
            "failed_wallets": 0,
            "gem_finders": [],
            "consistent": [],
            "scalpers": [],
            "flippers": [],
            "mixed": [],
            "failed_analyses": [],
            "rpc_provider": self._get_rpc_provider_name()
        }
        
        for i, wallet in enumerate(wallet_addresses, 1):
            logger.info(f"Analyzing wallet {i}/{len(wallet_addresses)}: {wallet}")
            
            try:
                analysis = self.analyze_wallet(wallet, days_back, include_contested)
                
                if analysis.get("success"):
                    # Filter by win rate
                    if analysis["metrics"]["win_rate"] >= min_winrate:
                        results["analyzed_wallets"] += 1
                        
                        # Categorize by type
                        wallet_type = analysis["wallet_type"]
                        if wallet_type == "gem_finder":
                            results["gem_finders"].append(analysis)
                        elif wallet_type == "consistent":
                            results["consistent"].append(analysis)
                        elif wallet_type == "scalper":
                            results["scalpers"].append(analysis)
                        elif wallet_type == "flipper":
                            results["flippers"].append(analysis)
                        else:
                            results["mixed"].append(analysis)
                else:
                    results["failed_wallets"] += 1
                    results["failed_analyses"].append({
                        "wallet_address": wallet,
                        "error": analysis.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing wallet {wallet}: {str(e)}")
                results["failed_wallets"] += 1
                results["failed_analyses"].append({
                    "wallet_address": wallet,
                    "error": str(e)
                })
        
        # Sort categories by performance
        for category in ["gem_finders", "consistent", "scalpers", "flippers", "mixed"]:
            results[category].sort(
                key=lambda x: (x["metrics"]["win_rate"], x["metrics"]["net_profit_usd"]),
                reverse=True
            )
        
        return results
    
    def export_wallet_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Export single wallet analysis."""
        if not analysis.get("success"):
            logger.warning(f"Cannot export failed analysis")
            return
        
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Main metrics file
            metrics_file = output_file.replace(".csv", "_metrics.csv")
            with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ["metric", "value"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Basic info
                writer.writerow({"metric": "wallet_address", "value": analysis["wallet_address"]})
                writer.writerow({"metric": "wallet_type", "value": analysis["wallet_type"]})
                writer.writerow({"metric": "analysis_type", "value": "hybrid"})
                writer.writerow({"metric": "rpc_provider", "value": analysis.get("rpc_provider", "Unknown")})
                
                # Metrics
                for key, value in analysis["metrics"].items():
                    if not isinstance(value, dict):
                        writer.writerow({"metric": key, "value": value})
                
                # Strategy
                strategy = analysis["strategy"]
                writer.writerow({"metric": "recommendation", "value": strategy["recommendation"]})
                writer.writerow({"metric": "confidence_level", "value": strategy.get("confidence_level", "MEDIUM")})
                writer.writerow({"metric": "entry_type", "value": strategy["entry_type"]})
                
                # Contested info
                if "contested_analysis" in analysis and analysis["contested_analysis"].get("success"):
                    contested = analysis["contested_analysis"]
                    writer.writerow({"metric": "contested_level", "value": contested["contested_level"]})
                    writer.writerow({"metric": "contested_classification", "value": contested["classification"]})
            
            # Recent tokens file
            if analysis.get("recent_tokens"):
                tokens_file = output_file.replace(".csv", "_recent_tokens.csv")
                with open(tokens_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ["token_symbol", "token_mint", "current_roi", "max_roi", "is_profitable"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for token in analysis["recent_tokens"]:
                        if token.get("success"):
                            writer.writerow({
                                "token_symbol": token.get("token_symbol", "UNKNOWN"),
                                "token_mint": token.get("token_mint", ""),
                                "current_roi": f"{token.get('current_roi', 0):.2f}%",
                                "max_roi": f"{token.get('max_roi', 0):.2f}%",
                                "is_profitable": token.get("is_profitable", False)
                            })
            
            logger.info(f"✅ Exported analysis to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error exporting analysis: {str(e)}")
    
    def export_batch_analysis(self, batch_analysis: Dict[str, Any], output_file: str) -> None:
        """Export batch wallet analysis results."""
        if not batch_analysis.get("success"):
            logger.error("Cannot export failed batch analysis")
            return
        
        try:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Main summary file
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'wallet_address', 'wallet_type', 'win_rate', 'net_profit_usd',
                    'total_trades', 'avg_holding_hours', 'strategy', 'confidence',
                    'contested_level', 'recent_token_performance'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write all analyzed wallets
                all_wallets = []
                for category in ["gem_finders", "consistent", "scalpers", "flippers", "mixed"]:
                    all_wallets.extend(batch_analysis.get(category, []))
                
                for wallet in all_wallets:
                    recent_tokens = wallet.get("recent_tokens", [])
                    profitable_recent = [t for t in recent_tokens if t.get("is_profitable", False)]
                    recent_performance = f"{len(profitable_recent)}/{len(recent_tokens)} profitable" if recent_tokens else "N/A"
                    
                    row = {
                        'wallet_address': wallet['wallet_address'],
                        'wallet_type': wallet['wallet_type'],
                        'win_rate': f"{wallet['metrics']['win_rate']:.2f}%",
                        'net_profit_usd': f"${wallet['metrics']['net_profit_usd']:.2f}",
                        'total_trades': wallet['metrics']['total_trades'],
                        'avg_holding_hours': f"{wallet['metrics']['avg_hold_time_hours']:.1f}",
                        'strategy': wallet['strategy']['recommendation'],
                        'confidence': wallet['strategy'].get('confidence_level', 'MEDIUM'),
                        'contested_level': wallet.get('contested_analysis', {}).get('classification', 'UNKNOWN'),
                        'recent_token_performance': recent_performance
                    }
                    writer.writerow(row)
            
            # Summary stats file
            summary_file = output_file.replace(".csv", "_summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=== WALLET ANALYSIS SUMMARY ===\n\n")
                f.write(f"RPC Provider: {batch_analysis.get('rpc_provider', 'Unknown')}\n")
                f.write(f"Total wallets analyzed: {batch_analysis['analyzed_wallets']}\n")
                f.write(f"Failed analyses: {batch_analysis['failed_wallets']}\n\n")
                
                f.write("WALLET DISTRIBUTION:\n")
                f.write(f"- Gem Finders: {len(batch_analysis['gem_finders'])}\n")
                f.write(f"- Consistent: {len(batch_analysis['consistent'])}\n")
                f.write(f"- Scalpers: {len(batch_analysis['scalpers'])}\n")
                f.write(f"- Flippers: {len(batch_analysis['flippers'])}\n")
                f.write(f"- Mixed: {len(batch_analysis['mixed'])}\n\n")
                
                f.write("TOP PERFORMERS:\n")
                top_wallets = sorted(
                    [w for cat in ["gem_finders", "consistent", "scalpers"] for w in batch_analysis.get(cat, [])],
                    key=lambda x: x['metrics']['win_rate'],
                    reverse=True
                )[:10]
                
                for i, wallet in enumerate(top_wallets, 1):
                    f.write(f"{i}. {wallet['wallet_address']}\n")
                    f.write(f"   Type: {wallet['wallet_type']}\n")
                    f.write(f"   Win Rate: {wallet['metrics']['win_rate']:.2f}%\n")
                    f.write(f"   P&L: ${wallet['metrics']['net_profit_usd']:.2f}\n")
                    f.write(f"   Strategy: {wallet['strategy']['recommendation']}\n\n")
            
            logger.info(f"✅ Exported batch analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting batch analysis: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    """Test the hybrid wallet analyzer."""
    import sys
    from cielo_api import CieloFinanceAPI
    
    # Get API key from command line or use test key
    api_key = sys.argv[1] if len(sys.argv) > 1 else input("Enter Cielo Finance API key: ")
    
    # Optional: Get custom RPC URL
    custom_rpc = None
    if len(sys.argv) > 2:
        custom_rpc = sys.argv[2]
    else:
        use_custom = input("Use custom RPC? (y/N): ").lower().strip()
        if use_custom == 'y':
            custom_rpc = input("Enter your RPC URL (QuickNode/Helius): ").strip()
    
    # Initialize
    cielo_api = CieloFinanceAPI(api_key)
    analyzer = WalletAnalyzer(cielo_api, rpc_url=custom_rpc)
    
    # Test wallet
    test_wallet = "2PnqznAKcwK6xu7mBjc2XAiwugMYzMx97vZSZsgLgVVd"
    
    print(f"\n🔍 Testing hybrid analysis on wallet: {test_wallet}")
    print("=" * 60)
    
    result = analyzer.analyze_wallet(test_wallet, include_contested=True)
    
    if result.get("success"):
        print("\n✅ Analysis successful!")
        print(f"\n🌐 RPC Provider: {result.get('rpc_provider', 'Unknown')}")
        print(f"\n📊 OVERALL STATS (from Cielo Finance):")
        print(f"  Wallet Type: {result['wallet_type']}")
        print(f"  Win Rate: {result['metrics']['win_rate']:.2f}%")
        print(f"  Net P&L: ${result['metrics']['net_profit_usd']:.2f}")
        print(f"  Total Trades: {result['metrics']['total_trades']}")
        print(f"  Avg Hold Time: {result['metrics']['avg_hold_time_hours']:.1f} hours")
        
        print(f"\n🪙 LAST 3 TOKENS:")
        for i, token in enumerate(result.get('recent_tokens', []), 1):
            if token.get('success'):
                print(f"  {i}. {token.get('token_symbol', 'UNKNOWN')}")
                print(f"     Current ROI: {token.get('current_roi', 0):.2f}%")
                print(f"     Max ROI: {token.get('max_roi', 0):.2f}%")
                print(f"     Status: {'✅ Profitable' if token.get('is_profitable') else '❌ Loss'}")
            else:
                print(f"  {i}. Token analysis failed: {token.get('error', 'Unknown error')}")
        
        print(f"\n🎯 CONTESTED ANALYSIS:")
        contested = result.get('contested_analysis', {})
        if contested.get('success'):
            print(f"  Level: {contested['contested_level']}% - {contested['classification']}")
            print(f"  {contested.get('recommendation', '')}")
        
        print(f"\n📋 STRATEGY:")
        strategy = result['strategy']
        print(f"  Recommendation: {strategy['recommendation']}")
        print(f"  Confidence: {strategy.get('confidence_level', 'MEDIUM')}")
        print(f"  Entry Type: {strategy['entry_type']}")
        print(f"  Notes: {strategy['notes']}")
    else:
        print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)