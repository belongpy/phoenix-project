"""
Wallet Analysis Module - Phoenix Project (Updated)

This module handles wallet analysis using Cielo Finance API and direct RPC calls
for contested wallet analysis.
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
            cielo_api: Cielo Finance API client
            birdeye_api: Birdeye API client (optional, for fallback)
            rpc_url: Solana RPC endpoint URL (P9 or other provider)
        """
        if not cielo_api:
            logger.warning("Cielo Finance API not provided. Will use Birdeye API if available.")
            if not birdeye_api:
                raise ValueError("Either Cielo Finance API or Birdeye API must be provided.")
        
        self.cielo_api = cielo_api
        self.birdeye_api = birdeye_api
        self.rpc_url = rpc_url
        
        # Verify API connectivity
        if cielo_api and not self._verify_api_connection():
            logger.warning("Cannot connect to Cielo Finance API. Will use Birdeye API if available.")
        
        # Track entry times for tokens to detect correlated wallets
        self.token_entries = {}  # token_address -> {wallet_address -> timestamp}
    
    def _verify_api_connection(self) -> bool:
        """
        Verify that the Cielo Finance API is accessible.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            if not self.cielo_api:
                return False
            # Try a simple API call to verify connectivity
            test_response = self.cielo_api.health_check() if hasattr(self.cielo_api, 'health_check') else True
            return bool(test_response)
        except Exception as e:
            logger.error(f"Cielo Finance API connection failed: {str(e)}")
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
    
    def _extract_trades(self, transactions: List[Dict[str, Any]], wallet_address: str) -> List[Dict[str, Any]]:
        """
        Extract and categorize trades from transaction history.
        
        Args:
            transactions (List[Dict[str, Any]]): Wallet transaction history
            wallet_address (str): The wallet address being analyzed
            
        Returns:
            List[Dict[str, Any]]: Categorized trades
        """
        if not transactions:
            logger.warning(f"No transactions provided for wallet {wallet_address}")
            return []
        
        trades = []
        
        try:
            for tx in transactions:
                # Skip transactions without proper data
                if not tx.get("data") or not isinstance(tx["data"], dict):
                    continue
                    
                tx_data = tx["data"]
                
                # Extract common fields
                trade = {
                    "tx_hash": tx_data.get("signature", ""),
                    "timestamp": tx_data.get("blockTime", 0),
                    "date": datetime.fromtimestamp(tx_data.get("blockTime", 0)) if tx_data.get("blockTime") else datetime.now(),
                    "token_address": "",
                    "token_symbol": "",
                    "type": "",
                    "amount": 0,
                    "price": 0,
                    "value_usd": 0,
                    "amount_sol": 0,
                    "market_cap_usd": 0,
                    "platform": ""
                }
                
                # Process token transfers
                if "tokenTransfers" in tx_data:
                    for transfer in tx_data["tokenTransfers"]:
                        if transfer.get("fromOwner") == "wallet":
                            # Token is going out - it's a sell
                            trade["type"] = "SELL"
                            trade["token_address"] = transfer.get("mint", "")
                            trade["token_symbol"] = transfer.get("symbol", "")
                            trade["amount"] = float(transfer.get("amount", 0))
                            
                            if "priceUsd" in transfer:
                                trade["price"] = float(transfer.get("priceUsd", 0))
                                trade["value_usd"] = trade["amount"] * trade["price"]
                            
                            # Extract SOL amount from native transfers
                            if "nativeTransfers" in tx_data:
                                for native_transfer in tx_data["nativeTransfers"]:
                                    if native_transfer.get("toOwner") == "wallet":
                                        trade["amount_sol"] = float(native_transfer.get("amount", 0))
                                        break
                            
                            trades.append(trade.copy())
                        
                        elif transfer.get("toOwner") == "wallet":
                            # Token is coming in - it's a buy
                            trade["type"] = "BUY"
                            trade["token_address"] = transfer.get("mint", "")
                            trade["token_symbol"] = transfer.get("symbol", "")
                            trade["amount"] = float(transfer.get("amount", 0))
                            
                            if "priceUsd" in transfer:
                                trade["price"] = float(transfer.get("priceUsd", 0))
                                trade["value_usd"] = trade["amount"] * trade["price"]
                            
                            # Extract SOL amount from native transfers
                            if "nativeTransfers" in tx_data:
                                for native_transfer in tx_data["nativeTransfers"]:
                                    if native_transfer.get("fromOwner") == "wallet":
                                        trade["amount_sol"] = float(native_transfer.get("amount", 0))
                                        break
                            
                            # Get additional token info
                            if trade["token_address"]:
                                try:
                                    # Try Cielo Finance first, fallback to Birdeye
                                    if self.cielo_api:
                                        token_info = self._get_token_info_cielo(trade["token_address"])
                                    elif self.birdeye_api:
                                        token_info = self.birdeye_api.get_token_info(trade["token_address"])
                                        token_info = token_info.get("data", {}) if token_info.get("success") else {}
                                    else:
                                        token_info = {}
                                        
                                    if token_info:
                                        trade["market_cap_usd"] = token_info.get("marketCap", 0)
                                        trade["platform"] = self._identify_platform(token_info)
                                except Exception as e:
                                    logger.warning(f"Error fetching token info for {trade['token_address']}: {str(e)}")
                            
                            # Store token entry for correlation analysis
                            if trade["token_address"]:
                                if trade["token_address"] not in self.token_entries:
                                    self.token_entries[trade["token_address"]] = {}
                                
                                self.token_entries[trade["token_address"]][wallet_address] = trade["timestamp"]
                            
                            trades.append(trade.copy())
            
            logger.info(f"Extracted {len(trades)} trades from {len(transactions)} transactions for wallet {wallet_address}")
            return trades
            
        except Exception as e:
            logger.error(f"Error extracting trades for wallet {wallet_address}: {str(e)}")
            return []
    
    def _get_token_info_cielo(self, token_address: str) -> Optional[Dict[str, Any]]:
        """Get token information using Cielo Finance API."""
        try:
            if not self.cielo_api or not hasattr(self.cielo_api, 'get_token_info'):
                return None
            
            token_info = self.cielo_api.get_token_info(token_address)
            
            if token_info and token_info.get("success"):
                return token_info.get("data", {})
            else:
                return None
                
        except Exception as e:
            logger.error(f"Cielo Finance API error getting token info for {token_address}: {str(e)}")
            return None
    
    def _identify_platform(self, token_info: Dict[str, Any]) -> str:
        """Identify platform using token data."""
        try:
            platforms = {
                "pumpfun": ["pump", "pf", "pump.fun"],
                "raydium": ["ray", "raydium"],
                "orca": ["orca"],
                "jupiter": ["jup", "jupiter"],
                "meteora": ["met", "meteora"],
                "lifinity": ["lfnty"],
                "saber": ["sbr", "saber"]
            }
            
            symbol = token_info.get("symbol", "").lower()
            name = token_info.get("name", "").lower()
            
            for platform, identifiers in platforms.items():
                for identifier in identifiers:
                    if identifier in symbol or identifier in name:
                        return platform
            
            tags = token_info.get("tags", [])
            for tag in tags:
                tag_lower = tag.lower() if isinstance(tag, str) else ""
                for platform, identifiers in platforms.items():
                    for identifier in identifiers:
                        if identifier in tag_lower:
                            return platform
            
            return ""
            
        except Exception as e:
            logger.error(f"Error identifying platform: {str(e)}")
            return ""
    
    def _pair_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pair buy and sell trades to calculate ROI."""
        if not trades:
            logger.warning("No trades to pair")
            return []
        
        # Group trades by token
        token_trades = defaultdict(list)
        for trade in trades:
            if trade["token_address"]:
                token_trades[trade["token_address"]].append(trade)
        
        # Pair buys with sells for each token
        paired_trades = []
        
        for token_address, token_trade_list in token_trades.items():
            # Sort trades by timestamp
            sorted_trades = sorted(token_trade_list, key=lambda x: x["timestamp"])
            
            buy_queue = []
            for trade in sorted_trades:
                if trade["type"] == "BUY":
                    buy_queue.append(trade)
                elif trade["type"] == "SELL" and buy_queue:
                    # Match with the earliest buy (FIFO)
                    buy_trade = buy_queue.pop(0)
                    
                    # Calculate ROI
                    buy_value = buy_trade["value_usd"]
                    sell_value = trade["value_usd"]
                    
                    if buy_value > 0:
                        roi = ((sell_value / buy_value) - 1) * 100
                    else:
                        roi = 0
                    
                    # Create paired trade
                    paired_trade = {
                        "token_address": token_address,
                        "token_symbol": trade["token_symbol"],
                        "buy_timestamp": buy_trade["timestamp"],
                        "buy_date": buy_trade["date"].isoformat(),
                        "buy_price": buy_trade["price"],
                        "buy_value_usd": buy_value,
                        "buy_value_sol": buy_trade["amount_sol"],
                        "market_cap_at_buy": buy_trade["market_cap_usd"],
                        "platform": buy_trade["platform"],
                        "sell_timestamp": trade["timestamp"],
                        "sell_date": trade["date"].isoformat(),
                        "sell_price": trade["price"],
                        "sell_value_usd": sell_value,
                        "sell_value_sol": trade["amount_sol"],
                        "holding_time_hours": (trade["timestamp"] - buy_trade["timestamp"]) / 3600,
                        "roi_percent": roi,
                        "is_win": roi > 0
                    }
                    
                    paired_trades.append(paired_trade)
        
        logger.info(f"Paired {len(paired_trades)} trades from {len(trades)} individual trades")
        return paired_trades
    
    def _calculate_metrics(self, paired_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from paired trades."""
        if not paired_trades:
            logger.warning("No paired trades for metrics calculation")
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
                "total_tokens_traded": 0
            }
        
        try:
            # Basic counts
            total_trades = len(paired_trades)
            win_count = sum(1 for trade in paired_trades if trade["is_win"])
            loss_count = total_trades - win_count
            
            # ROI statistics
            roi_values = [trade["roi_percent"] for trade in paired_trades]
            avg_roi = np.mean(roi_values) if roi_values else 0
            median_roi = np.median(roi_values) if roi_values else 0
            std_dev_roi = np.std(roi_values) if roi_values else 0
            max_roi = max(roi_values) if roi_values else 0
            min_roi = min(roi_values) if roi_values else 0
            
            # Profit/loss
            winning_trades = [t for t in paired_trades if t["is_win"]]
            losing_trades = [t for t in paired_trades if not t["is_win"]]
            
            total_profit_usd = sum(t["sell_value_usd"] - t["buy_value_usd"] for t in winning_trades)
            total_loss_usd = abs(sum(t["sell_value_usd"] - t["buy_value_usd"] for t in losing_trades))
            net_profit_usd = total_profit_usd - total_loss_usd
            profit_factor = total_profit_usd / total_loss_usd if total_loss_usd > 0 else float('inf')
            
            # Holding time
            holding_times = [trade["holding_time_hours"] for trade in paired_trades]
            avg_hold_time_hours = np.mean(holding_times) if holding_times else 0
            
            # Bet size
            bet_sizes = [trade["buy_value_usd"] for trade in paired_trades]
            total_bet_size_usd = sum(bet_sizes)
            avg_bet_size_usd = np.mean(bet_sizes) if bet_sizes else 0
            
            # Unique tokens
            unique_tokens = len(set(trade["token_address"] for trade in paired_trades))
            
            # ROI distribution buckets
            roi_buckets = {
                "10x_plus": len([t for t in paired_trades if t["roi_percent"] >= 1000]),
                "5x_to_10x": len([t for t in paired_trades if 500 <= t["roi_percent"] < 1000]),
                "2x_to_5x": len([t for t in paired_trades if 200 <= t["roi_percent"] < 500]),
                "1x_to_2x": len([t for t in paired_trades if 100 <= t["roi_percent"] < 200]),
                "50_to_100": len([t for t in paired_trades if 50 <= t["roi_percent"] < 100]),
                "0_to_50": len([t for t in paired_trades if 0 <= t["roi_percent"] < 50]),
                "minus50_to_0": len([t for t in paired_trades if -50 <= t["roi_percent"] < 0]),
                "below_minus50": len([t for t in paired_trades if t["roi_percent"] < -50])
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
            "roi_distribution": {}
        }
    
    def _determine_wallet_type(self, metrics: Dict[str, Any]) -> str:
        """Determine the wallet type based on metrics."""
        if metrics["total_trades"] < 5:
            return "unknown"
        
        try:
            win_rate = metrics["win_rate"]
            median_roi = metrics["median_roi"]
            std_dev_roi = metrics["std_dev_roi"]
            avg_hold_time_hours = metrics["avg_hold_time_hours"]
            roi_distribution = metrics["roi_distribution"]
            
            # Calculate key indicators
            big_win_count = roi_distribution.get("10x_plus", 0) + roi_distribution.get("5x_to_10x", 0) + roi_distribution.get("2x_to_5x", 0)
            big_win_ratio = big_win_count / metrics["total_trades"] if metrics["total_trades"] > 0 else 0
            
            # Gem finder: High ROI potential
            if big_win_ratio >= 0.15 and metrics["max_roi"] >= 300:
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
            # Base strategies by wallet type
            if wallet_type == "gem_finder":
                if metrics["max_roi"] >= 500:  # 5x+ potential
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
                if metrics["win_rate"] > 65 and metrics["median_roi"] > 50:
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
        Analyze a wallet for copy trading with optional contested analysis.
        
        Args:
            wallet_address (str): Wallet address
            days_back (int): Number of days to analyze
            include_contested (bool): Whether to include contested wallet analysis
            
        Returns:
            Dict[str, Any]: Wallet analysis results
        """
        logger.info(f"Analyzing wallet {wallet_address} for the past {days_back} days")
        
        try:
            # Get wallet transactions - try Cielo Finance first, fallback to Birdeye
            if self.cielo_api and hasattr(self.cielo_api, 'get_wallet_pnl_by_tokens'):
                # Use Cielo Finance for P&L analysis
                pnl_data = self.cielo_api.get_wallet_pnl_by_tokens(wallet_address)
                if pnl_data and pnl_data.get("success"):
                    transactions = self._convert_cielo_pnl_to_transactions(pnl_data)
                else:
                    transactions = []
            elif self.birdeye_api:
                # Fallback to Birdeye
                tx_response = self.birdeye_api.get_wallet_transactions(wallet_address)
                transactions = tx_response.get("data", []) if tx_response.get("success") else []
            else:
                return {
                    "success": False,
                    "error": "No API available for wallet analysis",
                    "wallet_address": wallet_address,
                    "error_type": "NO_API"
                }
            
            if not transactions:
                logger.warning(f"No transactions found for wallet {wallet_address}")
                return {
                    "success": False,
                    "error": "No transactions found for this wallet",
                    "wallet_address": wallet_address,
                    "error_type": "NO_DATA"
                }
            
            # Filter transactions by date
            date_limit = datetime.now() - timedelta(days=days_back)
            filtered_transactions = [
                tx for tx in transactions
                if "blockTime" in tx.get("data", {}) and 
                datetime.fromtimestamp(tx["data"]["blockTime"]) >= date_limit
            ]
            
            logger.info(f"Found {len(filtered_transactions)} transactions in the last {days_back} days")
            
            if not filtered_transactions:
                return {
                    "success": False,
                    "error": f"No transactions found in the last {days_back} days",
                    "wallet_address": wallet_address,
                    "error_type": "NO_RECENT_DATA"
                }
            
            # Extract and pair trades
            trades = self._extract_trades(filtered_transactions, wallet_address)
            paired_trades = self._pair_trades(trades)
            
            if not paired_trades:
                return {
                    "success": False,
                    "error": "No complete buy-sell pairs found",
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
                "analysis_period_days": days_back,
                "wallet_type": wallet_type,
                "metrics": metrics,
                "strategy": strategy,
                "trades": paired_trades,
                "correlated_wallets": correlated_wallets,
                "contested_analysis": contested_analysis,
                "api_source": "Cielo Finance + RPC" if self.cielo_api else "Birdeye + RPC"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "wallet_address": wallet_address,
                "error_type": "UNEXPECTED_ERROR"
            }
    
    def _convert_cielo_pnl_to_transactions(self, pnl_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert Cielo Finance P&L data to transaction format."""
        transactions = []
        
        try:
            if "data" in pnl_data and isinstance(pnl_data["data"], list):
                for token_pnl in pnl_data["data"]:
                    # Create mock transaction data from P&L info
                    # This is a simplified conversion - adjust based on actual Cielo Finance API response
                    transactions.append({
                        "data": {
                            "signature": f"cielo_{token_pnl.get('mint', '')[:8]}",
                            "blockTime": token_pnl.get("first_buy_time", int(datetime.now().timestamp())),
                            "tokenTransfers": [{
                                "mint": token_pnl.get("mint", ""),
                                "symbol": token_pnl.get("symbol", ""),
                                "amount": token_pnl.get("total_amount", 0),
                                "priceUsd": token_pnl.get("avg_price", 0),
                                "toOwner": "wallet"
                            }]
                        }
                    })
        except Exception as e:
            logger.error(f"Error converting Cielo P&L data: {str(e)}")
        
        return transactions
    
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
                            include_contested: bool = True) -> Dict[str, Any]:
        """
        Batch analyze multiple wallets with contested analysis.
        
        Args:
            wallet_addresses (List[str]): List of wallet addresses
            days_back (int): Number of days to analyze
            min_winrate (float): Minimum win rate percentage
            include_contested (bool): Whether to include contested analysis
            
        Returns:
            Dict[str, Any]: Categorized wallet analyses
        """
        logger.info(f"Batch analyzing {len(wallet_addresses)} wallets (contested: {include_contested})")
        
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
                    analysis = self.analyze_wallet(wallet_address, days_back, include_contested)
                    
                    if analysis.get("success"):
                        wallet_analyses.append(analysis)
                        # Log contested level if available
                        if include_contested and "contested_analysis" in analysis:
                            contested = analysis["contested_analysis"]
                            if contested.get("success"):
                                logger.info(f"  └─ Contested Level: {contested.get('contested_level', 0)}% ({contested.get('classification', 'UNKNOWN')})")
                    else:
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
                if analysis["metrics"]["win_rate"] >= min_winrate or
                analysis["metrics"]["profit_factor"] > 1.5
            ]
            
            # Categorize wallets
            gem_finders = [a for a in filtered_analyses if a["wallet_type"] == "gem_finder"]
            consistent = [a for a in filtered_analyses if a["wallet_type"] == "consistent"]
            flippers = [a for a in filtered_analyses if a["wallet_type"] == "flipper"]
            others = [a for a in filtered_analyses if a["wallet_type"] == "unknown"]
            
            # Sort each category by performance metrics
            gem_finders.sort(key=lambda x: x["metrics"]["max_roi"], reverse=True)
            consistent.sort(key=lambda x: x["metrics"]["median_roi"], reverse=True)
            flippers.sort(key=lambda x: x["metrics"]["win_rate"], reverse=True)
            
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
                "api_source": "Enhanced (Cielo + RPC)" if self.cielo_api else "Birdeye + RPC",
                "contested_analysis_included": include_contested
            }
            
        except Exception as e:
            logger.error(f"Error during batch analysis: {str(e)}")
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
                for key, value in analysis["metrics"].items():
                    if key != "roi_distribution":
                        writer.writerow({"metric": key, "value": value})
                
                # ROI distribution
                for key, value in analysis["metrics"].get("roi_distribution", {}).items():
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