"""
Wallet Analysis Module - Phoenix Project

This module handles wallet analysis for copy trading.
"""

import csv
import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger("phoenix.wallet")

class WalletAnalyzer:
    """Class for analyzing wallets for copy trading."""
    
    def __init__(self, birdeye_api: Any):
        """
        Initialize the wallet analyzer.
        
        Args:
            birdeye_api (BirdeyeAPI): Birdeye API client
        """
        self.birdeye_api = birdeye_api
        
        # Track entry times for tokens to detect correlated wallets
        self.token_entries = {}  # token_address -> {wallet_address -> timestamp}
    
    def _extract_trades(self, transactions: List[Dict[str, Any]], wallet_address: str) -> List[Dict[str, Any]]:
        """
        Extract and categorize trades from transaction history.
        
        Args:
            transactions (List[Dict[str, Any]]): Wallet transaction history
            wallet_address (str): The wallet address being analyzed
            
        Returns:
            List[Dict[str, Any]]: Categorized trades
        """
        trades = []
        
        for tx in transactions:
            # Skip transactions without proper data
            if not tx.get("data") or not isinstance(tx["data"], dict):
                continue
                
            tx_data = tx["data"]
            
            # Extract common fields
            trade = {
                "tx_hash": tx_data.get("signature", ""),
                "timestamp": tx_data.get("blockTime", 0),
                "date": datetime.fromtimestamp(tx_data.get("blockTime", 0)),
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
            
            # Determine if it's a buy or sell transaction
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
                        
                        # Try to extract SOL amount
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
                        
                        # Try to extract SOL amount
                        if "nativeTransfers" in tx_data:
                            for native_transfer in tx_data["nativeTransfers"]:
                                if native_transfer.get("fromOwner") == "wallet":
                                    trade["amount_sol"] = float(native_transfer.get("amount", 0))
                                    break
                        
                        # Fetch market cap if we have a token address
                        if trade["token_address"]:
                            try:
                                token_info = self.birdeye_api.get_token_info(trade["token_address"])
                                if token_info.get("success") and "data" in token_info:
                                    trade["market_cap_usd"] = token_info["data"].get("marketCap", 0)
                                    
                                    # Identify platform
                                    token_data = token_info["data"]
                                    platforms = {
                                        "letsbonk": ["BONK", "BK"],
                                        "raydium": ["RAY"],
                                        "pumpfun": ["PUMP", "PF"],
                                        "pumpswap": ["PUMP", "PS"],
                                        "meteora": ["MTR"],
                                        "launchpad": ["LP", "LAUNCH"]
                                    }
                                    
                                    symbol = token_data.get("symbol", "")
                                    for platform, identifiers in platforms.items():
                                        for identifier in identifiers:
                                            if identifier in symbol:
                                                trade["platform"] = platform
                                                break
                                        if trade["platform"]:
                                            break
                                    
                                    name = token_data.get("name", "")
                                    if not trade["platform"]:
                                        for platform in platforms.keys():
                                            if platform.lower() in name.lower():
                                                trade["platform"] = platform
                                                break
                            except Exception as e:
                                logger.warning(f"Error fetching market cap for {trade['token_address']}: {str(e)}")
                        
                        # Store token entry for correlation analysis
                        if trade["token_address"]:
                            if trade["token_address"] not in self.token_entries:
                                self.token_entries[trade["token_address"]] = {}
                            
                            self.token_entries[trade["token_address"]][wallet_address] = trade["timestamp"]
                        
                        trades.append(trade.copy())
        
        return trades
    
    def _pair_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Pair buy and sell trades to calculate ROI.
        
        Args:
            trades (List[Dict[str, Any]]): Categorized trades
            
        Returns:
            List[Dict[str, Any]]: Paired trades with ROI
        """
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
        
        return paired_trades
    
    def _calculate_metrics(self, paired_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate performance metrics from paired trades.
        
        Args:
            paired_trades (List[Dict[str, Any]]): Paired trades with ROI
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if not paired_trades:
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
            "5x_plus": len([t for t in paired_trades if t["roi_percent"] >= 500]),
            "2x_to_5x": len([t for t in paired_trades if 200 <= t["roi_percent"] < 500]),
            "1x_to_2x": len([t for t in paired_trades if 100 <= t["roi_percent"] < 200]),
            "50_to_100": len([t for t in paired_trades if 50 <= t["roi_percent"] < 100]),
            "20_to_50": len([t for t in paired_trades if 20 <= t["roi_percent"] < 50]),
            "0_to_20": len([t for t in paired_trades if 0 <= t["roi_percent"] < 20]),
            "minus20_to_0": len([t for t in paired_trades if -20 <= t["roi_percent"] < 0]),
            "minus50_to_minus20": len([t for t in paired_trades if -50 <= t["roi_percent"] < -20]),
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
    
    def _determine_wallet_type(self, metrics: Dict[str, Any]) -> str:
        """
        Determine the wallet type based on metrics.
        
        Args:
            metrics (Dict[str, Any]): Wallet performance metrics
            
        Returns:
            str: Wallet type (gem_finder, consistent, flipper)
        """
        # Not enough data
        if metrics["total_trades"] < 5:
            return "unknown"
        
        # Extract relevant metrics
        win_rate = metrics["win_rate"]
        median_roi = metrics["median_roi"]
        std_dev_roi = metrics["std_dev_roi"]
        avg_hold_time_hours = metrics["avg_hold_time_hours"]
        roi_distribution = metrics["roi_distribution"]
        
        # Calculate key indicators
        big_win_count = roi_distribution["5x_plus"] + roi_distribution["2x_to_5x"]
        big_win_ratio = big_win_count / metrics["total_trades"] if metrics["total_trades"] > 0 else 0
        
        # Gem finder: High ROI potential, might have lower win rate but finds significant gems
        if big_win_ratio >= 0.15 and metrics["max_roi"] >= 300:
            return "gem_finder"
        
        # Flipper: Quick trades, lower hold time, may have good win rate but smaller ROIs
        if avg_hold_time_hours < 12 and win_rate > 50:
            return "flipper"
        
        # Consistent: Good win rate, stable returns, less volatility
        if win_rate >= 45 and median_roi > 0 and std_dev_roi < 100:
            return "consistent"
        
        # Default to unknown if no clear pattern
        return "unknown"
    
    def _generate_strategy(self, wallet_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading strategy based on wallet type and metrics.
        
        Args:
            wallet_type (str): Wallet type
            metrics (Dict[str, Any]): Wallet performance metrics
            
        Returns:
            Dict[str, Any]: Trading strategy
        """
        if wallet_type == "gem_finder":
            # Check if this wallet has very high ROI values
            if metrics["max_roi"] >= 500:  # 5x+ potential
                return {
                    "recommendation": "HOLD_MOON",
                    "entry_type": "IMMEDIATE",
                    "entry": "FAST_FOLLOW",
                    "position_size": "SMALL",
                    "take_profit_1": 100,  # 100% ROI
                    "take_profit_2": 200,  # 200% ROI
                    "take_profit_3": 500,  # 500% ROI
                    "stop_loss": -30,  # 30% loss
                    "trailing_stop": {
                        "activation": 100,  # Activate at 100% profit
                        "trailing_percent": 25  # 25% trailing stop
                    },
                    "notes": "This wallet finds massive moonshots. Take 30% at TP1, hold rest for major gains. Use smaller position size."
                }
            else:
                return {
                    "recommendation": "SCALP_AND_HOLD",
                    "entry_type": "IMMEDIATE",
                    "entry": "FAST_FOLLOW",
                    "position_size": "SMALL",
                    "take_profit_1": 50,  # 50% ROI
                    "take_profit_2": 100,  # 100% ROI
                    "take_profit_3": 300,  # 300% ROI
                    "stop_loss": -30,  # 30% loss
                    "trailing_stop": {
                        "activation": 100,  # Activate at 100% profit
                        "trailing_percent": 25  # 25% trailing stop
                    },
                    "notes": "This wallet finds potential moonshots. Take partial profits at each level and let the rest ride with a trailing stop."
                }
        
        elif wallet_type == "consistent":
            if metrics["win_rate"] > 65 and metrics["median_roi"] > 50:
                return {
                    "recommendation": "SCALP_AND_HOLD",
                    "entry_type": "IMMEDIATE",
                    "entry": "FOLLOW",
                    "position_size": "MEDIUM",
                    "take_profit_1": 40,  # 40% ROI
                    "take_profit_2": 80,  # 80% ROI
                    "take_profit_3": 150,  # 150% ROI
                    "stop_loss": -25,  # 25% loss
                    "trailing_stop": {
                        "activation": 40,  # Activate at 40% profit
                        "trailing_percent": 20  # 20% trailing stop
                    },
                    "notes": "High-performance consistent wallet. Follow with confidence and use larger position size."
                }
            else:
                return {
                    "recommendation": "SCALP",
                    "entry_type": "IMMEDIATE",
                    "entry": "FOLLOW",
                    "position_size": "MEDIUM",
                    "take_profit_1": 30,  # 30% ROI
                    "take_profit_2": 50,  # 50% ROI
                    "take_profit_3": 100,  # 100% ROI
                    "stop_loss": -25,  # 25% loss
                    "trailing_stop": {
                        "activation": 30,  # Activate at 30% profit
                        "trailing_percent": 15  # 15% trailing stop
                    },
                    "notes": "This wallet has consistent returns. Follow their trades with confidence but maintain disciplined exits."
                }
        
        elif wallet_type == "flipper":
            return {
                "recommendation": "SCALP",
                "entry_type": "IMMEDIATE",
                "entry": "FAST_FOLLOW",
                "position_size": "MEDIUM",
                "take_profit_1": 15,  # 15% ROI
                "take_profit_2": 30,  # 30% ROI
                "take_profit_3": 60,  # 60% ROI
                "stop_loss": -15,  # 15% loss
                "trailing_stop": {
                    "activation": 15,  # Activate at 15% profit
                    "trailing_percent": 10  # 10% trailing stop
                },
                "notes": "This wallet makes quick flips. Enter and exit quickly, don't hold long-term."
            }
        
        else:  # unknown
            return {
                "recommendation": "CAUTIOUS",
                "entry_type": "WAIT_FOR_CONFIRMATION",
                "entry": "CAUTIOUS",
                "position_size": "SMALL",
                "take_profit_1": 20,  # 20% ROI
                "take_profit_2": 40,  # 40% ROI
                "take_profit_3": 80,  # 80% ROI
                "stop_loss": -20,  # 20% loss
                "trailing_stop": {
                    "activation": 20,  # Activate at 20% profit
                    "trailing_percent": 15  # 15% trailing stop
                },
                "notes": "Limited data or unclear pattern. Use caution when following this wallet."
            }
    
    def analyze_wallet(self, wallet_address: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze a wallet for copy trading.
        
        Args:
            wallet_address (str): Wallet address
            days_back (int): Number of days to analyze
            
        Returns:
            Dict[str, Any]: Wallet analysis results
        """
        logger.info(f"Analyzing wallet {wallet_address} for the past {days_back} days")
        
        try:
            # Get wallet transactions
            transactions = self.birdeye_api.get_wallet_transactions(wallet_address)
            
            if not transactions.get("data", []):
                logger.warning(f"No transactions found for wallet {wallet_address}")
                return {
                    "success": False,
                    "error": "No transactions found",
                    "wallet_address": wallet_address
                }
            
            # Filter transactions by date
            date_limit = datetime.now() - timedelta(days=days_back)
            filtered_transactions = [
                tx for tx in transactions.get("data", [])
                if "blockTime" in tx.get("data", {}) and 
                datetime.fromtimestamp(tx["data"]["blockTime"]) >= date_limit
            ]
            
            logger.info(f"Found {len(filtered_transactions)} transactions in the last {days_back} days")
            
            # Extract trades
            trades = self._extract_trades(filtered_transactions, wallet_address)
            logger.info(f"Extracted {len(trades)} trades")
            
            # Pair trades
            paired_trades = self._pair_trades(trades)
            logger.info(f"Paired {len(paired_trades)} trades")
            
            # Calculate metrics
            metrics = self._calculate_metrics(paired_trades)
            
            # Find correlated wallets
            correlated_wallets = self._find_correlated_wallets(wallet_address)
            
            # Determine wallet type
            wallet_type = self._determine_wallet_type(metrics)
            logger.info(f"Wallet type: {wallet_type}")
            
            # Generate strategy
            strategy = self._generate_strategy(wallet_type, metrics)
            
            return {
                "success": True,
                "wallet_address": wallet_address,
                "analysis_period_days": days_back,
                "wallet_type": wallet_type,
                "metrics": metrics,
                "strategy": strategy,
                "trades": paired_trades,
                "correlated_wallets": correlated_wallets
            }
            
        except Exception as e:
            logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "wallet_address": wallet_address
            }
    
    def _find_correlated_wallets(self, wallet_address: str, time_threshold: int = 300) -> List[Dict[str, Any]]:
        """
        Find wallets that entered the same tokens within a close time window.
        
        Args:
            wallet_address (str): The main wallet address
            time_threshold (int): Time threshold in seconds to consider wallets correlated
            
        Returns:
            List[Dict[str, Any]]: List of correlated wallets with correlation details
        """
        correlated_wallets = {}
        
        # Look for wallets that entered the same tokens within the time threshold
        for token_address, entries in self.token_entries.items():
            if wallet_address in entries:
                main_timestamp = entries[wallet_address]
                
                # Find other wallets that entered within the time threshold
                for other_wallet, other_timestamp in entries.items():
                    if other_wallet != wallet_address:
                        time_diff = abs(main_timestamp - other_timestamp)
                        
                        # If within threshold, consider correlated
                        if time_diff <= time_threshold:
                            if other_wallet not in correlated_wallets:
                                correlated_wallets[other_wallet] = {
                                    "wallet_address": other_wallet,
                                    "common_tokens": 0,
                                    "avg_time_diff": 0,
                                    "tokens": []
                                }
                            
                            # Update correlation data
                            correlated_wallets[other_wallet]["common_tokens"] += 1
                            current_avg = correlated_wallets[other_wallet]["avg_time_diff"]
                            current_count = len(correlated_wallets[other_wallet]["tokens"])
                            new_avg = (current_avg * current_count + time_diff) / (current_count + 1)
                            correlated_wallets[other_wallet]["avg_time_diff"] = new_avg
                            correlated_wallets[other_wallet]["tokens"].append({
                                "token_address": token_address,
                                "time_diff": time_diff
                            })
        
        # Sort by number of common tokens
        sorted_wallets = sorted(
            correlated_wallets.values(),
            key=lambda x: (x["common_tokens"], -x["avg_time_diff"]),
            reverse=True
        )
        
        return sorted_wallets
    
    def export_wallet_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """
        Export wallet analysis to CSV.
        
        Args:
            analysis (Dict[str, Any]): Wallet analysis data
            output_file (str): Output file path
        """
        if not analysis.get("success"):
            logger.warning(f"No successful analysis to export for {analysis.get('wallet_address')}")
            return
        
        try:
            # Ensure output directories exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            
            # Export metrics to CSV
            metrics_file = output_file.replace(".csv", "_metrics.csv")
            with open(metrics_file, 'w', newline='') as f:
                fieldnames = ["metric", "value"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write general info
                writer.writerow({
                    "metric": "wallet_address",
                    "value": analysis["wallet_address"]
                })
                writer.writerow({
                    "metric": "analysis_period_days",
                    "value": analysis["analysis_period_days"]
                })
                writer.writerow({
                    "metric": "wallet_type",
                    "value": analysis["wallet_type"]
                })
                
                # Write entry type for easy reference
                if "strategy" in analysis and "entry_type" in analysis["strategy"]:
                    writer.writerow({
                        "metric": "entry_type",
                        "value": analysis["strategy"]["entry_type"]
                    })
                
                # Write metrics
                metrics = analysis["metrics"]
                for key, value in metrics.items():
                    if key != "roi_distribution":
                        writer.writerow({
                            "metric": key,
                            "value": value
                        })
                
                # Write ROI distribution
                for key, value in metrics.get("roi_distribution", {}).items():
                    writer.writerow({
                        "metric": f"roi_dist_{key}",
                        "value": value
                    })
                
                logger.info(f"Exported wallet metrics to {metrics_file}")
            
            # Export strategy to CSV
            strategy_file = output_file.replace(".csv", "_strategy.csv")
            with open(strategy_file, 'w', newline='') as f:
                fieldnames = ["parameter", "value"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write strategy parameters
                strategy = analysis["strategy"]
                for key, value in strategy.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            writer.writerow({
                                "parameter": f"{key}_{sub_key}",
                                "value": sub_value
                            })
                    else:
                        writer.writerow({
                            "parameter": key,
                            "value": value
                        })
                
                logger.info(f"Exported wallet strategy to {strategy_file}")
            
            # Export trades to CSV
            if analysis.get("trades"):
                trades_file = output_file
                with open(trades_file, 'w', newline='') as f:
                    if not analysis["trades"]:
                        logger.warning("No trades to export")
                        return
                    
                    # Get fieldnames from the first trade
                    fieldnames = list(analysis["trades"][0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write trades
                    for trade in analysis["trades"]:
                        writer.writerow(trade)
                    
                    logger.info(f"Exported {len(analysis['trades'])} trades to {trades_file}")
            
            # Export correlated wallets if available
            if analysis.get("correlated_wallets"):
                correlated_file = output_file.replace(".csv", "_correlated_wallets.csv")
                with open(correlated_file, 'w', newline='') as f:
                    fieldnames = ["wallet_address", "common_tokens", "avg_time_diff", "tokens"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write correlated wallets
                    for wallet in analysis["correlated_wallets"]:
                        wallet_data = wallet.copy()
                        # Convert tokens list to string for CSV
                        if "tokens" in wallet_data:
                            wallet_data["tokens"] = str(wallet_data["tokens"])
                        writer.writerow(wallet_data)
                    
                    logger.info(f"Exported {len(analysis['correlated_wallets'])} correlated wallets to {correlated_file}")
        
        except Exception as e:
            logger.error(f"Error exporting wallet analysis: {str(e)}")
    
    def batch_analyze_wallets(self, wallet_addresses: List[str], 
                            days_back: int = 30,
                            min_winrate: float = 45.0) -> Dict[str, Any]:
        """
        Batch analyze multiple wallets and categorize them.
        
        Args:
            wallet_addresses (List[str]): List of wallet addresses
            days_back (int): Number of days to analyze
            min_winrate (float): Minimum win rate percentage
            
        Returns:
            Dict[str, Any]: Categorized wallet analyses
        """
        logger.info(f"Batch analyzing {len(wallet_addresses)} wallets")
        
        # Analyze each wallet
        wallet_analyses = []
        for wallet_address in wallet_addresses:
            try:
                analysis = self.analyze_wallet(wallet_address, days_back)
                if analysis.get("success"):
                    wallet_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing wallet {wallet_address}: {str(e)}")
        
        # Filter by minimum win rate
        filtered_analyses = [
            analysis for analysis in wallet_analyses
            if analysis["metrics"]["win_rate"] >= min_winrate or
            analysis["metrics"]["profit_factor"] > 1.5  # Allow lower win rate if profit factor is good
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
        
        # Find wallet clusters (groups of wallets that tend to trade together)
        wallet_clusters = self._identify_wallet_clusters(
            {a["wallet_address"]: a.get("correlated_wallets", []) for a in wallet_analyses}
        )
        
        return {
            "total_wallets": len(wallet_addresses),
            "analyzed_wallets": len(wallet_analyses),
            "filtered_wallets": len(filtered_analyses),
            "gem_finders": gem_finders,
            "consistent": consistent,
            "flippers": flippers,
            "others": others,
            "wallet_correlations": {a["wallet_address"]: a.get("correlated_wallets", []) for a in wallet_analyses},
            "wallet_clusters": wallet_clusters
        }
    
    def _identify_wallet_clusters(self, wallet_correlations: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Identify clusters of wallets that appear to be trading together.
        
        Args:
            wallet_correlations (Dict[str, List]): Dictionary of wallet addresses and their correlated wallets
            
        Returns:
            List[Dict[str, Any]]: List of wallet clusters
        """
        clusters = []
        processed_wallets = set()
        
        for wallet, correlations in wallet_correlations.items():
            if wallet in processed_wallets:
                continue
            
            # Start a new cluster with this wallet
            cluster_wallets = {wallet}
            cluster_correlation_strength = 0
            
            # Add all strongly correlated wallets
            for correlation in correlations:
                correlated_wallet = correlation["wallet_address"]
                if correlated_wallet not in processed_wallets and correlation["common_tokens"] >= 2:
                    cluster_wallets.add(correlated_wallet)
                    cluster_correlation_strength += correlation["common_tokens"]
            
            # Only create clusters with multiple wallets
            if len(cluster_wallets) > 1:
                clusters.append({
                    "wallets": list(cluster_wallets),
                    "size": len(cluster_wallets),
                    "correlation_strength": cluster_correlation_strength
                })
                
                # Mark all these wallets as processed
                processed_wallets.update(cluster_wallets)
        
        # Sort clusters by size and correlation strength
        clusters.sort(key=lambda x: (x["size"], x["correlation_strength"]), reverse=True)
        
        return clusters
    
    def export_batch_analysis(self, batch_analysis: Dict[str, Any], output_file: str) -> None:
        """
        Export batch wallet analysis to CSV.
        
        Args:
            batch_analysis (Dict[str, Any]): Batch analysis data
            output_file (str): Output file path
        """
        try:
            # Ensure output directories exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            
            # Prepare summary file
            summary_file = output_file.replace(".csv", "_summary.csv")
            with open(summary_file, 'w', newline='') as f:
                fieldnames = ["metric", "value"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write summary metrics
                writer.writerow({"metric": "total_wallets", "value": batch_analysis["total_wallets"]})
                writer.writerow({"metric": "analyzed_wallets", "value": batch_analysis["analyzed_wallets"]})
                writer.writerow({"metric": "filtered_wallets", "value": batch_analysis["filtered_wallets"]})
                writer.writerow({"metric": "gem_finder_count", "value": len(batch_analysis["gem_finders"])})
                writer.writerow({"metric": "consistent_count", "value": len(batch_analysis["consistent"])})
                writer.writerow({"metric": "flipper_count", "value": len(batch_analysis["flippers"])})
                writer.writerow({"metric": "other_count", "value": len(batch_analysis["others"])})
                
                logger.info(f"Exported batch analysis summary to {summary_file}")
            
            # Export categorized wallets
            categories = {
                "gem_finder": batch_analysis["gem_finders"],
                "consistent": batch_analysis["consistent"],
                "flipper": batch_analysis["flippers"],
                "other": batch_analysis["others"]
            }
            
            for category, wallets in categories.items():
                if not wallets:
                    continue
                
                category_file = output_file.replace(".csv", f"_{category}.csv")
                with open(category_file, 'w', newline='') as f:
                    fieldnames = [
                        "wallet_address", "wallet_type", "entry_type", "strategy", "win_rate", "profit_factor",
                        "median_roi", "std_dev_roi", "max_roi", "total_trades",
                        "avg_hold_time_hours", "avg_bet_size_usd", "buy_value_sol", "market_cap_at_buy", 
                        "total_tokens_traded", "platform"
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write wallet data
                    for wallet in wallets:
                        metrics = wallet["metrics"]
                        strategy = wallet["strategy"]
                        
                        # Calculate average SOL values and most common platform
                        sol_values = [t.get("buy_value_sol", 0) for t in wallet.get("trades", []) if "buy_value_sol" in t]
                        market_caps = [t.get("market_cap_at_buy", 0) for t in wallet.get("trades", []) if "market_cap_at_buy" in t]
                        platforms = [t.get("platform", "") for t in wallet.get("trades", []) if "platform" in t and t["platform"]]
                        
                        avg_sol = sum(sol_values) / len(sol_values) if sol_values else 0
                        avg_mcap = sum(market_caps) / len(market_caps) if market_caps else 0
                        
                        # Get most common platform
                        most_common_platform = ""
                        if platforms:
                            from collections import Counter
                            platform_counts = Counter(platforms)
                            most_common_platform = platform_counts.most_common(1)[0][0] if platform_counts else ""
                        
                        row = {
                            "wallet_address": wallet["wallet_address"],
                            "wallet_type": wallet["wallet_type"],
                            "entry_type": strategy.get("entry_type", "UNKNOWN"),
                            "strategy": strategy.get("recommendation", "CAUTIOUS"),
                            "win_rate": metrics["win_rate"],
                            "profit_factor": metrics["profit_factor"],
                            "median_roi": metrics["median_roi"],
                            "std_dev_roi": metrics["std_dev_roi"],
                            "max_roi": metrics["max_roi"],
                            "total_trades": metrics["total_trades"],
                            "avg_hold_time_hours": metrics["avg_hold_time_hours"],
                            "avg_bet_size_usd": metrics["avg_bet_size_usd"],
                            "buy_value_sol": avg_sol,
                            "market_cap_at_buy": avg_mcap,
                            "total_tokens_traded": metrics["total_tokens_traded"],
                            "platform": most_common_platform
                        }
                        writer.writerow(row)
                
                logger.info(f"Exported {len(wallets)} {category} wallets to {category_file}")
            
            # Export detailed analysis for each wallet
            for category, wallets in categories.items():
                for wallet in wallets:
                    wallet_file = output_file.replace(".csv", f"_detail_{wallet['wallet_address'][:8]}.csv")
                    self.export_wallet_analysis(wallet, wallet_file)
            
            # Export wallet clusters if available
            if "wallet_clusters" in batch_analysis and batch_analysis["wallet_clusters"]:
                clusters_file = output_file.replace(".csv", "_wallet_clusters.csv")
                with open(clusters_file, 'w', newline='') as f:
                    fieldnames = ["cluster_id", "size", "correlation_strength", "wallets"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write cluster data
                    for i, cluster in enumerate(batch_analysis["wallet_clusters"]):
                        row = {
                            "cluster_id": i + 1,
                            "size": cluster["size"],
                            "correlation_strength": cluster["correlation_strength"],
                            "wallets": ", ".join(cluster["wallets"])
                        }
                        writer.writerow(row)
                
                logger.info(f"Exported {len(batch_analysis['wallet_clusters'])} wallet clusters to {clusters_file}")
        
        except Exception as e:
            logger.error(f"Error exporting batch analysis: {str(e)}")