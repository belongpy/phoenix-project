"""
Birdeye API Module - Phoenix Project

This module handles all interactions with Birdeye Solana API.
"""

import requests
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("phoenix.birdeye")

class BirdeyeAPI:
    """Client for interacting with Birdeye Solana API."""
    
    BASE_URL = "https://public-api.birdeye.so"
    
    def __init__(self, api_key: str):
        """
        Initialize the Birdeye API client.
        
        Args:
            api_key (str): The Birdeye API key
        """
        self.api_key = api_key
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     retry_count: int = 3, retry_delay: int = 2) -> Dict[str, Any]:
        """
        Make a request to the Birdeye API with retry logic.
        
        Args:
            endpoint (str): API endpoint to call
            params (Dict[str, Any], optional): Query parameters
            retry_count (int): Number of retries on failure
            retry_delay (int): Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: Response data as JSON
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(retry_count):
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                return response.json()
            
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{retry_count}): {str(e)}")
                
                if attempt < retry_count - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Request failed after {retry_count} attempts")
                    raise
    
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get information about a token.
        
        Args:
            token_address (str): The token contract address
            
        Returns:
            Dict[str, Any]: Token information
        """
        endpoint = f"/v1/token/info"
        params = {"address": token_address}
        
        logger.debug(f"Fetching token info for {token_address}")
        return self._make_request(endpoint, params)
    
    def get_token_price(self, token_address: str) -> Dict[str, Any]:
        """
        Get the current price of a token.
        
        Args:
            token_address (str): The token contract address
            
        Returns:
            Dict[str, Any]: Token price information
        """
        endpoint = f"/v1/token/price"
        params = {"address": token_address}
        
        logger.debug(f"Fetching token price for {token_address}")
        return self._make_request(endpoint, params)
    
    def get_token_price_history(self, token_address: str, 
                              start_time: int, end_time: int, 
                              resolution: str = "5m") -> Dict[str, Any]:
        """
        Get historical price data for a token.
        
        Args:
            token_address (str): The token contract address
            start_time (int): Start timestamp in milliseconds
            end_time (int): End timestamp in milliseconds
            resolution (str): Time resolution (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            Dict[str, Any]: Historical price data
        """
        endpoint = f"/v1/token/chart"
        params = {
            "address": token_address,
            "startTime": start_time,
            "endTime": end_time,
            "resolution": resolution
        }
        
        logger.debug(f"Fetching price history for {token_address} from {start_time} to {end_time}")
        return self._make_request(endpoint, params)
    
    def get_wallet_transactions(self, wallet_address: str, 
                              limit: int = 100) -> Dict[str, Any]:
        """
        Get transactions for a wallet.
        
        Args:
            wallet_address (str): The wallet address
            limit (int): Maximum number of transactions to return
            
        Returns:
            Dict[str, Any]: Wallet transactions
        """
        endpoint = f"/v1/wallet/transaction_history"
        params = {
            "wallet": wallet_address,
            "limit": limit
        }
        
        logger.debug(f"Fetching transactions for wallet {wallet_address}")
        return self._make_request(endpoint, params)
    
    def get_wallet_tokens(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get token holdings for a wallet.
        
        Args:
            wallet_address (str): The wallet address
            
        Returns:
            Dict[str, Any]: Wallet token holdings
        """
        endpoint = f"/v1/wallet/tokens"
        params = {"wallet": wallet_address}
        
        logger.debug(f"Fetching token holdings for wallet {wallet_address}")
        return self._make_request(endpoint, params)
    
    def get_dex_trades(self, token_address: str, 
                     limit: int = 100) -> Dict[str, Any]:
        """
        Get recent DEX trades for a token.
        
        Args:
            token_address (str): The token contract address
            limit (int): Maximum number of trades to return
            
        Returns:
            Dict[str, Any]: DEX trades for the token
        """
        endpoint = f"/v1/dex/trades"
        params = {
            "address": token_address,
            "limit": limit
        }
        
        logger.debug(f"Fetching DEX trades for token {token_address}")
        return self._make_request(endpoint, params)
    
    def calculate_token_performance(self, token_address: str, 
                                  start_time: datetime) -> Dict[str, Any]:
        """
        Calculate performance metrics for a token since a specific time.
        
        Args:
            token_address (str): The token contract address
            start_time (datetime): The starting time for performance calculation
            
        Returns:
            Dict[str, Any]: Token performance metrics
        """
        # Convert datetime to millisecond timestamps
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(datetime.now().timestamp() * 1000)
        
        # Get historical price data
        price_history = self.get_token_price_history(
            token_address, 
            start_timestamp, 
            end_timestamp
        )
        
        # Get current price
        current_price_data = self.get_token_price(token_address)
        
        # Get market cap
        token_info = self.get_token_info(token_address)
        market_cap_usd = 0
        if token_info.get("success") and "data" in token_info:
            market_cap_usd = token_info["data"].get("marketCap", 0)
        
        # Process and calculate metrics
        if not price_history.get("data", []):
            logger.warning(f"No price history available for {token_address}")
            return {
                "success": False,
                "error": "No price history available"
            }
        
        prices = price_history.get("data", [])
        
        # Calculate metrics
        initial_price = prices[0]["value"] if prices else None
        current_price = current_price_data.get("data", {}).get("value")
        
        if not initial_price or not current_price:
            return {
                "success": False,
                "error": "Missing price data"
            }
        
        price_values = [p["value"] for p in prices if "value" in p]
        max_price = max(price_values) if price_values else initial_price
        min_price = min(price_values) if price_values else initial_price
        
        roi = ((current_price / initial_price) - 1) * 100
        max_roi = ((max_price / initial_price) - 1) * 100
        max_drawdown = ((min_price / initial_price) - 1) * 100
        
        # Extract SOL price ratio if available
        sol_price = 0
        sol_value = 0
        if "solPrice" in current_price_data.get("data", {}):
            sol_price = current_price_data["data"]["solPrice"]
            if sol_price > 0:
                sol_value = current_price / sol_price
        
        return {
            "success": True,
            "token_address": token_address,
            "initial_price": initial_price,
            "current_price": current_price,
            "max_price": max_price,
            "min_price": min_price,
            "roi_percent": roi,
            "max_roi_percent": max_roi,
            "max_drawdown_percent": max_drawdown,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "market_cap_usd": market_cap_usd,
            "sol_price": sol_price,
            "sol_value": sol_value
        }
    
    def analyze_wallet_performance(self, wallet_address: str, 
                                 days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze the trading performance of a wallet.
        
        Args:
            wallet_address (str): The wallet address
            days_back (int): Number of days to analyze
            
        Returns:
            Dict[str, Any]: Wallet performance analysis
        """
        # Get wallet transactions
        transactions = self.get_wallet_transactions(wallet_address)
        
        if not transactions.get("data", []):
            logger.warning(f"No transactions found for wallet {wallet_address}")
            return {
                "success": False,
                "error": "No transactions found"
            }
        
        # TODO: Implement detailed wallet performance analysis
        # This will require tracking buys and sells, calculating ROI per trade,
        # and computing statistics on trade performance
        
        return {
            "success": True,
            "wallet_address": wallet_address,
            "analysis_period_days": days_back,
            "transaction_count": len(transactions.get("data", [])),
            # Placeholder for actual analysis
            "win_rate": 0,
            "median_roi": 0,
            "std_dev_roi": 0,
            "wallet_type": "unknown"
        }