"""
Birdeye API Module - Phoenix Project (Complete Fixed Version)

This module handles all interactions with Birdeye Solana API with proper error handling.
FIXES:
- Added address validation to prevent invalid API calls
- Special handling for pump.fun tokens
- Better error handling and logging
- Improved price history resolution handling
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
    
    def _is_valid_solana_address(self, address: str) -> bool:
        """
        Validate Solana address format before making API calls.
        
        Args:
            address (str): Address to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Basic validation
        if not address or len(address) < 32 or len(address) > 44:
            return False
        
        # Check if it contains only base58 characters
        base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        if not all(c in base58_chars for c in address):
            return False
        
        # Reject if ALL lowercase (likely invalid)
        if address.islower():
            logger.warning(f"Rejecting all-lowercase address: {address}")
            return False
        
        # Reject known system programs
        system_programs = [
            "11111111111111111111111111111111",
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "So11111111111111111111111111111111111111112",
        ]
        if address in system_programs:
            return False
        
        return True
    
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
                
                # Handle specific error codes
                if response.status_code == 400:
                    logger.warning(f"Bad request for {endpoint} with params {params}: {response.text}")
                    return {
                        "success": False,
                        "error": "Invalid token address or bad request",
                        "data": None
                    }
                elif response.status_code == 404:
                    logger.warning(f"Token not found: {params}")
                    return {
                        "success": False,
                        "error": "Token not found",
                        "data": None
                    }
                elif response.status_code == 429:
                    logger.warning(f"Rate limit hit, waiting longer...")
                    time.sleep(retry_delay * 2)
                    continue
                
                response.raise_for_status()
                result = response.json()
                
                # Ensure consistent response format
                if isinstance(result, dict):
                    if "success" not in result:
                        result["success"] = True
                    return result
                else:
                    return {
                        "success": True,
                        "data": result
                    }
            
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{retry_count}): {str(e)}")
                
                if attempt < retry_count - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Request failed after {retry_count} attempts")
                    return {
                        "success": False,
                        "error": str(e),
                        "data": None
                    }
    
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get information about a token using the correct endpoint.
        
        Args:
            token_address (str): The token contract address
            
        Returns:
            Dict[str, Any]: Token information
        """
        # Validate address first
        if not self._is_valid_solana_address(token_address):
            logger.warning(f"Invalid token address format: {token_address}")
            return {
                "success": False,
                "error": "Invalid token address format",
                "data": None
            }
        
        # Use the correct endpoint
        endpoint = f"/defi/token_overview"
        params = {"address": token_address}
        
        logger.debug(f"Fetching token info for {token_address}")
        result = self._make_request(endpoint, params)
        
        # Handle pump.fun tokens that might not have data
        if not result.get("success") and token_address.endswith("pump"):
            logger.info(f"Token {token_address} appears to be a pump.fun token with limited data")
            return {
                "success": True,
                "data": {
                    "address": token_address,
                    "name": "Unknown Token",
                    "symbol": "UNKNOWN",
                    "marketCap": 0,
                    "platform": "pump.fun"
                }
            }
        
        return result
    
    def get_token_price(self, token_address: str) -> Dict[str, Any]:
        """
        Get the current price of a token.
        
        Args:
            token_address (str): The token contract address
            
        Returns:
            Dict[str, Any]: Token price information
        """
        # Validate address first
        if not self._is_valid_solana_address(token_address):
            logger.warning(f"Invalid token address format: {token_address}")
            return {
                "success": False,
                "error": "Invalid token address format",
                "data": {"value": 0, "solPrice": 0}
            }
        
        endpoint = f"/defi/price"
        params = {"address": token_address}
        
        logger.debug(f"Fetching token price for {token_address}")
        result = self._make_request(endpoint, params)
        
        # Handle tokens without price data
        if not result.get("success"):
            logger.warning(f"No price data available for {token_address}")
            return {
                "success": False,
                "error": "No price data available",
                "data": {"value": 0, "solPrice": 0}
            }
        
        return result
    
    def get_token_price_history(self, token_address: str, 
                              start_time: int, end_time: int, 
                              resolution: str = "5m") -> Dict[str, Any]:
        """
        Get historical price data for a token.
        
        Args:
            token_address (str): The token contract address
            start_time (int): Start timestamp in seconds (not milliseconds)
            end_time (int): End timestamp in seconds (not milliseconds)
            resolution (str): Time resolution (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            Dict[str, Any]: Historical price data
        """
        # Validate address first
        if not self._is_valid_solana_address(token_address):
            logger.warning(f"Invalid token address format: {token_address}")
            return {
                "success": False,
                "error": "Invalid token address format",
                "data": {"items": []}
            }
        
        # Special handling for pump.fun tokens
        if token_address.endswith("pump"):
            logger.info(f"Token {token_address} is a pump.fun token, limited price history available")
            return {
                "success": False,
                "error": "Limited data for pump.fun tokens",
                "data": {"items": []},
                "is_pump_token": True
            }
        
        endpoint = f"/defi/history_price"
        params = {
            "address": token_address,
            "address_type": "token",
            "type": resolution,
            "time_from": start_time,
            "time_to": end_time
        }
        
        logger.debug(f"Fetching price history for {token_address}")
        result = self._make_request(endpoint, params)
        
        # Handle tokens without price history
        if not result.get("success") or not result.get("data", {}).get("items"):
            logger.warning(f"No price history available for {token_address}")
            # Check if it's a pump.fun token
            if "pump" in token_address.lower():
                return {
                    "success": False,
                    "error": "No price history available for pump.fun token",
                    "data": {"items": []},
                    "is_pump_token": True
                }
            return {
                "success": False,
                "error": "No price history available",
                "data": {"items": []}
            }
        
        return result
    
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
        # Validate address first
        if not self._is_valid_solana_address(wallet_address):
            logger.warning(f"Invalid wallet address format: {wallet_address}")
            return {
                "success": False,
                "error": "Invalid wallet address format",
                "data": []
            }
        
        endpoint = f"/v1/wallet/tx"
        params = {
            "wallet": wallet_address,
            "limit": min(limit, 100)  # Birdeye has limits
        }
        
        logger.debug(f"Fetching transactions for wallet {wallet_address}")
        result = self._make_request(endpoint, params)
        
        if not result.get("success"):
            logger.warning(f"No transaction data available for {wallet_address}")
            return {
                "success": False,
                "error": "No transaction data available",
                "data": []
            }
        
        return result
    
    def get_wallet_tokens(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get token holdings for a wallet.
        
        Args:
            wallet_address (str): The wallet address
            
        Returns:
            Dict[str, Any]: Wallet token holdings
        """
        # Validate address first
        if not self._is_valid_solana_address(wallet_address):
            logger.warning(f"Invalid wallet address format: {wallet_address}")
            return {
                "success": False,
                "error": "Invalid wallet address format",
                "data": []
            }
        
        endpoint = f"/v1/wallet/token_list"
        params = {"wallet": wallet_address}
        
        logger.debug(f"Fetching token holdings for wallet {wallet_address}")
        result = self._make_request(endpoint, params)
        
        if not result.get("success"):
            logger.warning(f"No token holdings data available for {wallet_address}")
            return {
                "success": False,
                "error": "No token holdings data available",
                "data": []
            }
        
        return result
    
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
        # Validate address first
        if not self._is_valid_solana_address(token_address):
            logger.warning(f"Invalid token address format: {token_address}")
            return {
                "success": False,
                "error": "Invalid token address format",
                "data": {"items": []}
            }
        
        endpoint = f"/defi/txs/token"
        params = {
            "address": token_address,
            "tx_type": "swap",
            "limit": min(limit, 50)  # Birdeye has limits
        }
        
        logger.debug(f"Fetching DEX trades for token {token_address}")
        result = self._make_request(endpoint, params)
        
        if not result.get("success"):
            logger.warning(f"No trade data available for {token_address}")
            return {
                "success": False,
                "error": "No trade data available",
                "data": {"items": []}
            }
        
        return result
    
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
        # Validate address first
        if not self._is_valid_solana_address(token_address):
            logger.warning(f"Invalid token address format: {token_address}")
            return {
                "success": False,
                "error": "Invalid token address format"
            }
        
        # Convert datetime to second timestamps (not milliseconds)
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(datetime.now().timestamp())
        
        # Get historical price data
        price_history = self.get_token_price_history(
            token_address, 
            start_timestamp, 
            end_timestamp,
            "15m"  # Use 15m for better data availability
        )
        
        # Handle pump.fun tokens
        if price_history.get("is_pump_token"):
            logger.warning(f"Cannot calculate performance for pump.fun token {token_address}")
            return {
                "success": False,
                "error": "Cannot calculate performance for pump.fun token",
                "is_pump_token": True,
                "token_address": token_address,
                "note": "pump.fun tokens have limited price history"
            }
        
        # Get current price
        current_price_data = self.get_token_price(token_address)
        
        # Get market cap
        token_info = self.get_token_info(token_address)
        market_cap_usd = 0
        if token_info.get("success") and token_info.get("data"):
            market_cap_usd = token_info["data"].get("mc", 0) or token_info["data"].get("marketCap", 0)
        
        # Process and calculate metrics
        if not price_history.get("success") or not price_history.get("data", {}).get("items"):
            logger.warning(f"No price history available for performance calculation: {token_address}")
            
            # Try to get basic current price info
            if current_price_data.get("success") and current_price_data.get("data"):
                current_price = current_price_data["data"].get("value", 0)
                return {
                    "success": True,
                    "token_address": token_address,
                    "initial_price": current_price,  # Use current as initial
                    "current_price": current_price,
                    "max_price": current_price,
                    "min_price": current_price,
                    "roi_percent": 0,
                    "current_roi_percent": 0,
                    "max_roi_percent": 0,
                    "max_drawdown_percent": 0,
                    "start_time": start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "market_cap_usd": market_cap_usd,
                    "note": "Limited price history available"
                }
            else:
                return {
                    "success": False,
                    "error": "No price data available for performance calculation"
                }
        
        prices = price_history.get("data", {}).get("items", [])
        
        # Calculate metrics
        if not prices:
            return {
                "success": False,
                "error": "No price data points available"
            }
        
        initial_price = prices[0].get("value", 0)
        current_price = current_price_data.get("data", {}).get("value", 0) if current_price_data.get("success") else prices[-1].get("value", 0)
        
        if not initial_price or not current_price:
            return {
                "success": False,
                "error": "Invalid price data"
            }
        
        price_values = [p.get("value", 0) for p in prices if p.get("value", 0) > 0]
        max_price = max(price_values) if price_values else initial_price
        min_price = min(price_values) if price_values else initial_price
        
        roi = ((current_price / initial_price) - 1) * 100
        max_roi = ((max_price / initial_price) - 1) * 100
        max_drawdown = ((min_price / initial_price) - 1) * 100
        
        # Find time to max price
        time_to_max_roi_hours = 0
        for i, price_point in enumerate(prices):
            if price_point.get("value", 0) == max_price:
                max_timestamp = price_point.get("unixTime", start_timestamp)
                time_to_max_roi_hours = (max_timestamp - start_timestamp) / 3600
                break
        
        return {
            "success": True,
            "token_address": token_address,
            "initial_price": initial_price,
            "current_price": current_price,
            "max_price": max_price,
            "min_price": min_price,
            "roi_percent": roi,
            "current_roi_percent": roi,
            "max_roi_percent": max_roi,
            "max_drawdown_percent": max_drawdown,
            "time_to_max_roi_hours": time_to_max_roi_hours,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "market_cap_usd": market_cap_usd,
            "price_points_analyzed": len(prices)
        }
    
    def identify_platform(self, contract_address: str, token_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Identify which platform a token contract is associated with.
        Fixed to handle API response format issues.
        
        Args:
            contract_address (str): Token contract address
            token_info (Dict[str, Any], optional): Pre-fetched token info
            
        Returns:
            str: Platform name or empty string if not identified
        """
        # Known platform identifiers
        platform_identifiers = {
            "pump.fun": ["pump", "pumpfun"],
            "raydium": ["ray", "raydium"],
            "jupiter": ["jup", "jupiter"],
            "orca": ["orca"],
            "meteora": ["mtr", "meteora"]
        }
        
        try:
            # Check contract address for platform indicators
            contract_lower = contract_address.lower()
            for platform, identifiers in platform_identifiers.items():
                for identifier in identifiers:
                    if identifier in contract_lower:
                        return platform
            
            # Get token info if not provided
            if not token_info:
                token_info = self.get_token_info(contract_address)
            
            # Safely extract data with proper error handling
            if isinstance(token_info, dict) and token_info.get("success") and token_info.get("data"):
                token_data = token_info["data"]
                
                # Check token symbol
                symbol = token_data.get("symbol", "")
                if isinstance(symbol, str):
                    symbol_lower = symbol.lower()
                    for platform, identifiers in platform_identifiers.items():
                        for identifier in identifiers:
                            if identifier in symbol_lower:
                                return platform
                
                # Check token name
                name = token_data.get("name", "")
                if isinstance(name, str):
                    name_lower = name.lower()
                    for platform, identifiers in platform_identifiers.items():
                        for identifier in identifiers:
                            if identifier in name_lower:
                                return platform
            
            # Try to get DEX trade data to identify platform
            try:
                dex_trades = self.get_dex_trades(contract_address, limit=5)
                if dex_trades.get("success") and dex_trades.get("data", {}).get("items"):
                    for trade in dex_trades["data"]["items"]:
                        if isinstance(trade, dict):
                            source = trade.get("source", "").lower()
                            for platform, identifiers in platform_identifiers.items():
                                for identifier in identifiers:
                                    if identifier in source:
                                        return platform
            except Exception as e:
                logger.debug(f"Could not get DEX trades for platform identification: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Error identifying platform for {contract_address}: {str(e)}")
        
        return "unknown"