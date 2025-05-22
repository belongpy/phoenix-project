"""
Birdeye API Module - Phoenix Project

This module handles all interactions with Birdeye Solana API and Helius API.
"""

import requests
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("phoenix.birdeye")

class BirdeyeAPI:
    """Client for interacting with Birdeye Solana API and Helius API."""
    
    BIRDEYE_BASE_URL = "https://public-api.birdeye.so"
    HELIUS_BASE_URL = "https://api.helius.xyz"
    
    def __init__(self, api_key: str, helius_api_key: str = None, 
                 max_retries: int = 5, retry_delay: int = 2, 
                 rate_limit_pause: int = 10):
        """
        Initialize the API client.
        
        Args:
            api_key (str): The Birdeye API key
            helius_api_key (str): The Helius API key (optional, for wallet data)
            max_retries (int): Maximum number of retry attempts for failed requests
            retry_delay (int): Initial delay between retries in seconds
            rate_limit_pause (int): Pause time when rate limit is hit
        """
        self.api_key = api_key
        self.helius_api_key = helius_api_key
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_pause = rate_limit_pause
        self.request_count = 0
        self.last_request_time = 0
        
        if helius_api_key:
            logger.info("Helius API key configured - will use Helius for wallet transaction data")
        else:
            logger.warning("No Helius API key provided - will use Birdeye for wallet data (may have limitations)")
    
    def _make_request(self, url: str, headers: Dict[str, str], 
                     params: Optional[Dict[str, Any]] = None, 
                     retry_count: int = 3, retry_delay: int = 2) -> Dict[str, Any]:
        """
        Make a request to the API with retry logic.
        
        Args:
            url (str): Full URL to call
            headers (Dict[str, str]): Request headers
            params (Dict[str, Any], optional): Query parameters
            retry_count (int): Number of retries on failure
            retry_delay (int): Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: Response data as JSON
        """
        # Rate limiting - ensure we don't exceed 10 requests per second
        current_time = time.time()
        if current_time - self.last_request_time < 0.1:  # Less than 100ms since last request
            time.sleep(0.1)  # Wait at least 100ms between requests
        
        # Every 20 requests, pause briefly to avoid hitting rate limits
        self.request_count += 1
        if self.request_count >= 20:
            logger.debug(f"Pausing briefly after {self.request_count} requests to avoid rate limits")
            time.sleep(1)  # 1 second pause every 20 requests
            self.request_count = 0
        
        for attempt in range(retry_count):
            try:
                self.last_request_time = time.time()
                response = requests.get(url, headers=headers, params=params, timeout=30)
                
                # Handle rate limiting (429 status code)
                if response.status_code == 429:
                    logger.warning(f"Rate limit exceeded. Pausing for {self.rate_limit_pause} seconds...")
                    time.sleep(self.rate_limit_pause)
                    continue
                
                # Handle authentication errors
                if response.status_code in (401, 403):
                    logger.error(f"Authentication error: {response.status_code}. Check your API key.")
                    return {"success": False, "error": "Authentication failed. Check your API key."}
                
                # Handle not found errors
                if response.status_code == 404:
                    # For wallet transactions, a 404 might just mean no transactions
                    if "wallet" in url.lower() and ("tx_list" in url or "transaction" in url):
                        logger.warning(f"No transactions found for this wallet")
                        return {"success": True, "data": []}
                    else:
                        logger.warning(f"Resource not found: {url}")
                        return {"success": False, "error": f"Resource not found: {response.status_code}", "data": []}
                
                response.raise_for_status()
                result = response.json()
                
                # Handle Helius response format
                if isinstance(result, list):
                    return {"success": True, "data": result}
                
                return result
            
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{retry_count}): {str(e)}")
                
                if attempt < retry_count - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Request failed after {retry_count} attempts")
                    if "wallet" in url.lower():
                        return {"success": True, "data": []}  # Return empty for wallet queries
                    return {"success": False, "error": str(e), "data": []}
    
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get information about a token using Birdeye API.
        
        Args:
            token_address (str): The token contract address
            
        Returns:
            Dict[str, Any]: Token information
        """
        url = f"{self.BIRDEYE_BASE_URL}/defi/token_overview"
        params = {"address": token_address}
        
        logger.debug(f"Fetching token info for {token_address}")
        result = self._make_request(url, self.headers, params)
        
        return result
    
    def get_token_price(self, token_address: str) -> Dict[str, Any]:
        """
        Get the current price of a token using Birdeye API.
        
        Args:
            token_address (str): The token contract address
            
        Returns:
            Dict[str, Any]: Token price information
        """
        url = f"{self.BIRDEYE_BASE_URL}/defi/price"
        params = {"address": token_address}
        
        logger.debug(f"Fetching token price for {token_address}")
        return self._make_request(url, self.headers, params)
    
    def get_token_price_history(self, token_address: str, 
                              start_time: int, end_time: int, 
                              resolution: str = "5m") -> Dict[str, Any]:
        """
        Get historical price data for a token using Birdeye API.
        
        Args:
            token_address (str): The token contract address
            start_time (int): Start timestamp in milliseconds
            end_time (int): End timestamp in milliseconds
            resolution (str): Time resolution (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            Dict[str, Any]: Historical price data
        """
        url = f"{self.BIRDEYE_BASE_URL}/defi/history_price"
        params = {
            "address": token_address,
            "address_type": "token",
            "type": resolution,
            "time_from": start_time // 1000,  # Convert to seconds
            "time_to": end_time // 1000
        }
        
        logger.debug(f"Fetching price history for {token_address} from {start_time} to {end_time}")
        return self._make_request(url, self.headers, params)
    
    def get_wallet_transactions(self, wallet_address: str, 
                              limit: int = 100) -> Dict[str, Any]:
        """
        Get transactions for a wallet using Helius API (recommended) or Birdeye fallback.
        
        Args:
            wallet_address (str): The wallet address
            limit (int): Maximum number of transactions to return
            
        Returns:
            Dict[str, Any]: Wallet transactions
        """
        # Try Helius API first if available
        if self.helius_api_key:
            return self._get_wallet_transactions_helius(wallet_address, limit)
        else:
            return self._get_wallet_transactions_birdeye(wallet_address, limit)
    
    def _get_wallet_transactions_helius(self, wallet_address: str, 
                                      limit: int = 100) -> Dict[str, Any]:
        """
        Get transactions for a wallet using Helius API.
        
        Args:
            wallet_address (str): The wallet address
            limit (int): Maximum number of transactions to return
            
        Returns:
            Dict[str, Any]: Wallet transactions
        """
        url = f"{self.HELIUS_BASE_URL}/v0/addresses/{wallet_address}/transactions"
        params = {
            "api-key": self.helius_api_key,
            "limit": min(limit, 1000)  # Helius has a max limit
        }
        
        logger.debug(f"Fetching transactions for wallet {wallet_address} via Helius API")
        
        # Use empty headers for Helius (API key is in params)
        result = self._make_request(url, {}, params)
        
        # Transform Helius response to match expected format
        if isinstance(result, dict) and result.get("success", True) and "data" in result:
            return result
        elif isinstance(result, dict) and "data" in result:
            return result
        else:
            return {"success": True, "data": result if isinstance(result, list) else []}
    
    def _get_wallet_transactions_birdeye(self, wallet_address: str, 
                                       limit: int = 100) -> Dict[str, Any]:
        """
        Get transactions for a wallet using Birdeye API (fallback).
        
        Args:
            wallet_address (str): The wallet address
            limit (int): Maximum number of transactions to return
            
        Returns:
            Dict[str, Any]: Wallet transactions
        """
        # Try the updated Birdeye endpoint
        url = f"{self.BIRDEYE_BASE_URL}/v1/wallet/tx_list"
        params = {
            "wallet": wallet_address,
            "limit": limit
        }
        
        logger.debug(f"Fetching transactions for wallet {wallet_address} via Birdeye API")
        result = self._make_request(url, self.headers, params)
        
        # For wallet transactions, if we get an error, format it properly as an empty result
        if isinstance(result, dict) and result.get('success') is False:
            logger.warning(f"Birdeye wallet transaction error: {result.get('error', 'Unknown error')}")
            return {"success": True, "data": []}
            
        return result
    
    def get_wallet_tokens(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get token holdings for a wallet using Birdeye API.
        
        Args:
            wallet_address (str): The wallet address
            
        Returns:
            Dict[str, Any]: Wallet token holdings
        """
        url = f"{self.BIRDEYE_BASE_URL}/v1/wallet/token_list"
        params = {"wallet": wallet_address}
        
        logger.debug(f"Fetching token holdings for wallet {wallet_address}")
        return self._make_request(url, self.headers, params)
    
    def get_dex_trades(self, token_address: str, 
                     limit: int = 100) -> Dict[str, Any]:
        """
        Get recent DEX trades for a token using Birdeye API.
        
        Args:
            token_address (str): The token contract address
            limit (int): Maximum number of trades to return
            
        Returns:
            Dict[str, Any]: DEX trades for the token
        """
        url = f"{self.BIRDEYE_BASE_URL}/defi/txs/token"
        params = {
            "address": token_address,
            "limit": limit,
            "tx_type": "swap"
        }
        
        logger.debug(f"Fetching DEX trades for token {token_address}")
        return self._make_request(url, self.headers, params)
    
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
            market_cap_usd = token_info["data"].get("mc", 0)  # Birdeye uses "mc" for market cap
        
        # Process and calculate metrics
        if not price_history.get("data", {}).get("items", []):
            logger.warning(f"No price history available for {token_address}")
            return {
                "success": False,
                "error": "No price history available"
            }
        
        prices = price_history.get("data", {}).get("items", [])
        
        # Calculate metrics
        if not prices:
            return {
                "success": False,
                "error": "No price data available"
            }
        
        initial_price = prices[0].get("value", 0) if prices else 0
        current_price = current_price_data.get("data", {}).get("value", 0)
        
        if not initial_price or not current_price:
            return {
                "success": False,
                "error": "Missing price data"
            }
        
        price_values = [p.get("value", 0) for p in prices if p.get("value")]
        max_price = max(price_values) if price_values else initial_price
        min_price = min(price_values) if price_values else initial_price
        
        roi = ((current_price / initial_price) - 1) * 100
        max_roi = ((max_price / initial_price) - 1) * 100
        max_drawdown = ((min_price / initial_price) - 1) * 100
        
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
            "market_cap_usd": market_cap_usd
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
        
        if not transactions.get("success", True):
            logger.warning(f"Error getting transactions for wallet {wallet_address}: {transactions.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": transactions.get('error', 'No transactions found')
            }
            
        if not transactions.get("data", []):
            logger.warning(f"No transactions found for wallet {wallet_address}")
            return {
                "success": False,
                "error": "No transactions found"
            }
        
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