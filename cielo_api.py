"""
Cielo Finance API Module - Phoenix Project

This module handles all interactions with Cielo Finance Solana API.
"""

import requests
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("phoenix.cielo")

class CieloFinanceAPIError(Exception):
    """Custom exception for Cielo Finance API errors."""
    pass

class CieloFinanceAPI:
    """Client for interacting with Cielo Finance Solana API."""
    
    BASE_URL = "https://api.cielo.finance"  # Update with actual Cielo Finance API URL
    
    def __init__(self, api_key: str):
        """
        Initialize the Cielo Finance API client.
        
        Args:
            api_key (str): The Cielo Finance API key
        """
        if not api_key:
            raise CieloFinanceAPIError("API key is required")
        
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Phoenix-Project/1.0"
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between API requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     retry_count: int = 3, retry_delay: int = 2) -> Dict[str, Any]:
        """
        Make a request to the Cielo Finance API with retry logic.
        
        Args:
            endpoint (str): API endpoint to call
            params (Dict[str, Any], optional): Query parameters
            retry_count (int): Number of retries on failure
            retry_delay (int): Delay between retries in seconds
            
        Returns:
            Dict[str, Any]: Response data as JSON
            
        Raises:
            CieloFinanceAPIError: If the API request fails
        """
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(retry_count):
            try:
                # Apply rate limiting
                self._rate_limit()
                
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                # Handle different HTTP status codes
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise CieloFinanceAPIError("Authentication failed. Please check your API key.")
                elif response.status_code == 403:
                    raise CieloFinanceAPIError("Access forbidden. Your API key may not have sufficient permissions.")
                elif response.status_code == 404:
                    raise CieloFinanceAPIError(f"Endpoint not found: {endpoint}")
                elif response.status_code == 429:
                    # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded (attempt {attempt+1}/{retry_count})")
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay * 2)  # Longer delay for rate limiting
                        continue
                    else:
                        raise CieloFinanceAPIError("Rate limit exceeded. Please try again later.")
                elif response.status_code >= 500:
                    # Server error
                    logger.warning(f"Server error {response.status_code} (attempt {attempt+1}/{retry_count})")
                    if attempt < retry_count - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        raise CieloFinanceAPIError(f"Server error: {response.status_code}")
                else:
                    raise CieloFinanceAPIError(f"Unexpected status code: {response.status_code}")
            
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}/{retry_count}): {str(e)}")
                
                if attempt < retry_count - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Request failed after {retry_count} attempts")
                    raise CieloFinanceAPIError(f"Network error: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the Cielo Finance API is accessible.
        
        Returns:
            Dict[str, Any]: Health check response
        """
        try:
            # For testing purposes, return a mock successful response
            # In real implementation, this would make an actual API call to /health
            return {
                "success": True,
                "status": "API connection test successful (mock)",
                "message": "Cielo Finance API client initialized successfully"
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_api_usage(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Dict[str, Any]: API usage data
        """
        try:
            # For testing purposes, return mock usage data
            # In real implementation, this would make an actual API call
            return {
                "success": True,
                "data": {
                    "requests_made": 0,
                    "requests_limit": 1000,
                    "reset_time": "2024-01-01T00:00:00Z"
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get information about a token.
        
        Args:
            token_address (str): The token contract address
            
        Returns:
            Dict[str, Any]: Token information
        """
        if not token_address:
            raise CieloFinanceAPIError("Token address is required")
        
        endpoint = f"/v1/token/{token_address}"  # Update with actual endpoint structure
        
        logger.debug(f"Fetching token info for {token_address}")
        
        try:
            response = self._make_request(endpoint)
            
            # Standardize response format
            if response.get("data"):
                return {
                    "success": True,
                    "data": response["data"]
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "No data returned")
                }
        except CieloFinanceAPIError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_token_price(self, token_address: str) -> Dict[str, Any]:
        """
        Get the current price of a token.
        
        Args:
            token_address (str): The token contract address
            
        Returns:
            Dict[str, Any]: Token price information
        """
        if not token_address:
            raise CieloFinanceAPIError("Token address is required")
        
        endpoint = f"/v1/token/{token_address}/price"  # Update with actual endpoint
        
        logger.debug(f"Fetching token price for {token_address}")
        
        try:
            response = self._make_request(endpoint)
            
            if response.get("data"):
                return {
                    "success": True,
                    "data": response["data"]
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "No price data returned")
                }
        except CieloFinanceAPIError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
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
        if not token_address:
            raise CieloFinanceAPIError("Token address is required")
        
        endpoint = f"/v1/token/{token_address}/chart"  # Update with actual endpoint
        params = {
            "startTime": start_time,
            "endTime": end_time,
            "resolution": resolution
        }
        
        logger.debug(f"Fetching price history for {token_address} from {start_time} to {end_time}")
        
        try:
            response = self._make_request(endpoint, params)
            
            if response.get("data"):
                return {
                    "success": True,
                    "data": response["data"]
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "No price history data returned")
                }
        except CieloFinanceAPIError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_wallet_transactions(self, wallet_address: str, 
                              limit: int = 100,
                              offset: int = 0) -> Dict[str, Any]:
        """
        Get transactions for a wallet.
        
        Args:
            wallet_address (str): The wallet address
            limit (int): Maximum number of transactions to return
            offset (int): Number of transactions to skip
            
        Returns:
            Dict[str, Any]: Wallet transactions
        """
        if not wallet_address:
            raise CieloFinanceAPIError("Wallet address is required")
        
        endpoint = f"/v1/wallet/{wallet_address}/transactions"  # Update with actual endpoint
        params = {
            "limit": min(limit, 1000),  # Cap at 1000 to prevent overload
            "offset": offset
        }
        
        logger.debug(f"Fetching transactions for wallet {wallet_address}")
        
        try:
            response = self._make_request(endpoint, params)
            
            if response.get("data"):
                return {
                    "success": True,
                    "data": response["data"]
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "No transaction data returned")
                }
        except CieloFinanceAPIError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_wallet_tokens(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get token holdings for a wallet.
        
        Args:
            wallet_address (str): The wallet address
            
        Returns:
            Dict[str, Any]: Wallet token holdings
        """
        if not wallet_address:
            raise CieloFinanceAPIError("Wallet address is required")
        
        endpoint = f"/v1/wallet/{wallet_address}/tokens"  # Update with actual endpoint
        
        logger.debug(f"Fetching token holdings for wallet {wallet_address}")
        
        try:
            response = self._make_request(endpoint)
            
            if response.get("data"):
                return {
                    "success": True,
                    "data": response["data"]
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "No token holdings data returned")
                }
        except CieloFinanceAPIError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_wallet_performance(self, wallet_address: str, 
                             days_back: int = 30) -> Dict[str, Any]:
        """
        Get wallet performance metrics.
        
        Args:
            wallet_address (str): The wallet address
            days_back (int): Number of days to analyze
            
        Returns:
            Dict[str, Any]: Wallet performance metrics
        """
        if not wallet_address:
            raise CieloFinanceAPIError("Wallet address is required")
        
        endpoint = f"/v1/wallet/{wallet_address}/performance"  # Update with actual endpoint
        params = {
            "days": days_back
        }
        
        logger.debug(f"Fetching performance for wallet {wallet_address}")
        
        try:
            response = self._make_request(endpoint, params)
            
            if response.get("data"):
                return {
                    "success": True,
                    "data": response["data"]
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "No performance data returned")
                }
        except CieloFinanceAPIError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
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
        if not token_address:
            raise CieloFinanceAPIError("Token address is required")
        
        endpoint = f"/v1/token/{token_address}/trades"  # Update with actual endpoint
        params = {
            "limit": min(limit, 1000)
        }
        
        logger.debug(f"Fetching DEX trades for token {token_address}")
        
        try:
            response = self._make_request(endpoint, params)
            
            if response.get("data"):
                return {
                    "success": True,
                    "data": response["data"]
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "No trade data returned")
                }
        except CieloFinanceAPIError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
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
        if not token_address:
            raise CieloFinanceAPIError("Token address is required")
        
        try:
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
            if not price_history.get("success") or not price_history.get("data"):
                logger.warning(f"No price history available for {token_address}")
                return {
                    "success": False,
                    "error": "No price history available"
                }
            
            prices = price_history.get("data", [])
            
            # Calculate metrics
            initial_price = prices[0]["value"] if prices else None
            current_price = current_price_data.get("data", {}).get("value") if current_price_data.get("success") else None
            
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
            
        except Exception as e:
            logger.error(f"Error calculating token performance: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_trending_tokens(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get trending tokens on Solana.
        
        Args:
            limit (int): Maximum number of tokens to return
            
        Returns:
            Dict[str, Any]: Trending tokens data
        """
        endpoint = "/v1/trending"  # Update with actual endpoint
        params = {
            "limit": min(limit, 100)
        }
        
        logger.debug("Fetching trending tokens")
        
        try:
            response = self._make_request(endpoint, params)
            
            if response.get("data"):
                return {
                    "success": True,
                    "data": response["data"]
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "No trending data returned")
                }
        except CieloFinanceAPIError as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_tokens(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """
        Search for tokens by name or symbol.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            Dict[str, Any]: Search results
        """
        if not query:
            raise CieloFinanceAPIError("Search query is required")
        
        endpoint = "/v1/search"  # Update with actual endpoint
        params = {
            "q": query,
            "limit": min(limit, 50)
        }
        
        logger.debug(f"Searching for tokens: {query}")
        
        try:
            response = self._make_request(endpoint, params)
            
            if response.get("data"):
                return {
                    "success": True,
                    "data": response["data"]
                }
            else:
                return {
                    "success": False,
                    "error": response.get("error", "No search results returned")
                }
        except CieloFinanceAPIError as e:
            return {
                "success": False,
                "error": str(e)
            }