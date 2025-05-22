"""
Cielo Finance API Module - Phoenix Project

This module handles all interactions with Cielo Finance API for wallet analysis.
"""

import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("phoenix.cielo")

class CieloFinanceAPI:
    """Client for interacting with Cielo Finance API."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.cielo.finance"):
        """
        Initialize the Cielo Finance API client.
        
        Args:
            api_key (str): The Cielo Finance API key
            base_url (str): Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Phoenix-Project/1.0"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     method: str = "GET", timeout: int = 30) -> Dict[str, Any]:
        """
        Make a request to the Cielo Finance API.
        
        Args:
            endpoint (str): API endpoint
            params (Dict[str, Any], optional): Query parameters
            method (str): HTTP method
            timeout (int): Request timeout in seconds
            
        Returns:
            Dict[str, Any]: Response data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=params, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Try to return JSON, fallback to text if JSON parsing fails
            try:
                return response.json()
            except ValueError:
                return {"success": False, "error": "Invalid JSON response", "raw_response": response.text}
                
        except requests.RequestException as e:
            logger.error(f"Cielo Finance API request failed for {endpoint}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "endpoint": endpoint,
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
        except Exception as e:
            logger.error(f"Unexpected error in Cielo Finance API request: {str(e)}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def health_check(self) -> bool:
        """
        Check if the Cielo Finance API is accessible.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            # Try a simple endpoint or use a dedicated health check if available
            response = self._make_request("/health", timeout=10)
            
            # If there's no dedicated health endpoint, try getting a simple response
            if not response.get("success", True):
                # Try an alternative endpoint that should work
                test_response = self._make_request("/api/v1/status", timeout=10)
                return test_response.get("success", True)
            
            return True
            
        except Exception as e:
            logger.warning(f"Cielo Finance API health check failed: {str(e)}")
            return False
    
    def get_wallet_pnl_by_tokens(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get wallet P&L broken down by individual tokens.
        
        This endpoint retrieves PnL values for a specified wallet by different tokens.
        
        Args:
            wallet_address (str): The wallet address to analyze
            
        Returns:
            Dict[str, Any]: Wallet P&L data by tokens
        """
        logger.info(f"Fetching token P&L data for wallet {wallet_address}")
        
        response = self._make_request(
            "/gettokenspnl",
            params={"wallet": wallet_address}
        )
        
        if response.get("success", True):
            logger.info(f"Successfully retrieved token P&L data for {wallet_address}")
        else:
            logger.warning(f"Failed to retrieve token P&L data: {response.get('error', 'Unknown error')}")
        
        return response
    
    def get_wallet_total_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get aggregated P&L stats for a specified wallet.
        
        This endpoint retrieves aggregated PnL stats for a specified wallet.
        
        Args:
            wallet_address (str): The wallet address to analyze
            
        Returns:
            Dict[str, Any]: Aggregated wallet statistics
        """
        logger.info(f"Fetching total stats for wallet {wallet_address}")
        
        response = self._make_request(
            "/gettotalstats",
            params={"wallet": wallet_address}
        )
        
        if response.get("success", True):
            logger.info(f"Successfully retrieved total stats for {wallet_address}")
        else:
            logger.warning(f"Failed to retrieve total stats: {response.get('error', 'Unknown error')}")
        
        return response
    
    def get_wallet_tags(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get tags associated with a specific wallet address.
        
        Tags provide insights into wallet activity and behavior patterns.
        
        Args:
            wallet_address (str): The wallet address to analyze
            
        Returns:
            Dict[str, Any]: Wallet tags and activity insights
        """
        logger.info(f"Fetching wallet tags for {wallet_address}")
        
        response = self._make_request(
            "/getwalletstags",
            params={"wallet": wallet_address}
        )
        
        if response.get("success", True):
            logger.info(f"Successfully retrieved wallet tags for {wallet_address}")
        else:
            logger.warning(f"Failed to retrieve wallet tags: {response.get('error', 'Unknown error')}")
        
        return response
    
    def get_wallet_transactions(self, wallet_address: str, 
                              limit: int = 100,
                              offset: int = 0) -> Dict[str, Any]:
        """
        Get transaction history for a wallet (if supported by Cielo Finance).
        
        Args:
            wallet_address (str): The wallet address
            limit (int): Maximum number of transactions to return
            offset (int): Number of transactions to skip
            
        Returns:
            Dict[str, Any]: Wallet transaction history
        """
        logger.info(f"Fetching transaction history for wallet {wallet_address}")
        
        response = self._make_request(
            "/gettransactions",  # Adjust endpoint name based on actual Cielo Finance API
            params={
                "wallet": wallet_address,
                "limit": limit,
                "offset": offset
            }
        )
        
        if response.get("success", True):
            logger.info(f"Successfully retrieved {len(response.get('data', []))} transactions for {wallet_address}")
        else:
            logger.warning(f"Failed to retrieve transactions: {response.get('error', 'Unknown error')}")
        
        return response
    
    def get_wallet_performance_summary(self, wallet_address: str, 
                                     time_period: str = "30d") -> Dict[str, Any]:
        """
        Get a performance summary for a wallet over a specific time period.
        
        Args:
            wallet_address (str): The wallet address
            time_period (str): Time period (e.g., "7d", "30d", "90d")
            
        Returns:
            Dict[str, Any]: Wallet performance summary
        """
        logger.info(f"Fetching performance summary for wallet {wallet_address} ({time_period})")
        
        response = self._make_request(
            "/getperformance",  # Adjust endpoint name based on actual Cielo Finance API
            params={
                "wallet": wallet_address,
                "period": time_period
            }
        )
        
        if response.get("success", True):
            logger.info(f"Successfully retrieved performance summary for {wallet_address}")
        else:
            logger.warning(f"Failed to retrieve performance summary: {response.get('error', 'Unknown error')}")
        
        return response
    
    def batch_get_wallet_stats(self, wallet_addresses: List[str]) -> Dict[str, Any]:
        """
        Get stats for multiple wallets in a single request (if supported).
        
        Args:
            wallet_addresses (List[str]): List of wallet addresses
            
        Returns:
            Dict[str, Any]: Batch wallet statistics
        """
        logger.info(f"Fetching batch stats for {len(wallet_addresses)} wallets")
        
        response = self._make_request(
            "/getbatchstats",  # Adjust endpoint name based on actual Cielo Finance API
            params={"wallets": wallet_addresses},
            method="POST"
        )
        
        if response.get("success", True):
            logger.info(f"Successfully retrieved batch stats for {len(wallet_addresses)} wallets")
        else:
            logger.warning(f"Failed to retrieve batch stats: {response.get('error', 'Unknown error')}")
        
        return response
    
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get information about a specific token (if supported).
        
        Args:
            token_address (str): Token contract address
            
        Returns:
            Dict[str, Any]: Token information
        """
        logger.debug(f"Fetching token info for {token_address}")
        
        response = self._make_request(
            "/gettokeninfo",  # Adjust endpoint name based on actual Cielo Finance API
            params={"token": token_address}
        )
        
        return response
    
    def search_wallets_by_criteria(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for wallets based on specific criteria (if supported).
        
        Args:
            criteria (Dict[str, Any]): Search criteria (e.g., min_pnl, min_trades, etc.)
            
        Returns:
            Dict[str, Any]: Matching wallets
        """
        logger.info(f"Searching wallets with criteria: {criteria}")
        
        response = self._make_request(
            "/searchwallets",  # Adjust endpoint name based on actual Cielo Finance API
            params=criteria,
            method="POST"
        )
        
        if response.get("success", True):
            logger.info(f"Found {len(response.get('data', []))} wallets matching criteria")
        else:
            logger.warning(f"Wallet search failed: {response.get('error', 'Unknown error')}")
        
        return response
    
    def get_api_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics for the current API key.
        
        Returns:
            Dict[str, Any]: API usage statistics
        """
        logger.debug("Fetching API usage statistics")
        
        response = self._make_request("/getusage")  # Adjust endpoint name
        
        return response
    
    def validate_wallet_address(self, wallet_address: str) -> bool:
        """
        Validate if a wallet address is properly formatted for Solana.
        
        Args:
            wallet_address (str): Wallet address to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Basic Solana address validation
            if not wallet_address or len(wallet_address) < 32 or len(wallet_address) > 44:
                return False
            
            # Check if it's a valid base58 string
            import base58
            base58.b58decode(wallet_address)
            return True
            
        except Exception:
            return False
    
    def get_supported_endpoints(self) -> List[str]:
        """
        Get list of supported API endpoints.
        
        Returns:
            List[str]: List of available endpoints
        """
        return [
            "/gettokenspnl",      # Get wallet P&L by tokens
            "/gettotalstats",     # Get aggregated wallet stats
            "/getwalletstags",    # Get wallet tags and insights
            "/gettransactions",   # Get wallet transaction history
            "/getperformance",    # Get wallet performance summary
            "/getbatchstats",     # Get batch wallet statistics
            "/gettokeninfo",      # Get token information
            "/searchwallets",     # Search wallets by criteria
            "/getusage",          # Get API usage statistics
            "/health"             # API health check
        ]
    
    def __str__(self) -> str:
        """String representation of the API client."""
        return f"CieloFinanceAPI(base_url='{self.base_url}', authenticated={'✓' if self.api_key else '✗'})"
    
    def __repr__(self) -> str:
        """Detailed representation of the API client."""
        return f"CieloFinanceAPI(api_key='***', base_url='{self.base_url}')"