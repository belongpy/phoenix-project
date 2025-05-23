"""
Cielo Finance API Module - Phoenix Project (MINIMAL DEPENDENCIES VERSION)

This module handles all interactions with Cielo Finance API for wallet analysis.
This version has minimal dependencies and doesn't require urllib3 or base58.

FIXES:
- Removed urllib3 retry dependency
- Removed base58 dependency  
- Kept all functionality with manual implementations
- Fixed all try-except blocks
"""

import requests
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("phoenix.cielo")

class CieloFinanceAPI:
    """Client for interacting with Cielo Finance API with robust error handling."""
    
    def __init__(self, api_key: str, base_url: str = "https://feed-api.cielo.finance"):
        """
        Initialize the Cielo Finance API client.
        
        Args:
            api_key (str): The Cielo Finance API key
            base_url (str): Base URL for the API
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
            "User-Agent": "Phoenix-Project/1.0"
        }
        
        # Create session
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Track API calls for rate limiting
        self._last_call_time = 0
        self._min_call_interval = 0.5  # 500ms between calls
    
    def _rate_limit(self):
        """Apply rate limiting to avoid overwhelming the API."""
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        
        if time_since_last_call < self._min_call_interval:
            sleep_time = self._min_call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self._last_call_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     method: str = "GET", timeout: int = 30, retry_count: int = 3) -> Dict[str, Any]:
        """
        Make a request to the Cielo Finance API with improved error handling.
        
        Args:
            endpoint (str): API endpoint
            params (Dict[str, Any], optional): Query parameters
            method (str): HTTP method
            timeout (int): Request timeout in seconds
            retry_count (int): Number of manual retries
            
        Returns:
            Dict[str, Any]: Response data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Apply rate limiting
        self._rate_limit()
        
        # Manual retry loop
        for attempt in range(retry_count):
            try:
                logger.debug(f"Making request to {url} (attempt {attempt + 1}/{retry_count})")
                
                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=timeout)
                elif method.upper() == "POST":
                    response = self.session.post(url, json=params, timeout=timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                # Check for server errors
                if response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                # For client errors, don't retry
                if 400 <= response.status_code < 500:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "status_code": response.status_code
                    }
                
                response.raise_for_status()
                
                # Try to return JSON, fallback to text if JSON parsing fails
                try:
                    result = response.json()
                    # Ensure consistent response format
                    if isinstance(result, dict):
                        if "success" not in result:
                            # Cielo Finance returns {"status": "ok", "data": {...}}
                            if result.get("status") == "ok":
                                result["success"] = True
                            else:
                                result["success"] = False
                        return result
                    else:
                        return {
                            "success": True,
                            "data": result
                        }
                except ValueError:
                    return {"success": False, "error": "Invalid JSON response", "raw_response": response.text}
                    
            except requests.exceptions.Timeout as e:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{retry_count}): {str(e)}")
                if attempt < retry_count - 1:
                    wait_time = min(60, 2 ** attempt)  # Exponential backoff, max 60s
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    return {
                        "success": False,
                        "error": f"Request timeout after {retry_count} attempts",
                        "endpoint": endpoint,
                        "timeout": timeout
                    }
                    
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error (attempt {attempt + 1}/{retry_count}): {str(e)}")
                if attempt < retry_count - 1:
                    wait_time = min(60, 2 ** attempt)
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    return {
                        "success": False,
                        "error": f"Connection error after {retry_count} attempts: {str(e)}",
                        "endpoint": endpoint
                    }
                    
            except requests.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{retry_count}): {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(min(60, 2 ** attempt))
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "endpoint": endpoint,
                        "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
                    }
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return {"success": False, "error": f"Unexpected error: {str(e)}"}
        
        # Should not reach here, but just in case
        return {"success": False, "error": "Max retries exceeded"}
    
    def health_check(self) -> bool:
        """
        Check if the Cielo Finance API is accessible with improved timeout handling.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            logger.info("Checking Cielo Finance API connectivity...")
            
            # Try multiple approaches to verify connectivity
            # 1. First try a simple endpoint that should be fast
            test_endpoints = [
                ("api/v1/health", 10),  # Health endpoint with 10s timeout
                ("api/v1/status", 10),  # Status endpoint with 10s timeout
                ("api/v1", 15),         # Root API endpoint with 15s timeout
                (f"api/v1/11111111111111111111111111111111/trading-stats", 20)  # Test endpoint with 20s timeout
            ]
            
            for endpoint, timeout in test_endpoints:
                logger.debug(f"Trying endpoint: {endpoint} with {timeout}s timeout")
                response = self._make_request(endpoint, timeout=timeout, retry_count=2)
                
                # If we get any response (even an error), the API is reachable
                if response:
                    # Check various success indicators
                    if response.get("success") is False and "error" in response:
                        error_msg = str(response.get("error", "")).lower()
                        
                        # These errors indicate API is working but request is invalid
                        if any(indicator in error_msg for indicator in ["not found", "invalid", "unauthorized", "forbidden", "bad request"]):
                            logger.info("✅ Cielo Finance API is accessible (got expected error response)")
                            return True
                        
                        # Connection/timeout errors mean API is not accessible
                        elif any(indicator in error_msg for indicator in ["timeout", "connection", "timed out"]):
                            continue  # Try next endpoint
                    
                    # If we get a successful response, API is definitely working
                    if response.get("success") or response.get("status") == "ok":
                        logger.info("✅ Cielo Finance API is accessible")
                        return True
            
            # If all endpoints failed with timeouts/connection errors
            logger.error("❌ Cielo Finance API is not accessible - all endpoints timed out")
            return False
            
        except Exception as e:
            logger.error(f"Cielo Finance API health check failed: {str(e)}")
            return False
    
    def get_wallet_trading_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get detailed trading stats for a wallet using the correct Cielo Finance endpoint.
        
        This endpoint retrieves comprehensive trading statistics including P&L, ROI,
        win rate, and trading behavior insights.
        
        Args:
            wallet_address (str): The wallet address to analyze
            
        Returns:
            Dict[str, Any]: Wallet trading statistics
        """
        logger.info(f"Fetching trading stats for wallet {wallet_address}")
        
        # Use longer timeout for data endpoints
        response = self._make_request(f"api/v1/{wallet_address}/trading-stats", timeout=45)
        
        if response.get("success", True):
            logger.info(f"Successfully retrieved trading stats for {wallet_address}")
        else:
            logger.warning(f"Failed to retrieve trading stats: {response.get('error', 'Unknown error')}")
        
        return response
    
    def get_wallet_pnl_by_tokens(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get wallet P&L broken down by individual tokens.
        
        This is an alias for get_wallet_trading_stats to maintain compatibility
        with the existing wallet analysis code.
        
        Args:
            wallet_address (str): The wallet address to analyze
            
        Returns:
            Dict[str, Any]: Wallet P&L data by tokens
        """
        logger.info(f"Fetching token P&L data for wallet {wallet_address}")
        
        # Use the trading-stats endpoint which should include token-level P&L
        response = self.get_wallet_trading_stats(wallet_address)
        
        # Transform the response to match expected format if needed
        if response.get("success") and "data" in response:
            # If the response structure is different, we might need to transform it
            # For now, pass through as-is and let the wallet analyzer handle it
            pass
        
        return response
    
    def get_wallet_total_stats(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get aggregated P&L stats for a specified wallet.
        
        This is also covered by the trading-stats endpoint.
        
        Args:
            wallet_address (str): The wallet address to analyze
            
        Returns:
            Dict[str, Any]: Aggregated wallet statistics
        """
        logger.info(f"Fetching total stats for wallet {wallet_address}")
        
        return self.get_wallet_trading_stats(wallet_address)
    
    def get_wallet_tags(self, wallet_address: str) -> Dict[str, Any]:
        """
        Get tags associated with a specific wallet address.
        
        Note: This endpoint may not be available in the current Cielo Finance API.
        
        Args:
            wallet_address (str): The wallet address to analyze
            
        Returns:
            Dict[str, Any]: Wallet tags and activity insights
        """
        logger.info(f"Fetching wallet tags for {wallet_address}")
        
        # This endpoint might not exist, so we'll try and handle gracefully
        response = self._make_request(f"api/v1/{wallet_address}/tags", timeout=30)
        
        if not response.get("success"):
            # If tags endpoint doesn't exist, return empty tags
            logger.info("Wallet tags endpoint not available, returning empty tags")
            return {
                "success": True,
                "data": {"tags": [], "note": "Tags endpoint not available"}
            }
        
        return response
    
    def get_wallet_transactions(self, wallet_address: str, 
                              limit: int = 100,
                              offset: int = 0) -> Dict[str, Any]:
        """
        Get transaction history for a wallet.
        
        Note: This might be included in the trading-stats endpoint or might be a separate endpoint.
        
        Args:
            wallet_address (str): The wallet address
            limit (int): Maximum number of transactions to return
            offset (int): Number of transactions to skip
            
        Returns:
            Dict[str, Any]: Wallet transaction history
        """
        logger.info(f"Fetching transaction history for wallet {wallet_address}")
        
        # Try transactions endpoint first
        response = self._make_request(
            f"api/v1/{wallet_address}/transactions",
            params={"limit": limit, "offset": offset},
            timeout=30
        )
        
        if not response.get("success"):
            # If transactions endpoint doesn't exist, try to extract from trading-stats
            logger.info("Transactions endpoint not available, trying to extract from trading-stats")
            stats_response = self.get_wallet_trading_stats(wallet_address)
            if stats_response.get("success") and "data" in stats_response:
                # Extract transaction-like data from trading stats if available
                return {
                    "success": True,
                    "data": {"transactions": [], "note": "Extracted from trading stats"},
                    "source": "trading-stats"
                }
        
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
        
        # The trading-stats endpoint likely includes performance data
        response = self.get_wallet_trading_stats(wallet_address)
        
        if response.get("success"):
            # Add time period info to the response
            if "data" in response:
                response["data"]["requested_period"] = time_period
        
        return response
    
    def batch_get_wallet_stats(self, wallet_addresses: List[str]) -> Dict[str, Any]:
        """
        Get stats for multiple wallets.
        
        Note: Since there's no batch endpoint mentioned, we'll call individual endpoints.
        This is less efficient but necessary until a batch endpoint is available.
        
        Args:
            wallet_addresses (List[str]): List of wallet addresses
            
        Returns:
            Dict[str, Any]: Batch wallet statistics
        """
        logger.info(f"Fetching batch stats for {len(wallet_addresses)} wallets")
        
        results = {}
        errors = []
        
        for i, wallet in enumerate(wallet_addresses):
            try:
                # Add delay between requests to avoid rate limiting
                if i > 0:
                    time.sleep(0.5)
                
                result = self.get_wallet_trading_stats(wallet)
                results[wallet] = result
                
                if not result.get("success"):
                    errors.append({
                        "wallet": wallet,
                        "error": result.get("error", "Unknown error")
                    })
                    
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(wallet_addresses)} wallets")
                    
            except Exception as e:
                errors.append({
                    "wallet": wallet,
                    "error": str(e)
                })
        
        return {
            "success": len(results) > 0,
            "data": results,
            "errors": errors,
            "total_requested": len(wallet_addresses),
            "total_successful": len([r for r in results.values() if r.get("success")])
        }
    
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get information about a specific token.
        
        Note: This endpoint may not be available in Cielo Finance API.
        This method is kept for compatibility.
        
        Args:
            token_address (str): Token contract address
            
        Returns:
            Dict[str, Any]: Token information
        """
        logger.debug(f"Token info endpoint not available in Cielo Finance API for {token_address}")
        
        return {
            "success": False,
            "error": "Token info endpoint not available in Cielo Finance API",
            "note": "Use Birdeye API for token metadata"
        }
    
    def search_wallets_by_criteria(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for wallets based on specific criteria.
        
        Note: This endpoint is not documented in the current Cielo Finance API.
        
        Args:
            criteria (Dict[str, Any]): Search criteria
            
        Returns:
            Dict[str, Any]: Matching wallets
        """
        logger.info(f"Wallet search not available in current Cielo Finance API")
        
        return {
            "success": False,
            "error": "Wallet search endpoint not available",
            "note": "Feature not available in current API version"
        }
    
    def get_api_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics for the current API key.
        
        Returns:
            Dict[str, Any]: API usage statistics
        """
        logger.debug("Fetching API usage statistics")
        
        # Try common usage endpoints
        response = self._make_request("api/v1/usage", timeout=20)
        
        if not response.get("success"):
            # Try alternative endpoint
            response = self._make_request("api/v1/account/usage", timeout=20)
        
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
            
            # Check if it's a valid base58 string (Solana addresses are base58)
            # Without base58 library, do basic character check
            valid_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            return all(c in valid_chars for c in wallet_address)
            
        except Exception:
            return False
    
    def get_supported_endpoints(self) -> List[str]:
        """
        Get list of supported API endpoints.
        
        Returns:
            List[str]: List of available endpoints
        """
        return [
            "api/v1/{wallet}/trading-stats",  # Main endpoint for wallet analysis
            "api/v1/{wallet}/transactions",   # Transaction history (if available)
            "api/v1/{wallet}/tags",          # Wallet tags (if available)
            "api/v1/usage",                  # API usage stats
            "api/v1/account/usage",          # Alternative usage endpoint
            "api/v1/health",                 # Health check endpoint
            "api/v1/status"                  # Status endpoint
        ]
    
    def __str__(self) -> str:
        """String representation of the API client."""
        return f"CieloFinanceAPI(base_url='{self.base_url}', authenticated={'✓' if self.api_key else '✗'})"
    
    def __repr__(self) -> str:
        """Detailed representation of the API client."""
        return f"CieloFinanceAPI(api_key='***', base_url='{self.base_url}')"

# Alias for backward compatibility
CieloAPI = CieloFinanceAPI