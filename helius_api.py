"""
Helius API Module - Phoenix Project

This module handles all interactions with Helius API for enhanced transaction parsing
and pump.fun token analysis.
"""

import requests
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("phoenix.helius")

class HeliusAPI:
    """Client for interacting with Helius API for enhanced Solana data."""
    
    def __init__(self, api_key: str, rpc_url: Optional[str] = None):
        """
        Initialize the Helius API client.
        
        Args:
            api_key (str): Helius API key
            rpc_url (str, optional): Custom RPC URL (uses Helius RPC if not provided)
        """
        self.api_key = api_key
        self.base_url = "https://api.helius.xyz/v0"
        self.rpc_url = rpc_url or f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        
        # Rate limiting
        self._last_call_time = 0
        self._min_call_interval = 0.1  # 100ms between calls (10 requests/second)
        
        # Cache for token metadata
        self._token_cache = {}
        self._cache_ttl = 3600  # 1 hour cache
        
        logger.info("Helius API initialized for enhanced transaction parsing")
    
    def _rate_limit(self):
        """Apply rate limiting to avoid overwhelming the API."""
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        
        if time_since_last_call < self._min_call_interval:
            sleep_time = self._min_call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self._last_call_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     method: str = "GET", retry_count: int = 3) -> Dict[str, Any]:
        """
        Make a request to the Helius API with retry logic.
        
        Args:
            endpoint (str): API endpoint
            params (Dict[str, Any], optional): Query parameters
            method (str): HTTP method
            retry_count (int): Number of retries
            
        Returns:
            Dict[str, Any]: Response data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Apply rate limiting
        self._rate_limit()
        
        # Add API key to params
        if params is None:
            params = {}
        params["api-key"] = self.api_key
        
        for attempt in range(retry_count):
            try:
                logger.debug(f"Making request to {url} (attempt {attempt + 1}/{retry_count})")
                
                if method.upper() == "GET":
                    response = requests.get(url, params=params, timeout=30)
                elif method.upper() == "POST":
                    # For POST, API key goes in header
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    response = requests.post(url, json=params, headers=headers, timeout=30)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                
                result = response.json()
                return {
                    "success": True,
                    "data": result
                }
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{retry_count})")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {"success": False, "error": "Request timeout"}
                    
            except requests.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {"success": False, "error": str(e)}
                    
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Max retries exceeded"}
    
    def get_enhanced_transactions(self, wallet_address: str, 
                                limit: int = 100,
                                before_signature: Optional[str] = None) -> Dict[str, Any]:
        """
        Get enhanced parsed transactions for a wallet.
        
        Args:
            wallet_address (str): Wallet address
            limit (int): Maximum number of transactions
            before_signature (str, optional): Pagination cursor
            
        Returns:
            Dict[str, Any]: Enhanced transaction data
        """
        params = {
            "limit": min(limit, 100),
            "commitment": "confirmed"
        }
        
        if before_signature:
            params["before"] = before_signature
        
        logger.info(f"Fetching enhanced transactions for {wallet_address}")
        
        # Use POST endpoint for enhanced transactions
        response = self._make_request(
            f"addresses/{wallet_address}/transactions",
            params=params,
            method="GET"
        )
        
        if response.get("success"):
            logger.info(f"Retrieved {len(response.get('data', []))} enhanced transactions")
        
        return response
    
    def parse_transactions(self, transactions: List[str]) -> Dict[str, Any]:
        """
        Parse a list of transaction signatures to get enhanced data.
        
        Args:
            transactions (List[str]): List of transaction signatures
            
        Returns:
            Dict[str, Any]: Parsed transaction data
        """
        if not transactions:
            return {"success": True, "data": []}
        
        logger.info(f"Parsing {len(transactions)} transactions")
        
        # Helius accepts up to 100 transactions per request
        batch_size = 100
        all_parsed = []
        
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            
            response = self._make_request(
                "transactions",
                params={"transactions": batch},
                method="POST"
            )
            
            if response.get("success"):
                all_parsed.extend(response.get("data", []))
            else:
                logger.error(f"Failed to parse batch {i//batch_size + 1}: {response.get('error')}")
        
        return {
            "success": True,
            "data": all_parsed
        }
    
    def get_token_metadata(self, mint_addresses: List[str]) -> Dict[str, Any]:
        """
        Get metadata for multiple tokens including pump.fun tokens.
        
        Args:
            mint_addresses (List[str]): List of token mint addresses
            
        Returns:
            Dict[str, Any]: Token metadata
        """
        if not mint_addresses:
            return {"success": True, "data": []}
        
        # Check cache first
        uncached_mints = []
        cached_data = []
        
        current_time = time.time()
        for mint in mint_addresses:
            if mint in self._token_cache:
                cached_item, timestamp = self._token_cache[mint]
                if current_time - timestamp < self._cache_ttl:
                    cached_data.append(cached_item)
                else:
                    uncached_mints.append(mint)
            else:
                uncached_mints.append(mint)
        
        if not uncached_mints:
            return {"success": True, "data": cached_data}
        
        logger.info(f"Fetching metadata for {len(uncached_mints)} tokens")
        
        # DAS API endpoint for token metadata
        response = self._make_request(
            "token-metadata",
            params={"mints": uncached_mints},
            method="POST"
        )
        
        if response.get("success"):
            # Cache the results
            for item in response.get("data", []):
                if "mint" in item:
                    self._token_cache[item["mint"]] = (item, current_time)
            
            # Combine cached and new data
            all_data = cached_data + response.get("data", [])
            return {"success": True, "data": all_data}
        
        return response
    
    def get_pump_fun_token_price(self, token_address: str, 
                                timestamp: Optional[int] = None) -> Dict[str, Any]:
        """
        Get price data for pump.fun tokens using Helius enhanced data.
        
        Args:
            token_address (str): Token mint address
            timestamp (int, optional): Unix timestamp for historical price
            
        Returns:
            Dict[str, Any]: Price data
        """
        logger.info(f"Fetching pump.fun token price for {token_address}")
        
        # For pump.fun tokens, we need to analyze swap transactions
        # Get recent swaps for this token
        params = {
            "source": "PUMP_FUN",
            "type": "SWAP",
            "limit": 50
        }
        
        if timestamp:
            # Get swaps around the timestamp
            params["before"] = timestamp + 3600  # 1 hour after
            params["after"] = timestamp - 3600   # 1 hour before
        
        response = self._make_request(
            f"tokens/{token_address}/transactions",
            params=params,
            method="GET"
        )
        
        if not response.get("success") or not response.get("data"):
            logger.warning(f"No swap data found for pump.fun token {token_address}")
            return {
                "success": False,
                "error": "No price data available",
                "is_pump_token": True
            }
        
        # Extract price from swaps
        swaps = response.get("data", [])
        prices = []
        
        for swap in swaps:
            # Parse swap data to extract price
            if "tokenTransfers" in swap:
                # Calculate price from token transfers
                sol_amount = 0
                token_amount = 0
                
                for transfer in swap["tokenTransfers"]:
                    if transfer.get("mint") == "So11111111111111111111111111111111111111112":
                        sol_amount = transfer.get("tokenAmount", 0)
                    elif transfer.get("mint") == token_address:
                        token_amount = transfer.get("tokenAmount", 0)
                
                if sol_amount > 0 and token_amount > 0:
                    price = sol_amount / token_amount
                    prices.append({
                        "price": price,
                        "timestamp": swap.get("timestamp", 0),
                        "signature": swap.get("signature", "")
                    })
        
        if not prices:
            return {
                "success": False,
                "error": "Unable to calculate price from swaps",
                "is_pump_token": True
            }
        
        # Return the most relevant price
        if timestamp:
            # Find closest price to requested timestamp
            closest_price = min(prices, key=lambda p: abs(p["timestamp"] - timestamp))
            return {
                "success": True,
                "data": {
                    "price": closest_price["price"],
                    "timestamp": closest_price["timestamp"],
                    "source": "pump_fun_swaps"
                },
                "is_pump_token": True
            }
        else:
            # Return latest price
            latest_price = max(prices, key=lambda p: p["timestamp"])
            return {
                "success": True,
                "data": {
                    "price": latest_price["price"],
                    "timestamp": latest_price["timestamp"],
                    "source": "pump_fun_swaps"
                },
                "is_pump_token": True
            }
    
    def analyze_token_swaps(self, wallet_address: str, 
                          token_address: str,
                          limit: int = 100) -> Dict[str, Any]:
        """
        Analyze all swaps for a specific token by a wallet.
        
        Args:
            wallet_address (str): Wallet address
            token_address (str): Token mint address
            limit (int): Maximum number of swaps to analyze
            
        Returns:
            Dict[str, Any]: Detailed swap analysis
        """
        logger.info(f"Analyzing swaps for token {token_address} by wallet {wallet_address}")
        
        # Get all transactions for this wallet involving this token
        params = {
            "type": "SWAP",
            "limit": limit
        }
        
        response = self._make_request(
            f"addresses/{wallet_address}/transactions",
            params=params,
            method="GET"
        )
        
        if not response.get("success"):
            return response
        
        # Filter for specific token
        token_swaps = []
        for tx in response.get("data", []):
            if "tokenTransfers" in tx:
                for transfer in tx["tokenTransfers"]:
                    if transfer.get("mint") == token_address:
                        token_swaps.append(tx)
                        break
        
        # Analyze the swaps
        buy_swaps = []
        sell_swaps = []
        
        for swap in token_swaps:
            swap_type = swap.get("type", "")
            if "buy" in swap_type.lower():
                buy_swaps.append(swap)
            elif "sell" in swap_type.lower():
                sell_swaps.append(swap)
        
        return {
            "success": True,
            "data": {
                "token_address": token_address,
                "total_swaps": len(token_swaps),
                "buy_swaps": buy_swaps,
                "sell_swaps": sell_swaps,
                "first_swap": token_swaps[0] if token_swaps else None,
                "last_swap": token_swaps[-1] if token_swaps else None
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if the Helius API is accessible.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            logger.info("Checking Helius API connectivity...")
            
            # Simple test request
            response = self._make_request("health", retry_count=1)
            
            if response.get("success"):
                logger.info("✅ Helius API is accessible")
                return True
            else:
                logger.error("❌ Helius API health check failed")
                return False
                
        except Exception as e:
            logger.error(f"Helius API health check error: {str(e)}")
            return False