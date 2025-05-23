"""
Helius API Module - Phoenix Project (FIXED VERSION)

This module handles all interactions with Helius API for enhanced transaction parsing
and pump.fun token analysis.

FIXES:
- Use Enhanced Transactions API instead of token transactions endpoint
- Proper pump.fun token handling with parsed transactions
- Correct API endpoints according to Helius docs
"""

import requests
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("phoenix.helius")

class HeliusAPI:
    """Client for interacting with Helius API for enhanced Solana data."""
    
    BASE_URL = "https://api.helius.xyz"
    
    def __init__(self, api_key: str):
        """
        Initialize the Helius API client.
        
        Args:
            api_key (str): The Helius API key
        """
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        # Rate limiting
        self._last_call_time = 0
        self._min_call_interval = 0.1  # 10 requests per second
        
        logger.info("Helius API initialized for enhanced transaction parsing")
    
    def _rate_limit(self):
        """Simple rate limiting to avoid hitting API limits."""
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        
        if time_since_last_call < self._min_call_interval:
            sleep_time = self._min_call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self._last_call_time = time.time()
    
    def _make_request(self, endpoint: str, method: str = "GET", 
                     params: Optional[Dict[str, Any]] = None,
                     json_data: Optional[Dict[str, Any]] = None,
                     retry_count: int = 3) -> Dict[str, Any]:
        """
        Make a request to the Helius API.
        
        Args:
            endpoint (str): API endpoint
            method (str): HTTP method
            params (Dict[str, Any], optional): Query parameters
            json_data (Dict[str, Any], optional): JSON body data
            retry_count (int): Number of retries on failure
            
        Returns:
            Dict[str, Any]: Response data
        """
        # Add API key to params
        if params is None:
            params = {}
        params['api-key'] = self.api_key
        
        url = f"{self.BASE_URL}{endpoint}"
        
        # Apply rate limiting
        self._rate_limit()
        
        for attempt in range(retry_count):
            try:
                logger.debug(f"Making {method} request to {endpoint}")
                
                if method == "GET":
                    response = requests.get(url, params=params, headers=self.headers, timeout=30)
                elif method == "POST":
                    response = requests.post(url, params=params, json=json_data, headers=self.headers, timeout=30)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                # For 404s, return empty result instead of retrying
                if response.status_code == 404:
                    logger.debug(f"404 response for {endpoint}")
                    return {
                        "success": False,
                        "error": "Not found",
                        "data": None
                    }
                
                response.raise_for_status()
                
                # Try to parse JSON response
                try:
                    result = response.json()
                    return {
                        "success": True,
                        "data": result
                    }
                except ValueError:
                    return {
                        "success": False,
                        "error": "Invalid JSON response",
                        "raw_response": response.text
                    }
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt+1}/{retry_count})")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "success": False,
                        "error": "Request timeout"
                    }
                    
            except requests.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "success": False,
                        "error": str(e)
                    }
    
    def get_enhanced_transactions(self, wallet_address: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get enhanced parsed transactions for a wallet.
        
        This is the primary method for getting transaction data from Helius.
        It returns fully parsed and human-readable transaction data.
        
        Args:
            wallet_address (str): The wallet address
            limit (int): Maximum number of transactions to return
            
        Returns:
            Dict[str, Any]: Enhanced transaction data
        """
        endpoint = f"/v0/addresses/{wallet_address}/transactions"
        params = {
            "limit": min(limit, 100),  # Helius max is 100
            "type": "SWAP"  # Focus on swap transactions
        }
        
        logger.info(f"Fetching enhanced transactions for wallet {wallet_address}")
        return self._make_request(endpoint, params=params)
    
    def get_parsed_transaction(self, signature: str) -> Dict[str, Any]:
        """
        Get a single parsed transaction by signature.
        
        Args:
            signature (str): Transaction signature
            
        Returns:
            Dict[str, Any]: Parsed transaction data
        """
        endpoint = f"/v0/transactions/{signature}"
        params = {
            "transactions": [signature]
        }
        
        logger.debug(f"Fetching parsed transaction {signature}")
        return self._make_request(endpoint, method="POST", json_data=params)
    
    def get_token_metadata(self, mint_addresses: List[str]) -> Dict[str, Any]:
        """
        Get metadata for multiple tokens using DAS API.
        
        Args:
            mint_addresses (List[str]): List of token mint addresses
            
        Returns:
            Dict[str, Any]: Token metadata
        """
        endpoint = "/v0/token-metadata"
        json_data = {
            "mintAccounts": mint_addresses,
            "includeOffChain": True,
            "disableCache": False
        }
        
        logger.debug(f"Fetching metadata for {len(mint_addresses)} tokens")
        return self._make_request(endpoint, method="POST", json_data=json_data)
    
    def analyze_token_swaps(self, wallet_address: str, token_mint: str, 
                          limit: int = 50) -> Dict[str, Any]:
        """
        Analyze swaps for a specific token by a wallet using enhanced transactions.
        
        This replaces the token transactions endpoint that doesn't work for pump.fun.
        
        Args:
            wallet_address (str): The wallet address
            token_mint (str): The token mint address
            limit (int): Maximum number of transactions to analyze
            
        Returns:
            Dict[str, Any]: Token swap analysis
        """
        try:
            # Get enhanced transactions for the wallet
            tx_response = self.get_enhanced_transactions(wallet_address, limit)
            
            if not tx_response.get("success") or not tx_response.get("data"):
                logger.warning(f"No transaction data found for wallet {wallet_address}")
                return {
                    "success": False,
                    "error": "No transaction data found",
                    "data": {"swaps": []}
                }
            
            # Filter for swaps involving the specific token
            all_transactions = tx_response["data"]
            token_swaps = []
            
            for tx in all_transactions:
                # Check if this transaction involves our token
                if self._transaction_involves_token(tx, token_mint):
                    swap_info = self._extract_swap_info(tx, token_mint)
                    if swap_info:
                        token_swaps.append(swap_info)
            
            # Calculate price from swaps if available
            current_price = None
            if token_swaps:
                # Get the most recent swap price
                latest_swap = token_swaps[0]
                if latest_swap.get("price_per_token"):
                    current_price = latest_swap["price_per_token"]
            
            return {
                "success": True,
                "data": {
                    "swaps": token_swaps,
                    "current_price": current_price,
                    "swap_count": len(token_swaps)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing token swaps: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": {"swaps": []}
            }
    
    def _transaction_involves_token(self, tx: Dict[str, Any], token_mint: str) -> bool:
        """Check if a transaction involves a specific token."""
        try:
            # Check token transfers
            if "tokenTransfers" in tx:
                for transfer in tx["tokenTransfers"]:
                    if transfer.get("mint") == token_mint:
                        return True
            
            # Check account data for the token
            if "accountData" in tx:
                for account in tx["accountData"]:
                    if account.get("account") == token_mint:
                        return True
                    # Check token accounts
                    if account.get("tokenBalanceChanges"):
                        for change in account["tokenBalanceChanges"]:
                            if change.get("mint") == token_mint:
                                return True
            
            # Check instructions for swap programs
            if "instructions" in tx:
                for instruction in tx["instructions"]:
                    # Check if it's a swap instruction
                    program_id = instruction.get("programId", "")
                    if any(swap_program in program_id.lower() for swap_program in ["raydium", "orca", "jupiter", "pump"]):
                        # Check instruction accounts for our token
                        for account in instruction.get("accounts", []):
                            if account == token_mint:
                                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking transaction for token: {str(e)}")
            return False
    
    def _extract_swap_info(self, tx: Dict[str, Any], token_mint: str) -> Optional[Dict[str, Any]]:
        """Extract swap information from a transaction."""
        try:
            swap_info = {
                "signature": tx.get("signature", ""),
                "timestamp": tx.get("timestamp", 0),
                "slot": tx.get("slot", 0),
                "success": tx.get("meta", {}).get("err") is None,
                "type": "UNKNOWN",
                "token_amount": 0,
                "sol_amount": 0,
                "price_per_token": 0,
                "fee": tx.get("fee", 0) / 1e9  # Convert lamports to SOL
            }
            
            # Analyze token transfers to determine swap type and amounts
            token_in = 0
            token_out = 0
            sol_in = 0
            sol_out = 0
            
            if "tokenTransfers" in tx:
                for transfer in tx["tokenTransfers"]:
                    if transfer.get("mint") == token_mint:
                        amount = transfer.get("tokenAmount", 0)
                        if transfer.get("fromUserAccount") and not transfer.get("toUserAccount"):
                            # User sending token (selling)
                            token_out += amount
                        elif transfer.get("toUserAccount") and not transfer.get("fromUserAccount"):
                            # User receiving token (buying)
                            token_in += amount
            
            # Check native SOL transfers
            if "nativeTransfers" in tx:
                for transfer in tx["nativeTransfers"]:
                    amount = transfer.get("amount", 0) / 1e9  # Convert to SOL
                    if transfer.get("fromUserAccount"):
                        sol_out += amount
                    elif transfer.get("toUserAccount"):
                        sol_in += amount
            
            # Determine swap type and calculate price
            if token_in > 0 and sol_out > 0:
                # Buy transaction
                swap_info["type"] = "BUY"
                swap_info["token_amount"] = token_in
                swap_info["sol_amount"] = sol_out
                swap_info["price_per_token"] = sol_out / token_in if token_in > 0 else 0
            elif token_out > 0 and sol_in > 0:
                # Sell transaction
                swap_info["type"] = "SELL"
                swap_info["token_amount"] = token_out
                swap_info["sol_amount"] = sol_in
                swap_info["price_per_token"] = sol_in / token_out if token_out > 0 else 0
            else:
                # Could not determine swap type
                return None
            
            return swap_info
            
        except Exception as e:
            logger.debug(f"Error extracting swap info: {str(e)}")
            return None
    
    def get_pump_fun_token_price(self, token_mint: str, timestamp: Optional[int] = None) -> Dict[str, Any]:
        """
        Get pump.fun token price by analyzing recent swaps.
        
        Since pump.fun tokens don't have traditional price feeds, we calculate
        price from recent swap activity.
        
        Args:
            token_mint (str): The pump.fun token mint address
            timestamp (int, optional): Timestamp to get historical price
            
        Returns:
            Dict[str, Any]: Token price information
        """
        try:
            logger.info(f"Fetching pump.fun token price for {token_mint}")
            
            # For pump.fun tokens, we need to analyze recent swaps
            # First, try to get token metadata
            metadata_response = self.get_token_metadata([token_mint])
            
            token_info = {}
            if metadata_response.get("success") and metadata_response.get("data"):
                token_data = metadata_response["data"]
                if token_data and len(token_data) > 0:
                    token_info = token_data[0]
            
            # Since we can't get swaps directly for the token, we'll return
            # a response indicating we need wallet-specific analysis
            return {
                "success": True,
                "data": {
                    "price": 0,  # Will be calculated from wallet swaps
                    "token_info": token_info,
                    "note": "Price calculation requires wallet transaction analysis"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting pump.fun token price: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": {"price": 0}
            }
    
    def health_check(self) -> bool:
        """
        Check if the Helius API is accessible.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            # Try to get account info for a known address
            endpoint = "/v0/addresses/11111111111111111111111111111111/balances"
            response = self._make_request(endpoint)
            
            if response.get("success") or response.get("error") == "Not found":
                logger.info("✅ Helius API is accessible")
                return True
            else:
                logger.error("❌ Helius API health check failed")
                return False
                
        except Exception as e:
            logger.error(f"Helius API health check failed: {str(e)}")
            return False
    
    def get_supported_features(self) -> List[str]:
        """Get list of supported Helius features."""
        return [
            "Enhanced Transactions API",
            "DAS API (Token Metadata)",
            "Parsed Transaction Data",
            "Webhook Support",
            "Priority Fee API",
            "Compressed NFT Support"
        ]