"""
Helius API Module - Phoenix Project (FIXED VERSION)

This module handles Helius API interactions for pump.fun tokens and enhanced transaction parsing.
FIXES:
- Added enhanced transaction API endpoint
- Direct price calculation from swap amounts
- Fixed NoneType errors
- Proper fallback chain
"""

import requests
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("phoenix.helius")

class HeliusAPI:
    """Client for interacting with Helius API for enhanced Solana data."""
    
    def __init__(self, api_key: str):
        """
        Initialize the Helius API client.
        
        Args:
            api_key (str): Helius API key
        """
        self.api_key = api_key
        self.base_url = "https://api.helius.xyz"
        self.rpc_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Rate limiting
        self._last_call_time = 0
        self._min_call_interval = 0.1  # 100ms between calls
    
    def _rate_limit(self):
        """Apply rate limiting."""
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        
        if time_since_last_call < self._min_call_interval:
            sleep_time = self._min_call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self._last_call_time = time.time()
    
    def _make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """Make RPC call to Helius node."""
        self._rate_limit()
        
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
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    return {"success": False, "error": result["error"]}
                return {"success": True, "result": result.get("result")}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"RPC call failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_enhanced_transaction(self, signature: str) -> Dict[str, Any]:
        """
        Get enhanced transaction data from Helius with pump.fun swap details.
        
        Args:
            signature: Transaction signature
            
        Returns:
            Enhanced transaction data with parsed instructions
        """
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/v0/transactions/{signature}?api-key={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "data": data
                }
            else:
                logger.error(f"Enhanced transaction API error: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API returned {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Error getting enhanced transaction: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def extract_pump_price_from_enhanced_tx(self, tx_data: Dict[str, Any], token_address: str) -> Optional[Dict[str, Any]]:
        """
        Extract pump.fun token price from enhanced transaction data.
        
        Args:
            tx_data: Enhanced transaction data from Helius
            token_address: Token mint address
            
        Returns:
            Price data if found
        """
        try:
            # Look for swap instructions
            instructions = tx_data.get("instructions", [])
            
            for instruction in instructions:
                # Check if this is a pump.fun swap
                if instruction.get("programId") == "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P":
                    # Parse pump.fun swap data
                    inner_instructions = instruction.get("innerInstructions", [])
                    
                    for inner in inner_instructions:
                        # Look for token transfers
                        if inner.get("parsedInfo", {}).get("type") == "transfer":
                            info = inner.get("parsedInfo", {}).get("info", {})
                            
                            # Check if this involves our token
                            if token_address in [info.get("mint"), info.get("source"), info.get("destination")]:
                                amount = float(info.get("amount", 0))
                                lamports = float(info.get("lamports", 0))
                                
                                if amount > 0 and lamports > 0:
                                    # Calculate price in SOL
                                    sol_amount = lamports / 1e9
                                    price = sol_amount / amount
                                    
                                    return {
                                        "price": price,
                                        "sol_amount": sol_amount,
                                        "token_amount": amount,
                                        "timestamp": tx_data.get("timestamp", 0)
                                    }
            
            # Alternative: Check token balance changes
            token_balance_changes = tx_data.get("tokenBalanceChanges", [])
            native_balance_changes = tx_data.get("nativeBalanceChanges", [])
            
            # Find token change
            token_change = None
            for change in token_balance_changes:
                if change.get("mint") == token_address:
                    token_change = change
                    break
            
            # Find SOL change for the same account
            if token_change:
                user_account = token_change.get("userAccount")
                sol_change = 0
                
                for native_change in native_balance_changes:
                    if native_change.get("account") == user_account:
                        sol_change = abs(native_change.get("amount", 0)) / 1e9
                        break
                
                token_amount = abs(float(token_change.get("rawTokenAmount", {}).get("tokenAmount", 0)))
                
                if token_amount > 0 and sol_change > 0:
                    price = sol_change / token_amount
                    return {
                        "price": price,
                        "sol_amount": sol_change,
                        "token_amount": token_amount,
                        "timestamp": tx_data.get("timestamp", 0)
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting pump price: {str(e)}")
            return None
    
    def calculate_price_from_swap_amounts(self, sol_amount: float, token_amount: float) -> float:
        """
        Calculate token price directly from swap amounts.
        
        Args:
            sol_amount: Amount of SOL in the swap
            token_amount: Amount of tokens in the swap
            
        Returns:
            Price per token in SOL
        """
        if token_amount > 0 and sol_amount > 0:
            return sol_amount / token_amount
        return 0
    
    def health_check(self) -> bool:
        """Check if Helius API is accessible."""
        try:
            result = self._make_rpc_call("getHealth", [])
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def get_pump_fun_token_price(self, token_address: str, timestamp: Optional[int] = None) -> Dict[str, Any]:
        """
        Get pump.fun token price using enhanced transaction data.
        
        Args:
            token_address: Token mint address
            timestamp: Optional timestamp to get historical price
            
        Returns:
            Price data or error
        """
        try:
            # First, try to get recent transactions for this token
            signatures_result = self.get_token_transactions(token_address, limit=10)
            
            if not signatures_result.get("success") or not signatures_result.get("data"):
                return {
                    "success": False,
                    "error": "No transactions found for token",
                    "data": {"price": 0}
                }
            
            signatures = signatures_result["data"]
            
            # Try to get price from recent transactions
            for sig_info in signatures:
                if not isinstance(sig_info, dict):
                    continue
                    
                signature = sig_info.get("signature")
                if not signature:
                    continue
                
                # Get enhanced transaction data
                enhanced_tx = self.get_enhanced_transaction(signature)
                
                if enhanced_tx.get("success") and enhanced_tx.get("data"):
                    price_data = self.extract_pump_price_from_enhanced_tx(
                        enhanced_tx["data"], 
                        token_address
                    )
                    
                    if price_data:
                        return {
                            "success": True,
                            "data": {
                                "price": price_data["price"],
                                "sol_amount": price_data["sol_amount"],
                                "token_amount": price_data["token_amount"],
                                "source": "enhanced_transaction"
                            }
                        }
            
            # If no price found from enhanced transactions, return error
            return {
                "success": False,
                "error": "Could not determine price from transactions",
                "data": {"price": 0}
            }
            
        except Exception as e:
            logger.error(f"Error getting pump token price: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": {"price": 0}
            }
    
    def get_token_transactions(self, token_address: str, limit: int = 100) -> Dict[str, Any]:
        """Get recent transactions for a token."""
        try:
            # Get signatures for token
            result = self._make_rpc_call(
                "getSignaturesForAddress",
                [token_address, {"limit": limit}]
            )
            
            if result.get("success") and result.get("result"):
                return {
                    "success": True,
                    "data": result["result"]
                }
            else:
                return {
                    "success": False,
                    "error": "No transactions found"
                }
                
        except Exception as e:
            logger.error(f"Error getting token transactions: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_token_metadata(self, token_addresses: List[str]) -> Dict[str, Any]:
        """Get metadata for pump.fun tokens."""
        try:
            # For pump.fun tokens, we'll use basic metadata
            metadata = []
            
            for address in token_addresses:
                if address.endswith("pump"):
                    metadata.append({
                        "address": address,
                        "symbol": "PUMP",
                        "name": "Pump.fun Token",
                        "decimals": 6,
                        "platform": "pump.fun"
                    })
            
            return {
                "success": True,
                "data": metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting token metadata: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_enhanced_transactions(self, wallet_address: str, limit: int = 100) -> Dict[str, Any]:
        """Get enhanced parsed transactions for a wallet."""
        try:
            # Get signatures
            sig_result = self._make_rpc_call(
                "getSignaturesForAddress",
                [wallet_address, {"limit": limit}]
            )
            
            if not sig_result.get("success"):
                return sig_result
            
            signatures = sig_result.get("result", [])
            enhanced_txs = []
            
            # Get enhanced data for each transaction
            for sig_info in signatures[:20]:  # Limit to prevent rate limiting
                if isinstance(sig_info, dict) and sig_info.get("signature"):
                    enhanced = self.get_enhanced_transaction(sig_info["signature"])
                    if enhanced.get("success") and enhanced.get("data"):
                        enhanced_txs.append(enhanced["data"])
                
                time.sleep(0.1)  # Rate limiting
            
            return {
                "success": True,
                "data": enhanced_txs
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced transactions: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_token_swaps(self, wallet_address: str, token_address: str, limit: int = 50) -> Dict[str, Any]:
        """
        Analyze swaps for a specific token with direct price calculation.
        
        Args:
            wallet_address: Wallet to analyze (empty string for all)
            token_address: Token mint address
            limit: Number of transactions to analyze
            
        Returns:
            Swap analysis with calculated prices
        """
        try:
            logger.info(f"Analyzing pump.fun token swaps for {token_address}")
            
            # Get transactions for the token
            tx_result = self.get_token_transactions(token_address, limit)
            
            if not tx_result.get("success") or not tx_result.get("data"):
                return {
                    "success": False,
                    "error": "No transactions found",
                    "swaps": []
                }
            
            swaps = []
            
            for sig_info in tx_result["data"][:limit]:
                if not isinstance(sig_info, dict):
                    continue
                    
                signature = sig_info.get("signature")
                if not signature:
                    continue
                
                # Get transaction details
                tx_result = self._make_rpc_call(
                    "getTransaction",
                    [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0}]
                )
                
                if not tx_result.get("success") or not tx_result.get("result"):
                    continue
                
                tx = tx_result["result"]
                
                # Extract swap data from transaction
                swap_data = self._extract_swap_from_transaction(tx, token_address, wallet_address)
                
                if swap_data:
                    # Calculate price directly
                    if swap_data.get("sol_amount", 0) > 0 and swap_data.get("token_amount", 0) > 0:
                        swap_data["price"] = self.calculate_price_from_swap_amounts(
                            swap_data["sol_amount"],
                            swap_data["token_amount"]
                        )
                    
                    swaps.append(swap_data)
            
            return {
                "success": True,
                "swaps": swaps,
                "count": len(swaps)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing token swaps: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "swaps": []
            }
    
    def _extract_swap_from_transaction(self, tx: Dict[str, Any], token_address: str, 
                                     wallet_address: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Extract swap data from a transaction with direct calculation."""
        try:
            if not tx or "meta" not in tx:
                return None
            
            meta = tx["meta"]
            
            # Get token balance changes
            pre_token_balances = meta.get("preTokenBalances", [])
            post_token_balances = meta.get("postTokenBalances", [])
            
            # Get SOL balance changes
            pre_balances = meta.get("preBalances", [])
            post_balances = meta.get("postBalances", [])
            
            # Find token change for this mint
            token_change = 0
            token_decimals = 6
            involved_account = None
            
            # Check pre balances
            for balance in pre_token_balances:
                if balance.get("mint") == token_address:
                    if not wallet_address or balance.get("owner") == wallet_address:
                        involved_account = balance.get("accountIndex")
                        token_decimals = balance.get("uiTokenAmount", {}).get("decimals", 6)
                        pre_amount = float(balance.get("uiTokenAmount", {}).get("amount", 0))
                        
                        # Find post balance
                        post_amount = 0
                        for post_balance in post_token_balances:
                            if (post_balance.get("mint") == token_address and 
                                post_balance.get("accountIndex") == involved_account):
                                post_amount = float(post_balance.get("uiTokenAmount", {}).get("amount", 0))
                                break
                        
                        token_change = post_amount - pre_amount
                        break
            
            # Check if we didn't find in pre, check post (new token)
            if involved_account is None:
                for balance in post_token_balances:
                    if balance.get("mint") == token_address:
                        if not wallet_address or balance.get("owner") == wallet_address:
                            involved_account = balance.get("accountIndex")
                            token_decimals = balance.get("uiTokenAmount", {}).get("decimals", 6)
                            token_change = float(balance.get("uiTokenAmount", {}).get("amount", 0))
                            break
            
            if involved_account is None or token_change == 0:
                return None
            
            # Get SOL change for the involved account
            sol_change = 0
            if involved_account < len(pre_balances) and involved_account < len(post_balances):
                sol_change = (post_balances[involved_account] - pre_balances[involved_account]) / 1e9
            
            # Determine swap type
            swap_type = "buy" if token_change > 0 else "sell"
            
            # Convert token amount to UI amount
            token_amount = abs(token_change) / (10 ** token_decimals)
            sol_amount = abs(sol_change)
            
            # Calculate price directly
            price = self.calculate_price_from_swap_amounts(sol_amount, token_amount)
            
            return {
                "signature": tx.get("transaction", {}).get("signatures", [""])[0],
                "type": swap_type,
                "token_amount": token_amount,
                "sol_amount": sol_amount,
                "price": price,
                "timestamp": tx.get("blockTime", 0),
                "token_address": token_address,
                "wallet_address": wallet_address or "unknown",
                "source": "direct_calculation"
            }
            
        except Exception as e:
            logger.error(f"Error extracting swap: {str(e)}")
            return None
    
    def __str__(self) -> str:
        """String representation."""
        return f"HeliusAPI(api_key='***')"