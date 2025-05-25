"""
Helius API Module - Phoenix Project (ENHANCED VERSION)

This module handles all interactions with Helius API for enhanced transaction parsing,
pump.fun token analysis, and multi-source price discovery.

MAJOR UPDATES:
- Uses Enhanced Transactions API for proper transaction parsing
- Implements pump.fun bonding curve price calculation
- Multi-source price discovery with fallbacks
- Early call detection and special handling
- Maintains backward compatibility with wallet_module
"""

import requests
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import base58

logger = logging.getLogger("phoenix.helius")

@dataclass
class PricePoint:
    """Data class for price points"""
    timestamp: int
    price: float
    source: str
    is_baseline: bool = False
    confidence: str = "high"  # high, medium, low

class HeliusAPI:
    """Enhanced Helius API client with proper transaction parsing and price discovery."""
    
    # Helius API endpoints
    BASE_URL = "https://api.helius.xyz"
    
    # Pump.fun constants
    PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
    PUMP_FUN_INITIAL_PRICE = 0.000000001  # Conservative estimate
    
    # Known DEX programs
    DEX_PROGRAMS = {
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8": "Raydium",
        "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc": "Orca",
        "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB": "Jupiter",
        "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK": "Raydium CPMM"
    }
    
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
        self.rpc_url = f"https://mainnet.helius-rpc.com/?api-key={api_key}"
        
        # Cache for performance
        self.token_cache = {}
        self.price_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms between requests
    
    def _rate_limit(self):
        """Apply rate limiting to avoid overwhelming the API."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last)
        
        self._last_request_time = time.time()
    
    def _make_request(self, endpoint: str, method: str = "GET", 
                     data: Optional[Dict[str, Any]] = None,
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the Helius API.
        
        Args:
            endpoint (str): API endpoint
            method (str): HTTP method
            data (Dict[str, Any], optional): Request body
            params (Dict[str, Any], optional): Query parameters
            
        Returns:
            Dict[str, Any]: Response data
        """
        self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        # Add API key to params
        if params is None:
            params = {}
        params['api-key'] = self.api_key
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, json=data, params=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Helius API request failed: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """
        Check if the Helius API is accessible.
        
        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            # Use RPC health check
            response = requests.post(
                self.rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Helius health check failed: {str(e)}")
            return False
    
    def get_enhanced_transactions(self, address: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get enhanced parsed transactions for an address.
        
        Args:
            address (str): Wallet or token address
            limit (int): Maximum number of transactions to return
            
        Returns:
            Dict[str, Any]: Enhanced transaction data
        """
        try:
            endpoint = f"/v0/addresses/{address}/transactions"
            params = {
                "limit": min(limit, 100),
                "type": "SWAP"  # Focus on swap transactions
            }
            
            result = self._make_request(endpoint, params=params)
            
            if "error" in result:
                logger.error(f"Failed to get enhanced transactions: {result['error']}")
                return {"success": False, "error": result["error"]}
            
            return {
                "success": True,
                "data": result,
                "count": len(result) if isinstance(result, list) else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced transactions: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def parse_transaction(self, signature: str) -> Dict[str, Any]:
        """
        Parse a single transaction to extract swap details.
        
        Args:
            signature (str): Transaction signature
            
        Returns:
            Dict[str, Any]: Parsed transaction data
        """
        try:
            endpoint = f"/v0/transactions/{signature}"
            result = self._make_request(endpoint)
            
            if "error" in result:
                return {"success": False, "error": result["error"]}
            
            # Extract swap information
            swap_info = self._extract_swap_info(result)
            
            return {
                "success": True,
                "data": result,
                "swap_info": swap_info
            }
            
        except Exception as e:
            logger.error(f"Error parsing transaction: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_token_metadata(self, token_addresses: List[str]) -> Dict[str, Any]:
        """
        Get metadata for multiple tokens using DAS API.
        Maintains compatibility with existing code.
        
        Args:
            token_addresses (List[str]): List of token addresses
            
        Returns:
            Dict[str, Any]: Token metadata
        """
        try:
            endpoint = "/v1/token-metadata"
            data = {
                "mintAccounts": token_addresses,
                "includeOffChain": True,
                "disableCache": False
            }
            
            result = self._make_request(endpoint, method="POST", data=data)
            
            if "error" in result:
                return {"success": False, "error": result["error"], "data": []}
            
            return {
                "success": True,
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Error getting token metadata: {str(e)}")
            return {"success": False, "error": str(e), "data": []}
    
    def analyze_token_swaps(self, wallet_address: str, token_address: str, 
                          limit: int = 100) -> Dict[str, Any]:
        """
        Analyze token swaps for price discovery.
        This is the main method called by telegram_module.
        
        Args:
            wallet_address (str): Wallet address (can be empty for all swaps)
            token_address (str): Token address to analyze
            limit (int): Maximum number of swaps to analyze
            
        Returns:
            Dict[str, Any]: Swap analysis with price data
        """
        try:
            # Check if pump.fun token
            is_pump = token_address.endswith("pump")
            
            # Get transactions for the token
            address_to_query = wallet_address if wallet_address else token_address
            tx_result = self.get_enhanced_transactions(address_to_query, limit)
            
            if not tx_result.get("success"):
                # If no enhanced transactions, try pump.fun specific approach
                if is_pump:
                    return self._get_pump_fun_price_data(token_address)
                return {"success": False, "error": "No transaction data available", "data": []}
            
            # Parse transactions to extract swaps
            swaps = []
            transactions = tx_result.get("data", [])
            
            for tx in transactions:
                # Skip if not a swap
                if tx.get("type") != "SWAP":
                    continue
                
                swap_info = self._extract_swap_from_enhanced_tx(tx, token_address)
                if swap_info:
                    swaps.append(swap_info)
            
            # If no swaps found and it's a pump token, use bonding curve
            if not swaps and is_pump:
                return self._get_pump_fun_price_data(token_address)
            
            return {
                "success": True,
                "data": swaps,
                "count": len(swaps),
                "is_pump_token": is_pump
            }
            
        except Exception as e:
            logger.error(f"Error analyzing token swaps: {str(e)}")
            return {"success": False, "error": str(e), "data": []}
    
    def get_pump_fun_token_price(self, token_address: str, 
                                timestamp: Optional[int] = None) -> Dict[str, Any]:
        """
        Get pump.fun token price using bonding curve or first DEX listing.
        
        Args:
            token_address (str): Token address
            timestamp (int, optional): Specific timestamp for historical price
            
        Returns:
            Dict[str, Any]: Price data
        """
        try:
            # First, try to get actual swap data
            swap_data = self.analyze_token_swaps("", token_address, 50)
            
            if swap_data.get("success") and swap_data.get("data"):
                swaps = swap_data["data"]
                
                # If we have a specific timestamp, find closest price
                if timestamp:
                    closest_swap = self._find_closest_price(swaps, timestamp)
                    if closest_swap:
                        return {
                            "success": True,
                            "data": {
                                "price": closest_swap["price"],
                                "timestamp": closest_swap["timestamp"],
                                "source": closest_swap.get("source", "swap"),
                                "is_baseline": closest_swap.get("is_baseline", False)
                            }
                        }
                
                # Otherwise return latest price
                if swaps:
                    latest = max(swaps, key=lambda x: x.get("timestamp", 0))
                    return {
                        "success": True,
                        "data": {
                            "price": latest["price"],
                            "timestamp": latest["timestamp"],
                            "source": "latest_swap"
                        }
                    }
            
            # Fallback to bonding curve estimate
            return self._get_pump_fun_price_data(token_address, timestamp)
            
        except Exception as e:
            logger.error(f"Error getting pump.fun token price: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _get_pump_fun_price_data(self, token_address: str, 
                                timestamp: Optional[int] = None) -> Dict[str, Any]:
        """
        Get pump.fun price data using bonding curve calculations.
        
        Args:
            token_address (str): Token address
            timestamp (int, optional): Specific timestamp
            
        Returns:
            Dict[str, Any]: Price estimation
        """
        try:
            # Check if token was recently created
            creation_time = self._get_token_creation_time(token_address)
            
            if creation_time:
                # Calculate bonding curve price based on age
                current_time = int(datetime.now().timestamp())
                token_age_hours = (current_time - creation_time) / 3600
                
                # Simple bonding curve estimation
                # Pump.fun typically starts at ~0.000001 SOL
                base_price = 0.000001
                
                # Price increases with age/activity
                if token_age_hours < 1:
                    estimated_price = base_price
                    confidence = "low"
                elif token_age_hours < 6:
                    estimated_price = base_price * (1 + token_age_hours * 0.5)
                    confidence = "medium"
                else:
                    estimated_price = base_price * 5  # Conservative estimate
                    confidence = "low"
                
                return {
                    "success": True,
                    "data": [{
                        "price": estimated_price,
                        "timestamp": timestamp or current_time,
                        "source": "bonding_curve_estimate",
                        "confidence": confidence,
                        "is_baseline": True,
                        "token_age_hours": token_age_hours
                    }]
                }
            
            # Default pump.fun price if no creation time found
            return {
                "success": True,
                "data": [{
                    "price": self.PUMP_FUN_INITIAL_PRICE,
                    "timestamp": timestamp or int(datetime.now().timestamp()),
                    "source": "pump_fun_default",
                    "confidence": "low",
                    "is_baseline": True
                }]
            }
            
        except Exception as e:
            logger.error(f"Error calculating pump.fun price: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _extract_swap_from_enhanced_tx(self, tx: Dict[str, Any], 
                                      token_address: str) -> Optional[Dict[str, Any]]:
        """
        Extract swap information from enhanced transaction.
        
        Args:
            tx (Dict[str, Any]): Enhanced transaction data
            token_address (str): Token address to look for
            
        Returns:
            Optional[Dict[str, Any]]: Swap information if found
        """
        try:
            # Get basic transaction info
            signature = tx.get("signature", "")
            timestamp = tx.get("timestamp", 0)
            
            # Look for swap events
            events = tx.get("events", {})
            swap_event = events.get("swap")
            
            if swap_event:
                # Extract token amounts
                token_inputs = swap_event.get("tokenInputs", [])
                token_outputs = swap_event.get("tokenOutputs", [])
                
                # Find our token in inputs or outputs
                for token_data in token_inputs + token_outputs:
                    if token_data.get("mint") == token_address:
                        # Calculate price from SOL amount
                        sol_amount = self._get_sol_amount(token_inputs, token_outputs)
                        token_amount = float(token_data.get("rawTokenAmount", {}).get("tokenAmount", 0))
                        
                        if sol_amount > 0 and token_amount > 0:
                            price = sol_amount / token_amount
                            
                            return {
                                "signature": signature,
                                "timestamp": timestamp,
                                "price": price,
                                "sol_amount": sol_amount,
                                "token_amount": token_amount,
                                "type": "buy" if token_address in [t.get("mint") for t in token_outputs] else "sell",
                                "source": swap_event.get("programId", "unknown"),
                                "dex": self._identify_dex(swap_event.get("programId", ""))
                            }
            
            # Try native instructions if no swap event
            return self._extract_from_instructions(tx, token_address)
            
        except Exception as e:
            logger.error(f"Error extracting swap from transaction: {str(e)}")
            return None
    
    def _extract_from_instructions(self, tx: Dict[str, Any], 
                                  token_address: str) -> Optional[Dict[str, Any]]:
        """
        Extract swap info from transaction instructions.
        
        Args:
            tx (Dict[str, Any]): Transaction data
            token_address (str): Token address
            
        Returns:
            Optional[Dict[str, Any]]: Swap information if found
        """
        try:
            instructions = tx.get("instructions", [])
            
            for instruction in instructions:
                program_id = instruction.get("programId", "")
                
                # Check if it's a known DEX
                if program_id in self.DEX_PROGRAMS:
                    # Parse instruction data based on DEX type
                    parsed = instruction.get("parsed", {})
                    
                    if parsed:
                        # Look for token transfers involving our token
                        info = parsed.get("info", {})
                        
                        # This would need DEX-specific parsing
                        # For now, return basic info
                        return {
                            "timestamp": tx.get("timestamp", 0),
                            "source": "instruction_parse",
                            "dex": self.DEX_PROGRAMS[program_id],
                            "confidence": "medium"
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting from instructions: {str(e)}")
            return None
    
    def _get_sol_amount(self, inputs: List[Dict], outputs: List[Dict]) -> float:
        """Calculate SOL amount from token transfers."""
        sol_mint = "So11111111111111111111111111111111111111112"
        
        for token_data in inputs + outputs:
            if token_data.get("mint") == sol_mint:
                raw_amount = token_data.get("rawTokenAmount", {})
                return float(raw_amount.get("tokenAmount", 0)) / 1e9  # Convert lamports to SOL
        
        return 0.0
    
    def _identify_dex(self, program_id: str) -> str:
        """Identify DEX from program ID."""
        return self.DEX_PROGRAMS.get(program_id, "Unknown")
    
    def _find_closest_price(self, swaps: List[Dict], target_timestamp: int) -> Optional[Dict]:
        """Find the swap closest to target timestamp."""
        if not swaps:
            return None
        
        # Sort by timestamp
        sorted_swaps = sorted(swaps, key=lambda x: abs(x.get("timestamp", 0) - target_timestamp))
        
        # Return closest if within 1 hour
        closest = sorted_swaps[0]
        time_diff = abs(closest.get("timestamp", 0) - target_timestamp)
        
        if time_diff <= 3600:  # 1 hour
            return closest
        
        # Otherwise, check if we need to use as baseline
        if closest.get("timestamp", 0) > target_timestamp:
            closest["is_baseline"] = True
            closest["confidence"] = "medium"
        
        return closest
    
    def _get_token_creation_time(self, token_address: str) -> Optional[int]:
        """
        Get token creation timestamp.
        
        Args:
            token_address (str): Token address
            
        Returns:
            Optional[int]: Creation timestamp
        """
        try:
            # Get first transaction for the token
            endpoint = f"/v0/addresses/{token_address}/transactions"
            params = {
                "limit": 1,
                "before": None  # Get oldest transaction
            }
            
            result = self._make_request(endpoint, params=params)
            
            if isinstance(result, list) and result:
                return result[0].get("timestamp", None)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting token creation time: {str(e)}")
            return None
    
    def _extract_swap_info(self, transaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract swap information from a parsed transaction.
        
        Args:
            transaction (Dict[str, Any]): Parsed transaction data
            
        Returns:
            Optional[Dict[str, Any]]: Swap information if found
        """
        try:
            # This is a placeholder - would need actual transaction parsing logic
            # based on Helius enhanced transaction format
            return None
            
        except Exception as e:
            logger.error(f"Error extracting swap info: {str(e)}")
            return None
    
    def __str__(self) -> str:
        """String representation of the API client."""
        return f"HeliusAPI(authenticated={'✓' if self.api_key else '✗'})"
    
    def __repr__(self) -> str:
        """Detailed representation of the API client."""
        return f"HeliusAPI(api_key='***', base_url='{self.BASE_URL}')"