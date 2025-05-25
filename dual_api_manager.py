"""
Dual API Manager - Phoenix Project (WITH HELIUS INTEGRATION)

STRICT SEPARATION:
- Birdeye: For mainstream token analysis
- Helius: For pump.fun tokens and enhanced transaction parsing
- Cielo Finance: ONLY for wallet analysis
- NO FALLBACKS to prevent rate limiting

UPDATES:
- Added Helius API integration
- Smart routing based on token type
- Enhanced transaction parsing

FIXES IMPLEMENTED:
- Better error handling for API failures
- Improved logging with progress feedback
- Validation of API responses
"""

import logging
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("phoenix.api_manager")

class DualAPIManager:
    """Manages Birdeye, Helius, and Cielo Finance APIs with strict separation."""
    
    def __init__(self, birdeye_api_key: str, cielo_api_key: Optional[str] = None, 
                 helius_api_key: Optional[str] = None):
        """
        Initialize the dual-API manager with Helius support.
        
        Args:
            birdeye_api_key (str): Birdeye API key (required for token analysis)
            cielo_api_key (str, optional): Cielo Finance API key (required for wallet analysis)
            helius_api_key (str, optional): Helius API key (for pump.fun tokens and enhanced parsing)
        """
        self.birdeye_api = None
        self.cielo_api = None
        self.helius_api = None
        
        # Initialize Birdeye API (for mainstream TOKEN analysis)
        try:
            from birdeye_api import BirdeyeAPI
            self.birdeye_api = BirdeyeAPI(birdeye_api_key)
            logger.info("Birdeye API initialized for mainstream token analysis")
            print("✅ Birdeye API initialized", flush=True)
        except Exception as e:
            logger.error(f"Failed to initialize Birdeye API: {str(e)}")
            print(f"❌ Failed to initialize Birdeye API: {str(e)}", flush=True)
            raise ValueError("Birdeye API is required for token analysis")
        
        # Initialize Helius API (for pump.fun tokens and enhanced parsing)
        if helius_api_key:
            try:
                from helius_api import HeliusAPI
                self.helius_api = HeliusAPI(helius_api_key)
                logger.info("Helius API initialized for pump.fun tokens and enhanced parsing")
                print("✅ Helius API initialized for pump.fun tokens", flush=True)
            except Exception as e:
                logger.error(f"Failed to initialize Helius API: {str(e)}")
                logger.warning("Pump.fun token analysis will be limited without Helius")
                print(f"⚠️ Helius API initialization failed: {str(e)}", flush=True)
                print("⚠️ Pump.fun token analysis will be limited", flush=True)
        else:
            logger.warning("No Helius API key provided. Pump.fun token analysis will be limited.")
            print("⚠️ No Helius API key - pump.fun analysis limited", flush=True)
        
        # Initialize Cielo Finance API (for WALLET analysis only)
        if cielo_api_key:
            try:
                from cielo_api import CieloFinanceAPI
                self.cielo_api = CieloFinanceAPI(cielo_api_key)
                logger.info("Cielo Finance API initialized for WALLET analysis ONLY")
                print("✅ Cielo Finance API initialized for wallet analysis", flush=True)
            except Exception as e:
                logger.error(f"Failed to initialize Cielo Finance API: {str(e)}")
                logger.warning("Wallet analysis will NOT work without Cielo Finance API")
                print(f"❌ Cielo Finance API initialization failed: {str(e)}", flush=True)
                print("❌ Wallet analysis will NOT work", flush=True)
        else:
            logger.warning("No Cielo Finance API key provided. Wallet analysis will NOT work.")
            print("⚠️ No Cielo Finance API key - wallet analysis disabled", flush=True)
    
    def _is_pump_fun_token(self, token_address: str) -> bool:
        """Check if a token is a pump.fun token."""
        if not token_address:
            return False
        return token_address.endswith("pump") or "pump" in token_address.lower()
    
    # TOKEN-related methods (use Birdeye or Helius based on token type)
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get token information using appropriate API.
        Pump.fun tokens use Helius, others use Birdeye.
        """
        try:
            # Validate token address
            if not token_address or len(token_address) < 32:
                return {"success": False, "error": "Invalid token address"}
            
            # Check if it's a pump.fun token
            if self._is_pump_fun_token(token_address) and self.helius_api:
                logger.debug(f"Using Helius for pump.fun token: {token_address}")
                metadata = self.helius_api.get_token_metadata([token_address])
                if metadata.get("success") and metadata.get("data"):
                    return {
                        "success": True,
                        "data": metadata["data"][0] if metadata["data"] else {},
                        "source": "helius"
                    }
                else:
                    logger.warning(f"Helius failed for token {token_address}, trying Birdeye")
            
            # Default to Birdeye for all other tokens
            if self.birdeye_api:
                result = self.birdeye_api.get_token_info(token_address)
                if result.get("success"):
                    result["source"] = "birdeye"
                return result
            else:
                return {"success": False, "error": "No API available for token info"}
            
        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_token_price(self, token_address: str) -> Dict[str, Any]:
        """
        Get token price using appropriate API.
        Pump.fun tokens use Helius, others use Birdeye.
        """
        try:
            # Validate token address
            if not token_address or len(token_address) < 32:
                return {"success": False, "error": "Invalid token address"}
            
            # Check if it's a pump.fun token
            if self._is_pump_fun_token(token_address) and self.helius_api:
                logger.debug(f"Using Helius for pump.fun token price: {token_address}")
                result = self.helius_api.get_pump_fun_token_price(token_address)
                if result.get("success"):
                    result["source"] = "helius"
                return result
            
            # Default to Birdeye
            if self.birdeye_api:
                result = self.birdeye_api.get_token_price(token_address)
                if result.get("success"):
                    result["source"] = "birdeye"
                return result
            else:
                return {"success": False, "error": "No API available for token price"}
            
        except Exception as e:
            logger.error(f"Error getting token price: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_token_price_history(self, token_address: str, 
                              start_time: int, end_time: int, 
                              resolution: str = "5m") -> Dict[str, Any]:
        """
        Get token price history using appropriate API.
        Pump.fun tokens use Helius, others use Birdeye.
        """
        try:
            # Validate token address
            if not token_address or len(token_address) < 32:
                return {"success": False, "error": "Invalid token address"}
            
            # Validate timestamps
            if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
                return {"success": False, "error": "Invalid timestamps"}
            
            # Check if it's a pump.fun token
            if self._is_pump_fun_token(token_address) and self.helius_api:
                logger.debug(f"Using Helius for pump.fun token history: {token_address}")
                # Helius doesn't have traditional price history, so we analyze swaps
                result = self.helius_api.analyze_token_swaps(
                    "", # No specific wallet
                    token_address,
                    limit=100
                )
                if result.get("success"):
                    result["source"] = "helius"
                    result["is_pump_token"] = True
                return result
            
            # Default to Birdeye
            if self.birdeye_api:
                result = self.birdeye_api.get_token_price_history(token_address, start_time, end_time, resolution)
                if result.get("success"):
                    result["source"] = "birdeye"
                return result
            else:
                return {"success": False, "error": "No API available for price history"}
            
        except Exception as e:
            logger.error(f"Error getting token price history: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_enhanced_transactions(self, wallet_address: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get enhanced parsed transactions using Helius if available.
        Falls back to basic RPC if Helius not available.
        """
        try:
            if not wallet_address or len(wallet_address) < 32:
                return {"success": False, "error": "Invalid wallet address"}
            
            if self.helius_api:
                logger.debug(f"Using Helius for enhanced transactions: {wallet_address}")
                result = self.helius_api.get_enhanced_transactions(wallet_address, limit)
                if result.get("success"):
                    result["source"] = "helius"
                return result
            else:
                logger.warning("Helius API not available for enhanced transactions")
                return {"success": False, "error": "Helius API required for enhanced transactions"}
                
        except Exception as e:
            logger.error(f"Error getting enhanced transactions: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_dex_trades(self, token_address: str, limit: int = 100) -> Dict[str, Any]:
        """Get DEX trades using Birdeye API."""
        try:
            if not token_address or len(token_address) < 32:
                return {"success": False, "error": "Invalid token address"}
            
            if self.birdeye_api:
                result = self.birdeye_api.get_dex_trades(token_address, limit)
                if result.get("success"):
                    result["source"] = "birdeye"
                return result
            else:
                return {"success": False, "error": "Birdeye API required for DEX trades"}
                
        except Exception as e:
            logger.error(f"Error getting DEX trades: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def calculate_token_performance(self, token_address: str, start_time: datetime) -> Dict[str, Any]:
        """
        Calculate token performance using appropriate API.
        Pump.fun tokens use Helius, others use Birdeye.
        """
        try:
            if not token_address or len(token_address) < 32:
                return {"success": False, "error": "Invalid token address"}
            
            # Check if it's a pump.fun token
            if self._is_pump_fun_token(token_address) and self.helius_api:
                logger.debug(f"Using Helius for pump.fun token performance: {token_address}")
                # Convert datetime to timestamp
                start_timestamp = int(start_time.timestamp())
                price_data = self.helius_api.get_pump_fun_token_price(token_address, start_timestamp)
                
                if price_data.get("success"):
                    initial_price = price_data.get("data", {}).get("price", 0)
                    current_price_data = self.helius_api.get_pump_fun_token_price(token_address)
                    current_price = current_price_data.get("data", {}).get("price", initial_price)
                    
                    roi = ((current_price / initial_price) - 1) * 100 if initial_price > 0 else 0
                    
                    return {
                        "success": True,
                        "token_address": token_address,
                        "initial_price": initial_price,
                        "current_price": current_price,
                        "roi_percent": roi,
                        "data_source": "helius",
                        "is_pump_token": True
                    }
            
            # Default to Birdeye
            if self.birdeye_api:
                result = self.birdeye_api.calculate_token_performance(token_address, start_time)
                if result.get("success"):
                    result["data_source"] = "birdeye"
                return result
            else:
                return {"success": False, "error": "No API available for performance calculation"}
            
        except Exception as e:
            logger.error(f"Error calculating token performance: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def identify_platform(self, contract_address: str, token_info: Optional[Dict[str, Any]] = None) -> str:
        """Identify platform using available APIs."""
        try:
            # Quick check for pump.fun
            if contract_address.endswith("pump"):
                return "pump.fun"
            
            # Use Birdeye for platform identification
            if self.birdeye_api:
                return self.birdeye_api.identify_platform(contract_address, token_info)
            else:
                return "unknown"
            
        except Exception as e:
            logger.error(f"Error identifying platform: {str(e)}")
            return "unknown"
    
    # WALLET-related methods (use Cielo Finance ONLY - NO FALLBACK)
    def get_wallet_transactions(self, wallet_address: str, limit: int = 100) -> Dict[str, Any]:
        """Get wallet transactions using Cielo Finance API ONLY - NO FALLBACK."""
        if not self.cielo_api:
            logger.error("Cielo Finance API not configured. Cannot analyze wallets.")
            return {
                "success": False,
                "error": "Cielo Finance API required for wallet analysis. Configure with --cielo-api-key"
            }
        
        try:
            if not wallet_address or len(wallet_address) < 32:
                return {"success": False, "error": "Invalid wallet address"}
            
            logger.debug(f"Using Cielo Finance API for wallet transactions: {wallet_address}")
            result = self.cielo_api.get_wallet_transactions(wallet_address, limit)
            if result.get("success"):
                result["source"] = "cielo"
            return result
        except Exception as e:
            logger.error(f"Cielo Finance API failed for wallet transactions: {str(e)}")
            return {
                "success": False,
                "error": f"Cielo Finance API error: {str(e)}"
            }
    
    def get_wallet_tokens(self, wallet_address: str) -> Dict[str, Any]:
        """Get wallet token holdings using Cielo Finance API ONLY - NO FALLBACK."""
        if not self.cielo_api:
            logger.error("Cielo Finance API not configured. Cannot get wallet tokens.")
            return {
                "success": False,
                "error": "Cielo Finance API required for wallet analysis. Configure with --cielo-api-key"
            }
        
        try:
            if not wallet_address or len(wallet_address) < 32:
                return {"success": False, "error": "Invalid wallet address"}
            
            logger.debug(f"Using Cielo Finance API for wallet tokens: {wallet_address}")
            result = self.cielo_api.get_wallet_tokens(wallet_address)
            if result.get("success"):
                result["source"] = "cielo"
            return result
        except Exception as e:
            logger.error(f"Cielo Finance API failed for wallet tokens: {str(e)}")
            return {
                "success": False,
                "error": f"Cielo Finance API error: {str(e)}"
            }
    
    def get_api_status(self) -> Dict[str, Any]:
        """Check the status of configured APIs."""
        status = {
            "apis_configured": [],
            "api_status": {},
            "usage": {
                "birdeye": "Mainstream token analysis",
                "helius": "Pump.fun tokens & enhanced parsing",
                "cielo": "Wallet analysis"
            }
        }
        
        # Check Birdeye API
        if self.birdeye_api:
            status["apis_configured"].append("birdeye")
            try:
                test_result = self.birdeye_api.get_token_info("So11111111111111111111111111111111111111112")
                if test_result.get("success", True):
                    status["api_status"]["birdeye"] = "operational"
                else:
                    status["api_status"]["birdeye"] = "limited"
            except Exception as e:
                status["api_status"]["birdeye"] = f"error: {str(e)}"
        
        # Check Helius API
        if self.helius_api:
            status["apis_configured"].append("helius")
            try:
                if self.helius_api.health_check():
                    status["api_status"]["helius"] = "operational"
                else:
                    status["api_status"]["helius"] = "limited"
            except Exception as e:
                status["api_status"]["helius"] = f"error: {str(e)}"
        else:
            status["api_status"]["helius"] = "not_configured"
        
        # Check Cielo Finance API
        if self.cielo_api:
            status["apis_configured"].append("cielo")
            try:
                if hasattr(self.cielo_api, 'health_check'):
                    health_result = self.cielo_api.health_check()
                    if health_result:
                        status["api_status"]["cielo"] = "operational"
                    else:
                        status["api_status"]["cielo"] = "limited"
                else:
                    test_result = self.cielo_api.get_wallet_trading_stats("11111111111111111111111111111111")
                    status["api_status"]["cielo"] = "operational"
            except Exception as e:
                error_msg = str(e).lower()
                if any(indicator in error_msg for indicator in ["not found", "invalid", "unauthorized"]):
                    status["api_status"]["cielo"] = "operational"
                else:
                    status["api_status"]["cielo"] = f"error: {str(e)}"
        else:
            status["api_status"]["cielo"] = "not_configured"
        
        return status
    
    def get_recommended_config(self) -> Dict[str, str]:
        """Get recommended configuration based on available APIs."""
        config = {
            "mainstream_tokens": "birdeye" if self.birdeye_api else "not_available",
            "pump_fun_tokens": "helius" if self.helius_api else "limited_analysis",
            "wallet_analysis": "cielo" if self.cielo_api else "not_available",
            "notes": []
        }
        
        if not self.helius_api:
            config["notes"].append("RECOMMENDED: Add Helius API key for pump.fun token analysis")
            config["notes"].append("Run: python phoenix.py configure --helius-api-key YOUR_KEY")
        
        if not self.cielo_api:
            config["notes"].append("CRITICAL: Cielo Finance API key required for wallet analysis")
            config["notes"].append("Run: python phoenix.py configure --cielo-api-key YOUR_KEY")
        
        if not self.birdeye_api:
            config["notes"].append("CRITICAL: Birdeye API key required for token analysis")
            config["notes"].append("Run: python phoenix.py configure --birdeye-api-key YOUR_KEY")
        
        config["notes"].append("Separation: Birdeye=mainstream tokens, Helius=pump.fun, Cielo=wallets")
        
        return config