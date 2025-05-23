"""
Dual API Manager - Phoenix Project (Fixed - No Fallback)

STRICT SEPARATION:
- Birdeye: ONLY for token analysis (Telegram channels)
- Cielo Finance: ONLY for wallet analysis
- NO FALLBACKS to prevent rate limiting

FIXED: Correct import for CieloFinanceAPI
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("phoenix.api_manager")

class DualAPIManager:
    """Manages Birdeye for token analysis and Cielo Finance for wallet analysis - STRICT SEPARATION."""
    
    def __init__(self, birdeye_api_key: str, cielo_api_key: Optional[str] = None):
        """
        Initialize the dual-API manager.
        
        Args:
            birdeye_api_key (str): Birdeye API key (required for token analysis)
            cielo_api_key (str, optional): Cielo Finance API key (required for wallet analysis)
        """
        self.birdeye_api = None
        self.cielo_api = None
        
        # Initialize Birdeye API (for TOKEN analysis only)
        try:
            from birdeye_api import BirdeyeAPI
            self.birdeye_api = BirdeyeAPI(birdeye_api_key)
            logger.info("Birdeye API initialized for TOKEN analysis ONLY")
        except Exception as e:
            logger.error(f"Failed to initialize Birdeye API: {str(e)}")
            raise ValueError("Birdeye API is required for token analysis")
        
        # Initialize Cielo Finance API (for WALLET analysis only)
        if cielo_api_key:
            try:
                from cielo_api import CieloFinanceAPI  # FIXED: Correct class name
                self.cielo_api = CieloFinanceAPI(cielo_api_key)  # FIXED: Correct class name
                logger.info("Cielo Finance API initialized for WALLET analysis ONLY")
            except Exception as e:
                logger.error(f"Failed to initialize Cielo Finance API: {str(e)}")
                logger.warning("Wallet analysis will NOT work without Cielo Finance API")
        else:
            logger.warning("No Cielo Finance API key provided. Wallet analysis will NOT work.")
    
    # TOKEN-related methods (use Birdeye ONLY)
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """Get token information using Birdeye API ONLY."""
        try:
            return self.birdeye_api.get_token_info(token_address)
        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_token_price(self, token_address: str) -> Dict[str, Any]:
        """Get token price using Birdeye API ONLY."""
        try:
            return self.birdeye_api.get_token_price(token_address)
        except Exception as e:
            logger.error(f"Error getting token price: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_token_price_history(self, token_address: str, 
                              start_time: int, end_time: int, 
                              resolution: str = "5m") -> Dict[str, Any]:
        """Get token price history using Birdeye API ONLY."""
        try:
            return self.birdeye_api.get_token_price_history(token_address, start_time, end_time, resolution)
        except Exception as e:
            logger.error(f"Error getting token price history: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_dex_trades(self, token_address: str, limit: int = 100) -> Dict[str, Any]:
        """Get DEX trades using Birdeye API ONLY."""
        try:
            return self.birdeye_api.get_dex_trades(token_address, limit)
        except Exception as e:
            logger.error(f"Error getting DEX trades: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def calculate_token_performance(self, token_address: str, start_time: datetime) -> Dict[str, Any]:
        """Calculate token performance using Birdeye API ONLY."""
        try:
            return self.birdeye_api.calculate_token_performance(token_address, start_time)
        except Exception as e:
            logger.error(f"Error calculating token performance: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def identify_platform(self, contract_address: str, token_info: Optional[Dict[str, Any]] = None) -> str:
        """Identify platform using Birdeye API ONLY."""
        try:
            return self.birdeye_api.identify_platform(contract_address, token_info)
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
            logger.debug(f"Using Cielo Finance API for wallet transactions: {wallet_address}")
            return self.cielo_api.get_wallet_transactions(wallet_address, limit)
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
            logger.debug(f"Using Cielo Finance API for wallet tokens: {wallet_address}")
            return self.cielo_api.get_wallet_tokens(wallet_address)
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
                "birdeye": "Token analysis (Telegram channels)",
                "cielo": "Wallet analysis"
            }
        }
        
        # Check Birdeye API (for tokens)
        if self.birdeye_api:
            status["apis_configured"].append("birdeye")
            try:
                # Test with a known token (Wrapped SOL)
                test_result = self.birdeye_api.get_token_info("So11111111111111111111111111111111111111112")
                if test_result.get("success", True):
                    status["api_status"]["birdeye"] = "operational"
                else:
                    status["api_status"]["birdeye"] = "limited"
            except Exception as e:
                status["api_status"]["birdeye"] = f"error: {str(e)}"
        
        # Check Cielo Finance API (for wallets)
        if self.cielo_api:
            status["apis_configured"].append("cielo")
            try:
                # Test with a simple API call
                if hasattr(self.cielo_api, 'health_check'):
                    # Use health check if available
                    health_result = self.cielo_api.health_check()
                    if health_result:
                        status["api_status"]["cielo"] = "operational"
                    else:
                        status["api_status"]["cielo"] = "limited"
                else:
                    # Fallback to testing with a wallet
                    test_result = self.cielo_api.get_wallet_trading_stats("11111111111111111111111111111111")
                    # Even if the wallet doesn't exist, if we get a proper API response, it's operational
                    status["api_status"]["cielo"] = "operational"
            except Exception as e:
                # Check if it's just an invalid wallet error (which means API is working)
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
            "token_analysis": "birdeye" if self.birdeye_api else "not_available",
            "wallet_analysis": "cielo" if self.cielo_api else "not_available",
            "notes": []
        }
        
        if not self.cielo_api:
            config["notes"].append("CRITICAL: Cielo Finance API key required for wallet analysis")
            config["notes"].append("Run: python phoenix.py configure --cielo-api-key YOUR_KEY")
        
        if not self.birdeye_api:
            config["notes"].append("CRITICAL: Birdeye API key required for token analysis")
            config["notes"].append("Run: python phoenix.py configure --birdeye-api-key YOUR_KEY")
        
        config["notes"].append("Strict separation: Birdeye=tokens, Cielo=wallets (no fallbacks)")
        
        return config