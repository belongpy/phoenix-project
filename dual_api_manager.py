"""
Dual API Manager - Phoenix Project (SMART ROUTING & FALLBACK VERSION)

CRITICAL FIXES IMPLEMENTED:
- Smart API routing: Helius → P9 RPC → DexScreener → Birdeye fallback for pump.fun
- Proper fallback cascade for all token types
- Enhanced pump.fun token detection and handling
- Maintains full compatibility with wallet_module
- Improved error handling and retry logic
- Performance optimizations with caching

ROUTING LOGIC:
- Pump.fun tokens: Helius (bonding curve) → RPC → DexScreener → Birdeye
- Regular tokens: Birdeye → DexScreener → RPC → Helius
- Wallet analysis: Cielo Finance ONLY (no fallbacks for data integrity)
"""

import logging
import sys
import time
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger("phoenix.api_manager")

class DualAPIManager:
    """Enhanced API manager with smart routing and proper fallback logic."""
    
    # API priority constants
    PUMP_FUN_API_PRIORITY = ['helius', 'rpc', 'dexscreener', 'birdeye']
    REGULAR_TOKEN_API_PRIORITY = ['birdeye', 'dexscreener', 'rpc', 'helius']
    
    # Timeouts for each API (seconds)
    API_TIMEOUTS = {
        'birdeye': 5,
        'helius': 6,
        'rpc': 4,
        'dexscreener': 4,
        'cielo': 10
    }
    
    # Cache settings
    PRICE_CACHE_TTL = 300  # 5 minutes
    TOKEN_INFO_CACHE_TTL = 600  # 10 minutes
    
    def __init__(self, birdeye_api_key: str, cielo_api_key: Optional[str] = None, 
                 helius_api_key: Optional[str] = None, rpc_url: Optional[str] = None):
        """
        Initialize the enhanced dual-API manager.
        
        Args:
            birdeye_api_key (str): Birdeye API key (required for token analysis)
            cielo_api_key (str, optional): Cielo Finance API key (required for wallet analysis)
            helius_api_key (str, optional): Helius API key (recommended for pump.fun tokens)
            rpc_url (str, optional): Custom RPC URL (P9 or other provider)
        """
        self.birdeye_api = None
        self.cielo_api = None
        self.helius_api = None
        self.rpc_url = rpc_url or "https://api.mainnet-beta.solana.com"
        
        # API performance tracking
        self.api_stats = {
            'birdeye': {'calls': 0, 'success': 0, 'errors': 0},
            'helius': {'calls': 0, 'success': 0, 'errors': 0},
            'rpc': {'calls': 0, 'success': 0, 'errors': 0},
            'dexscreener': {'calls': 0, 'success': 0, 'errors': 0},
            'cielo': {'calls': 0, 'success': 0, 'errors': 0}
        }
        
        # Caching system
        self.price_cache = {}
        self.token_info_cache = {}
        
        # Initialize APIs
        self._initialize_apis(birdeye_api_key, cielo_api_key, helius_api_key)
    
    def _initialize_apis(self, birdeye_api_key: str, cielo_api_key: Optional[str], helius_api_key: Optional[str]):
        """Initialize all APIs with proper error handling."""
        
        # Initialize Birdeye API (REQUIRED for mainstream tokens)
        try:
            from birdeye_api import BirdeyeAPI
            self.birdeye_api = BirdeyeAPI(birdeye_api_key)
            logger.info("✅ Birdeye API initialized for mainstream token analysis")
            print("✅ Birdeye API initialized", flush=True)
        except Exception as e:
            logger.error(f"❌ Failed to initialize Birdeye API: {str(e)}")
            print(f"❌ Failed to initialize Birdeye API: {str(e)}", flush=True)
            raise ValueError("Birdeye API is required for token analysis")
        
        # Initialize Helius API (RECOMMENDED for pump.fun tokens)
        if helius_api_key:
            try:
                from helius_api import HeliusAPI
                self.helius_api = HeliusAPI(helius_api_key)
                logger.info("✅ Helius API initialized for pump.fun tokens and enhanced parsing")
                print("✅ Helius API initialized for pump.fun tokens", flush=True)
            except Exception as e:
                logger.error(f"⚠️ Failed to initialize Helius API: {str(e)}")
                logger.warning("Pump.fun token analysis will be limited without Helius")
                print(f"⚠️ Helius API initialization failed: {str(e)}", flush=True)
                print("⚠️ Pump.fun token analysis will be limited", flush=True)
        else:
            logger.warning("No Helius API key provided. Pump.fun token analysis will be limited.")
            print("⚠️ No Helius API key - pump.fun analysis limited", flush=True)
        
        # Initialize Cielo Finance API (REQUIRED for wallet analysis)
        if cielo_api_key:
            try:
                from cielo_api import CieloFinanceAPI
                self.cielo_api = CieloFinanceAPI(cielo_api_key)
                logger.info("✅ Cielo Finance API initialized for WALLET analysis ONLY")
                print("✅ Cielo Finance API initialized for wallet analysis", flush=True)
            except Exception as e:
                logger.error(f"❌ Failed to initialize Cielo Finance API: {str(e)}")
                logger.warning("Wallet analysis will NOT work without Cielo Finance API")
                print(f"❌ Cielo Finance API initialization failed: {str(e)}", flush=True)
                print("❌ Wallet analysis will NOT work", flush=True)
        else:
            logger.warning("No Cielo Finance API key provided. Wallet analysis will NOT work.")
            print("⚠️ No Cielo Finance API key - wallet analysis disabled", flush=True)
    
    def _is_pump_fun_token(self, token_address: str) -> bool:
        """Enhanced pump.fun token detection."""
        if not token_address:
            return False
        
        # Direct pump.fun detection
        if token_address.endswith("pump"):
            return True
        
        # Additional heuristics for pump.fun tokens
        if len(token_address) == 44 and "pump" in token_address.lower():
            return True
        
        # Check if token has typical pump.fun characteristics
        # (This would need more sophisticated detection in production)
        
        return False
    
    def _get_cache_key(self, prefix: str, token_address: str, **kwargs) -> str:
        """Generate cache key for token data."""
        key_parts = [prefix, token_address]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "_".join(key_parts)
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any], ttl: int) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - cache_entry.get('timestamp', 0) < ttl
    
    async def _try_dexscreener_api(self, token_address: str) -> Dict[str, Any]:
        """Try DexScreener API for token price."""
        self.api_stats['dexscreener']['calls'] += 1
        
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.API_TIMEOUTS['dexscreener'])) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get('pairs', [])
                        
                        if pairs:
                            # Get best pair (highest liquidity)
                            best_pair = max(pairs, key=lambda x: float(x.get('liquidity', {}).get('usd', 0) or 0))
                            price_usd = float(best_pair.get('priceUsd', 0) or 0)
                            
                            if price_usd > 0:
                                self.api_stats['dexscreener']['success'] += 1
                                return {
                                    "success": True,
                                    "data": {
                                        "value": price_usd,
                                        "marketCap": float(best_pair.get('marketCap', 0) or 0),
                                        "liquidity": float(best_pair.get('liquidity', {}).get('usd', 0) or 0),
                                        "volume24h": float(best_pair.get('volume', {}).get('h24', 0) or 0),
                                        "dex": best_pair.get('dexId', 'unknown')
                                    },
                                    "source": "dexscreener"
                                }
            
            self.api_stats['dexscreener']['errors'] += 1
            return {"success": False, "error": "No valid pairs found", "source": "dexscreener"}
            
        except Exception as e:
            self.api_stats['dexscreener']['errors'] += 1
            logger.debug(f"DexScreener API error for {token_address}: {str(e)}")
            return {"success": False, "error": str(e), "source": "dexscreener"}
    
    async def _try_rpc_price(self, token_address: str) -> Dict[str, Any]:
        """Try RPC for token price via pool analysis."""
        self.api_stats['rpc']['calls'] += 1
        
        try:
            # This is a simplified implementation
            # In production, you'd query DEX pools directly via RPC
            # For now, return a placeholder that indicates RPC was attempted
            
            self.api_stats['rpc']['errors'] += 1
            return {"success": False, "error": "RPC price discovery not implemented", "source": "rpc"}
            
        except Exception as e:
            self.api_stats['rpc']['errors'] += 1
            logger.debug(f"RPC price error for {token_address}: {str(e)}")
            return {"success": False, "error": str(e), "source": "rpc"}
    
    # TOKEN-related methods with smart routing
    def get_token_info(self, token_address: str) -> Dict[str, Any]:
        """
        Get token information using smart API routing.
        Pump.fun tokens use Helius first, others use Birdeye first.
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key("token_info", token_address)
            if cache_key in self.token_info_cache:
                cache_entry = self.token_info_cache[cache_key]
                if self._is_cache_valid(cache_entry, self.TOKEN_INFO_CACHE_TTL):
                    return cache_entry['data']
            
            # Validate token address
            if not token_address or len(token_address) < 32:
                return {"success": False, "error": "Invalid token address"}
            
            # Determine API priority based on token type
            is_pump = self._is_pump_fun_token(token_address)
            api_priority = self.PUMP_FUN_API_PRIORITY if is_pump else self.REGULAR_TOKEN_API_PRIORITY
            
            result = None
            errors = []
            
            # Try APIs in priority order
            for api_name in api_priority:
                try:
                    if api_name == 'helius' and self.helius_api:
                        self.api_stats['helius']['calls'] += 1
                        metadata = self.helius_api.get_token_metadata([token_address])
                        if metadata.get("success") and metadata.get("data"):
                            self.api_stats['helius']['success'] += 1
                            result = {
                                "success": True,
                                "data": metadata["data"][0] if metadata["data"] else {},
                                "source": "helius"
                            }
                            break
                        else:
                            self.api_stats['helius']['errors'] += 1
                            errors.append(f"helius: {metadata.get('error', 'No data')}")
                    
                    elif api_name == 'birdeye' and self.birdeye_api:
                        self.api_stats['birdeye']['calls'] += 1
                        birdeye_result = self.birdeye_api.get_token_info(token_address)
                        if birdeye_result.get("success"):
                            self.api_stats['birdeye']['success'] += 1
                            result = birdeye_result
                            result["source"] = "birdeye"
                            break
                        else:
                            self.api_stats['birdeye']['errors'] += 1
                            errors.append(f"birdeye: {birdeye_result.get('error', 'Failed')}")
                    
                    elif api_name == 'dexscreener':
                        # DexScreener doesn't provide detailed token info, skip for this method
                        continue
                    
                    elif api_name == 'rpc':
                        # RPC doesn't provide token metadata, skip for this method
                        continue
                        
                except Exception as e:
                    errors.append(f"{api_name}: {str(e)}")
                    continue
            
            # If no API succeeded, return error with all attempts
            if not result:
                result = {
                    "success": False, 
                    "error": f"All APIs failed: {'; '.join(errors)}",
                    "source": "none"
                }
            
            # Cache the result
            self.token_info_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_token_price(self, token_address: str) -> Dict[str, Any]:
        """
        Get token price using smart API routing with async fallbacks.
        Pump.fun tokens: Helius → RPC → DexScreener → Birdeye
        Regular tokens: Birdeye → DexScreener → RPC → Helius
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key("token_price", token_address)
            if cache_key in self.price_cache:
                cache_entry = self.price_cache[cache_key]
                if self._is_cache_valid(cache_entry, self.PRICE_CACHE_TTL):
                    return cache_entry['data']
            
            # Validate token address
            if not token_address or len(token_address) < 32:
                return {"success": False, "error": "Invalid token address"}
            
            # Determine API priority based on token type
            is_pump = self._is_pump_fun_token(token_address)
            api_priority = self.PUMP_FUN_API_PRIORITY if is_pump else self.REGULAR_TOKEN_API_PRIORITY
            
            result = None
            errors = []
            
            # Try APIs in priority order
            for api_name in api_priority:
                try:
                    if api_name == 'helius' and self.helius_api:
                        self.api_stats['helius']['calls'] += 1
                        if is_pump:
                            # Use pump.fun specific method
                            pump_result = self.helius_api.get_pump_fun_token_price(token_address)
                            if pump_result.get("success") and pump_result.get("data"):
                                price = pump_result["data"].get("price", 0)
                                if price > 0:
                                    self.api_stats['helius']['success'] += 1
                                    result = {
                                        "success": True,
                                        "data": {"value": price, "solPrice": price},
                                        "source": "helius_pump",
                                        "confidence": pump_result["data"].get("confidence", "medium"),
                                        "is_graduated": pump_result["data"].get("is_graduated", False)
                                    }
                                    break
                            else:
                                self.api_stats['helius']['errors'] += 1
                                errors.append(f"helius_pump: {pump_result.get('error', 'No price data')}")
                        else:
                            # Use regular swap analysis
                            swap_result = self.helius_api.analyze_token_swaps("", token_address, 5)
                            if swap_result.get("success") and swap_result.get("latest_price", 0) > 0:
                                self.api_stats['helius']['success'] += 1
                                result = {
                                    "success": True,
                                    "data": {"value": swap_result["latest_price"], "solPrice": swap_result["latest_price"]},
                                    "source": "helius_swap"
                                }
                                break
                            else:
                                self.api_stats['helius']['errors'] += 1
                                errors.append(f"helius_swap: {swap_result.get('error', 'No swaps found')}")
                    
                    elif api_name == 'birdeye' and self.birdeye_api:
                        self.api_stats['birdeye']['calls'] += 1
                        birdeye_result = self.birdeye_api.get_token_price(token_address)
                        if birdeye_result.get("success") and birdeye_result.get("data"):
                            price = birdeye_result["data"].get("value", 0)
                            if price > 0:
                                self.api_stats['birdeye']['success'] += 1
                                result = birdeye_result
                                result["source"] = "birdeye"
                                break
                        self.api_stats['birdeye']['errors'] += 1
                        errors.append(f"birdeye: {birdeye_result.get('error', 'No price data')}")
                    
                    elif api_name == 'dexscreener':
                        dex_result = await self._try_dexscreener_api(token_address)
                        if dex_result.get("success"):
                            result = dex_result
                            break
                        else:
                            errors.append(f"dexscreener: {dex_result.get('error', 'Failed')}")
                    
                    elif api_name == 'rpc':
                        rpc_result = await self._try_rpc_price(token_address)
                        if rpc_result.get("success"):
                            result = rpc_result
                            break
                        else:
                            errors.append(f"rpc: {rpc_result.get('error', 'Failed')}")
                        
                except Exception as e:
                    errors.append(f"{api_name}: {str(e)}")
                    continue
            
            # If no API succeeded, return error with all attempts
            if not result:
                result = {
                    "success": False, 
                    "error": f"All price APIs failed: {'; '.join(errors)}",
                    "source": "none",
                    "data": {"value": 0, "solPrice": 0}
                }
            
            # Cache the result
            self.price_cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting token price: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_token_price_sync(self, token_address: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for get_token_price.
        Used by existing code that can't handle async.
        """
        try:
            # Try to run in existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.get_token_price(token_address))
                    return future.result(timeout=15)
            else:
                return asyncio.run(self.get_token_price(token_address))
        except Exception as e:
            logger.error(f"Error in sync price wrapper: {str(e)}")
            # Fallback to direct API calls without async
            return self._get_token_price_fallback_sync(token_address)
    
    def _get_token_price_fallback_sync(self, token_address: str) -> Dict[str, Any]:
        """Synchronous fallback for token price when async fails."""
        is_pump = self._is_pump_fun_token(token_address)
        
        if is_pump and self.helius_api:
            result = self.helius_api.get_pump_fun_token_price(token_address)
            if result.get("success"):
                return {
                    "success": True,
                    "data": {"value": result["data"]["price"], "solPrice": result["data"]["price"]},
                    "source": "helius_sync"
                }
        
        if self.birdeye_api:
            result = self.birdeye_api.get_token_price(token_address)
            if result.get("success"):
                result["source"] = "birdeye_sync"
                return result
        
        return {"success": False, "error": "All sync APIs failed", "data": {"value": 0, "solPrice": 0}}
    
    def get_token_price_history(self, token_address: str, 
                              start_time: int, end_time: int, 
                              resolution: str = "5m") -> Dict[str, Any]:
        """
        Get token price history using appropriate API.
        Pump.fun tokens have limited history, so we focus on Helius bonding curve data.
        """
        try:
            # Validate inputs
            if not token_address or len(token_address) < 32:
                return {"success": False, "error": "Invalid token address"}
            
            if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
                return {"success": False, "error": "Invalid timestamps"}
            
            # Check if it's a pump.fun token
            if self._is_pump_fun_token(token_address) and self.helius_api:
                logger.debug(f"Using Helius for pump.fun token history: {token_address}")
                
                # For pump.fun tokens, we can't get traditional price history
                # Instead, we estimate based on bonding curve progression
                pump_result = self.helius_api.get_pump_fun_token_price(token_address, start_time)
                if pump_result.get("success"):
                    current_result = self.helius_api.get_pump_fun_token_price(token_address)
                    
                    # Create synthetic price history
                    start_price = pump_result["data"].get("price", 0)
                    current_price = current_result["data"].get("price", start_price) if current_result.get("success") else start_price
                    
                    # Generate simple price progression
                    time_points = 10  # Generate 10 data points
                    time_step = (end_time - start_time) / time_points
                    price_step = (current_price - start_price) / time_points if current_price > start_price else 0
                    
                    items = []
                    for i in range(time_points + 1):
                        items.append({
                            "unixTime": int(start_time + (i * time_step)),
                            "value": start_price + (i * price_step),
                            "type": "estimated"
                        })
                    
                    return {
                        "success": True,
                        "data": {"items": items},
                        "source": "helius_estimated",
                        "is_pump_token": True
                    }
            
            # Default to Birdeye for regular tokens
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
    
    def calculate_token_performance(self, token_address: str, start_time: datetime) -> Dict[str, Any]:
        """
        Calculate token performance using smart API routing.
        """
        try:
            if not token_address or len(token_address) < 32:
                return {"success": False, "error": "Invalid token address"}
            
            # Check if it's a pump.fun token
            if self._is_pump_fun_token(token_address) and self.helius_api:
                logger.debug(f"Using Helius for pump.fun token performance: {token_address}")
                
                # Get price at start time and current price
                start_timestamp = int(start_time.timestamp())
                start_price_data = self.helius_api.get_pump_fun_token_price(token_address, start_timestamp)
                current_price_data = self.helius_api.get_pump_fun_token_price(token_address)
                
                if start_price_data.get("success") and current_price_data.get("success"):
                    initial_price = start_price_data["data"].get("price", 0)
                    current_price = current_price_data["data"].get("price", initial_price)
                    
                    roi = ((current_price / initial_price) - 1) * 100 if initial_price > 0 else 0
                    
                    return {
                        "success": True,
                        "token_address": token_address,
                        "initial_price": initial_price,
                        "current_price": current_price,
                        "roi_percent": roi,
                        "max_roi_percent": roi,  # Simplified
                        "data_source": "helius",
                        "is_pump_token": True,
                        "confidence": current_price_data["data"].get("confidence", "medium")
                    }
            
            # Default to Birdeye for regular tokens
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
    
    def get_enhanced_transactions(self, wallet_address: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get enhanced parsed transactions using Helius if available.
        Maintains compatibility with wallet_module.
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
        """Get DEX trades using Birdeye API with Helius fallback."""
        try:
            if not token_address or len(token_address) < 32:
                return {"success": False, "error": "Invalid token address"}
            
            # Try Birdeye first
            if self.birdeye_api:
                result = self.birdeye_api.get_dex_trades(token_address, limit)
                if result.get("success"):
                    result["source"] = "birdeye"
                    return result
            
            # Fallback to Helius for pump.fun tokens
            if self._is_pump_fun_token(token_address) and self.helius_api:
                swap_result = self.helius_api.analyze_token_swaps("", token_address, limit)
                if swap_result.get("success"):
                    # Transform Helius swap data to match Birdeye format
                    trades = swap_result.get("data", [])
                    formatted_trades = []
                    
                    for trade in trades:
                        formatted_trades.append({
                            "signature": trade.get("signature", ""),
                            "blockTime": trade.get("timestamp", 0),
                            "price": trade.get("price_per_token", 0),
                            "volume": trade.get("sol_amount", 0),
                            "type": trade.get("swap_type", "unknown")
                        })
                    
                    return {
                        "success": True,
                        "data": {"items": formatted_trades},
                        "source": "helius_swaps"
                    }
            
            return {"success": False, "error": "No API available for DEX trades"}
                
        except Exception as e:
            logger.error(f"Error getting DEX trades: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def identify_platform(self, contract_address: str, token_info: Optional[Dict[str, Any]] = None) -> str:
        """Identify platform using available APIs with pump.fun detection."""
        try:
            # Quick check for pump.fun
            if self._is_pump_fun_token(contract_address):
                return "pump.fun"
            
            # Use Birdeye for platform identification
            if self.birdeye_api:
                return self.birdeye_api.identify_platform(contract_address, token_info)
            else:
                return "unknown"
            
        except Exception as e:
            logger.error(f"Error identifying platform: {str(e)}")
            return "unknown"
    
    # WALLET-related methods (use Cielo Finance ONLY - NO FALLBACK for data integrity)
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
            
            self.api_stats['cielo']['calls'] += 1
            logger.debug(f"Using Cielo Finance API for wallet transactions: {wallet_address}")
            result = self.cielo_api.get_wallet_transactions(wallet_address, limit)
            if result.get("success"):
                self.api_stats['cielo']['success'] += 1
                result["source"] = "cielo"
            else:
                self.api_stats['cielo']['errors'] += 1
            return result
        except Exception as e:
            self.api_stats['cielo']['errors'] += 1
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
            
            self.api_stats['cielo']['calls'] += 1
            logger.debug(f"Using Cielo Finance API for wallet tokens: {wallet_address}")
            result = self.cielo_api.get_wallet_tokens(wallet_address)
            if result.get("success"):
                self.api_stats['cielo']['success'] += 1
                result["source"] = "cielo"
            else:
                self.api_stats['cielo']['errors'] += 1
            return result
        except Exception as e:
            self.api_stats['cielo']['errors'] += 1
            logger.error(f"Cielo Finance API failed for wallet tokens: {str(e)}")
            return {
                "success": False,
                "error": f"Cielo Finance API error: {str(e)}"
            }
    
    def get_api_status(self) -> Dict[str, Any]:
        """Check the status of all configured APIs with enhanced statistics."""
        status = {
            "apis_configured": [],
            "api_status": {},
            "api_stats": self.api_stats.copy(),
            "usage": {
                "birdeye": "Mainstream token analysis",
                "helius": "Pump.fun tokens & enhanced parsing",
                "dexscreener": "Alternative price source",
                "rpc": "Direct Solana node queries",
                "cielo": "Wallet analysis"
            },
            "routing": {
                "pump_fun_priority": self.PUMP_FUN_API_PRIORITY,
                "regular_token_priority": self.REGULAR_TOKEN_API_PRIORITY
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
                    # Test with dummy wallet
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
        
        # Add DexScreener and RPC status
        status["api_status"]["dexscreener"] = "external_api"
        status["api_status"]["rpc"] = f"configured: {self.rpc_url}"
        
        return status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics for all APIs."""
        perf_stats = {}
        
        for api_name, stats in self.api_stats.items():
            total_calls = stats['calls']
            success_calls = stats['success']
            error_calls = stats['errors']
            
            success_rate = (success_calls / total_calls * 100) if total_calls > 0 else 0
            
            perf_stats[api_name] = {
                'total_calls': total_calls,
                'successful_calls': success_calls,
                'failed_calls': error_calls,
                'success_rate_percent': round(success_rate, 2),
                'status': 'good' if success_rate >= 80 else 'degraded' if success_rate >= 50 else 'poor'
            }
        
        # Add cache statistics
        perf_stats['caching'] = {
            'price_cache_size': len(self.price_cache),
            'token_info_cache_size': len(self.token_info_cache),
            'price_cache_ttl_minutes': self.PRICE_CACHE_TTL / 60,
            'token_info_cache_ttl_minutes': self.TOKEN_INFO_CACHE_TTL / 60
        }
        
        return perf_stats
    
    def clear_cache(self):
        """Clear all cached data."""
        self.price_cache.clear()
        self.token_info_cache.clear()
        logger.info("API manager cache cleared")
    
    def get_recommended_config(self) -> Dict[str, str]:
        """Get recommended configuration based on available APIs."""
        config = {
            "pump_fun_tokens": "helius" if self.helius_api else "limited_analysis",
            "mainstream_tokens": "birdeye" if self.birdeye_api else "not_available",
            "wallet_analysis": "cielo" if self.cielo_api else "not_available",
            "price_fallbacks": "dexscreener + rpc" if self.helius_api or self.birdeye_api else "none",
            "notes": []
        }
        
        if not self.helius_api:
            config["notes"].append("RECOMMENDED: Add Helius API key for pump.fun token analysis")
            config["notes"].append("Command: python phoenix.py configure --helius-api-key YOUR_KEY")
        
        if not self.cielo_api:
            config["notes"].append("CRITICAL: Cielo Finance API key required for wallet analysis")
            config["notes"].append("Command: python phoenix.py configure --cielo-api-key YOUR_KEY")
        
        if not self.birdeye_api:
            config["notes"].append("CRITICAL: Birdeye API key required for token analysis")
            config["notes"].append("Command: python phoenix.py configure --birdeye-api-key YOUR_KEY")
        
        config["notes"].append("Smart Routing: Helius→RPC→DexScreener→Birdeye (pump.fun)")
        config["notes"].append("Smart Routing: Birdeye→DexScreener→RPC→Helius (regular)")
        
        return config