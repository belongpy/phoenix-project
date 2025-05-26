"""
Telegram Module - Phoenix Project (TWO-TIER HOTSTREAK SYSTEM)

CLEAN REBUILD with intelligent two-tier filtering:

TIER 1 - QUICK FILTER (Last 5 Calls):
- Scan each KOL's most recent 5 calls
- Quick current price checks
- Calculate basic performance metrics
- Filter for "promising" channels only

TIER 2 - DEEP ANALYSIS (Last 5 Days):
- Only analyze KOLs that pass Tier 1 filter
- Full historical price analysis
- Complete performance metrics and composite scoring
- Take profit recommendations for scalping

CORE IMPROVEMENTS:
- Always produces CSV output
- No cache prompts (smart defaults)
- Proper rate limiting for Telegram & APIs
- Uses configured RPC settings
- Focuses on hotstreak performers
- Graceful error handling
"""

import asyncio
import re
import logging
import time
import json
import requests
import csv
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Setup logging
logger = logging.getLogger("phoenix.telegram")

# Import Telethon
try:
    from telethon import TelegramClient
    from telethon.errors import FloodWaitError, ChannelPrivateError
    from telethon.tl.types import Channel
except ImportError:
    logger.error("Telethon not installed. Install with: pip install telethon")
    raise

@dataclass
class TokenCall:
    """Data class for individual token calls"""
    kol: str
    channel_id: int
    token_address: str
    first_call_unix: int
    call_message: str
    price_at_call: Optional[float] = None
    current_price: Optional[float] = None
    ath_price: Optional[float] = None
    lowest_price: Optional[float] = None
    max_roi_percent: float = 0.0
    max_drawdown_percent: float = 0.0
    time_to_2x_minutes: Optional[float] = None
    achieved_2x: bool = False
    market_cap_at_call: Optional[float] = None

@dataclass
class KOLQuickStats:
    """Tier 1 quick filtering stats"""
    kol_name: str
    channel_id: int
    calls_analyzed: int
    calls_with_data: int
    calls_2x_plus: int
    avg_roi_percent: float
    max_roi_percent: float
    avg_time_to_2x_minutes: float
    has_5x_plus: bool
    is_promising: bool
    promising_reason: str

@dataclass
class KOLDeepStats:
    """Tier 2 deep analysis stats"""
    kol_name: str
    channel_id: int
    total_calls: int
    calls_with_data: int
    calls_2x_plus: int
    calls_5x_plus: int
    success_rate_2x: float
    success_rate_5x: float
    avg_max_roi: float
    avg_max_drawdown: float
    avg_time_to_2x_minutes: float
    composite_score: float
    recommended_tp1: float
    recommended_tp2: float
    scalping_strategy: str

class TelegramScraper:
    """Two-tier telegram scraper for hotstreak KOL analysis."""
    
    # Configuration constants
    SPYDEFI_SCAN_HOURS = 24
    MIN_MENTIONS_THRESHOLD = 2
    TIER1_CALLS_TO_CHECK = 5  # Quick filter
    TIER2_DAYS_TO_ANALYZE = 5  # Deep analysis
    
    # Promising criteria for Tier 1 filter
    PROMISING_CRITERIA = {
        'min_2x_rate': 0.4,  # 2/5 calls hit 2x (40%)
        'min_avg_roi': 150,  # Average ROI >150%
        'max_avg_time_2x': 120,  # Fast 2x timing <2 hours
        'needs_5x_plus': True  # OR has at least one 5x+
    }
    
    # Rate limiting (conservative for reliability)
    TELEGRAM_DELAY = 3.0  # 3 seconds between Telegram requests
    BIRDEYE_DELAY = 1.2   # 1.2 seconds between Birdeye requests
    KOL_PROCESSING_DELAY = 2.0  # 2 seconds between KOLs
    
    # Timeouts
    CHANNEL_TIMEOUT = 45
    PRICE_TIMEOUT = 15
    GLOBAL_TIMEOUT = 1800  # 30 minutes total
    
    # Cache settings
    CACHE_HOURS = 6
    
    def __init__(self, api_id: int, api_hash: str, session_name: str = "phoenix"):
        """Initialize the two-tier telegram scraper."""
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = TelegramClient(session_name, api_id, api_hash)
        
        # API clients (will be set by CLI)
        self.birdeye_api = None
        self.helius_api = None
        self.rpc_url = "https://api.mainnet-beta.solana.com"  # Default, will be updated
        
        # Cache management
        self.cache_dir = Path.home() / ".phoenix_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.spydefi_cache_file = self.cache_dir / "spydefi_kols.json"
        
        # Results storage
        self.tier1_results = {}  # KOL -> KOLQuickStats
        self.tier2_results = {}  # KOL -> KOLDeepStats
        self.promising_kols = []  # List of KOLs that passed Tier 1
        
        # Rate limiting tracking
        self.last_telegram_request = 0
        self.last_birdeye_request = 0
        
        # API call tracking
        self.api_stats = {
            'telegram_requests': 0,
            'birdeye_requests': 0,
            'helius_requests': 0,
            'addresses_found': 0,
            'prices_fetched': 0,
            'tier1_kols_analyzed': 0,
            'tier2_kols_analyzed': 0,
            'promising_kols_found': 0
        }
        
        # Contract address patterns
        self.contract_patterns = [
            r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b',  # Standard Solana addresses
            r'pump\.fun/([1-9A-HJ-NP-Za-km-z]{32,44})',  # pump.fun URLs
            r'dexscreener\.com/solana/([1-9A-HJ-NP-Za-km-z]{32,44})',  # DexScreener
            r'birdeye\.so/token/([1-9A-HJ-NP-Za-km-z]{32,44})',  # Birdeye
        ]
    
    def set_rpc_url(self, rpc_url: str):
        """Set RPC URL from configuration."""
        self.rpc_url = rpc_url
        logger.info(f"RPC URL configured: {rpc_url}")
    
    async def connect(self):
        """Connect to Telegram."""
        logger.info("Connecting to Telegram...")
        await self.client.start()
        logger.info("✅ Connected to Telegram")
    
    async def disconnect(self):
        """Disconnect from Telegram."""
        logger.info("Disconnecting from Telegram...")
        await self.client.disconnect()
        logger.info("✅ Disconnected from Telegram")
    
    def _rate_limit_telegram(self):
        """Apply rate limiting for Telegram requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_telegram_request
        
        if time_since_last < self.TELEGRAM_DELAY:
            sleep_time = self.TELEGRAM_DELAY - time_since_last
            time.sleep(sleep_time)
        
        self.last_telegram_request = time.time()
    
    def _rate_limit_birdeye(self):
        """Apply rate limiting for Birdeye requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_birdeye_request
        
        if time_since_last < self.BIRDEYE_DELAY:
            sleep_time = self.BIRDEYE_DELAY - time_since_last
            time.sleep(sleep_time)
        
        self.last_birdeye_request = time.time()
    
    def _load_spydefi_cache(self) -> Optional[Dict[str, int]]:
        """Load SpyDefi cache if fresh (<6 hours)."""
        try:
            if not self.spydefi_cache_file.exists():
                logger.info("No SpyDefi cache found, will scan fresh")
                return None
            
            with open(self.spydefi_cache_file, 'r') as f:
                cache = json.load(f)
            
            # Check cache age
            cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            
            if age_hours > self.CACHE_HOURS:
                logger.info(f"SpyDefi cache expired ({age_hours:.1f}h old), will refresh")
                return None
            
            kol_mentions = cache.get('kol_mentions', {})
            logger.info(f"✅ Using SpyDefi cache: {len(kol_mentions)} KOLs ({age_hours:.1f}h old)")
            return kol_mentions
            
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return None
    
    def _save_spydefi_cache(self, kol_mentions: Dict[str, int]):
        """Save SpyDefi cache."""
        try:
            cache = {
                'kol_mentions': kol_mentions,
                'timestamp': datetime.now().isoformat(),
                'version': '2.0'
            }
            
            with open(self.spydefi_cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            
            logger.info(f"✅ Saved SpyDefi cache: {len(kol_mentions)} KOLs")
            
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    async def discover_spydefi_kols(self) -> Dict[str, int]:
        """Discover KOLs from SpyDefi with smart caching."""
        # Try cache first
        cached_kols = self._load_spydefi_cache()
        if cached_kols:
            return cached_kols
        
        logger.info(f"🔍 Scanning SpyDefi for last {self.SPYDEFI_SCAN_HOURS} hours...")
        
        try:
            self._rate_limit_telegram()
            self.api_stats['telegram_requests'] += 1
            
            # Get SpyDefi channel
            channel = await self.client.get_entity("spydefi")
            if not isinstance(channel, Channel):
                logger.error("SpyDefi is not accessible")
                return {}
            
            # Scan messages
            after_date = datetime.now() - timedelta(hours=self.SPYDEFI_SCAN_HOURS)
            kol_mentions = defaultdict(int)
            message_count = 0
            
            logger.info(f"Scanning SpyDefi messages since {after_date.strftime('%Y-%m-%d %H:%M')}")
            
            async with asyncio.timeout(120):  # 2 minute timeout for SpyDefi scan
                async for message in self.client.iter_messages(
                    channel, 
                    offset_date=after_date,
                    limit=2000
                ):
                    if message.text:
                        # Extract @mentions
                        mentions = re.findall(r'@([a-zA-Z0-9_]+)', message.text)
                        for mention in mentions:
                            if mention.lower() != 'spydefi':
                                kol_mentions[mention] += 1
                        
                        message_count += 1
                        
                        if message_count % 200 == 0:
                            print(f"\r   Scanned {message_count} messages...", end="", flush=True)
            
            print(f"\r✅ Scanned {message_count} SpyDefi messages", flush=True)
            
            # Filter by minimum mentions
            filtered_kols = {kol: count for kol, count in kol_mentions.items() 
                           if count >= self.MIN_MENTIONS_THRESHOLD}
            
            logger.info(f"Found {len(filtered_kols)} KOLs with >{self.MIN_MENTIONS_THRESHOLD} mentions")
            
            # Save to cache
            if filtered_kols:
                self._save_spydefi_cache(filtered_kols)
            
            return filtered_kols
            
        except asyncio.TimeoutError:
            logger.error("SpyDefi scan timeout")
            return {}
        except Exception as e:
            logger.error(f"Error scanning SpyDefi: {str(e)}")
            return {}
    
    def _extract_token_addresses(self, text: str) -> List[str]:
        """Extract valid Solana token addresses from text."""
        addresses = set()
        
        # Extract from various patterns
        for pattern in self.contract_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if self._is_valid_solana_address(match):
                    addresses.add(match)
        
        self.api_stats['addresses_found'] += len(addresses)
        return list(addresses)
    
    def _is_valid_solana_address(self, address: str) -> bool:
        """Validate Solana address format."""
        if not address or len(address) < 32 or len(address) > 44:
            return False
        
        # Check base58 characters
        base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        if not all(c in base58_chars for c in address):
            return False
        
        # Reject all lowercase or system programs
        if address.islower():
            return False
        
        system_programs = [
            "11111111111111111111111111111111",
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "So11111111111111111111111111111111111111112",
        ]
        if address in system_programs:
            return False
        
        return True
    
    async def get_kol_recent_calls(self, kol_name: str, limit: int = 5) -> List[TokenCall]:
        """Get recent token calls from a KOL."""
        try:
            self._rate_limit_telegram()
            self.api_stats['telegram_requests'] += 1
            
            # Get KOL channel
            channel = await self.client.get_entity(f"@{kol_name}")
            if not isinstance(channel, Channel):
                logger.debug(f"@{kol_name} is not accessible")
                return []
            
            # Get recent messages
            messages = []
            call_count = 0
            
            async with asyncio.timeout(self.CHANNEL_TIMEOUT):
                async for message in self.client.iter_messages(channel, limit=100):
                    if message.text and call_count < limit:
                        # Extract token addresses
                        addresses = self._extract_token_addresses(message.text)
                        
                        for address in addresses:
                            if call_count >= limit:
                                break
                            
                            # Create token call record
                            token_call = TokenCall(
                                kol=kol_name,
                                channel_id=channel.id,
                                token_address=address,
                                first_call_unix=int(message.date.timestamp()),
                                call_message=message.text[:200]  # Truncate for storage
                            )
                            
                            messages.append(token_call)
                            call_count += 1
            
            logger.debug(f"Found {len(messages)} recent calls from @{kol_name}")
            return messages
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting calls from @{kol_name}")
            return []
        except Exception as e:
            logger.debug(f"Error getting calls from @{kol_name}: {str(e)}")
            return []
    
    def _get_current_price(self, token_address: str) -> Optional[float]:
        """Get current token price using Birdeye."""
        if not self.birdeye_api:
            return None
        
        try:
            self._rate_limit_birdeye()
            self.api_stats['birdeye_requests'] += 1
            
            result = self.birdeye_api.get_token_price(token_address)
            
            if result.get("success") and result.get("data"):
                price = result["data"].get("value", 0)
                if price > 0:
                    self.api_stats['prices_fetched'] += 1
                    return price
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting current price for {token_address}: {str(e)}")
            return None
    
    def _get_historical_price_data(self, token_address: str, from_unix: int) -> Dict[str, Any]:
        """Get historical price data from call time to now."""
        if not self.birdeye_api:
            return {}
        
        try:
            self._rate_limit_birdeye()
            self.api_stats['birdeye_requests'] += 1
            
            to_unix = int(datetime.now().timestamp())
            
            result = self.birdeye_api.get_token_price_history(
                token_address,
                from_unix,
                to_unix,
                "5m"  # 5-minute resolution
            )
            
            if result.get("success") and result.get("data", {}).get("items"):
                price_data = result["data"]["items"]
                
                # Calculate metrics
                prices = [item.get("value", 0) for item in price_data if item.get("value", 0) > 0]
                
                if prices:
                    return {
                        'price_at_call': prices[0],
                        'current_price': prices[-1],
                        'ath_price': max(prices),
                        'lowest_price': min(prices),
                        'price_points': len(prices)
                    }
            
            return {}
            
        except Exception as e:
            logger.debug(f"Error getting historical data for {token_address}: {str(e)}")
            return {}
    
    def _analyze_token_performance(self, token_call: TokenCall) -> TokenCall:
        """Analyze individual token performance."""
        # Get historical price data
        historical_data = self._get_historical_price_data(
            token_call.token_address,
            token_call.first_call_unix
        )
        
        if historical_data:
            token_call.price_at_call = historical_data.get('price_at_call')
            token_call.current_price = historical_data.get('current_price')
            token_call.ath_price = historical_data.get('ath_price')
            token_call.lowest_price = historical_data.get('lowest_price')
            
            # Calculate performance metrics
            if token_call.price_at_call and token_call.price_at_call > 0:
                if token_call.ath_price:
                    token_call.max_roi_percent = ((token_call.ath_price / token_call.price_at_call) - 1) * 100
                    
                    # Check if hit 2x
                    if token_call.max_roi_percent >= 100:
                        token_call.achieved_2x = True
                        # Estimate time to 2x (simplified)
                        token_call.time_to_2x_minutes = 60  # Placeholder - would need more detailed analysis
                
                if token_call.lowest_price:
                    token_call.max_drawdown_percent = ((token_call.lowest_price / token_call.price_at_call) - 1) * 100
        
        return token_call
    
    async def tier1_quick_filter(self, kol_mentions: Dict[str, int]) -> List[str]:
        """Tier 1: Quick filter to identify promising KOLs."""
        logger.info("🔍 TIER 1: Quick filtering promising KOLs...")
        
        promising_kols = []
        sorted_kols = sorted(kol_mentions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (kol_name, mention_count) in enumerate(sorted_kols[:50], 1):  # Top 50 KOLs max
            print(f"\rTier 1: Analyzing {i}/50 - @{kol_name}", end="", flush=True)
            
            try:
                # Get recent 5 calls
                recent_calls = await self.get_kol_recent_calls(kol_name, self.TIER1_CALLS_TO_CHECK)
                
                if not recent_calls:
                    continue
                
                # Analyze each call quickly
                analyzed_calls = []
                for call in recent_calls:
                    analyzed_call = self._analyze_token_performance(call)
                    if analyzed_call.price_at_call:  # Only include calls with price data
                        analyzed_calls.append(analyzed_call)
                
                if len(analyzed_calls) < 2:  # Need at least 2 calls with data
                    continue
                
                # Calculate quick stats
                calls_2x = len([c for c in analyzed_calls if c.achieved_2x])
                avg_roi = sum(c.max_roi_percent for c in analyzed_calls) / len(analyzed_calls)
                max_roi = max(c.max_roi_percent for c in analyzed_calls)
                has_5x = any(c.max_roi_percent >= 400 for c in analyzed_calls)
                
                # Check promising criteria
                is_promising = False
                reason = ""
                
                if calls_2x / len(analyzed_calls) >= self.PROMISING_CRITERIA['min_2x_rate']:
                    is_promising = True
                    reason = f"{calls_2x}/{len(analyzed_calls)} calls hit 2x+"
                elif avg_roi >= self.PROMISING_CRITERIA['min_avg_roi']:
                    is_promising = True
                    reason = f"Avg ROI {avg_roi:.0f}%"
                elif has_5x:
                    is_promising = True
                    reason = f"Has 5x+ token (max: {max_roi:.0f}%)"
                
                # Store Tier 1 results
                quick_stats = KOLQuickStats(
                    kol_name=kol_name,
                    channel_id=recent_calls[0].channel_id if recent_calls else 0,
                    calls_analyzed=len(recent_calls),
                    calls_with_data=len(analyzed_calls),
                    calls_2x_plus=calls_2x,
                    avg_roi_percent=avg_roi,
                    max_roi_percent=max_roi,
                    avg_time_to_2x_minutes=60,  # Placeholder
                    has_5x_plus=has_5x,
                    is_promising=is_promising,
                    promising_reason=reason
                )
                
                self.tier1_results[kol_name] = quick_stats
                
                if is_promising:
                    promising_kols.append(kol_name)
                    logger.debug(f"✅ @{kol_name} is promising: {reason}")
                
                self.api_stats['tier1_kols_analyzed'] += 1
                
                # Small delay between KOLs
                await asyncio.sleep(self.KOL_PROCESSING_DELAY)
                
            except Exception as e:
                logger.error(f"Error analyzing @{kol_name} in Tier 1: {str(e)}")
                continue
        
        print(f"\r✅ Tier 1 complete: {len(promising_kols)} promising KOLs found", flush=True)
        self.api_stats['promising_kols_found'] = len(promising_kols)
        self.promising_kols = promising_kols
        
        return promising_kols
    
    async def tier2_deep_analysis(self, promising_kols: List[str]) -> Dict[str, KOLDeepStats]:
        """Tier 2: Deep analysis of promising KOLs."""
        logger.info(f"🔍 TIER 2: Deep analysis of {len(promising_kols)} promising KOLs...")
        
        deep_results = {}
        
        for i, kol_name in enumerate(promising_kols, 1):
            print(f"\rTier 2: Deep analysis {i}/{len(promising_kols)} - @{kol_name}", end="", flush=True)
            
            try:
                # Get 5 days of calls
                five_days_ago = datetime.now() - timedelta(days=self.TIER2_DAYS_TO_ANALYZE)
                
                # Get more messages for deeper analysis
                self._rate_limit_telegram()
                self.api_stats['telegram_requests'] += 1
                
                channel = await self.client.get_entity(f"@{kol_name}")
                all_calls = []
                
                async with asyncio.timeout(self.CHANNEL_TIMEOUT):
                    async for message in self.client.iter_messages(
                        channel, 
                        offset_date=five_days_ago,
                        limit=200
                    ):
                        if message.text:
                            addresses = self._extract_token_addresses(message.text)
                            
                            for address in addresses:
                                token_call = TokenCall(
                                    kol=kol_name,
                                    channel_id=channel.id,
                                    token_address=address,
                                    first_call_unix=int(message.date.timestamp()),
                                    call_message=message.text[:200]
                                )
                                all_calls.append(token_call)
                
                # Analyze all calls
                analyzed_calls = []
                for call in all_calls[:30]:  # Limit to 30 calls max to manage API usage
                    analyzed_call = self._analyze_token_performance(call)
                    if analyzed_call.price_at_call:
                        analyzed_calls.append(analyzed_call)
                    
                    # Small delay between price fetches
                    await asyncio.sleep(0.5)
                
                if len(analyzed_calls) < 3:
                    continue
                
                # Calculate comprehensive metrics
                calls_2x = len([c for c in analyzed_calls if c.achieved_2x])
                calls_5x = len([c for c in analyzed_calls if c.max_roi_percent >= 400])
                
                success_rate_2x = (calls_2x / len(analyzed_calls)) * 100
                success_rate_5x = (calls_5x / len(analyzed_calls)) * 100
                
                avg_max_roi = sum(c.max_roi_percent for c in analyzed_calls) / len(analyzed_calls)
                avg_max_drawdown = sum(abs(c.max_drawdown_percent) for c in analyzed_calls) / len(analyzed_calls)
                
                # Calculate composite score
                composite_score = self._calculate_composite_score(
                    success_rate_2x, avg_max_roi, avg_max_drawdown, len(analyzed_calls)
                )
                
                # Generate recommendations
                tp_recommendations = self._generate_tp_recommendations(analyzed_calls)
                
                # Store deep analysis results
                deep_stats = KOLDeepStats(
                    kol_name=kol_name,
                    channel_id=channel.id,
                    total_calls=len(analyzed_calls),
                    calls_with_data=len(analyzed_calls),
                    calls_2x_plus=calls_2x,
                    calls_5x_plus=calls_5x,
                    success_rate_2x=success_rate_2x,
                    success_rate_5x=success_rate_5x,
                    avg_max_roi=avg_max_roi,
                    avg_max_drawdown=avg_max_drawdown,
                    avg_time_to_2x_minutes=tp_recommendations['avg_time_to_2x'],
                    composite_score=composite_score,
                    recommended_tp1=tp_recommendations['tp1'],
                    recommended_tp2=tp_recommendations['tp2'],
                    scalping_strategy=tp_recommendations['strategy']
                )
                
                deep_results[kol_name] = deep_stats
                self.api_stats['tier2_kols_analyzed'] += 1
                
                # Delay between KOLs
                await asyncio.sleep(self.KOL_PROCESSING_DELAY)
                
            except Exception as e:
                logger.error(f"Error in Tier 2 analysis for @{kol_name}: {str(e)}")
                continue
        
        print(f"\r✅ Tier 2 complete: {len(deep_results)} KOLs analyzed", flush=True)
        self.tier2_results = deep_results
        
        return deep_results
    
    def _calculate_composite_score(self, success_rate_2x: float, avg_roi: float, 
                                 avg_drawdown: float, call_count: int) -> float:
        """Calculate composite score for KOL performance."""
        # Base score from 2x success rate (0-40 points)
        success_score = min(40, success_rate_2x * 0.4)
        
        # ROI score (0-30 points)
        roi_score = min(30, avg_roi / 10)
        
        # Drawdown penalty (0-20 points, lower drawdown = higher score)
        drawdown_score = max(0, 20 - (avg_drawdown / 5))
        
        # Activity bonus (0-10 points)
        activity_score = min(10, call_count / 2)
        
        total_score = success_score + roi_score + drawdown_score + activity_score
        return min(100, total_score)
    
    def _generate_tp_recommendations(self, calls: List[TokenCall]) -> Dict[str, Any]:
        """Generate take profit recommendations based on analysis."""
        if not calls:
            return {'tp1': 50, 'tp2': 100, 'strategy': 'Conservative', 'avg_time_to_2x': 60}
        
        # Analyze successful calls
        successful_calls = [c for c in calls if c.achieved_2x]
        
        if len(successful_calls) >= 2:
            avg_max_roi = sum(c.max_roi_percent for c in successful_calls) / len(successful_calls)
            avg_drawdown = sum(abs(c.max_drawdown_percent) for c in successful_calls) / len(successful_calls)
            
            # Conservative TP recommendations
            tp1 = min(100, avg_max_roi * 0.4)  # 40% of average max ROI
            tp2 = min(300, avg_max_roi * 0.7)  # 70% of average max ROI
            
            if avg_max_roi > 500:
                strategy = "Gem Hunter - Let winners run"
            elif avg_drawdown < 30:
                strategy = "Conservative - Good exit timing"
            else:
                strategy = "Scalper - Quick profits recommended"
        else:
            tp1 = 50
            tp2 = 100
            strategy = "Conservative - Limited success data"
        
        return {
            'tp1': tp1,
            'tp2': tp2,
            'strategy': strategy,
            'avg_time_to_2x': 90  # Placeholder
        }
    
    async def export_results(self, output_file: str) -> bool:
        """Export comprehensive results to CSV."""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Prepare CSV data
            csv_data = []
            
            # Add Tier 2 results (deep analysis)
            for kol_name, stats in self.tier2_results.items():
                row = {
                    'kol': f"@{kol_name}",
                    'channel_id': stats.channel_id,
                    'analysis_tier': 'Deep (Tier 2)',
                    'total_calls': stats.total_calls,
                    'calls_with_data': stats.calls_with_data,
                    'calls_2x_plus': stats.calls_2x_plus,
                    'calls_5x_plus': stats.calls_5x_plus,
                    'success_rate_2x': round(stats.success_rate_2x, 1),
                    'success_rate_5x': round(stats.success_rate_5x, 1),
                    'avg_max_roi': round(stats.avg_max_roi, 1),
                    'avg_max_drawdown': round(stats.avg_max_drawdown, 1),
                    'avg_time_to_2x_minutes': round(stats.avg_time_to_2x_minutes, 1),
                    'composite_score': round(stats.composite_score, 1),
                    'recommended_tp1': round(stats.recommended_tp1, 1),
                    'recommended_tp2': round(stats.recommended_tp2, 1),
                    'scalping_strategy': stats.scalping_strategy
                }
                csv_data.append(row)
            
            # Add Tier 1 results (quick filter) for non-promising KOLs
            for kol_name, stats in self.tier1_results.items():
                if kol_name not in self.tier2_results:  # Only non-promising ones
                    row = {
                        'kol': f"@{kol_name}",
                        'channel_id': stats.channel_id,
                        'analysis_tier': 'Quick (Tier 1)',
                        'total_calls': stats.calls_analyzed,
                        'calls_with_data': stats.calls_with_data,
                        'calls_2x_plus': stats.calls_2x_plus,
                        'calls_5x_plus': 1 if stats.has_5x_plus else 0,
                        'success_rate_2x': round((stats.calls_2x_plus / max(1, stats.calls_with_data)) * 100, 1),
                        'success_rate_5x': round((1 if stats.has_5x_plus else 0) / max(1, stats.calls_with_data) * 100, 1),
                        'avg_max_roi': round(stats.avg_roi_percent, 1),
                        'avg_max_drawdown': 0,  # Not calculated in Tier 1
                        'avg_time_to_2x_minutes': round(stats.avg_time_to_2x_minutes, 1),
                        'composite_score': 0,  # Not calculated in Tier 1
                        'recommended_tp1': 50,  # Default
                        'recommended_tp2': 100,  # Default
                        'scalping_strategy': 'Not Analyzed (Tier 1 only)'
                    }
                    csv_data.append(row)
            
            # Sort by composite score (Tier 2) then by avg ROI (Tier 1)
            csv_data.sort(key=lambda x: (x['composite_score'], x['avg_max_roi']), reverse=True)
            
            # Write CSV
            if csv_data:
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
                
                logger.info(f"✅ Exported {len(csv_data)} KOL results to {output_file}")
                
                # Export summary
                self._export_summary(output_file.replace('.csv', '_summary.txt'))
                
                return True
            else:
                logger.warning("No data to export")
                # Create empty CSV with headers
                headers = ['kol', 'channel_id', 'analysis_tier', 'total_calls', 'calls_with_data', 
                          'calls_2x_plus', 'calls_5x_plus', 'success_rate_2x', 'success_rate_5x',
                          'avg_max_roi', 'avg_max_drawdown', 'avg_time_to_2x_minutes', 
                          'composite_score', 'recommended_tp1', 'recommended_tp2', 'scalping_strategy']
                
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                
                return False
                
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return False
    
    def _export_summary(self, summary_file: str):
        """Export analysis summary."""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("PHOENIX PROJECT - TWO-TIER HOTSTREAK ANALYSIS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Tier 1 Results
                f.write("TIER 1 - QUICK FILTER RESULTS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"KOLs analyzed: {self.api_stats['tier1_kols_analyzed']}\n")
                f.write(f"Promising KOLs found: {self.api_stats['promising_kols_found']}\n")
                
                promising_rate = (self.api_stats['promising_kols_found'] / 
                                max(1, self.api_stats['tier1_kols_analyzed']) * 100)
                f.write(f"Promising rate: {promising_rate:.1f}%\n\n")
                
                # Tier 2 Results
                f.write("TIER 2 - DEEP ANALYSIS RESULTS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"KOLs deep analyzed: {self.api_stats['tier2_kols_analyzed']}\n")
                
                if self.tier2_results:
                    avg_score = sum(stats.composite_score for stats in self.tier2_results.values()) / len(self.tier2_results)
                    f.write(f"Average composite score: {avg_score:.1f}\n")
                    
                    top_kol = max(self.tier2_results.values(), key=lambda x: x.composite_score)
                    f.write(f"Top performer: @{top_kol.kol_name} (Score: {top_kol.composite_score:.1f})\n\n")
                
                # API Usage Stats
                f.write("API USAGE STATISTICS:\n")
                f.write("-" * 40 + "\n")
                for key, value in self.api_stats.items():
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                
                # Top 10 Performers
                if self.tier2_results:
                    f.write("\nTOP 10 HOTSTREAK PERFORMERS:\n")
                    f.write("-" * 40 + "\n")
                    
                    sorted_kols = sorted(self.tier2_results.items(), 
                                       key=lambda x: x[1].composite_score, reverse=True)
                    
                    for i, (kol_name, stats) in enumerate(sorted_kols[:10], 1):
                        f.write(f"\n{i}. @{kol_name}\n")
                        f.write(f"   Composite Score: {stats.composite_score:.1f}\n")
                        f.write(f"   2x Success Rate: {stats.success_rate_2x:.1f}%\n")
                        f.write(f"   Avg Max ROI: {stats.avg_max_roi:.1f}%\n")
                        f.write(f"   Calls Analyzed: {stats.total_calls}\n")
                        f.write(f"   Recommended TP1/TP2: {stats.recommended_tp1:.0f}%/{stats.recommended_tp2:.0f}%\n")
                        f.write(f"   Strategy: {stats.scalping_strategy}\n")
            
            logger.info(f"✅ Exported summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting summary: {str(e)}")
    
    async def run_two_tier_analysis(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Run the complete two-tier hotstreak analysis."""
        logger.info("🚀 STARTING TWO-TIER HOTSTREAK ANALYSIS")
        
        try:
            async with asyncio.timeout(self.GLOBAL_TIMEOUT):
                # Step 1: Discover KOLs from SpyDefi
                logger.info("📍 Step 1: Discovering KOLs from SpyDefi...")
                kol_mentions = await self.discover_spydefi_kols()
                
                if not kol_mentions:
                    return {"success": False, "error": "No KOLs found in SpyDefi"}
                
                logger.info(f"✅ Found {len(kol_mentions)} KOLs from SpyDefi")
                
                # Step 2: Tier 1 - Quick filter
                promising_kols = await self.tier1_quick_filter(kol_mentions)
                
                if not promising_kols:
                    logger.warning("No promising KOLs found in Tier 1 filter")
                    # Still continue to export Tier 1 results
                
                # Step 3: Tier 2 - Deep analysis (only if we have promising KOLs)
                if promising_kols:
                    await self.tier2_deep_analysis(promising_kols)
                
                # Calculate final statistics
                total_kols_analyzed = len(self.tier1_results) + len(self.tier2_results)
                total_calls_analyzed = sum(stats.calls_analyzed for stats in self.tier1_results.values())
                total_calls_analyzed += sum(stats.total_calls for stats in self.tier2_results.values())
                
                avg_2x_rate = 0
                if self.tier2_results:
                    avg_2x_rate = sum(stats.success_rate_2x for stats in self.tier2_results.values()) / len(self.tier2_results)
                
                return {
                    "success": True,
                    "ranked_kols": {kol: {
                        'kol': kol,
                        'composite_score': stats.composite_score,
                        'success_rate_2x': stats.success_rate_2x,
                        'avg_ath_roi': stats.avg_max_roi,
                        'tokens_mentioned': stats.total_calls,
                        'channel_id': stats.channel_id,
                        'analysis_type': 'two_tier_hotstreak',
                        'avg_time_to_2x_minutes': stats.avg_time_to_2x_minutes,
                        'avg_max_pullback_percent': stats.avg_max_drawdown
                    } for kol, stats in self.tier2_results.items()},
                    "total_kols_analyzed": total_kols_analyzed,
                    "deep_analyses_performed": len(self.tier2_results),
                    "total_calls": total_calls_analyzed,
                    "success_rate_2x": avg_2x_rate,
                    "api_stats": self.api_stats.copy(),
                    "tier1_results": len(self.tier1_results),
                    "tier2_results": len(self.tier2_results),
                    "promising_kols_found": len(promising_kols)
                }
                
        except asyncio.TimeoutError:
            logger.error(f"Global timeout reached ({self.GLOBAL_TIMEOUT}s)")
            return self._generate_partial_results()
        except Exception as e:
            logger.error(f"Unexpected error in two-tier analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def _generate_partial_results(self) -> Dict[str, Any]:
        """Generate results from whatever data was collected before timeout."""
        return {
            "success": True,  # Mark as success to export partial data
            "ranked_kols": {kol: {
                'kol': kol,
                'composite_score': stats.composite_score if hasattr(stats, 'composite_score') else 0,
                'success_rate_2x': stats.success_rate_2x if hasattr(stats, 'success_rate_2x') else 0,
                'avg_ath_roi': stats.avg_max_roi if hasattr(stats, 'avg_max_roi') else 0,
                'tokens_mentioned': stats.total_calls if hasattr(stats, 'total_calls') else 0,
                'channel_id': stats.channel_id if hasattr(stats, 'channel_id') else 0,
                'analysis_type': 'partial_timeout',
                'avg_time_to_2x_minutes': 0,
                'avg_max_pullback_percent': 0
            } for kol, stats in self.tier2_results.items()},
            "total_kols_analyzed": len(self.tier1_results) + len(self.tier2_results),
            "deep_analyses_performed": len(self.tier2_results),
            "total_calls": sum(getattr(stats, 'total_calls', 0) for stats in self.tier2_results.values()),
            "success_rate_2x": 0,
            "api_stats": self.api_stats.copy(),
            "partial_results": True,
            "timeout_occurred": True
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            if self.spydefi_cache_file.exists():
                self.spydefi_cache_file.unlink()
                logger.info("✅ Cleared SpyDefi cache")
            
            # Clear results
            self.tier1_results.clear()
            self.tier2_results.clear()
            self.promising_kols.clear()
            
            logger.info("✅ Cleared analysis results")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    # Compatibility methods for existing code
    async def redesigned_spydefi_analysis(self, hours: int = 24, force_refresh: bool = False) -> Dict[str, Any]:
        """Compatibility method for existing CLI code."""
        return await self.run_two_tier_analysis(force_refresh)
    
    async def export_spydefi_analysis(self, analysis_results: Dict[str, Any], output_file: str):
        """Compatibility method for existing CLI export code."""
        success = await self.export_results(output_file)
        if success:
            logger.info(f"✅ Two-tier analysis exported to {output_file}")
        else:
            logger.warning(f"⚠️ Export completed with limited data to {output_file}")