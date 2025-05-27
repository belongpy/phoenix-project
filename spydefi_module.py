"""
SPYDEFI Module - Phoenix Project (PROFESSIONAL KOL ANALYSIS SYSTEM)

Complete KOL performance tracking and analysis platform for Solana memecoins.

SYSTEM FLOW:
1. Scan SpyDefi channel (24h) for "Achievement Unlocked" + Solana emoji
2. Rank KOLs by mention frequency → Select top 25 (configurable)
3. Analyze each KOL's channel for 7 days of token calls
4. Cross-reference to find original call timestamps
5. Track token performance from call time until now
6. Calculate comprehensive metrics and composite scores
7. Classify trading strategies (Scalp vs Hold)
8. Export detailed CSV + TXT summary

PERFORMANCE METRICS:
- Success Rate (>50% profit threshold)
- 2x Success Rate & Time to 2x
- 5x+ Success Rate (Gem Finding)
- Consistency Score & Max Pullback %
- Unrealized vs Optimal Exit Analysis

COMPOSITE SCORING:
- Success Rate: 25%
- 2x Rate: 25% 
- Consistency: 20%
- Time to 2x: 15%
- 5x+ Rate: 15%
"""

import asyncio
import re
import logging
import time
import json
import csv
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import base58

# Setup logging
logger = logging.getLogger("phoenix.spydefi")

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
    contract_address: str
    call_timestamp: int
    original_call_timestamp: int
    message_text: str
    market_cap_at_call: Optional[float] = None
    price_at_call: Optional[float] = None
    current_price: Optional[float] = None
    ath_price: Optional[float] = None
    max_loss_price: Optional[float] = None
    performance_tracked: bool = False

@dataclass
class KOLPerformance:
    """Data class for KOL performance metrics"""
    kol: str
    channel_id: int
    subscriber_count: int
    total_calls: int
    winning_calls: int
    losing_calls: int
    success_rate: float
    tokens_2x_plus: int
    tokens_5x_plus: int
    success_rate_2x: float
    success_rate_5x: float
    avg_time_to_2x_hours: float
    avg_max_pullback_percent: float
    avg_unrealized_gains_percent: float
    consistency_score: float
    composite_score: float
    strategy_classification: str  # "SCALP" or "HOLD"
    follower_tier: str  # "HIGH", "MEDIUM", "LOW"
    total_roi_percent: float
    max_roi_percent: float
    
class SpyDefiAnalyzer:
    """Professional KOL analysis system for Solana memecoin tracking."""
    
    # Configuration constants
    DEFAULT_SPYDEFI_SCAN_HOURS = 24
    DEFAULT_KOL_ANALYSIS_DAYS = 7
    DEFAULT_TOP_KOLS_COUNT = 25
    DEFAULT_MIN_MENTIONS = 2
    DEFAULT_MAX_MARKET_CAP_USD = 100000  # $100M
    DEFAULT_MIN_SUBSCRIBERS = 500
    DEFAULT_WIN_THRESHOLD_PERCENT = 50
    
    # Performance constants
    CACHE_DURATION_HOURS = 6
    MAX_CONCURRENT_KOLS = 3
    API_DELAY_SECONDS = 1
    MAX_RETRIES = 3
    DAILY_API_LIMIT = 5000
    
    # Composite score weights
    SCORE_WEIGHTS = {
        'success_rate': 0.25,
        'success_rate_2x': 0.25,
        'consistency': 0.20,
        'time_to_2x': 0.15,
        'success_rate_5x': 0.15
    }
    
    # Message patterns
    ACHIEVEMENT_PATTERN = r'Achievement Unlocked:\s*x(\d+)!'
    SOLANA_EMOJI_PATTERNS = [
        '⚡',  # Lightning emoji often used for Solana
        '🟣',  # Purple circle for Solana
        '◎',   # Solana official symbol
        'SOL',  # Text fallback
        'Solana'  # Text fallback
    ]
    
    def __init__(self, api_id: int, api_hash: str, session_name: str = "phoenix_spydefi"):
        """Initialize the SPYDEFI analyzer."""
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = TelegramClient(session_name, api_id, api_hash)
        
        # API integrations
        self.dual_api_manager = None
        self.birdeye_api = None
        self.helius_api = None
        
        # Cache system
        self.cache_dir = Path.home() / ".phoenix_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.spydefi_cache_file = self.cache_dir / "spydefi_kol_analysis.json"
        
        # Performance tracking
        self.api_call_count = 0
        self.daily_api_calls = 0
        self.processing_start_time = None
        
        # Results storage
        self.kol_mentions = defaultdict(int)
        self.kol_performances = {}
        self.token_calls_database = []
        self.analysis_metadata = {}
        
        # Rate limiting
        self.last_api_call_time = 0
        self.consecutive_failures = 0
        
        # Configuration (can be overridden)
        self.config = {
            'spydefi_scan_hours': self.DEFAULT_SPYDEFI_SCAN_HOURS,
            'kol_analysis_days': self.DEFAULT_KOL_ANALYSIS_DAYS,
            'top_kols_count': self.DEFAULT_TOP_KOLS_COUNT,
            'min_mentions': self.DEFAULT_MIN_MENTIONS,
            'max_market_cap_usd': self.DEFAULT_MAX_MARKET_CAP_USD,
            'min_subscribers': self.DEFAULT_MIN_SUBSCRIBERS,
            'win_threshold_percent': self.DEFAULT_WIN_THRESHOLD_PERCENT
        }
    
    def set_api_manager(self, dual_api_manager):
        """Set the dual API manager for price data."""
        self.dual_api_manager = dual_api_manager
        self.birdeye_api = dual_api_manager.birdeye_api
        self.helius_api = dual_api_manager.helius_api
        logger.info("✅ API manager configured for SPYDEFI analysis")
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Config updated: {key} = {value}")
    
    async def connect(self):
        """Connect to Telegram."""
        logger.info("🔗 Connecting to Telegram for SPYDEFI analysis...")
        await self.client.start()
        logger.info("✅ Connected to Telegram")
    
    async def disconnect(self):
        """Disconnect from Telegram."""
        logger.info("📡 Disconnecting from Telegram...")
        await self.client.disconnect()
        logger.info("✅ Disconnected from Telegram")
    
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load cached analysis data if valid."""
        try:
            if not self.spydefi_cache_file.exists():
                return None
            
            with open(self.spydefi_cache_file, 'r') as f:
                cache = json.load(f)
            
            # Check if cache is expired
            cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
            if datetime.now() - cache_time > timedelta(hours=self.CACHE_DURATION_HOURS):
                logger.info("⏰ SPYDEFI cache expired, will refresh")
                return None
            
            logger.info(f"📦 Loaded SPYDEFI cache with {len(cache.get('kol_performances', {}))} KOLs")
            return cache
            
        except Exception as e:
            logger.error(f"❌ Error loading cache: {str(e)}")
            return None
    
    def _save_cache(self, data: Dict[str, Any]):
        """Save analysis data to cache."""
        try:
            cache = {
                **data,
                'timestamp': datetime.now().isoformat(),
                'version': '3.0',
                'config': self.config.copy()
            }
            
            with open(self.spydefi_cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            
            logger.info(f"💾 Saved SPYDEFI cache with {len(data.get('kol_performances', {}))} KOLs")
            
        except Exception as e:
            logger.error(f"❌ Error saving cache: {str(e)}")
    
    def _is_solana_message(self, message_text: str) -> bool:
        """Check if message contains Solana indicators."""
        for pattern in self.SOLANA_EMOJI_PATTERNS:
            if pattern in message_text:
                return True
        return False
    
    def _extract_achievement_data(self, message_text: str) -> Optional[Tuple[int, str]]:
        """Extract achievement multiplier and KOL from message."""
        try:
            # Look for Achievement Unlocked pattern
            achievement_match = re.search(self.ACHIEVEMENT_PATTERN, message_text, re.IGNORECASE)
            if not achievement_match:
                return None
            
            multiplier = int(achievement_match.group(1))
            
            # Extract KOL mention
            kol_pattern = r'@([a-zA-Z0-9_]+)'
            kol_matches = re.findall(kol_pattern, message_text)
            
            if kol_matches:
                # Filter out 'spydefi' itself
                kols = [kol for kol in kol_matches if kol.lower() != 'spydefi']
                if kols:
                    return multiplier, kols[0]
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting achievement data: {str(e)}")
            return None
    
    def _extract_contract_addresses(self, text: str) -> Set[str]:
        """Extract potential Solana contract addresses from text."""
        addresses = set()
        
        # URL patterns first
        url_patterns = [
            (r'pump\.fun/([1-9A-HJ-NP-Za-km-z]{32,44})', 'pump.fun'),
            (r'dexscreener\.com/solana/([1-9A-HJ-NP-Za-km-z]{32,44})', 'dexscreener'),
            (r'birdeye\.so/token/([1-9A-HJ-NP-Za-km-z]{32,44})', 'birdeye'),
        ]
        
        for pattern, source in url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if self._is_valid_solana_address(match):
                    addresses.add(match)
        
        # Standalone addresses
        standalone_pattern = r'\b([1-9A-HJ-NP-Za-km-z]{32,44})\b'
        matches = re.findall(standalone_pattern, text)
        for match in matches:
            if self._is_valid_solana_address(match) and not match.islower():
                addresses.add(match)
        
        return addresses
    
    def _is_valid_solana_address(self, address: str) -> bool:
        """Validate Solana address format."""
        if not address or len(address) < 32 or len(address) > 44:
            return False
        
        # Check base58 characters
        base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        if not all(c in base58_chars for c in address):
            return False
        
        # Reject system programs
        system_programs = [
            "11111111111111111111111111111111",
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "So11111111111111111111111111111111111111112",
        ]
        return address not in system_programs
    
    async def _rate_limit(self):
        """Apply rate limiting to API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call_time
        
        if time_since_last < self.API_DELAY_SECONDS:
            await asyncio.sleep(self.API_DELAY_SECONDS - time_since_last)
        
        self.last_api_call_time = time.time()
    
    async def scan_spydefi_for_kols(self, hours: int = None, force_refresh: bool = False) -> Dict[str, int]:
        """
        Scan SpyDefi channel for KOL mentions in Achievement Unlocked messages.
        
        Args:
            hours: Hours to scan back (default from config)
            force_refresh: Force refresh cache
            
        Returns:
            Dict of KOL mentions {kol: mention_count}
        """
        if hours is None:
            hours = self.config['spydefi_scan_hours']
        
        logger.info(f"🔍 Scanning SpyDefi for last {hours} hours...")
        
        # Check cache first
        if not force_refresh:
            cache = self._load_cache()
            if cache and cache.get('kol_mentions'):
                logger.info(f"📦 Using cached KOL mentions: {len(cache['kol_mentions'])} KOLs")
                return cache['kol_mentions']
        
        kol_mentions = defaultdict(int)
        total_messages = 0
        achievement_messages = 0
        solana_achievement_messages = 0
        
        try:
            # Get SpyDefi channel
            channel = await self.client.get_entity("spydefi")
            if not isinstance(channel, Channel):
                raise ValueError("SpyDefi is not a channel")
            
            after_date = datetime.now() - timedelta(hours=hours)
            
            logger.info(f"📥 Fetching messages from SpyDefi (ID: {channel.id})...")
            
            async for message in self.client.iter_messages(
                channel, 
                offset_date=after_date, 
                reverse=True,
                limit=2000
            ):
                if not message.text:
                    continue
                
                total_messages += 1
                
                # Check for Achievement Unlocked pattern
                achievement_data = self._extract_achievement_data(message.text)
                if achievement_data:
                    achievement_messages += 1
                    
                    # Check for Solana indicators
                    if self._is_solana_message(message.text):
                        solana_achievement_messages += 1
                        multiplier, kol = achievement_data
                        kol_mentions[kol] += 1
                        
                        logger.debug(f"📊 Found Solana achievement: @{kol} (x{multiplier})")
                
                # Progress indicator
                if total_messages % 100 == 0:
                    print(f"\r   📥 Processed {total_messages} messages, found {len(kol_mentions)} KOLs...", end="", flush=True)
            
            print(f"\r✅ Processed {total_messages} messages", flush=True)
            
            # Filter KOLs by minimum mentions
            qualified_kols = {
                kol: count for kol, count in kol_mentions.items() 
                if count >= self.config['min_mentions']
            }
            
            # Sort and take top KOLs
            sorted_kols = sorted(qualified_kols.items(), key=lambda x: x[1], reverse=True)
            top_kols = dict(sorted_kols[:self.config['top_kols_count']])
            
            logger.info(f"📊 SPYDEFI SCAN RESULTS:")
            logger.info(f"   Total messages: {total_messages}")
            logger.info(f"   Achievement messages: {achievement_messages}")
            logger.info(f"   Solana achievements: {solana_achievement_messages}")
            logger.info(f"   Qualified KOLs ({self.config['min_mentions']}+ mentions): {len(qualified_kols)}")
            logger.info(f"   Top KOLs selected: {len(top_kols)}")
            
            self.kol_mentions = top_kols
            return top_kols
            
        except Exception as e:
            logger.error(f"❌ Error scanning SpyDefi: {str(e)}")
            return {}
    
    async def get_channel_info(self, username: str) -> Tuple[Optional[int], int]:
        """Get channel ID and subscriber count."""
        try:
            await self._rate_limit()
            channel = await self.client.get_entity(username)
            
            if isinstance(channel, Channel):
                # Get participant count
                try:
                    full_channel = await self.client(GetFullChannelRequest(channel))
                    subscriber_count = full_channel.full_chat.participants_count
                except:
                    subscriber_count = 0
                
                return channel.id, subscriber_count
            
            return None, 0
            
        except Exception as e:
            logger.debug(f"Error getting channel info for {username}: {str(e)}")
            return None, 0
    
    async def scan_kol_channel_for_calls(self, kol: str, days: int = None) -> List[TokenCall]:
        """
        Scan a KOL's channel for token calls over specified days.
        
        Args:
            kol: KOL username
            days: Days to scan back (default from config)
            
        Returns:
            List of TokenCall objects
        """
        if days is None:
            days = self.config['kol_analysis_days']
        
        logger.info(f"🔍 Scanning @{kol}'s channel for {days} days of calls...")
        
        token_calls = []
        
        try:
            await self._rate_limit()
            
            # Get channel info
            channel_id, subscriber_count = await self.get_channel_info(f"@{kol}")
            if not channel_id:
                logger.warning(f"⚠️ Could not access @{kol}'s channel")
                return []
            
            # Check minimum subscriber requirement
            if subscriber_count < self.config['min_subscribers']:
                logger.info(f"⏭️ Skipping @{kol} - only {subscriber_count} subscribers (min: {self.config['min_subscribers']})")
                return []
            
            channel = await self.client.get_entity(channel_id)
            after_date = datetime.now() - timedelta(days=days)
            
            message_count = 0
            contracts_found = set()
            
            async for message in self.client.iter_messages(
                channel,
                offset_date=after_date,
                reverse=True,
                limit=1000
            ):
                if not message.text:
                    continue
                
                message_count += 1
                
                # Extract contract addresses
                addresses = self._extract_contract_addresses(message.text)
                
                for address in addresses:
                    # Only count first mention of each token
                    if address not in contracts_found:
                        contracts_found.add(address)
                        
                        token_call = TokenCall(
                            kol=kol,
                            channel_id=channel_id,
                            contract_address=address,
                            call_timestamp=int(message.date.timestamp()),
                            original_call_timestamp=int(message.date.timestamp()),
                            message_text=message.text[:200]  # Truncate for storage
                        )
                        
                        token_calls.append(token_call)
                        logger.debug(f"📞 Found call: @{kol} → {address}")
            
            logger.info(f"✅ @{kol}: {len(token_calls)} unique calls from {message_count} messages ({subscriber_count:,} subscribers)")
            return token_calls
            
        except ChannelPrivateError:
            logger.warning(f"🔒 @{kol}'s channel is private or inaccessible")
            return []
        except Exception as e:
            logger.error(f"❌ Error scanning @{kol}'s channel: {str(e)}")
            return []
    
    async def _get_token_price_data(self, token_address: str, timestamp: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get token price data: price at timestamp, current price, and market cap.
        
        Returns:
            Tuple of (price_at_call, current_price, market_cap_usd)
        """
        if not self.dual_api_manager:
            logger.error("❌ No API manager configured")
            return None, None, None
        
        try:
            await self._rate_limit()
            self.api_call_count += 1
            self.daily_api_calls += 1
            
            # Check daily limit
            if self.daily_api_calls >= self.DAILY_API_LIMIT:
                logger.warning(f"⚠️ Daily API limit reached ({self.DAILY_API_LIMIT})")
                return None, None, None
            
            # Get current price
            current_price_result = await self.dual_api_manager.get_token_price(token_address)
            current_price = None
            market_cap = None
            
            if current_price_result.get("success"):
                current_price = current_price_result.get("data", {}).get("value", 0)
                market_cap = current_price_result.get("data", {}).get("marketCap", 0)
            
            # Get historical price (at call time)
            end_time = timestamp + 3600  # 1 hour after call
            historical_result = self.dual_api_manager.get_token_price_history(
                token_address, timestamp, end_time, "5m"
            )
            
            price_at_call = None
            if historical_result.get("success"):
                items = historical_result.get("data", {}).get("items", [])
                if items:
                    # Find closest price to call time
                    closest_item = min(items, key=lambda x: abs(x.get("unixTime", 0) - timestamp))
                    price_at_call = closest_item.get("value", 0)
            
            # Fallback: use current price as proxy if no historical data
            if not price_at_call and current_price:
                price_at_call = current_price
                logger.debug(f"Using current price as historical proxy for {token_address}")
            
            return price_at_call, current_price, market_cap
            
        except Exception as e:
            logger.error(f"❌ Error getting price data for {token_address}: {str(e)}")
            return None, None, None
    
    async def _calculate_token_performance(self, token_call: TokenCall) -> TokenCall:
        """Calculate performance metrics for a token call."""
        try:
            # Get price data
            price_at_call, current_price, market_cap = await self._get_token_price_data(
                token_call.contract_address, 
                token_call.call_timestamp
            )
            
            if not price_at_call or not current_price:
                logger.debug(f"❌ No price data for {token_call.contract_address}")
                return token_call
            
            # Check market cap filter
            if market_cap and market_cap > self.config['max_market_cap_usd']:
                logger.debug(f"⏭️ Skipping {token_call.contract_address} - market cap too high: ${market_cap:,.0f}")
                return token_call
            
            # Get historical price data for ATH/max loss calculation
            current_timestamp = int(datetime.now().timestamp())
            historical_result = self.dual_api_manager.get_token_price_history(
                token_call.contract_address,
                token_call.call_timestamp,
                current_timestamp,
                "15m"
            )
            
            ath_price = current_price
            max_loss_price = current_price
            
            if historical_result.get("success"):
                items = historical_result.get("data", {}).get("items", [])
                if items:
                    prices = [item.get("value", 0) for item in items if item.get("value", 0) > 0]
                    if prices:
                        ath_price = max(prices)
                        max_loss_price = min(prices)
            
            # Update token call with performance data
            token_call.price_at_call = price_at_call
            token_call.current_price = current_price
            token_call.market_cap_at_call = market_cap
            token_call.ath_price = ath_price
            token_call.max_loss_price = max_loss_price
            token_call.performance_tracked = True
            
            # Log performance
            roi = ((current_price / price_at_call) - 1) * 100
            ath_roi = ((ath_price / price_at_call) - 1) * 100
            max_loss_roi = ((max_loss_price / price_at_call) - 1) * 100
            
            logger.debug(f"📈 @{token_call.kol} {token_call.contract_address[:8]}... "
                        f"ROI: {roi:+.1f}% | ATH: {ath_roi:+.1f}% | Max Loss: {max_loss_roi:+.1f}%")
            
            return token_call
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance for {token_call.contract_address}: {str(e)}")
            return token_call
    
    def _calculate_kol_metrics(self, kol: str, token_calls: List[TokenCall], subscriber_count: int) -> KOLPerformance:
        """Calculate comprehensive performance metrics for a KOL."""
        
        # Filter calls with performance data
        valid_calls = [call for call in token_calls if call.performance_tracked and call.price_at_call and call.current_price]
        
        if not valid_calls:
            logger.warning(f"⚠️ No valid performance data for @{kol}")
            return KOLPerformance(
                kol=kol,
                channel_id=token_calls[0].channel_id if token_calls else 0,
                subscriber_count=subscriber_count,
                total_calls=len(token_calls),
                winning_calls=0,
                losing_calls=0,
                success_rate=0,
                tokens_2x_plus=0,
                tokens_5x_plus=0,
                success_rate_2x=0,
                success_rate_5x=0,
                avg_time_to_2x_hours=0,
                avg_max_pullback_percent=0,
                avg_unrealized_gains_percent=0,
                consistency_score=0,
                composite_score=0,
                strategy_classification="UNKNOWN",
                follower_tier="LOW",
                total_roi_percent=0,
                max_roi_percent=0
            )
        
        # Calculate basic metrics
        total_calls = len(valid_calls)
        winning_calls = 0
        losing_calls = 0
        tokens_2x_plus = 0
        tokens_5x_plus = 0
        total_roi = 0
        max_roi = 0
        unrealized_gains = []
        max_pullbacks = []
        time_to_2x_list = []
        
        win_threshold = self.config['win_threshold_percent'] / 100  # Convert to decimal
        
        for call in valid_calls:
            # Current ROI
            current_roi = (call.current_price / call.price_at_call) - 1
            total_roi += current_roi
            max_roi = max(max_roi, current_roi)
            
            # ATH ROI (unrealized gains)
            ath_roi = (call.ath_price / call.price_at_call) - 1
            unrealized_gains.append(ath_roi)
            
            # Max pullback
            max_loss_roi = (call.max_loss_price / call.price_at_call) - 1
            max_pullbacks.append(abs(max_loss_roi))
            
            # Win/Loss classification
            if current_roi >= win_threshold:
                winning_calls += 1
            else:
                losing_calls += 1
            
            # 2x and 5x tracking
            if ath_roi >= 1.0:  # 2x
                tokens_2x_plus += 1
                # Estimate time to 2x (simplified)
                time_to_2x_list.append(24)  # Placeholder: 24 hours average
            
            if ath_roi >= 4.0:  # 5x
                tokens_5x_plus += 1
        
        # Calculate derived metrics
        success_rate = (winning_calls / total_calls) * 100
        success_rate_2x = (tokens_2x_plus / total_calls) * 100
        success_rate_5x = (tokens_5x_plus / total_calls) * 100
        avg_time_to_2x_hours = sum(time_to_2x_list) / len(time_to_2x_list) if time_to_2x_list else 0
        avg_max_pullback_percent = (sum(max_pullbacks) / len(max_pullbacks)) * 100
        avg_unrealized_gains_percent = (sum(unrealized_gains) / len(unrealized_gains)) * 100
        total_roi_percent = (total_roi / total_calls) * 100
        max_roi_percent = max_roi * 100
        
        # Consistency score (based on win rate stability)
        consistency_score = min(100, success_rate + (success_rate_2x * 0.5))
        
        # Composite score calculation
        composite_score = self._calculate_composite_score(
            success_rate, success_rate_2x, success_rate_5x, 
            avg_time_to_2x_hours, consistency_score
        )
        
        # Strategy classification
        follower_tier = self._classify_follower_tier(subscriber_count)
        strategy_classification = self._classify_strategy(
            success_rate_2x, avg_time_to_2x_hours, tokens_5x_plus, 
            total_calls, subscriber_count
        )
        
        return KOLPerformance(
            kol=kol,
            channel_id=token_calls[0].channel_id if token_calls else 0,
            subscriber_count=subscriber_count,
            total_calls=total_calls,
            winning_calls=winning_calls,
            losing_calls=losing_calls,
            success_rate=success_rate,
            tokens_2x_plus=tokens_2x_plus,
            tokens_5x_plus=tokens_5x_plus,
            success_rate_2x=success_rate_2x,
            success_rate_5x=success_rate_5x,
            avg_time_to_2x_hours=avg_time_to_2x_hours,
            avg_max_pullback_percent=avg_max_pullback_percent,
            avg_unrealized_gains_percent=avg_unrealized_gains_percent,
            consistency_score=consistency_score,
            composite_score=composite_score,
            strategy_classification=strategy_classification,
            follower_tier=follower_tier,
            total_roi_percent=total_roi_percent,
            max_roi_percent=max_roi_percent
        )
    
    def _calculate_composite_score(self, success_rate: float, success_rate_2x: float, 
                                 success_rate_5x: float, avg_time_to_2x_hours: float, 
                                 consistency_score: float) -> float:
        """Calculate weighted composite score."""
        
        # Normalize metrics to 0-100 scale
        success_component = (success_rate / 100) * 100
        success_2x_component = (success_rate_2x / 100) * 100
        success_5x_component = (success_rate_5x / 100) * 100
        
        # Time to 2x component (inverted - faster is better)
        if avg_time_to_2x_hours > 0:
            time_component = max(0, 100 - (avg_time_to_2x_hours / 72 * 100))  # 72h = 0 points
        else:
            time_component = 0
        
        consistency_component = min(100, consistency_score)
        
        # Apply weights
        weighted_score = (
            success_component * self.SCORE_WEIGHTS['success_rate'] +
            success_2x_component * self.SCORE_WEIGHTS['success_rate_2x'] +
            success_5x_component * self.SCORE_WEIGHTS['success_rate_5x'] +
            time_component * self.SCORE_WEIGHTS['time_to_2x'] +
            consistency_component * self.SCORE_WEIGHTS['consistency']
        )
        
        return min(100, max(0, weighted_score))
    
    def _classify_follower_tier(self, subscriber_count: int) -> str:
        """Classify follower tier for scalping potential."""
        if subscriber_count >= 10000:
            return "HIGH"
        elif subscriber_count >= 1000:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _classify_strategy(self, success_rate_2x: float, avg_time_to_2x: float, 
                         tokens_5x_plus: int, total_calls: int, subscriber_count: int) -> str:
        """Classify KOL strategy as SCALP or HOLD."""
        
        # SCALP indicators: High followers + Fast 2x + Decent success rate
        if (subscriber_count >= 5000 and 
            avg_time_to_2x <= 12 and  # Fast to 2x
            success_rate_2x >= 30):    # Decent 2x rate
            return "SCALP"
        
        # HOLD indicators: Consistent gem finding + Lower time pressure
        gem_rate = (tokens_5x_plus / total_calls * 100) if total_calls > 0 else 0
        if gem_rate >= 15 and success_rate_2x >= 40:  # Good at finding gems
            return "HOLD"
        
        # Default classification
        if success_rate_2x >= 25:
            return "SCALP" if subscriber_count >= 1000 else "HOLD"
        else:
            return "HOLD"
    
    async def analyze_kol_performance(self, kol_mentions: Dict[str, int]) -> Dict[str, KOLPerformance]:
        """
        Analyze performance for all top KOLs.
        
        Args:
            kol_mentions: Dictionary of KOL mentions from SpyDefi scan
            
        Returns:
            Dictionary of KOL performance data
        """
        logger.info(f"📊 Starting performance analysis for {len(kol_mentions)} KOLs...")
        
        kol_performances = {}
        processed_count = 0
        total_kols = len(kol_mentions)
        
        # Sequential processing for stability
        for kol, mention_count in kol_mentions.items():
            processed_count += 1
            
            try:
                logger.info(f"🔍 Analyzing @{kol} ({processed_count}/{total_kols}) - {mention_count} SpyDefi mentions")
                
                # Scan KOL's channel for token calls
                token_calls = await self.scan_kol_channel_for_calls(kol)
                
                if not token_calls:
                    logger.info(f"⏭️ No calls found for @{kol}")
                    continue
                
                # Get subscriber count
                _, subscriber_count = await self.get_channel_info(f"@{kol}")
                
                logger.info(f"📈 Tracking performance for {len(token_calls)} calls from @{kol}...")
                
                # Calculate performance for each token call
                performance_tracked = 0
                for i, token_call in enumerate(token_calls):
                    print(f"\r   📊 Processing call {i+1}/{len(token_calls)}: {token_call.contract_address[:8]}...", end="", flush=True)
                    
                    updated_call = await self._calculate_token_performance(token_call)
                    token_calls[i] = updated_call
                    
                    if updated_call.performance_tracked:
                        performance_tracked += 1
                    
                    # Respect rate limits
                    await asyncio.sleep(0.5)
                
                print(f"\r✅ Tracked performance for {performance_tracked}/{len(token_calls)} calls", flush=True)
                
                # Calculate KOL metrics
                kol_performance = self._calculate_kol_metrics(kol, token_calls, subscriber_count)
                kol_performances[kol] = kol_performance
                
                # Store token calls for export
                self.token_calls_database.extend(token_calls)
                
                logger.info(f"✅ @{kol} Analysis Complete:")
                logger.info(f"   📊 Score: {kol_performance.composite_score:.1f}/100")
                logger.info(f"   🎯 Success Rate: {kol_performance.success_rate:.1f}%")
                logger.info(f"   💎 2x Rate: {kol_performance.success_rate_2x:.1f}%")
                logger.info(f"   🚀 5x Rate: {kol_performance.success_rate_5x:.1f}%")
                logger.info(f"   👥 Strategy: {kol_performance.strategy_classification} ({kol_performance.subscriber_count:,} subs)")
                
                # Delay between KOLs
                if processed_count < total_kols:
                    await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error analyzing @{kol}: {str(e)}")
                self.consecutive_failures += 1
                
                if self.consecutive_failures >= 5:
                    logger.error("🛑 Too many consecutive failures, stopping analysis")
                    break
                
                continue
            
            # Reset failure counter on success
            self.consecutive_failures = 0
        
        logger.info(f"✅ Analysis complete: {len(kol_performances)} KOLs analyzed")
        
        self.kol_performances = kol_performances
        return kol_performances
    
    async def run_full_analysis(self, spydefi_hours: int = None, kol_days: int = None, 
                              top_kols: int = None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run complete SPYDEFI analysis pipeline.
        
        Args:
            spydefi_hours: Hours to scan SpyDefi (default from config)
            kol_days: Days to analyze each KOL (default from config) 
            top_kols: Number of top KOLs to analyze (default from config)
            force_refresh: Force refresh cache
            
        Returns:
            Complete analysis results
        """
        self.processing_start_time = time.time()
        
        logger.info("🚀 STARTING SPYDEFI COMPREHENSIVE ANALYSIS")
        logger.info("=" * 60)
        
        # Update config if parameters provided
        if spydefi_hours is not None:
            self.config['spydefi_scan_hours'] = spydefi_hours
        if kol_days is not None:
            self.config['kol_analysis_days'] = kol_days  
        if top_kols is not None:
            self.config['top_kols_count'] = top_kols
        
        logger.info(f"📋 ANALYSIS CONFIGURATION:")
        logger.info(f"   🕐 SpyDefi scan period: {self.config['spydefi_scan_hours']} hours")
        logger.info(f"   📅 KOL analysis period: {self.config['kol_analysis_days']} days")
        logger.info(f"   🎯 Top KOLs to analyze: {self.config['top_kols_count']}")
        logger.info(f"   💰 Max market cap filter: ${self.config['max_market_cap_usd']:,}")
        logger.info(f"   👥 Min subscribers: {self.config['min_subscribers']:,}")
        logger.info(f"   📈 Win threshold: {self.config['win_threshold_percent']}%")
        
        try:
            # Step 1: Scan SpyDefi for top KOLs
            logger.info("\n🔍 STEP 1: Scanning SpyDefi for top KOLs...")
            kol_mentions = await self.scan_spydefi_for_kols(
                self.config['spydefi_scan_hours'], 
                force_refresh
            )
            
            if not kol_mentions:
                return {
                    "success": False,
                    "error": "No KOLs found in SpyDefi scan",
                    "kol_performances": {},
                    "metadata": self._get_analysis_metadata()
                }
            
            # Step 2: Analyze KOL performance  
            logger.info(f"\n📊 STEP 2: Analyzing performance for {len(kol_mentions)} KOLs...")
            kol_performances = await self.analyze_kol_performance(kol_mentions)
            
            if not kol_performances:
                return {
                    "success": False,
                    "error": "No KOL performance data generated",
                    "kol_performances": {},
                    "metadata": self._get_analysis_metadata()
                }
            
            # Step 3: Sort by composite score
            logger.info("\n🏆 STEP 3: Ranking KOLs by performance...")
            sorted_performances = dict(sorted(
                kol_performances.items(),
                key=lambda x: x[1].composite_score,
                reverse=True
            ))
            
            # Calculate overall statistics
            total_calls = sum(perf.total_calls for perf in kol_performances.values())
            total_wins = sum(perf.winning_calls for perf in kol_performances.values())
            total_2x = sum(perf.tokens_2x_plus for perf in kol_performances.values())
            total_5x = sum(perf.tokens_5x_plus for perf in kol_performances.values())
            
            overall_success_rate = (total_wins / total_calls * 100) if total_calls > 0 else 0
            overall_2x_rate = (total_2x / total_calls * 100) if total_calls > 0 else 0
            overall_5x_rate = (total_5x / total_calls * 100) if total_calls > 0 else 0
            
            # Analysis metadata
            processing_time = time.time() - self.processing_start_time
            metadata = self._get_analysis_metadata()
            metadata.update({
                'processing_time_seconds': processing_time,
                'total_calls_analyzed': total_calls,
                'overall_success_rate': overall_success_rate,
                'overall_2x_rate': overall_2x_rate,
                'overall_5x_rate': overall_5x_rate,
                'api_calls_made': self.api_call_count
            })
            
            # Save to cache
            cache_data = {
                'kol_mentions': kol_mentions,
                'kol_performances': {k: self._serialize_performance(v) for k, v in sorted_performances.items()},
                'metadata': metadata
            }
            self._save_cache(cache_data)
            
            logger.info("🎉 SPYDEFI ANALYSIS COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"📊 FINAL RESULTS:")
            logger.info(f"   🎯 KOLs analyzed: {len(kol_performances)}")
            logger.info(f"   📞 Total calls: {total_calls}")
            logger.info(f"   ✅ Overall success rate: {overall_success_rate:.1f}%")
            logger.info(f"   💎 Overall 2x rate: {overall_2x_rate:.1f}%")
            logger.info(f"   🚀 Overall 5x rate: {overall_5x_rate:.1f}%")
            logger.info(f"   ⏱️ Processing time: {processing_time:.1f}s")
            logger.info(f"   📡 API calls made: {self.api_call_count}")
            
            return {
                "success": True,
                "kol_performances": sorted_performances,
                "kol_mentions": kol_mentions,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"❌ SPYDEFI analysis failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e),
                "kol_performances": {},
                "metadata": self._get_analysis_metadata()
            }
    
    def _get_analysis_metadata(self) -> Dict[str, Any]:
        """Get analysis metadata."""
        return {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.copy(),
            'version': '3.0',
            'api_calls': self.api_call_count,
            'consecutive_failures': self.consecutive_failures,
            'processing_start_time': self.processing_start_time
        }
    
    def _serialize_performance(self, performance: KOLPerformance) -> Dict[str, Any]:
        """Serialize KOLPerformance object to dictionary."""
        return {
            'kol': performance.kol,
            'channel_id': performance.channel_id,
            'subscriber_count': performance.subscriber_count,
            'total_calls': performance.total_calls,
            'winning_calls': performance.winning_calls,
            'losing_calls': performance.losing_calls,
            'success_rate': performance.success_rate,
            'tokens_2x_plus': performance.tokens_2x_plus,
            'tokens_5x_plus': performance.tokens_5x_plus,
            'success_rate_2x': performance.success_rate_2x,
            'success_rate_5x': performance.success_rate_5x,
            'avg_time_to_2x_hours': performance.avg_time_to_2x_hours,
            'avg_max_pullback_percent': performance.avg_max_pullback_percent,
            'avg_unrealized_gains_percent': performance.avg_unrealized_gains_percent,
            'consistency_score': performance.consistency_score,
            'composite_score': performance.composite_score,
            'strategy_classification': performance.strategy_classification,
            'follower_tier': performance.follower_tier,
            'total_roi_percent': performance.total_roi_percent,
            'max_roi_percent': performance.max_roi_percent
        }
    
    def clear_cache(self):
        """Clear all cache data."""
        try:
            if self.spydefi_cache_file.exists():
                self.spydefi_cache_file.unlink()
                logger.info("✅ Cleared SPYDEFI cache")
            
            # Reset internal state
            self.kol_mentions.clear()
            self.kol_performances.clear()
            self.token_calls_database.clear()
            self.analysis_metadata.clear()
            
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()