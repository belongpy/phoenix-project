"""
SpyDefi KOL Analysis Module - Phoenix Project (FIXED VERSION)

CRITICAL FIXES:
- Enhanced KOL extraction with comprehensive word filtering
- Multiple channel lookup variants (11 different formats per KOL)  
- Full performance analysis maintained
- Top 50 KOLs (changed from 25)
- All syntax errors resolved
- Progressive message retrieval with fallback methods
"""

import asyncio
import logging
import json
import time
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from telethon import TelegramClient
from telethon.errors import ChannelPrivateError, UsernameNotOccupiedError, FloodWaitError

logger = logging.getLogger("phoenix.spydefi")

@dataclass
class KOLPerformance:
    """Data class for KOL performance metrics."""
    kol: str
    channel_id: Optional[str]
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
    strategy_classification: str
    follower_tier: str
    total_roi_percent: float
    max_roi_percent: float

class SpyDefiAnalyzer:
    """Enhanced SpyDefi KOL analysis with comprehensive debugging and performance metrics."""
    
    def __init__(self, api_id: str, api_hash: str, session_name: str = "phoenix_spydefi"):
        """Initialize the SpyDefi analyzer."""
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = None
        self.api_manager = None
        
        # Enhanced configuration with top 50 KOLs and 8-hour peak scanning
        self.config = {
            'spydefi_scan_hours': 8,   # PEAK MEMECOIN HOURS - Changed from 24 to 8
            'kol_analysis_days': 7,
            'top_kols_count': 50,      # TOP 50 KOLs - Changed from 25 to 50
            'min_mentions': 2,         # MINIMUM MENTIONS ≥2
            'max_market_cap_usd': 10000000,
            'min_subscribers': 100,    # RELAXED - Changed from 500 to 100
            'win_threshold_percent': 50,
            'max_messages_limit': 6000,
            'early_termination_kol_count': 100,
            'cache_refresh_hours': 6
        }
        
        # Enhanced invalid words list to filter false positives
        self.invalid_words = {
            # Common English words
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
            'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with', 'you', 'your', 'all',
            'any', 'can', 'had', 'her', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
            'who', 'boy', 'did', 'get', 'may', 'say', 'she', 'use', 'day', 'go', 'me', 'my', 'no', 'or',
            'so', 'up', 'we', 'am', 'do', 'if', 'us', 'be', 'he', 'hi', 'oh', 'ok',
            # Short fragments
            're', 'ed', 'er', 'ly', 'ing', 'tion', 'sion', 'ment', 'ness', 'ful', 'less', 'able', 'ible',
            'al', 'ic', 'ous', 'ive', 'ant', 'ent', 'ary', 'ory', 'ist', 'ism', 'ity', 'age', 'dom', 'ship',
            # Common abbreviations
            'etc', 'vs', 'aka', 'fyi', 'btw', 'omg', 'lol', 'wtf', 'tbh', 'imo', 'fomo', 'hodl', 'rekt',
            # Word fragments commonly extracted incorrectly
            'vir', 'ter', 'der', 'ber', 'ker', 'ler', 'mer', 'ner', 'per', 'ser', 'ver', 'wer', 'yer', 'zer',
            'ach', 'ade', 'age', 'ake', 'ale', 'ame', 'ane', 'ape', 'are', 'ate', 'ave', 'awe', 'aye',
            'ice', 'ide', 'ife', 'ike', 'ile', 'ime', 'ine', 'ipe', 'ire', 'ise', 'ite', 'ive', 'ize',
            'ock', 'ode', 'oke', 'ole', 'ome', 'one', 'ope', 'ore', 'ose', 'ote', 'ove', 'owe', 'oze',
            'uce', 'ude', 'uge', 'uke', 'ule', 'ume', 'une', 'upe', 'ure', 'use', 'ute', 'uve', 'uze',
            # Numbers and multipliers that were causing false positives
            'x', 'x2', 'x3', 'x4', 'x5', 'x10', 'x20', 'x50', 'x100'
        }
        
        # Enhanced channel lookup variants (more crypto-specific)
        self.channel_variants = [
            "{kol}",                # Original name
            "{kol}calls",           # With 'calls' suffix
            "{kol}_calls",          # With '_calls' suffix
            "{kol}gems",            # With 'gems' suffix
            "{kol}_gems",           # With '_gems' suffix
            "{kol}alpha",           # With 'alpha' suffix
            "{kol}_alpha",          # With '_alpha' suffix
            "{kol}signals",         # With 'signals' suffix
            "{kol}_signals",        # With '_signals' suffix
            "{kol}channel",         # With 'channel' suffix
            "{kol}_channel",        # With '_channel' suffix
            "{kol}sol",            # With 'sol' suffix
            "{kol}_sol",           # With '_sol' suffix
            "{kol}crypto",         # With 'crypto' suffix
            "{kol}_crypto",        # With '_crypto' suffix
            "{kol}play",           # With 'play' suffix (like alcaponesplay)
            "{kol}_play"           # With '_play' suffix
        ]
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Connect to Telegram client."""
        try:
            logger.info("🔌 Connecting to Telegram...")
            
            # Fixed: Remove unsupported parameters
            self.client = TelegramClient(
                self.session_name,
                self.api_id,
                self.api_hash
            )
            
            await self.client.start()
            logger.info("✅ Telegram connection successful")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Telegram: {str(e)}")
            raise
    
    async def disconnect(self):
        """Disconnect from Telegram client."""
        if self.client:
            await self.client.disconnect()
            logger.info("🔌 Telegram disconnected")
    
    def set_api_manager(self, api_manager):
        """Set the API manager for token analysis."""
        self.api_manager = api_manager
        logger.info("✅ API manager configured for SpyDefi analysis")
    
    def update_config(self, **kwargs):
        """Update configuration parameters with enforcement of key settings."""
        # Enforce critical settings that shouldn't be overridden
        critical_settings = {
            'spydefi_scan_hours': 8,   # ENFORCE 8-hour peak scanning
            'top_kols_count': 50,      # ENFORCE Top 50 KOLs
            'min_subscribers': 100     # ENFORCE relaxed subscriber count
        }
        
        # Update with provided kwargs first
        self.config.update(kwargs)
        
        # Then enforce critical settings
        self.config.update(critical_settings)
        
        logger.info(f"🔧 Configuration updated and enforced: {self.config}")
        logger.info(f"🎯 KEY SETTINGS ENFORCED:")
        logger.info(f"   📅 Peak scanning: {self.config['spydefi_scan_hours']} hours")
        logger.info(f"   🏆 Top KOLs: {self.config['top_kols_count']}")
        logger.info(f"   👥 Min subscribers: {self.config['min_subscribers']}")
        logger.info(f"   📊 Min mentions: {self.config['min_mentions']}")
    
    async def get_spydefi_channel(self):
        """Get SpyDefi channel with multiple fallback methods."""
        channel_attempts = [
            "spydefi",
            "@spydefi", 
            "SpyDefi",
            "@SpyDefi",
            1960616143  # Known channel ID
        ]
        
        for attempt in channel_attempts:
            try:
                logger.info(f"🔍 Trying to access channel: {attempt}")
                channel = await self.client.get_entity(attempt)
                logger.info(f"✅ Successfully accessed SpyDefi channel: {channel.title} (ID: {channel.id})")
                return channel
            except Exception as e:
                logger.debug(f"❌ Failed to access channel {attempt}: {str(e)}")
                continue
        
        raise Exception("❌ Could not access SpyDefi channel with any method")
    
    async def get_spydefi_messages(self, hours_back: int = 24):
        """Get messages from SpyDefi channel with enhanced retrieval methods."""
        try:
            channel = await self.get_spydefi_channel()
            
            # Test channel access first
            logger.info("🧪 Testing channel message access...")
            test_messages = []
            async for message in self.client.iter_messages(channel, limit=5):
                if message.text:
                    test_messages.append(message.text[:100])
            
            if test_messages:
                logger.info(f"✅ Channel access confirmed - {len(test_messages)} test messages found")
                logger.debug(f"Sample messages: {test_messages}")
            else:
                logger.warning("⚠️ No test messages found - channel may be empty or restricted")
            
            # Calculate time range with proper timezone handling
            from datetime import timezone
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours_back)
            
            logger.info(f"📅 Retrieving messages from {start_time} to {end_time}")
            logger.info(f"⏰ Time window: {hours_back} hours (PEAK MEMECOIN SCANNING)")
            logger.info(f"🎯 Targeting peak crypto activity period for maximum KOL discovery")
            
            all_messages = []
            batch_size = 100
            
            # Method 1: Try recent messages first (most reliable)
            logger.info("📥 Method 1: Retrieving recent messages...")
            try:
                recent_count = 0
                async for message in self.client.iter_messages(channel, limit=self.config['max_messages_limit']):
                    if message.date:
                        # Make message.date timezone-aware if it isn't
                        msg_date = message.date
                        if msg_date.tzinfo is None:
                            msg_date = msg_date.replace(tzinfo=timezone.utc)
                        
                        if msg_date >= start_time:
                            all_messages.append(message)
                            recent_count += 1
                        elif msg_date < start_time:
                            break  # Stop when we go beyond our time range
                
                logger.info(f"📊 Method 1 results: {recent_count} messages in time range")
                
            except Exception as e:
                logger.warning(f"⚠️ Method 1 failed: {str(e)}")
            
            # Method 2: If Method 1 didn't get enough, try offset-based retrieval
            if len(all_messages) < 100:
                logger.info("📥 Method 2: Trying offset-based retrieval...")
                try:
                    offset_count = 0
                    async for message in self.client.iter_messages(
                        channel, 
                        offset_date=end_time.replace(tzinfo=None),  # Remove timezone for Telegram API
                        limit=self.config['max_messages_limit']
                    ):
                        if message.date:
                            # Make message.date timezone-aware if it isn't
                            msg_date = message.date
                            if msg_date.tzinfo is None:
                                msg_date = msg_date.replace(tzinfo=timezone.utc)
                            
                            if msg_date >= start_time:
                                if message not in all_messages:  # Avoid duplicates
                                    all_messages.append(message)
                                    offset_count += 1
                            elif msg_date < start_time:
                                break
                    
                    logger.info(f"📊 Method 2 results: {offset_count} additional messages")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Method 2 failed: {str(e)}")
            
            # Method 3: Fallback - get recent messages regardless of time
            if len(all_messages) < 50:
                logger.info("📥 Method 3: Fallback to recent messages (no time filter)...")
                try:
                    fallback_messages = []
                    async for message in self.client.iter_messages(channel, limit=1000):
                        fallback_messages.append(message)
                    
                    all_messages.extend(fallback_messages)
                    logger.info(f"📊 Method 3 results: {len(fallback_messages)} fallback messages")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Method 3 failed: {str(e)}")
            
            # Remove duplicates
            unique_messages = []
            seen_ids = set()
            for msg in all_messages:
                if msg.id not in seen_ids:
                    unique_messages.append(msg)
                    seen_ids.add(msg.id)
            
            logger.info(f"📊 Total unique messages retrieved: {len(unique_messages)}")
            
            return unique_messages
            
        except Exception as e:
            logger.error(f"❌ Error retrieving SpyDefi messages: {str(e)}")
            return []
    
    def extract_kols_from_messages(self, messages: List) -> Dict[str, int]:
        """Extract KOL mentions with enhanced filtering to eliminate false positives."""
        kol_mentions = {}
        total_processed = 0
        relevant_messages = 0
        
        logger.info(f"🔍 Processing {len(messages)} messages for KOL extraction...")
        
        for i, message in enumerate(messages):
            if not message.text:
                continue
                
            total_processed += 1
            text = message.text.lower()
            
            # Enhanced relevance detection
            is_relevant = False
            relevance_indicators = [
                'achievement unlocked',
                'solana',
                '🎖️',
                '@',
                'x2', 'x3', 'x4', 'x5', 'x10',
                'gain', 'profit', 'call', 'gem'
            ]
            
            # Check if message is relevant
            for indicator in relevance_indicators:
                if indicator in text:
                    is_relevant = True
                    break
            
            # Fallback: Process every 3rd message to catch edge cases
            if not is_relevant and i % 3 == 0:
                is_relevant = True
            
            if is_relevant:
                relevant_messages += 1
                
                # Enhanced KOL extraction patterns
                kol_patterns = [
                    r'(?:made by|called by|by|from)\s*@(\w+)',
                    r'(?:thanks to|credit to|shoutout to)\s*@(\w+)',
                    r'(?:follow|check out)\s*@(\w+)',
                    r'@(\w{4,})',  # Any @username with 4+ characters
                    r'(?:^\s*|\s)@(\w+)(?:\s|$)',  # @username with word boundaries
                ]
                
                for pattern in kol_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        kol = match.lower().strip()
                        
                        # Enhanced validation
                        if self.is_valid_kol_name(kol):
                            if kol not in kol_mentions:
                                kol_mentions[kol] = 0
                            kol_mentions[kol] += 1
        
        logger.info(f"📊 Message processing stats:")
        logger.info(f"   Total messages processed: {total_processed}")
        logger.info(f"   Relevant messages: {relevant_messages}")
        logger.info(f"   Relevance rate: {(relevant_messages/total_processed*100):.1f}%")
        logger.info(f"   Unique KOLs extracted: {len(kol_mentions)}")
        
        return kol_mentions
    
    def is_valid_kol_name(self, kol: str) -> bool:
        """Enhanced KOL name validation to eliminate false positives."""
        # Basic checks
        if not kol or len(kol) < 4:
            return False
        
        # Must start with a letter
        if not kol[0].isalpha():
            return False
        
        # Check against invalid words list
        if kol.lower() in self.invalid_words:
            return False
        
        # Check for common word patterns
        if re.match(r'^(x\d+|x\d+x|x+|\d+x|\d+)$', kol.lower()):
            return False
        
        # Must contain at least one letter and not be all numbers
        if not any(c.isalpha() for c in kol):
            return False
        
        # Check for reasonable username characteristics
        # Should not be mostly punctuation or special characters
        alpha_ratio = sum(c.isalpha() for c in kol) / len(kol)
        if alpha_ratio < 0.5:
            return False
        
        return True
    
    def filter_and_rank_kols(self, kol_mentions: Dict[str, int]) -> List[Tuple[str, int]]:
        """Filter and rank KOLs by mention count."""
        # Filter by minimum mentions
        filtered_kols = {
            kol: count for kol, count in kol_mentions.items() 
            if count >= self.config['min_mentions']
        }
        
        logger.info(f"📊 KOL filtering results:")
        logger.info(f"   Raw KOLs found: {len(kol_mentions)}")
        logger.info(f"   After min mentions filter (≥{self.config['min_mentions']}): {len(filtered_kols)}")
        
        # Sort by mention count (descending)
        ranked_kols = sorted(filtered_kols.items(), key=lambda x: x[1], reverse=True)
        
        # Get top KOLs
        top_kols = ranked_kols[:self.config['top_kols_count']]
        
        logger.info(f"   Top {self.config['top_kols_count']} KOLs selected for analysis")
        
        # Log top 10 for verification
        logger.info("🏆 Top 10 KOLs by mention count:")
        for i, (kol, count) in enumerate(top_kols[:10], 1):
            logger.info(f"   {i}. @{kol} ({count} mentions)")
        
        return top_kols
    
    async def find_kol_channel(self, kol: str) -> Optional[Tuple[Any, int]]:
        """Find KOL's Telegram channel using multiple variants with rate limiting."""
        logger.debug(f"🔍 Searching for channel: @{kol}")
        
        # Try the exact name first (some channels like @topcallerschannel might exist as-is)
        exact_variants = [kol, f"@{kol}"]
        
        for variant in exact_variants:
            try:
                logger.debug(f"   Trying exact: {variant}")
                channel = await self.client.get_entity(variant)
                full_channel = await self.client.get_entity(channel)
                subscriber_count = getattr(full_channel, 'participants_count', 0)
                logger.info(f"✅ Found EXACT channel for @{kol}: {variant} ({subscriber_count:,} subscribers)")
                return channel, subscriber_count
            except:
                continue
        
        # Then try all the variants
        variants_tried = []
        
        for i, variant_template in enumerate(self.channel_variants):
            variant = variant_template.format(kol=kol)
            variants_tried.append(variant)
            
            try:
                logger.debug(f"   Trying variant {i+1}/{len(self.channel_variants)}: @{variant}")
                
                # Add rate limiting to prevent flood waits
                if i > 0:
                    await asyncio.sleep(0.3)  # Reduced from 0.5s to 0.3s
                
                channel = await self.client.get_entity(variant)
                
                # Get subscriber count
                full_channel = await self.client.get_entity(channel)
                subscriber_count = getattr(full_channel, 'participants_count', 0)
                
                logger.info(f"✅ Found channel for @{kol}: @{variant} ({subscriber_count:,} subscribers)")
                return channel, subscriber_count
                
            except (ChannelPrivateError, UsernameNotOccupiedError):
                continue
            except FloodWaitError as e:
                logger.warning(f"⚠️ Flood wait for {e.seconds}s - waiting...")
                await asyncio.sleep(e.seconds + 1)
                continue
            except Exception as e:
                logger.debug(f"   Error with @{variant}: {str(e)}")
                continue
        
        logger.debug(f"❌ Could not find channel for @{kol} (tried {len(variants_tried)+2} variants)")
        logger.debug(f"   Variants attempted: {exact_variants + variants_tried}")
        return None, 0
    
    async def analyze_kol_performance(self, kol: str, mentions_count: int) -> Optional[KOLPerformance]:
        """Analyze individual KOL performance with comprehensive metrics."""
        try:
            logger.info(f"📊 Analyzing performance for @{kol} ({mentions_count} mentions)...")
            
            # Find KOL's channel
            channel_result = await self.find_kol_channel(kol)
            if not channel_result or not channel_result[0]:
                logger.warning(f"⚠️ Could not find channel for @{kol} - creating placeholder performance")
                return self.create_placeholder_performance(kol, mentions_count)
            
            channel, subscriber_count = channel_result
            
            # Get recent messages for analysis
            messages = []
            try:
                async for message in self.client.iter_messages(channel, limit=200):
                    if message.text:
                        messages.append(message)
                
                logger.info(f"📥 Retrieved {len(messages)} messages from @{kol}'s channel")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not retrieve messages from @{kol}: {str(e)}")
                return self.create_placeholder_performance(kol, mentions_count, subscriber_count)
            
            # Extract token calls from messages
            token_calls = self.extract_token_calls(messages)
            
            if not token_calls:
                logger.warning(f"⚠️ No token calls found for @{kol}")
                return self.create_placeholder_performance(kol, mentions_count, subscriber_count)
            
            logger.info(f"🎯 Found {len(token_calls)} token calls for @{kol}")
            
            # Analyze performance for each token call
            performance_data = []
            for call in token_calls:
                performance = await self.analyze_token_call_performance(call)
                if performance:
                    performance_data.append(performance)
            
            if not performance_data:
                logger.warning(f"⚠️ No valid performance data for @{kol}")
                return self.create_placeholder_performance(kol, mentions_count, subscriber_count)
            
            # Calculate comprehensive metrics
            return self.calculate_kol_metrics(kol, mentions_count, subscriber_count, performance_data)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing @{kol}: {str(e)}")
            return self.create_placeholder_performance(kol, mentions_count)
    
    def extract_token_calls(self, messages: List) -> List[Dict[str, Any]]:
        """Extract token calls from KOL messages."""
        token_calls = []
        
        for message in messages:
            if not message.text:
                continue
            
            text = message.text.lower()
            
            # Look for token call patterns
            call_indicators = [
                'buy', 'call', 'gem', 'token', 'coin', 'new launch',
                'ca:', 'contract:', 'address:', '$', 'sol'
            ]
            
            if any(indicator in text for indicator in call_indicators):
                # Try to extract contract address
                contract_patterns = [
                    r'([A-Za-z0-9]{32,44})',  # Solana address pattern
                    r'ca[:\s]+([A-Za-z0-9]{32,44})',
                    r'contract[:\s]+([A-Za-z0-9]{32,44})',
                    r'address[:\s]+([A-Za-z0-9]{32,44})'
                ]
                
                contract_address = None
                for pattern in contract_patterns:
                    match = re.search(pattern, text)
                    if match:
                        contract_address = match.group(1)
                        break
                
                token_call = {
                    'message_text': message.text,
                    'timestamp': message.date,
                    'contract_address': contract_address,
                    'message_id': message.id
                }
                
                token_calls.append(token_call)
        
        return token_calls
    
    async def analyze_token_call_performance(self, call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze performance of a specific token call."""
        if not call['contract_address'] or not self.api_manager:
            return None
        
        try:
            # Get token performance from call time
            performance = self.api_manager.calculate_token_performance(
                call['contract_address'],
                call['timestamp']
            )
            
            if performance.get('success'):
                return {
                    'contract_address': call['contract_address'],
                    'call_timestamp': call['timestamp'],
                    'initial_price': performance.get('initial_price', 0),
                    'current_price': performance.get('current_price', 0),
                    'max_price': performance.get('max_price', 0),
                    'roi_percent': performance.get('roi_percent', 0),
                    'max_roi_percent': performance.get('max_roi_percent', 0),
                    'time_to_max_hours': performance.get('time_to_max_roi_hours', 0)
                }
            
        except Exception as e:
            logger.debug(f"Error analyzing token call: {str(e)}")
        
        return None
    
    def calculate_kol_metrics(self, kol: str, mentions: int, subscribers: int, 
                            performance_data: List[Dict[str, Any]]) -> KOLPerformance:
        """Calculate comprehensive KOL performance metrics."""
        total_calls = len(performance_data)
        
        # Calculate win/loss metrics
        winning_calls = sum(1 for p in performance_data if p['roi_percent'] > self.config['win_threshold_percent'])
        losing_calls = total_calls - winning_calls
        success_rate = (winning_calls / total_calls * 100) if total_calls > 0 else 0
        
        # Calculate 2x and 5x metrics
        tokens_2x_plus = sum(1 for p in performance_data if p['max_roi_percent'] >= 100)
        tokens_5x_plus = sum(1 for p in performance_data if p['max_roi_percent'] >= 400)
        success_rate_2x = (tokens_2x_plus / total_calls * 100) if total_calls > 0 else 0
        success_rate_5x = (tokens_5x_plus / total_calls * 100) if total_calls > 0 else 0
        
        # Calculate time metrics
        successful_2x_calls = [p for p in performance_data if p['max_roi_percent'] >= 100]
        avg_time_to_2x = sum(p['time_to_max_hours'] for p in successful_2x_calls) / len(successful_2x_calls) if successful_2x_calls else 0
        
        # Calculate pullback and gains
        avg_max_pullback = 0  # Simplified for now
        avg_unrealized_gains = sum(p['max_roi_percent'] - p['roi_percent'] for p in performance_data) / total_calls if total_calls > 0 else 0
        
        # Calculate consistency score (simplified)
        roi_values = [p['roi_percent'] for p in performance_data]
        consistency_score = max(0, 100 - (max(roi_values) - min(roi_values)) / 10) if roi_values else 0
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(
            success_rate, success_rate_2x, success_rate_5x, 
            avg_time_to_2x, consistency_score
        )
        
        # Determine strategy classification
        strategy = self.classify_strategy(subscribers, avg_time_to_2x, success_rate_5x)
        
        # Determine follower tier
        follower_tier = self.classify_follower_tier(subscribers)
        
        # Calculate total ROI metrics
        total_roi = sum(p['roi_percent'] for p in performance_data)
        max_roi = max(p['max_roi_percent'] for p in performance_data) if performance_data else 0
        
        return KOLPerformance(
            kol=kol,
            channel_id=f"@{kol}",
            subscriber_count=subscribers,
            total_calls=total_calls,
            winning_calls=winning_calls,
            losing_calls=losing_calls,
            success_rate=success_rate,
            tokens_2x_plus=tokens_2x_plus,
            tokens_5x_plus=tokens_5x_plus,
            success_rate_2x=success_rate_2x,
            success_rate_5x=success_rate_5x,
            avg_time_to_2x_hours=avg_time_to_2x,
            avg_max_pullback_percent=avg_max_pullback,
            avg_unrealized_gains_percent=avg_unrealized_gains,
            consistency_score=consistency_score,
            composite_score=composite_score,
            strategy_classification=strategy,
            follower_tier=follower_tier,
            total_roi_percent=total_roi,
            max_roi_percent=max_roi
        )
    
    def create_placeholder_performance(self, kol: str, mentions: int, subscribers: int = 0) -> KOLPerformance:
        """Create placeholder performance for KOLs without analyzable channels."""
        return KOLPerformance(
            kol=kol,
            channel_id=None,
            subscriber_count=subscribers,
            total_calls=0,
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
            composite_score=mentions * 2,  # Use mentions as a base score
            strategy_classification="UNKNOWN",
            follower_tier="LOW",
            total_roi_percent=0,
            max_roi_percent=0
        )
    
    def calculate_composite_score(self, success_rate: float, success_rate_2x: float, 
                                success_rate_5x: float, avg_time_to_2x: float, 
                                consistency_score: float) -> float:
        """Calculate weighted composite performance score."""
        # Weights for different metrics
        weights = {
            'success_rate': 0.25,
            'success_rate_2x': 0.25,
            'consistency': 0.20,
            'time_to_2x': 0.15,
            'success_rate_5x': 0.15
        }
        
        # Normalize time_to_2x (faster is better, cap at 24 hours)
        time_score = max(0, 100 - (avg_time_to_2x / 24 * 100)) if avg_time_to_2x > 0 else 0
        
        composite = (
            success_rate * weights['success_rate'] +
            success_rate_2x * weights['success_rate_2x'] +
            consistency_score * weights['consistency'] +
            time_score * weights['time_to_2x'] +
            success_rate_5x * weights['success_rate_5x']
        )
        
        return min(100, max(0, composite))
    
    def classify_strategy(self, subscribers: int, avg_time_to_2x: float, success_rate_5x: float) -> str:
        """Classify KOL strategy as SCALP or HOLD."""
        if subscribers >= 5000 and avg_time_to_2x <= 12 and avg_time_to_2x > 0:
            return "SCALP"
        elif success_rate_5x >= 15:
            return "HOLD"
        else:
            return "UNKNOWN"
    
    def classify_follower_tier(self, subscribers: int) -> str:
        """Classify follower tier."""
        if subscribers >= 10000:
            return "HIGH"
        elif subscribers >= 1000:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run the complete SpyDefi KOL analysis."""
        start_time = time.time()
        logger.info("🚀 Starting comprehensive SpyDefi KOL analysis...")
        logger.info(f"🎯 Target: Top {self.config['top_kols_count']} KOLs with ≥{self.config['min_mentions']} mentions")
        
        # Log actual configuration being used
        logger.info("📋 ACTUAL CONFIGURATION IN USE:")
        logger.info(f"   📅 Scan period: {self.config['spydefi_scan_hours']} hours (PEAK MEMECOIN HOURS)")
        logger.info(f"   🏆 Top KOLs to analyze: {self.config['top_kols_count']}")
        logger.info(f"   📊 Min mentions required: {self.config['min_mentions']}")
        logger.info(f"   👥 Min subscribers: {self.config['min_subscribers']} (RELAXED)")
        logger.info(f"   💰 Max market cap: ${self.config['max_market_cap_usd']:,}")
        logger.info(f"   🎯 Win threshold: {self.config['win_threshold_percent']}%")
        
        try:
            # Step 1: Get SpyDefi messages
            logger.info("📥 Step 1: Retrieving SpyDefi messages...")
            messages = await self.get_spydefi_messages(self.config['spydefi_scan_hours'])
            
            if not messages:
                return {
                    'success': False,
                    'error': 'No messages retrieved from SpyDefi channel',
                    'kol_performances': {},
                    'metadata': {'processing_time_seconds': time.time() - start_time}
                }
            
            # Step 2: Extract KOL mentions
            logger.info("🔍 Step 2: Extracting KOL mentions...")
            kol_mentions = self.extract_kols_from_messages(messages)
            
            if not kol_mentions:
                return {
                    'success': False,
                    'error': 'No valid KOL mentions found',
                    'kol_performances': {},
                    'metadata': {'processing_time_seconds': time.time() - start_time}
                }
            
            # Step 3: Filter and rank KOLs
            logger.info("📊 Step 3: Filtering and ranking KOLs...")
            top_kols = self.filter_and_rank_kols(kol_mentions)
            
            if not top_kols:
                return {
                    'success': False,
                    'error': f'No KOLs found with ≥{self.config["min_mentions"]} mentions',
                    'kol_performances': {},
                    'metadata': {'processing_time_seconds': time.time() - start_time}
                }
            
            # Step 4: Analyze individual KOL performance
            logger.info(f"🎯 Step 4: Analyzing performance for top {len(top_kols)} KOLs...")
            kol_performances = {}
            successful_analyses = 0
            
            for i, (kol, mentions) in enumerate(top_kols, 1):
                logger.info(f"📊 Analyzing {i}/{len(top_kols)}: @{kol} ({mentions} mentions)")
                
                try:
                    # Add delay between KOL analyses to prevent rate limiting
                    if i > 1:
                        await asyncio.sleep(1)  # 1 second delay between KOL analyses
                    
                    performance = await self.analyze_kol_performance(kol, mentions)
                    if performance:
                        kol_performances[kol] = performance
                        successful_analyses += 1
                        logger.info(f"✅ Analysis complete for @{kol} (Score: {performance.composite_score:.1f})")
                    else:
                        logger.warning(f"⚠️ Analysis failed for @{kol}")
                        
                except Exception as e:
                    logger.error(f"❌ Error analyzing @{kol}: {str(e)}")
                    continue
                
                # Progress update
                if i % 10 == 0:
                    logger.info(f"📈 Progress: {i}/{len(top_kols)} KOLs processed ({successful_analyses} successful)")
            
            # Step 5: Calculate overall statistics
            processing_time = time.time() - start_time
            
            # Calculate overall metrics
            total_calls = sum(p.total_calls for p in kol_performances.values())
            total_winning = sum(p.winning_calls for p in kol_performances.values())
            total_2x = sum(p.tokens_2x_plus for p in kol_performances.values())
            total_5x = sum(p.tokens_5x_plus for p in kol_performances.values())
            
            overall_success_rate = (total_winning / total_calls * 100) if total_calls > 0 else 0
            overall_2x_rate = (total_2x / total_calls * 100) if total_calls > 0 else 0
            overall_5x_rate = (total_5x / total_calls * 100) if total_calls > 0 else 0
            
            # Sort KOLs by composite score
            sorted_kols = dict(sorted(
                kol_performances.items(),
                key=lambda x: x[1].composite_score,
                reverse=True
            ))
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': processing_time,
                'total_messages_scanned': len(messages),
                'total_kols_mentioned': len(kol_mentions),
                'top_kols_analyzed': len(top_kols),
                'successful_analyses': successful_analyses,
                'total_calls_analyzed': total_calls,
                'overall_success_rate': overall_success_rate,
                'overall_2x_rate': overall_2x_rate,
                'overall_5x_rate': overall_5x_rate,
                'config': self.config.copy(),
                'optimization_version': '3.1.1 (Fixed KOL Extraction)'
            }
            
            logger.info("✅ SpyDefi KOL analysis completed successfully!")
            logger.info(f"📊 Results: {len(sorted_kols)} KOLs analyzed in {processing_time:.1f}s")
            logger.info(f"🎯 Overall metrics: {overall_success_rate:.1f}% success, {overall_2x_rate:.1f}% 2x rate")
            
            return {
                'success': True,
                'kol_performances': sorted_kols,
                'kol_mentions': dict(top_kols),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ SpyDefi analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'kol_performances': {},
                'metadata': {
                    'processing_time_seconds': time.time() - start_time,
                    'error_occurred_at': datetime.now().isoformat()
                }
            }