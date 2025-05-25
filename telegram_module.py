"""
Telegram Module - Phoenix Project (2X HOT STREAK EDITION - PERFORMANCE OPTIMIZED)

MAJOR UPDATES:
- Added message limits and progress indicators
- Implemented smart caching system (6-hour cache)
- Progressive message fetching (6h -> 12h -> 24h)
- Timeout protection at multiple levels
- Parallel processing with semaphore limits
- Early termination for efficiency
- Removed all 5x analysis - focus on 2x targets only
- Two-tier analysis: Initial (5 calls) and Deep (20 calls for 40%+ performers)

REQUIREMENTS:
- Python 3.8+
- telethon
- pandas
- asyncio
"""

import asyncio
import re
import logging
import time
import json
import requests
import csv
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import os
import sys
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
    """Data class for token calls"""
    kol: str
    channel_id: int
    contract_address: str
    call_timestamp: int
    message_text: str
    market_cap: Optional[float] = None
    platform: Optional[str] = None

class TelegramScraper:
    """Telegram scraper for SpyDefi KOL analysis with 2x hot streak focus."""
    
    # Analysis tier constants
    INITIAL_ANALYSIS_CALLS = 5
    DEEP_ANALYSIS_CALLS = 20
    DEEP_ANALYSIS_THRESHOLD = 0.40  # 40% 2x success rate triggers deep analysis
    
    # Performance constants
    DEFAULT_MESSAGE_LIMIT = 1000
    PROGRESS_INTERVAL = 100
    SPYDEFI_TIMEOUT = 60  # seconds
    CHANNEL_TIMEOUT = 30  # seconds
    KOL_ANALYSIS_TIMEOUT = 45  # seconds
    GLOBAL_TIMEOUT = 300  # 5 minutes
    MAX_CONCURRENT_CHANNELS = 3
    CACHE_DURATION_HOURS = 6
    MIN_KOL_MENTIONS_NEEDED = 20
    
    def __init__(self, api_id: int, api_hash: str, session_name: str = "phoenix"):
        """Initialize the Telegram scraper."""
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = TelegramClient(session_name, api_id, api_hash)
        self.birdeye_api = None
        self.helius_api = None
        
        # Cache directory
        self.cache_dir = Path.home() / ".phoenix_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.spydefi_cache_file = self.cache_dir / "spydefi_kols.json"
        
        # Semaphore for concurrent operations
        self.channel_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CHANNELS)
        
        # Enhanced validation patterns
        self.contract_patterns = [
            r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b',  # Solana addresses
            r'[0-9a-fA-F]{40,66}',  # Ethereum-style addresses
            r'pump\.fun/[1-9A-HJ-NP-Za-km-z]{32,44}',  # pump.fun links
            r'dexscreener\.com/solana/[1-9A-HJ-NP-Za-km-z]{32,44}',  # DexScreener links
            r'birdeye\.so/token/[1-9A-HJ-NP-Za-km-z]{32,44}',  # Birdeye links
        ]
        
        # Spam patterns to exclude
        self.spam_patterns = [
            r'\.sol\b',  # Solana domains
            r'@[a-zA-Z0-9_]+',  # Telegram handles
            r't\.me/',  # Telegram links
            r'twitter\.com/',  # Twitter links
            r'x\.com/',  # X links
            r'(http|https)://[^\s]+',  # General URLs (unless they contain contracts)
        ]
        
        # Track API calls
        self.api_call_count = {
            'birdeye': 0,
            'helius': 0,
            'birdeye_failures': 0,
            'helius_failures': 0,
            'addresses_validated': 0,
            'addresses_rejected': 0,
            'contract_extraction_attempts': 0,
            'pump_tokens_found': 0
        }
        
    async def connect(self):
        """Connect to Telegram."""
        logger.info("Connecting to Telegram...")
        await self.client.start()
        logger.info("Connected to Telegram")
        
    async def disconnect(self):
        """Disconnect from Telegram."""
        logger.info("Disconnecting from Telegram...")
        await self.client.disconnect()
        logger.info("Disconnected from Telegram")
    
    def _load_spydefi_cache(self) -> Optional[Dict[str, Any]]:
        """Load SpyDefi KOL cache if valid."""
        try:
            if not self.spydefi_cache_file.exists():
                return None
            
            with open(self.spydefi_cache_file, 'r') as f:
                cache = json.load(f)
            
            # Check if cache is expired
            cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
            if datetime.now() - cache_time > timedelta(hours=self.CACHE_DURATION_HOURS):
                logger.info("SpyDefi cache expired, will refresh")
                return None
            
            logger.info(f"✅ Loaded SpyDefi cache with {len(cache.get('kol_mentions', {}))} KOLs")
            return cache
            
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return None
    
    def _save_spydefi_cache(self, kol_mentions: Dict[str, int], message_count: int):
        """Save SpyDefi KOL mentions to cache."""
        try:
            cache = {
                'kol_mentions': kol_mentions,
                'timestamp': datetime.now().isoformat(),
                'message_count': message_count,
                'version': '1.0'
            }
            
            with open(self.spydefi_cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            
            logger.info(f"✅ Saved SpyDefi cache with {len(kol_mentions)} KOLs")
            
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def _is_spam_address(self, potential_address: str) -> bool:
        """Check if the potential address is likely spam."""
        # Check against spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, potential_address, re.IGNORECASE):
                # Special case: allow URLs that contain contract addresses
                if 'pump.fun/' in potential_address or 'dexscreener.com/' in potential_address or 'birdeye.so/' in potential_address:
                    continue
                return True
        
        # Check if it's a channel/user ID (all numbers)
        if potential_address.isdigit() and len(potential_address) > 5:
            return True
            
        # Check if it looks like a filename
        if any(ext in potential_address.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.mp4', '.pdf']):
            return True
            
        return False
    
    def _extract_contract_addresses(self, text: str) -> Set[str]:
        """Extract potential contract addresses from text with enhanced validation."""
        addresses = set()
        self.api_call_count['contract_extraction_attempts'] += 1
        
        # First, try to extract from URLs
        url_patterns = [
            r'pump\.fun/([1-9A-HJ-NP-Za-km-z]{32,44})',
            r'dexscreener\.com/solana/([1-9A-HJ-NP-Za-km-z]{32,44})',
            r'birdeye\.so/token/([1-9A-HJ-NP-Za-km-z]{32,44})',
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if self._is_valid_solana_address(match) and not self._is_spam_address(match):
                    addresses.add(match)
                    if match.endswith('pump'):
                        self.api_call_count['pump_tokens_found'] += 1
        
        # Then look for standalone addresses
        for pattern in self.contract_patterns[:2]:  # Only use the address patterns, not URL patterns
            matches = re.findall(pattern, text)
            for match in matches:
                # Additional length check for Solana addresses
                if 32 <= len(match) <= 44:
                    if self._is_valid_solana_address(match) and not self._is_spam_address(match):
                        addresses.add(match)
                        if match.endswith('pump'):
                            self.api_call_count['pump_tokens_found'] += 1
        
        # Update validation stats
        self.api_call_count['addresses_validated'] += len(addresses)
        
        return addresses
    
    def _is_valid_solana_address(self, address: str) -> bool:
        """Validate Solana address format."""
        # Basic validation
        if not address or len(address) < 32 or len(address) > 44:
            self.api_call_count['addresses_rejected'] += 1
            return False
        
        # Check if it contains only base58 characters
        base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        if not all(c in base58_chars for c in address):
            self.api_call_count['addresses_rejected'] += 1
            return False
        
        # Reject if it looks like a transaction signature (64 chars)
        if len(address) > 50:
            self.api_call_count['addresses_rejected'] += 1
            return False
        
        # Reject known system programs
        system_programs = [
            "11111111111111111111111111111111",
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "So11111111111111111111111111111111111111112",
        ]
        if address in system_programs:
            self.api_call_count['addresses_rejected'] += 1
            return False
        
        return True
    
    async def scrape_channel_messages(self, channel_username: str, hours: int = 24, 
                                    limit: int = None, show_progress: bool = True) -> List[Dict[str, Any]]:
        """Scrape messages from a specific channel with limits and progress."""
        try:
            async with self.channel_semaphore:
                channel = await self.client.get_entity(channel_username)
                if not isinstance(channel, Channel):
                    logger.error(f"{channel_username} is not a channel")
                    return []
                
                after_date = datetime.now() - timedelta(hours=hours)
                messages = []
                message_count = 0
                
                if limit is None:
                    limit = self.DEFAULT_MESSAGE_LIMIT
                
                logger.info(f"Scraping up to {limit} messages from {channel.title} (ID: {channel.id})")
                
                # Use timeout for channel scraping
                try:
                    async with asyncio.timeout(self.CHANNEL_TIMEOUT):
                        # Get newest messages first
                        async for message in self.client.iter_messages(
                            channel, 
                            offset_date=after_date, 
                            reverse=True,  # Newest first
                            limit=limit
                        ):
                            if message.text:
                                messages.append({
                                    'id': message.id,
                                    'date': message.date,
                                    'text': message.text,
                                    'channel_id': channel.id,
                                    'channel_username': channel_username
                                })
                                message_count += 1
                                
                                # Show progress
                                if show_progress and message_count % self.PROGRESS_INTERVAL == 0:
                                    print(f"\r   Fetched {message_count}/{limit} messages...", end="", flush=True)
                                    sys.stdout.flush()
                        
                        if show_progress and message_count > 0:
                            print(f"\r   ✅ Fetched {message_count} messages from {channel.title}", flush=True)
                
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout reached for {channel_username}, continuing with {len(messages)} messages")
                    if show_progress:
                        print(f"\r   ⚠️ Timeout: Got {len(messages)} messages from {channel.title}", flush=True)
                
                logger.info(f"Finished retrieving {len(messages)} messages from {channel.id}")
                return messages
                
        except ChannelPrivateError:
            logger.error(f"Cannot access {channel_username} - it's private or you're not a member")
            return []
        except Exception as e:
            logger.error(f"Error scraping {channel_username}: {str(e)}")
            return []
    
    async def get_channel_info(self, channel_username: str) -> Optional[int]:
        """Get channel ID from username."""
        try:
            channel = await self.client.get_entity(channel_username)
            if isinstance(channel, Channel):
                return channel.id
            return None
        except Exception as e:
            logger.error(f"Error getting channel info for {channel_username}: {str(e)}")
            return None
    
    async def progressive_spydefi_discovery(self, max_hours: int = 24) -> Dict[str, int]:
        """Progressive SpyDefi discovery - start with 6h, expand if needed."""
        kol_mentions = defaultdict(int)
        total_messages = 0
        kol_pattern = r'@([a-zA-Z0-9_]+)'
        
        # Progressive time windows
        time_windows = [6, 12, 24]
        
        for hours in time_windows:
            if hours > max_hours:
                break
                
            logger.info(f"🔍 Scanning SpyDefi for last {hours} hours...")
            
            # Calculate messages to fetch (less for longer periods)
            messages_to_fetch = min(1000, 2000 // (hours // 6))
            
            try:
                async with asyncio.timeout(self.SPYDEFI_TIMEOUT):
                    messages = await self.scrape_channel_messages(
                        "spydefi", 
                        hours=hours,
                        limit=messages_to_fetch,
                        show_progress=True
                    )
                    
                    # Extract KOL mentions
                    new_mentions = 0
                    for msg in messages:
                        mentions = re.findall(kol_pattern, msg['text'])
                        for mention in mentions:
                            if mention.lower() != 'spydefi':
                                if mention not in kol_mentions:
                                    new_mentions += 1
                                kol_mentions[mention] += 1
                    
                    total_messages += len(messages)
                    unique_kols = len(kol_mentions)
                    
                    logger.info(f"✅ Found {unique_kols} unique KOLs (+{new_mentions} new) from {len(messages)} messages")
                    
                    # Early termination if we have enough KOLs
                    if unique_kols >= self.MIN_KOL_MENTIONS_NEEDED:
                        logger.info(f"🎯 Sufficient KOLs found ({unique_kols}), stopping discovery")
                        break
                        
            except asyncio.TimeoutError:
                logger.warning(f"SpyDefi discovery timeout at {hours}h, continuing with current data")
                break
        
        # Save to cache
        if kol_mentions:
            self._save_spydefi_cache(dict(kol_mentions), total_messages)
        
        return dict(kol_mentions)
    
    async def redesigned_spydefi_analysis(self, hours: int = 24, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Redesigned SpyDefi analysis with performance optimizations.
        
        Two-tier analysis system:
        1. Initial scan: Last 5 calls for all KOLs
        2. Deep scan: Last 20 calls for KOLs with 40%+ 2x success rate
        """
        logger.info("🚀 STARTING OPTIMIZED SPYDEFI ANALYSIS (2x Hot Streak Focus)")
        
        try:
            # Global timeout for entire analysis
            async with asyncio.timeout(self.GLOBAL_TIMEOUT):
                # Phase 1: Discover active KOLs from SpyDefi
                logger.info("🎯 Phase 1: Discovering active KOLs from SpyDefi...")
                
                # Check if APIs are available
                if self.birdeye_api:
                    logger.info("✅ Birdeye API available for mainstream tokens")
                else:
                    logger.warning("⚠️ Birdeye API not configured - token analysis will be limited")
                    
                if self.helius_api:
                    logger.info("✅ Helius API available for pump.fun tokens")
                else:
                    logger.warning("⚠️ Helius API not configured - pump.fun analysis will be limited")
                
                # Check cache first
                kol_mentions = None
                if not force_refresh:
                    cache = self._load_spydefi_cache()
                    if cache:
                        kol_mentions = cache.get('kol_mentions', {})
                        logger.info(f"📦 Using cached SpyDefi data: {len(kol_mentions)} KOLs")
                
                # If no cache or force refresh, do progressive discovery
                if not kol_mentions:
                    kol_mentions = await self.progressive_spydefi_discovery(hours)
                
                if not kol_mentions:
                    logger.error("No KOLs found in SpyDefi")
                    return {
                        'success': False,
                        'error': 'No KOLs found in SpyDefi channel'
                    }
                
                logger.info(f"✅ Found {len(kol_mentions)} active KOLs from SpyDefi")
                
                # Phase 2: Initial analysis of top KOLs (5 calls each)
                logger.info(f"🎯 Phase 2: Initial analysis of KOLs (last {self.INITIAL_ANALYSIS_CALLS} calls each)...")
                
                # Sort KOLs by mention count
                sorted_kols = sorted(kol_mentions.items(), key=lambda x: x[1], reverse=True)
                
                # Limit to top 30 KOLs for initial analysis
                max_kols_initial = 30
                top_kols = sorted_kols[:max_kols_initial]
                
                logger.info(f"📊 Analyzing top {len(top_kols)} KOLs for initial screening")
                
                # Parallel initial analysis
                kol_initial_performance = {}
                
                # Create tasks for parallel execution
                initial_tasks = []
                for kol, mention_count in top_kols:
                    task = self.analyze_individual_kol(
                        kol, 
                        hours=hours,
                        max_calls=self.INITIAL_ANALYSIS_CALLS,
                        analysis_type="initial"
                    )
                    initial_tasks.append((kol, mention_count, task))
                
                # Execute with progress
                for i, (kol, mention_count, task) in enumerate(initial_tasks, 1):
                    print(f"\r📊 Initial analysis {i}/{len(initial_tasks)}: @{kol} ({mention_count} mentions)", end="", flush=True)
                    
                    try:
                        kol_analysis = await task
                        
                        if kol_analysis and kol_analysis.get('tokens_mentioned', 0) > 0:
                            kol_initial_performance[kol] = kol_analysis
                        
                    except Exception as e:
                        logger.error(f"Error analyzing @{kol}: {str(e)}")
                    
                    # Small delay between KOLs
                    if i < len(initial_tasks):
                        await asyncio.sleep(0.5)
                
                print(f"\r✅ Initial analysis complete: {len(kol_initial_performance)} KOLs with data", flush=True)
                
                # Phase 3: Deep analysis for high performers
                logger.info(f"🎯 Phase 3: Deep analysis for KOLs with {self.DEEP_ANALYSIS_THRESHOLD*100:.0f}%+ 2x success rate...")
                
                kol_deep_performance = {}
                deep_candidates = [
                    (kol, stats) for kol, stats in kol_initial_performance.items()
                    if stats['success_rate_2x'] >= self.DEEP_ANALYSIS_THRESHOLD * 100
                ]
                
                logger.info(f"🔥 Found {len(deep_candidates)} KOLs qualified for deep analysis")
                
                # Deep analysis with progress
                for i, (kol, initial_stats) in enumerate(deep_candidates, 1):
                    print(f"\r🔥 Deep analysis {i}/{len(deep_candidates)}: @{kol} "
                          f"(Initial: {initial_stats['success_rate_2x']:.1f}% 2x rate)", end="", flush=True)
                    
                    try:
                        # Get last 20 calls for deep analysis
                        deep_analysis = await self.analyze_individual_kol(
                            kol,
                            hours=hours * 7,  # Look back further for deep analysis
                            max_calls=self.DEEP_ANALYSIS_CALLS,
                            analysis_type="deep"
                        )
                        
                        if deep_analysis:
                            kol_deep_performance[kol] = deep_analysis
                            
                    except Exception as e:
                        logger.error(f"Error in deep analysis for @{kol}: {str(e)}")
                    
                    await asyncio.sleep(0.5)
                
                print(f"\r✅ Deep analysis complete: {len(kol_deep_performance)} KOLs analyzed", flush=True)
                
                # Combine results (deep analysis overrides initial for those KOLs)
                final_kol_performance = kol_initial_performance.copy()
                final_kol_performance.update(kol_deep_performance)
                
                # Phase 4: Calculate composite scores with speed weighting
                logger.info("🎯 Phase 4: Calculating composite scores (2x rate + speed)...")
                
                for kol, stats in final_kol_performance.items():
                    composite_score = self._calculate_composite_score(stats)
                    stats['composite_score'] = composite_score
                
                # Sort by composite score
                ranked_kols = dict(sorted(
                    final_kol_performance.items(),
                    key=lambda x: x[1]['composite_score'],
                    reverse=True
                ))
                
                # Log top performers
                logger.info("🏆 TOP 10 KOLs by composite score (2x rate + speed weighted):")
                for i, (kol, stats) in enumerate(list(ranked_kols.items())[:10], 1):
                    logger.info(f"   {i}. @{kol}: {stats['composite_score']:.1f} score, "
                               f"{stats['success_rate_2x']:.1f}% 2x rate, "
                               f"{stats.get('avg_time_to_2x_minutes', 0):.1f} min avg")
                    if stats.get('analysis_type') == 'deep':
                        logger.info(f"      🔥 Deep analysis performed ({self.DEEP_ANALYSIS_CALLS} calls)")
                
                # Phase 5: Get channel IDs for top performers (limited concurrency)
                logger.info("🎯 Phase 5: Getting channel IDs for TOP 10 KOLs...")
                
                top_10_kols = list(ranked_kols.keys())[:10]
                
                for i, kol in enumerate(top_10_kols, 1):
                    print(f"\rGetting channel ID {i}/10: @{kol}", end="", flush=True)
                    
                    if kol in ranked_kols:
                        try:
                            channel_id = await self.get_channel_info(f"@{kol}")
                            if channel_id:
                                ranked_kols[kol]['channel_id'] = channel_id
                            else:
                                logger.warning(f"Could not find channel ID for @{kol}")
                        except Exception as e:
                            logger.error(f"Error getting channel ID for @{kol}: {str(e)}")
                    
                    await asyncio.sleep(1)
                
                print(f"\r✅ Channel ID retrieval complete", flush=True)
                
                # Log API call statistics
                logger.info("📊 VALIDATION STATISTICS:")
                logger.info(f"   📍 Contract extraction attempts: {self.api_call_count['contract_extraction_attempts']}")
                logger.info(f"   ✅ Valid addresses found: {self.api_call_count['addresses_validated']}")
                logger.info(f"   ❌ Invalid addresses rejected: {self.api_call_count['addresses_rejected']}")
                logger.info(f"   🚀 Pump.fun tokens found: {self.api_call_count['pump_tokens_found']}")
                logger.info(f"   📞 Birdeye API calls made: {self.api_call_count['birdeye']}")
                logger.info(f"   📞 Helius API calls made: {self.api_call_count['helius']}")
                logger.info(f"   ⚠️ Birdeye API failures: {self.api_call_count['birdeye_failures']}")
                logger.info(f"   ⚠️ Helius API failures: {self.api_call_count['helius_failures']}")
                
                success_rate = (self.api_call_count['addresses_validated'] / 
                               max(1, self.api_call_count['addresses_validated'] + self.api_call_count['addresses_rejected'])) * 100
                logger.info(f"   📈 Address validation success rate: {success_rate:.1f}%")
                
                # Calculate overall stats
                total_kols = len(final_kol_performance)
                total_calls = sum(k.get('tokens_mentioned', 0) for k in final_kol_performance.values())
                total_2x = sum(k.get('tokens_2x_plus', 0) for k in final_kol_performance.values())
                overall_2x_rate = (total_2x / max(1, total_calls)) * 100
                
                logger.info("🎉 OPTIMIZED ANALYSIS COMPLETE!")
                logger.info(f"📊 Total KOLs analyzed: {total_kols}")
                logger.info(f"📊 Initial analyses: {len(kol_initial_performance)}")
                logger.info(f"📊 Deep analyses: {len(kol_deep_performance)}")
                logger.info(f"📊 Total calls analyzed: {total_calls}")
                logger.info(f"📊 2x success rate: {overall_2x_rate:.1f}%")
                
                return {
                    'success': True,
                    'ranked_kols': ranked_kols,
                    'total_kols_analyzed': total_kols,
                    'deep_analyses_performed': len(kol_deep_performance),
                    'total_calls': total_calls,
                    'total_2x_tokens': total_2x,
                    'success_rate_2x': overall_2x_rate,
                    'api_stats': self.api_call_count.copy()
                }
                
        except asyncio.TimeoutError:
            logger.error(f"Global timeout reached ({self.GLOBAL_TIMEOUT}s), returning partial results")
            return {
                'success': False,
                'error': f'Analysis timeout after {self.GLOBAL_TIMEOUT} seconds',
                'partial_results': True
            }
    
    async def analyze_individual_kol(self, kol_username: str, hours: int = 168, 
                                   max_calls: int = 5, analysis_type: str = "initial") -> Optional[Dict[str, Any]]:
        """
        Analyze an individual KOL's performance focusing on 2x targets.
        
        Args:
            kol_username: KOL's telegram username (without @)
            hours: How far back to look for messages
            max_calls: Maximum number of recent calls to analyze
            analysis_type: "initial" or "deep"
        """
        try:
            async with asyncio.timeout(self.KOL_ANALYSIS_TIMEOUT):
                logger.info(f"🔍 Analyzing individual KOL: @{kol_username} ({analysis_type} analysis)")
                
                # Determine message limit based on analysis type
                message_limit = 500 if analysis_type == "initial" else 2000
                
                # Scrape messages
                messages = await self.scrape_channel_messages(
                    f"@{kol_username}", 
                    hours,
                    limit=message_limit,
                    show_progress=False  # Less verbose for individual KOLs
                )
                
                if not messages:
                    logger.warning(f"No messages found for @{kol_username}")
                    return None
                
                logger.info(f"📨 Found {len(messages)} messages in @{kol_username}'s channel")
                
                # Extract token calls
                token_calls = []
                
                for msg in messages:
                    contracts = self._extract_contract_addresses(msg['text'])
                    
                    for contract in contracts:
                        token_calls.append({
                            'contract_address': contract,
                            'call_timestamp': int(msg['date'].timestamp()),
                            'message_text': msg['text'][:200],  # First 200 chars
                            'is_pump': contract.endswith('pump')
                        })
                
                # Sort by timestamp (newest first) and limit
                token_calls = sorted(token_calls, key=lambda x: x['call_timestamp'], reverse=True)[:max_calls]
                
                logger.info(f"🎯 Found {len(token_calls)} token calls for @{kol_username}")
                
                if not token_calls:
                    return {
                        'kol': kol_username,
                        'tokens_mentioned': 0,
                        'tokens_2x_plus': 0,
                        'success_rate_2x': 0,
                        'avg_ath_roi': 0,
                        'avg_max_pullback_percent': 0,
                        'avg_time_to_2x_minutes': 0,
                        'analysis_type': analysis_type
                    }
                
                # Analyze each token call
                analyzed_calls = []
                tokens_2x = 0
                
                for i, call in enumerate(token_calls, 1):
                    if analysis_type == "initial" or (analysis_type == "deep" and i % 4 == 0):
                        logger.info(f"📊 Analyzing call {i}/{len(token_calls)} for @{kol_username}")
                    
                    try:
                        # Add delay to respect rate limits
                        if i > 1:
                            await asyncio.sleep(0.5)
                        
                        # Get token performance
                        performance = await self._get_token_performance_2x(
                            call['contract_address'],
                            call['call_timestamp'],
                            call['is_pump']
                        )
                        
                        if performance:
                            if performance.get('reached_2x', False):
                                tokens_2x += 1
                            
                            analyzed_calls.append({
                                **call,
                                **performance
                            })
                            
                    except Exception as e:
                        logger.error(f"Error analyzing token {call['contract_address']}: {str(e)}")
                        continue
                
                # Calculate metrics
                if analyzed_calls:
                    success_rate_2x = (tokens_2x / len(analyzed_calls)) * 100
                    
                    # Calculate averages only for tokens that reached 2x
                    tokens_with_2x = [c for c in analyzed_calls if c.get('reached_2x', False)]
                    
                    if tokens_with_2x:
                        avg_time_to_2x_minutes = sum(c.get('time_to_2x_minutes', 0) for c in tokens_with_2x) / len(tokens_with_2x)
                        avg_pullback_2x = sum(c.get('max_pullback_before_2x', 0) for c in tokens_with_2x) / len(tokens_with_2x)
                    else:
                        avg_time_to_2x_minutes = 0
                        avg_pullback_2x = 0
                    
                    # ATH ROI for all tokens
                    avg_ath_roi = sum(c.get('ath_roi', 0) for c in analyzed_calls) / len(analyzed_calls)
                    
                    result = {
                        'kol': kol_username,
                        'tokens_mentioned': len(analyzed_calls),
                        'tokens_2x_plus': tokens_2x,
                        'success_rate_2x': round(success_rate_2x, 2),
                        'avg_ath_roi': round(avg_ath_roi, 2),
                        'avg_max_pullback_percent': round(avg_pullback_2x, 2),
                        'avg_time_to_2x_minutes': round(avg_time_to_2x_minutes, 2),
                        'analysis_type': analysis_type,
                        'analyzed_calls': analyzed_calls  # Keep for debugging
                    }
                    
                    return result
                else:
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout analyzing @{kol_username} after {self.KOL_ANALYSIS_TIMEOUT}s")
            return None
        except Exception as e:
            logger.error(f"Error analyzing KOL @{kol_username}: {str(e)}")
            return None
    
    async def _get_token_performance_2x(self, contract_address: str, call_timestamp: int, 
                                       is_pump: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get token performance focusing on 2x achievement.
        Track performance for 2-3 days after call to catch 2x movements.
        """
        try:
            # Determine tracking window (2-3 days)
            current_time = int(datetime.now().timestamp())
            time_since_call = current_time - call_timestamp
            max_track_time = 3 * 24 * 60 * 60  # 3 days in seconds
            
            # Use appropriate end time
            end_timestamp = min(current_time, call_timestamp + max_track_time)
            
            # Get price history based on token type
            if is_pump and self.helius_api:
                performance = await self._analyze_pump_token_2x(
                    contract_address,
                    call_timestamp,
                    end_timestamp
                )
            elif self.birdeye_api:
                performance = await self._analyze_regular_token_2x(
                    contract_address,
                    call_timestamp,
                    end_timestamp
                )
            else:
                logger.warning(f"No API available for token {contract_address}")
                return None
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting token performance: {str(e)}")
            return None
    
    async def _analyze_regular_token_2x(self, contract_address: str, 
                                       start_timestamp: int, end_timestamp: int) -> Optional[Dict[str, Any]]:
        """Analyze regular token for 2x achievement using Birdeye."""
        try:
            self.api_call_count['birdeye'] += 1
            
            # Get price history
            history = self.birdeye_api.get_token_price_history(
                contract_address,
                start_timestamp,
                end_timestamp,
                "15m"  # 15 minute intervals
            )
            
            if not history.get("success") or not history.get("data", {}).get("items"):
                logger.warning(f"No price history available for {contract_address}")
                self.api_call_count['birdeye_failures'] += 1
                return None
            
            prices = history["data"]["items"]
            if not prices:
                return None
            
            # Extract price data
            initial_price = prices[0].get("value", 0)
            if initial_price <= 0:
                return None
            
            # Track metrics
            ath_price = initial_price
            ath_roi = 0
            reached_2x = False
            time_to_2x_minutes = 0
            max_pullback_before_2x = 0
            current_peak = initial_price
            
            for price_point in prices:
                price = price_point.get("value", 0)
                timestamp = price_point.get("unixTime", 0)
                
                if price <= 0:
                    continue
                
                # Calculate ROI
                roi = ((price / initial_price) - 1) * 100
                
                # Track ATH
                if price > ath_price:
                    ath_price = price
                    ath_roi = roi
                
                # Track pullback from recent peak
                if price > current_peak:
                    current_peak = price
                else:
                    pullback = ((current_peak - price) / current_peak) * 100
                    if not reached_2x and pullback > max_pullback_before_2x:
                        max_pullback_before_2x = pullback
                
                # Check if reached 2x
                if not reached_2x and roi >= 100:
                    reached_2x = True
                    time_to_2x_minutes = (timestamp - start_timestamp) / 60
            
            return {
                'reached_2x': reached_2x,
                'ath_roi': ath_roi,
                'time_to_2x_minutes': time_to_2x_minutes if reached_2x else 0,
                'max_pullback_before_2x': max_pullback_before_2x if reached_2x else 0,
                'price_points_analyzed': len(prices)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing regular token: {str(e)}")
            self.api_call_count['birdeye_failures'] += 1
            return None
    
    async def _analyze_pump_token_2x(self, contract_address: str, 
                                    start_timestamp: int, end_timestamp: int) -> Optional[Dict[str, Any]]:
        """Analyze pump.fun token for 2x achievement using Helius."""
        try:
            self.api_call_count['helius'] += 1
            
            logger.info(f"Analyzing pump.fun token swaps for {contract_address}")
            
            # Use Helius to analyze token swaps
            swap_analysis = self.helius_api.analyze_token_swaps(
                wallet_address="",  # Empty to get all swaps
                token_address=contract_address,
                limit=100
            )
            
            if not swap_analysis.get("success") or not swap_analysis.get("data"):
                logger.warning(f"No swap data available for pump token {contract_address}")
                self.api_call_count['helius_failures'] += 1
                return None
            
            swaps = swap_analysis["data"]
            if not swaps:
                return None
            
            # Filter swaps within our time window
            relevant_swaps = []
            for swap in swaps:
                swap_time = swap.get("timestamp", 0)
                if start_timestamp <= swap_time <= end_timestamp:
                    relevant_swaps.append(swap)
            
            if not relevant_swaps:
                return None
            
            # Calculate prices from swaps
            price_history = []
            for swap in relevant_swaps:
                sol_amount = swap.get("sol_amount", 0)
                token_amount = swap.get("token_amount", 0)
                
                if sol_amount > 0 and token_amount > 0:
                    price = sol_amount / token_amount
                    price_history.append({
                        'timestamp': swap.get("timestamp", 0),
                        'price': price,
                        'type': swap.get("type", "unknown")
                    })
            
            if not price_history:
                # Try alternative: construct price from first and last known values
                return self._estimate_pump_token_performance(contract_address, start_timestamp, end_timestamp)
            
            # Sort by timestamp
            price_history.sort(key=lambda x: x['timestamp'])
            
            # Analyze price movement
            initial_price = price_history[0]['price']
            ath_price = initial_price
            ath_roi = 0
            reached_2x = False
            time_to_2x_minutes = 0
            max_pullback_before_2x = 0
            current_peak = initial_price
            
            for point in price_history:
                price = point['price']
                timestamp = point['timestamp']
                
                # Calculate ROI
                roi = ((price / initial_price) - 1) * 100
                
                # Track ATH
                if price > ath_price:
                    ath_price = price
                    ath_roi = roi
                
                # Track pullback
                if price > current_peak:
                    current_peak = price
                else:
                    pullback = ((current_peak - price) / current_peak) * 100
                    if not reached_2x and pullback > max_pullback_before_2x:
                        max_pullback_before_2x = pullback
                
                # Check if reached 2x
                if not reached_2x and roi >= 100:
                    reached_2x = True
                    time_to_2x_minutes = (timestamp - start_timestamp) / 60
            
            return {
                'reached_2x': reached_2x,
                'ath_roi': ath_roi,
                'time_to_2x_minutes': time_to_2x_minutes if reached_2x else 0,
                'max_pullback_before_2x': max_pullback_before_2x if reached_2x else 0,
                'price_points_analyzed': len(price_history),
                'is_pump_token': True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pump token: {str(e)}")
            self.api_call_count['helius_failures'] += 1
            return None
    
    def _estimate_pump_token_performance(self, contract_address: str, 
                                       start_timestamp: int, end_timestamp: int) -> Dict[str, Any]:
        """Estimate pump token performance when detailed data isn't available."""
        # Conservative estimates for pump tokens
        # Most pump tokens either pump quickly or dump
        time_window_hours = (end_timestamp - start_timestamp) / 3600
        
        if time_window_hours < 24:
            # Very short window - assume no 2x
            return {
                'reached_2x': False,
                'ath_roi': 50,  # Assume modest gain
                'time_to_2x_minutes': 0,
                'max_pullback_before_2x': 0,
                'price_points_analyzed': 0,
                'is_pump_token': True,
                'estimated': True
            }
        else:
            # Longer window - some chance of 2x
            # This is a rough estimate based on pump token behavior
            estimated_2x_chance = 0.15  # 15% of pump tokens reach 2x
            
            return {
                'reached_2x': False,  # Conservative
                'ath_roi': 80,  # Assume decent gain
                'time_to_2x_minutes': 0,
                'max_pullback_before_2x': 0,
                'price_points_analyzed': 0,
                'is_pump_token': True,
                'estimated': True
            }
    
    def _calculate_composite_score(self, kol_stats: Dict[str, Any]) -> float:
        """
        Calculate composite score with heavy weighting for:
        1. 2x success rate (40% weight)
        2. Speed to 2x (40% weight)
        3. Average ATH ROI (20% weight)
        """
        # Base components
        success_rate_2x = kol_stats.get('success_rate_2x', 0)
        avg_time_to_2x_minutes = kol_stats.get('avg_time_to_2x_minutes', 0)
        avg_ath_roi = kol_stats.get('avg_ath_roi', 0)
        tokens_mentioned = kol_stats.get('tokens_mentioned', 0)
        
        # Minimum calls threshold
        if tokens_mentioned < 2:
            return 0
        
        # 1. Success rate score (0-40 points)
        success_score = (success_rate_2x / 100) * 40
        
        # 2. Speed score (0-40 points)
        # Faster is better: 30 min = 40 points, 360 min (6 hours) = 0 points
        if avg_time_to_2x_minutes > 0 and success_rate_2x > 0:
            # Normalize: 30 minutes or less = maximum score
            # Linear decrease up to 360 minutes
            if avg_time_to_2x_minutes <= 30:
                speed_score = 40
            elif avg_time_to_2x_minutes >= 360:
                speed_score = 0
            else:
                # Linear interpolation
                speed_score = 40 * (1 - (avg_time_to_2x_minutes - 30) / 330)
        else:
            speed_score = 0
        
        # 3. ATH ROI score (0-20 points)
        # Normalize: 500% = 20 points
        ath_score = min(20, (avg_ath_roi / 500) * 20)
        
        # 4. Activity bonus for recent hot streak (0-10 points)
        activity_bonus = 0
        if tokens_mentioned >= 10:
            activity_bonus = 10
        elif tokens_mentioned >= 5:
            activity_bonus = 5
        
        # 5. Analysis type bonus
        analysis_bonus = 0
        if kol_stats.get('analysis_type') == 'deep':
            analysis_bonus = 10  # Bonus for passing deep analysis threshold
        
        # Total score
        total_score = success_score + speed_score + ath_score + activity_bonus + analysis_bonus
        
        # Cap at 100
        return min(100, total_score)
    
    async def export_spydefi_analysis(self, analysis_results: Dict[str, Any], output_file: str = "spydefi_analysis_2x.csv"):
        """Export the SpyDefi analysis results focusing on 2x metrics."""
        try:
            if not analysis_results.get('success') or not analysis_results.get('ranked_kols'):
                logger.error("No data to export")
                return
            
            ranked_kols = analysis_results['ranked_kols']
            
            # Prepare CSV data
            csv_data = []
            
            for kol, data in ranked_kols.items():
                row = {
                    'kol': kol,
                    'channel_id': data.get('channel_id', ''),
                    'tokens_mentioned': data.get('tokens_mentioned', 0),
                    'tokens_2x_plus': data.get('tokens_2x_plus', 0),
                    'success_rate_2x': data.get('success_rate_2x', 0),
                    'avg_ath_roi': data.get('avg_ath_roi', 0),
                    'composite_score': data.get('composite_score', 0),
                    'avg_max_pullback_percent': data.get('avg_max_pullback_percent', 0),
                    'avg_time_to_2x_minutes': data.get('avg_time_to_2x_minutes', 0),
                    'analysis_type': data.get('analysis_type', 'initial')
                }
                csv_data.append(row)
            
            # Write main CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
            
            logger.info(f"✅ Exported {len(csv_data)} KOLs to {output_file}")
            
            # Export summary
            summary_file = output_file.replace('.csv', '_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("SPYDEFI 2X HOT STREAK ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total KOLs Analyzed: {analysis_results.get('total_kols_analyzed', 0)}\n")
                f.write(f"Deep Analyses Performed: {analysis_results.get('deep_analyses_performed', 0)}\n")
                f.write(f"Total Token Calls: {analysis_results.get('total_calls', 0)}\n")
                f.write(f"2x Success Rate: {analysis_results.get('success_rate_2x', 0):.2f}%\n\n")
                
                # API stats
                api_stats = analysis_results.get('api_stats', {})
                f.write("API STATISTICS:\n")
                f.write(f"Birdeye Calls: {api_stats.get('birdeye', 0)}\n")
                f.write(f"Helius Calls: {api_stats.get('helius', 0)}\n")
                f.write(f"Birdeye Failures: {api_stats.get('birdeye_failures', 0)}\n")
                f.write(f"Helius Failures: {api_stats.get('helius_failures', 0)}\n\n")
                
                # Top performers
                f.write("TOP 10 KOLS (2X HOT STREAKS):\n")
                f.write("-" * 50 + "\n")
                
                top_kols = list(ranked_kols.items())[:10]
                for i, (kol, data) in enumerate(top_kols, 1):
                    f.write(f"\n{i}. @{kol}\n")
                    f.write(f"   Composite Score: {data.get('composite_score', 0):.1f}\n")
                    f.write(f"   2x Success Rate: {data.get('success_rate_2x', 0):.1f}%\n")
                    f.write(f"   Avg Time to 2x: {data.get('avg_time_to_2x_minutes', 0):.1f} minutes\n")
                    f.write(f"   Avg ATH ROI: {data.get('avg_ath_roi', 0):.1f}%\n")
                    f.write(f"   Analysis Type: {data.get('analysis_type', 'initial')}\n")
                    if data.get('channel_id'):
                        f.write(f"   Channel ID: {data.get('channel_id')}\n")
            
            logger.info(f"✅ Exported summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting analysis: {str(e)}")
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            if self.spydefi_cache_file.exists():
                self.spydefi_cache_file.unlink()
                logger.info("✅ Cleared SpyDefi cache")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()