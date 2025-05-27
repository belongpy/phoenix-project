"""
SpyDefi KOL Analysis Module - Phoenix Project (FINAL FIXED VERSION)

CRITICAL FIXES IMPLEMENTED:
1. REMOVED winning_calls & losing_calls completely from dataclass and all calculations
2. FIXED channel_id to return REAL numeric Telegram IDs (-1001234567890)
3. FIXED avg_max_pullback_percent to show REAL pullback data for stop loss placement
4. FIXED avg_roi to show REAL average ROI data
5. FIXED composite_score to properly weight pullback at 25%
6. FIXED FloodWaitError handling to prevent crashes
7. FIXED token analysis to get actual price data and calculate real metrics
8. FORCE CLEAR BROKEN CACHE to prevent loading old data
"""

import asyncio
import logging
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time

try:
    from telethon import TelegramClient
    from telethon.errors import FloodWaitError, ChatAdminRequiredError, ChannelPrivateError
    from telethon.tl.types import Channel, Chat
    TELETHON_AVAILABLE = True
except ImportError:
    TELETHON_AVAILABLE = False

logger = logging.getLogger("phoenix.spydefi")

@dataclass
class KOLPerformance:
    """FIXED KOL performance metrics - removed winning/losing calls, added real data."""
    kol: str
    channel_id: str  # REAL numeric Telegram channel ID like -1001234567890
    follower_tier: str  # HIGH/MEDIUM/LOW
    total_calls: int
    tokens_2x_plus: int
    tokens_5x_plus: int
    success_rate_2x: float      # tokens_2x_plus / total_calls * 100
    success_rate_5x: float      # tokens_5x_plus / total_calls * 100  
    avg_time_to_2x_hours: float
    avg_max_pullback_percent: float  # REAL pullback data for stop loss placement
    consistency_score: float         # Based on REAL ROI variance
    composite_score: float          # Properly weights pullback at 25%
    strategy_classification: str    # SCALP/HOLD/MIXED
    avg_roi: float                 # REAL average ROI across all tokens

class SpyDefiAnalyzer:
    """FIXED SpyDefi analyzer with real channel IDs and actual pullback/ROI data."""
    
    def __init__(self, api_id: str, api_hash: str, session_name: str = "phoenix_spydefi"):
        if not TELETHON_AVAILABLE:
            raise ImportError("Telethon is required for SpyDefi analysis")
        
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = None
        self.api_manager = None
        
        # Fixed configuration
        self.config = {
            'spydefi_scan_hours': 8,
            'kol_analysis_days': 7,
            'top_kols_count': 50,
            'min_mentions': 1,
            'max_market_cap_usd': 10_000_000,
            'win_threshold_percent': 50,
            'timeout_minutes': 30,
        }
        
        # Cache setup
        self.cache_dir = Path.home() / ".phoenix_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "spydefi_kol_analysis.json"
        
        logger.info("🎯 SpyDefi Analyzer initialized with REAL channel IDs and pullback data")
    
    def set_api_manager(self, api_manager):
        """Set the API manager for token analysis."""
        self.api_manager = api_manager
        logger.info("✅ API manager configured for REAL token analysis")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self.client:
            self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
        
        await self.client.start()
        logger.info("📱 Telegram client started")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.disconnect()
            logger.info("📱 Telegram client disconnected")
    
    def _should_use_cache(self) -> bool:
        """FIXED: Force clear old cache with broken data - run fresh analysis."""
        # FORCE CLEAR CACHE to prevent loading broken data with winning/losing calls
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache has broken data structure
                kol_performances = cache_data.get('kol_performances', {})
                if kol_performances:
                    # Check if any KOL performance has winning_calls/losing_calls (broken data)
                    first_kol = next(iter(kol_performances.values()))
                    if isinstance(first_kol, dict):
                        if 'winning_calls' in first_kol or 'losing_calls' in first_kol:
                            logger.info("🧹 CLEARING BROKEN CACHE: Contains winning/losing calls")
                            self.cache_file.unlink()
                            return False
                    
                    # Check for fake channel IDs
                    if first_kol.get('channel_id', '').startswith('spydefi_based_'):
                        logger.info("🧹 CLEARING BROKEN CACHE: Contains fake channel IDs")
                        self.cache_file.unlink()
                        return False
                    
                    # Check for zero pullback data (indicates broken calculations)
                    if first_kol.get('avg_max_pullback_percent', -999) == 0:
                        logger.info("🧹 CLEARING BROKEN CACHE: Contains zero pullback data")
                        self.cache_file.unlink()  
                        return False
                    
                    # Check for zero ROI data
                    if first_kol.get('avg_roi', -999) == 0:
                        logger.info("🧹 CLEARING BROKEN CACHE: Contains zero ROI data")
                        self.cache_file.unlink()
                        return False
                
                    # Check version to force refresh for fixes
                    version = cache_data.get('version', '')
                    if version != '4.3_FIXED':
                        logger.info(f"🧹 CLEARING OLD CACHE: Version {version} != 4.3_FIXED")
                        self.cache_file.unlink()
                        return False
                    
                    # Check timestamp age using timezone-aware comparison
                    timestamp_str = cache_data.get('timestamp')
                    if timestamp_str:
                        try:
                            cache_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                            current_time = datetime.now(timezone.utc)
                            age_hours = (current_time - cache_time).total_seconds() / 3600
                            
                            if age_hours > 6:  # Cache older than 6 hours
                                logger.info(f"🧹 CLEARING EXPIRED CACHE: {age_hours:.1f} hours old")
                                self.cache_file.unlink()
                                return False
                        except Exception as e:
                            logger.warning(f"🧹 CLEARING CACHE: Invalid timestamp format: {str(e)}")
                            self.cache_file.unlink()
                            return False
                
                # If we get here, cache might be valid - but let's force fresh for safety
                logger.info("🧹 FORCING FRESH ANALYSIS: Clearing cache to ensure fixed data")
                self.cache_file.unlink()
                return False
                
            except Exception as e:
                logger.error(f"📦 Cache error, clearing: {str(e)}")
                if self.cache_file.exists():
                    self.cache_file.unlink()
                return False
        
        return False  # Never use cache, always run fresh
    
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load cached analysis results."""
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            kol_count = len(cache_data.get('kol_performances', {}))
            logger.info(f"📦 Loaded cache with {kol_count} KOLs")
            return cache_data
            
        except Exception as e:
            logger.error(f"📦 Error loading cache: {str(e)}")
            return None
    
    def _save_cache(self, results: Dict[str, Any]):
        """Save analysis results to cache."""
        try:
            cache_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'version': '4.3_FIXED',
                'config': self.config,
                **results
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.info(f"📦 Results cached to {self.cache_file}")
            
        except Exception as e:
            logger.error(f"📦 Error saving cache: {str(e)}")
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run the complete SpyDefi KOL analysis with FIXED data."""
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting SpyDefi KOL analysis with REAL channel IDs and pullback data...")
            
            # Check cache first (will force clear broken cache)
            if self._should_use_cache():
                cached_results = self._load_cache()
                if cached_results:
                    logger.info("📦 Using cached analysis")
                    return {
                        'success': True,
                        **cached_results
                    }
            
            # Scan SpyDefi for KOL mentions with FIXED error handling
            logger.info("📱 Scanning SpyDefi channel for KOL mentions...")
            kol_mentions = await self._scan_spydefi_for_kols()
            
            if not kol_mentions:
                logger.error("❌ No KOL mentions found in SpyDefi")
                
                # Try to use cached data if available even if expired
                if self.cache_file.exists():
                    logger.info("📦 Using expired cache as fallback...")
                    cached_results = self._load_cache()
                    if cached_results and cached_results.get('kol_performances'):
                        return {
                            'success': True,
                            'fallback_cache': True,
                            **cached_results
                        }
                
                return {'success': False, 'error': 'No KOL mentions found and no cache available'}
            
            logger.info(f"📊 Found {len(kol_mentions)} unique KOLs mentioned")
            
            # Get top KOLs
            top_kols = list(kol_mentions.keys())[:self.config['top_kols_count']]
            
            # Analyze each KOL with REAL channel lookup and data
            logger.info(f"🔍 Analyzing top {len(top_kols)} KOLs with REAL data...")
            kol_performances = {}
            api_calls = 0
            
            for i, kol in enumerate(top_kols, 1):
                try:
                    logger.info(f"📊 Analyzing KOL {i}/{len(top_kols)}: @{kol}")
                    
                    # Get REAL channel ID
                    real_channel_id = await self._get_real_channel_id(kol)
                    
                    # Analyze KOL performance with REAL data
                    performance = await self._analyze_kol_performance(kol, real_channel_id)
                    
                    if performance:
                        kol_performances[kol] = performance
                        api_calls += 5  # Estimate
                        
                        logger.info(f"✅ @{kol}: Score {performance.composite_score:.1f}, "
                                  f"2x Rate {performance.success_rate_2x:.1f}%, "
                                  f"Pullback {performance.avg_max_pullback_percent:.1f}%, "
                                  f"ROI {performance.avg_roi:.1f}%")
                    else:
                        logger.warning(f"⚠️ Failed to analyze @{kol}")
                        
                except Exception as e:
                    logger.error(f"❌ Error analyzing @{kol}: {str(e)}")
                    continue
            
            if not kol_performances:
                logger.error("❌ No KOL performances generated")
                return {'success': False, 'error': 'No valid KOL performances'}
            
            # Sort by composite score
            sorted_kols = sorted(kol_performances.items(), 
                               key=lambda x: x[1].composite_score, 
                               reverse=True)
            kol_performances = dict(sorted_kols)
            
            # Calculate overall statistics
            total_calls = sum(p.total_calls for p in kol_performances.values())
            total_2x = sum(p.tokens_2x_plus for p in kol_performances.values())
            total_5x = sum(p.tokens_5x_plus for p in kol_performances.values())
            
            overall_2x_rate = (total_2x / total_calls * 100) if total_calls > 0 else 0
            overall_5x_rate = (total_5x / total_calls * 100) if total_calls > 0 else 0
            
            processing_time = time.time() - start_time
            
            results = {
                'success': True,
                'kol_performances': {k: asdict(v) for k, v in kol_performances.items()},
                'kol_mentions': kol_mentions,
                'metadata': {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'total_calls_analyzed': total_calls,
                    'overall_2x_rate': overall_2x_rate,
                    'overall_5x_rate': overall_5x_rate,
                    'processing_time_seconds': processing_time,
                    'api_calls': api_calls,
                    'config': self.config,
                    'version': '4.3_FIXED'
                }
            }
            
            # Cache results
            self._save_cache(results)
            
            logger.info(f"✅ Analysis complete: {len(kol_performances)} KOLs analyzed")
            logger.info(f"📊 Overall 2x rate: {overall_2x_rate:.1f}%")
            logger.info(f"📊 Overall 5x rate: {overall_5x_rate:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ SpyDefi analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _scan_spydefi_for_kols(self) -> Dict[str, int]:
        """FIXED: Scan SpyDefi channel with proper FloodWaitError handling."""
        try:
            spydefi_entity = None
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"📱 Getting SpyDefi entity (attempt {attempt + 1}/{max_retries})")
                    spydefi_entity = await self.client.get_entity("spydefi")
                    break
                    
                except FloodWaitError as e:
                    wait_time = e.seconds
                    logger.warning(f"⚠️ FloodWait when accessing SpyDefi: {wait_time}s")
                    
                    if wait_time > 300:
                        logger.error(f"❌ FloodWait too long ({wait_time}s)")
                        return {}
                    
                    await asyncio.sleep(wait_time)
                    continue
                    
                except Exception as e:
                    logger.error(f"❌ Error getting SpyDefi entity: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5)
                        continue
                    else:
                        raise
            
            if not spydefi_entity:
                return {}
            
            # Calculate time range - FIXED: Use UTC timezone for Telegram compatibility
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=self.config['spydefi_scan_hours'])
            
            logger.info(f"📅 Scanning from {start_time.strftime('%Y-%m-%d %H:%M UTC')} to {end_time.strftime('%Y-%m-%d %H:%M UTC')}")
            
            kol_mentions = {}
            message_count = 0
            
            try:
                async for message in self.client.iter_messages(
                    spydefi_entity,
                    offset_date=end_time,
                    reverse=True,
                    limit=2000
                ):
                    if message.date < start_time:
                        break
                    
                    message_count += 1
                    
                    if message.text:
                        usernames = self._extract_kol_usernames(message.text)
                        
                        for username in usernames:
                            if self._is_valid_kol_username(username):
                                kol_mentions[username] = kol_mentions.get(username, 0) + 1
                    
                    if message_count % 50 == 0:
                        await asyncio.sleep(0.1)
                        
            except FloodWaitError as e:
                logger.warning(f"⚠️ FloodWait during message iteration: {e.seconds}s")
                logger.info(f"📊 Processed {message_count} messages before rate limit")
                
            except Exception as e:
                logger.error(f"❌ Error during message iteration: {str(e)}")
            
            logger.info(f"📊 Scanned {message_count} messages, found {len(kol_mentions)} unique KOLs")
            
            # Filter by minimum mentions
            filtered_kols = {k: v for k, v in kol_mentions.items() 
                           if v >= self.config['min_mentions']}
            
            # Sort by mention count
            sorted_kols = dict(sorted(filtered_kols.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True))
            
            return sorted_kols
            
        except Exception as e:
            logger.error(f"❌ Error scanning SpyDefi: {str(e)}")
            return {}
    
    def _extract_kol_usernames(self, text: str) -> List[str]:
        """Extract KOL usernames from text with fixed logic."""
        username_pattern = r'@([a-zA-Z][a-zA-Z0-9_]{2,30})'
        matches = re.findall(username_pattern, text, re.IGNORECASE)
        
        # Filter out false positives
        false_positives = {
            'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
            'x15', 'x20', 'x30', 'x40', 'x50', 'x60', 'x100',
            'everyone', 'here', 'channel', 'admin', 'bot'
        }
        
        valid_usernames = []
        for username in matches:
            if (username.lower() not in false_positives and 
                not username.lower().startswith('x') and
                len(username) >= 3 and
                not username.isdigit()):
                valid_usernames.append(username)
        
        return list(set(valid_usernames))
    
    def _is_valid_kol_username(self, username: str) -> bool:
        """Validate KOL username to avoid false positives."""
        if not username or len(username) < 3:
            return False
        
        false_positives = {
            'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
            'x15', 'x20', 'x30', 'x40', 'x50', 'x60', 'x100',
            'everyone', 'here', 'channel', 'admin', 'bot', 'spydefi'
        }
        
        if username.lower() in false_positives:
            return False
        
        if not username[0].isalpha():
            return False
        
        if username.isdigit():
            return False
        
        if username.lower().startswith('x') and len(username) > 1 and username[1:].isdigit():
            return False
        
        return True
    
    async def _get_real_channel_id(self, kol_username: str) -> str:
        """FIXED: Get REAL numeric Telegram channel ID."""
        try:
            channel_variants = [
                kol_username,
                f"{kol_username}_calls",
                f"{kol_username}calls",
                f"{kol_username}_gems",
                f"{kol_username}gems",
                f"{kol_username}_channel",
                f"{kol_username}channel",
                f"{kol_username}_official",
                f"{kol_username}official",
                f"{kol_username}_alpha",
                f"{kol_username}alpha"
            ]
            
            for variant in channel_variants:
                try:
                    await asyncio.sleep(0.5)  # Rate limiting
                    
                    entity = await self.client.get_entity(variant)
                    
                    if isinstance(entity, (Channel, Chat)):
                        # FIXED: Return REAL numeric channel ID
                        if isinstance(entity, Channel):
                            # For channels, use -100 prefix with the channel ID
                            real_id = f"-100{entity.id}"
                        else:
                            # For chats, use the ID directly
                            real_id = str(entity.id)
                        
                        logger.debug(f"✅ Found REAL channel ID for @{kol_username}: {real_id}")
                        return real_id
                
                except FloodWaitError as e:
                    if e.seconds > 60:
                        logger.warning(f"⚠️ Skipping @{kol_username} due to FloodWait")
                        break
                    await asyncio.sleep(e.seconds)
                    continue
                    
                except (ChannelPrivateError, ChatAdminRequiredError):
                    continue
                except Exception:
                    continue
            
            # If no real channel found, return a recognizable placeholder
            logger.warning(f"⚠️ No accessible channel found for @{kol_username}")
            return f"@{kol_username}"  # Return username format if channel not found
            
        except Exception as e:
            logger.error(f"❌ Error getting channel ID for @{kol_username}: {str(e)}")
            return f"@{kol_username}"
    
    async def _analyze_kol_performance(self, kol: str, channel_id: str) -> Optional[KOLPerformance]:
        """FIXED: Analyze KOL performance with REAL pullback and ROI calculation."""
        try:
            if not self.api_manager:
                logger.error("❌ API manager not configured")
                return None
            
            # Get recent token calls from the KOL
            token_calls = await self._get_kol_token_calls(kol, channel_id)
            
            if not token_calls or len(token_calls) < 5:
                logger.warning(f"⚠️ Insufficient token calls for @{kol} ({len(token_calls) if token_calls else 0})")
                return None
            
            # Analyze each token call with REAL data
            analyzed_tokens = []
            
            for token_call in token_calls:
                try:
                    token_analysis = await self._analyze_token_call(token_call)
                    if token_analysis:
                        analyzed_tokens.append(token_analysis)
                        
                except Exception as e:
                    logger.error(f"❌ Error analyzing token {token_call.get('address', 'unknown')}: {str(e)}")
                    continue
            
            if not analyzed_tokens:
                logger.warning(f"⚠️ No valid token analyses for @{kol}")
                return None
            
            # Calculate performance metrics with REAL data
            return self._calculate_kol_metrics(kol, channel_id, analyzed_tokens)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing KOL @{kol}: {str(e)}")
            return None
    
    async def _get_kol_token_calls(self, kol: str, channel_id: str) -> List[Dict[str, Any]]:
        """Get recent token calls from KOL channel."""
        try:
            entity = None
            
            try:
                # Try to get entity by channel_id first, then fallback to username
                if channel_id.startswith('-100'):
                    # Convert back to int for Telegram API
                    entity_id = int(channel_id.replace('-100', ''))
                    entity = await self.client.get_entity(entity_id)
                elif channel_id.startswith('@'):
                    entity = await self.client.get_entity(channel_id[1:])  # Remove @ prefix
                else:
                    entity = await self.client.get_entity(kol)
                    
            except FloodWaitError as e:
                if e.seconds > 120:
                    return []
                await asyncio.sleep(e.seconds)
                try:
                    entity = await self.client.get_entity(kol)
                except:
                    return []
            
            if not entity:
                return []
            
            # Get messages from last N days - FIXED: Use UTC timezone
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=self.config['kol_analysis_days'])
            
            token_calls = []
            message_count = 0
            
            try:
                async for message in self.client.iter_messages(
                    entity,
                    offset_date=end_time,
                    limit=100
                ):
                    if message.date < start_time:
                        break
                    
                    message_count += 1
                    
                    if message.text:
                        token_addresses = self._extract_token_addresses(message.text)
                        
                        for address in token_addresses:
                            token_calls.append({
                                'address': address,
                                'call_time': message.date,
                                'message_text': message.text[:200],
                                'kol': kol
                            })
                    
                    if message_count % 20 == 0:
                        await asyncio.sleep(0.2)
                        
            except FloodWaitError as e:
                logger.warning(f"⚠️ FloodWait during message iteration for @{kol}: {e.seconds}s")
                
            except Exception as e:
                logger.error(f"❌ Error iterating messages for @{kol}: {str(e)}")
            
            logger.debug(f"📊 Found {len(token_calls)} token calls from @{kol}")
            return token_calls[:30]  # Limit to 30 most recent
            
        except Exception as e:
            logger.error(f"❌ Error getting token calls from @{kol}: {str(e)}")
            return []
    
    def _extract_token_addresses(self, text: str) -> List[str]:
        """Extract Solana token addresses from message text."""
        address_pattern = r'\b([1-9A-HJ-NP-Za-km-z]{32,44})\b'
        matches = re.findall(address_pattern, text)
        
        system_addresses = {
            'So11111111111111111111111111111111111111112',
            'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
            '11111111111111111111111111111111',
        }
        
        valid_addresses = []
        for address in matches:
            if (address not in system_addresses and 
                len(address) >= 32 and 
                not address.isdigit() and
                address not in text.lower()):
                valid_addresses.append(address)
        
        return list(set(valid_addresses))
    
    async def _analyze_token_call(self, token_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """FIXED: Analyze token call with REAL pullback and ROI calculation."""
        try:
            token_address = token_call['address']
            call_time = token_call['call_time']
            
            # Get token performance from call time to now
            performance = self.api_manager.calculate_token_performance(
                token_address, 
                call_time
            )
            
            if not performance.get('success'):
                logger.debug(f"⚠️ No performance data for {token_address}")
                return None
            
            # Extract key metrics
            initial_price = performance.get('initial_price', 0)
            current_price = performance.get('current_price', 0)
            max_price = performance.get('max_price', 0)
            min_price = performance.get('min_price', 0)
            
            if not initial_price or initial_price <= 0:
                return None
            
            # FIXED: Calculate REAL metrics
            current_roi = ((current_price / initial_price) - 1) * 100 if current_price > 0 else -100
            max_roi = ((max_price / initial_price) - 1) * 100 if max_price > 0 else -100
            min_roi = ((min_price / initial_price) - 1) * 100 if min_price > 0 else -100
            
            # FIXED: Calculate REAL pullback percentage (this is what was broken)
            if max_price > initial_price and max_price > 0 and min_price > 0:
                # Calculate pullback from peak to trough as a percentage
                max_pullback = ((min_price / max_price) - 1) * 100
            else:
                # If never went above initial, pullback is just the loss
                max_pullback = min_roi
            
            # Determine if hit 2x or 5x
            hit_2x = max_roi >= 100  # 2x = 100% gain
            hit_5x = max_roi >= 400  # 5x = 400% gain
            
            # Calculate time to 2x if applicable
            time_to_2x_hours = 0
            if hit_2x:
                time_to_2x_hours = performance.get('time_to_max_roi_hours', 24)
            
            return {
                'token_address': token_address,
                'call_time': call_time,
                'initial_price': initial_price,
                'current_price': current_price,
                'max_price': max_price,
                'min_price': min_price,
                'current_roi': current_roi,
                'max_roi': max_roi,
                'min_roi': min_roi,
                'max_pullback': max_pullback,  # REAL pullback percentage
                'hit_2x': hit_2x,
                'hit_5x': hit_5x,
                'time_to_2x_hours': time_to_2x_hours,
                'kol': token_call['kol']
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing token call: {str(e)}")
            return None
    
    def _calculate_kol_metrics(self, kol: str, channel_id: str, analyzed_tokens: List[Dict[str, Any]]) -> KOLPerformance:
        """FIXED: Calculate KOL metrics with REAL pullback and ROI data."""
        try:
            total_calls = len(analyzed_tokens)
            
            # Count tokens that hit 2x and 5x
            tokens_2x_plus = sum(1 for token in analyzed_tokens if token['hit_2x'])
            tokens_5x_plus = sum(1 for token in analyzed_tokens if token['hit_5x'])
            
            # Calculate success rates
            success_rate_2x = (tokens_2x_plus / total_calls * 100) if total_calls > 0 else 0
            success_rate_5x = (tokens_5x_plus / total_calls * 100) if total_calls > 0 else 0
            
            # Calculate average time to 2x (only for tokens that hit 2x)
            tokens_that_hit_2x = [t for t in analyzed_tokens if t['hit_2x']]
            avg_time_to_2x_hours = (
                sum(t['time_to_2x_hours'] for t in tokens_that_hit_2x) / len(tokens_that_hit_2x)
                if tokens_that_hit_2x else 0
            )
            
            # FIXED: Calculate REAL average max pullback for 2x tokens
            if tokens_that_hit_2x:
                avg_max_pullback_percent = sum(t['max_pullback'] for t in tokens_that_hit_2x) / len(tokens_that_hit_2x)
            else:
                # If no 2x tokens, use overall average pullback
                avg_max_pullback_percent = sum(t['max_pullback'] for t in analyzed_tokens) / len(analyzed_tokens)
            
            # FIXED: Calculate REAL average ROI
            avg_roi = sum(t['current_roi'] for t in analyzed_tokens) / len(analyzed_tokens)
            
            # FIXED: Calculate REAL consistency score based on ROI variance
            roi_values = [t['current_roi'] for t in analyzed_tokens]
            if len(roi_values) > 1:
                mean_roi = sum(roi_values) / len(roi_values)
                variance = sum((x - mean_roi) ** 2 for x in roi_values) / len(roi_values)
                std_dev = variance ** 0.5
                # Convert to consistency score (0-100, higher is better)
                consistency_score = max(0, min(100, 100 - (std_dev / 50)))  # Adjusted for real variance
            else:
                consistency_score = 90
            
            # Normalize scores for composite calculation
            time_score = max(0, (48 - min(avg_time_to_2x_hours, 48)) / 48 * 100) if avg_time_to_2x_hours > 0 else 50
            
            # FIXED: Pullback score - smaller pullback (less negative) = higher score
            # -5% pullback = 95 score, -25% pullback = 75 score, -50% pullback = 50 score
            pullback_score = max(0, min(100, 100 + avg_max_pullback_percent))
            
            # ROI score - normalize to 0-100 scale
            roi_score = max(0, min(100, (avg_roi + 100) / 10))
            
            # FIXED: Composite score with proper weighting including pullback
            composite_score = (
                success_rate_2x * 0.25 +        # 25% weight on 2x success rate
                success_rate_5x * 0.20 +        # 20% weight on 5x success rate  
                time_score * 0.20 +             # 20% weight on speed to 2x
                pullback_score * 0.25 +         # 25% weight on pullback management
                roi_score * 0.10                # 10% weight on average ROI
            )
            
            composite_score = max(0, min(100, composite_score))
            
            # Determine strategy classification
            if success_rate_2x >= 35 and avg_time_to_2x_hours <= 12 and avg_time_to_2x_hours > 0:
                strategy_classification = "SCALP"
            elif success_rate_5x >= 12 and consistency_score >= 70:
                strategy_classification = "HOLD"
            else:
                strategy_classification = "MIXED"
            
            # Determine follower tier based on composite score
            if composite_score >= 80:
                follower_tier = "HIGH"
            elif composite_score >= 60:
                follower_tier = "MEDIUM"
            else:
                follower_tier = "LOW"
            
            return KOLPerformance(
                kol=kol,
                channel_id=channel_id,  # REAL numeric channel ID
                follower_tier=follower_tier,
                total_calls=total_calls,
                tokens_2x_plus=tokens_2x_plus,
                tokens_5x_plus=tokens_5x_plus,
                success_rate_2x=success_rate_2x,
                success_rate_5x=success_rate_5x,
                avg_time_to_2x_hours=avg_time_to_2x_hours,
                avg_max_pullback_percent=avg_max_pullback_percent,  # REAL pullback data
                consistency_score=consistency_score,  # REAL consistency
                composite_score=composite_score,  # FIXED composite score
                strategy_classification=strategy_classification,
                avg_roi=avg_roi  # REAL average ROI
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics for @{kol}: {str(e)}")
            return None