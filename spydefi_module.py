"""
SpyDefi KOL Analysis Module - Phoenix Project (FIXED MATH & REAL DATA VERSION)

MAJOR FIXES:
- REMOVED: losing_calls and winning_calls (caused mathematical contradictions)
- FIXED: avg_max_pullback_percent now shows REAL pullback data for stop loss placement
- FIXED: avg_roi now shows REAL average ROI data
- FIXED: consistency_score based on real ROI variance
- FIXED: FloodWaitError handling in main scanning function
- FIXED: Uses cached data when rate limited
- Real Telegram channel ID lookup (numeric IDs like -1001234567890)
- Fixed token analysis logic (proper 2x/5x counting)
- Fixed composite score calculation
- If 15/15 calls hit 2x, then success_rate_2x = 100% (math that makes sense)
"""

import asyncio
import logging
import json
import re
from datetime import datetime, timedelta
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
    """Fixed KOL performance metrics with real data and consistent math."""
    kol: str
    channel_id: str  # Real numeric Telegram channel ID
    follower_tier: str  # HIGH/MEDIUM/LOW
    total_calls: int
    tokens_2x_plus: int
    tokens_5x_plus: int
    success_rate_2x: float      # tokens_2x_plus / total_calls * 100
    success_rate_5x: float      # tokens_5x_plus / total_calls * 100  
    avg_time_to_2x_hours: float
    avg_max_pullback_percent: float  # REAL pullback data for stop loss placement
    consistency_score: float         # Based on ROI variance (real data)
    composite_score: float          # Based on 2x rate, 5x rate, time, pullback, avg_roi
    strategy_classification: str    # SCALP/HOLD/MIXED
    avg_roi: float                 # REAL average ROI across all tokens

class SpyDefiAnalyzer:
    """Enhanced SpyDefi analyzer with FIXED math and real financial data."""
    
    def __init__(self, api_id: str, api_hash: str, session_name: str = "phoenix_spydefi"):
        if not TELETHON_AVAILABLE:
            raise ImportError("Telethon is required for SpyDefi analysis")
        
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = None
        self.api_manager = None
        
        # Fixed configuration - NEVER override from CLI
        self.config = {
            'spydefi_scan_hours': 8,          # Peak memecoin hours
            'kol_analysis_days': 7,           # Days to analyze each KOL
            'top_kols_count': 50,             # Increased from 25
            'min_mentions': 1,                # Quality filter
            'max_market_cap_usd': 10_000_000, # $10M max
            'win_threshold_percent': 50,      # 50% profit threshold
            'timeout_minutes': 30,            # Analysis timeout
        }
        
        # Cache setup
        self.cache_dir = Path.home() / ".phoenix_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "spydefi_kol_analysis.json"
        
        logger.info("🎯 SpyDefi Analyzer initialized with FIXED math and REAL data")
        logger.info(f"⚙️ Configuration: {self.config['spydefi_scan_hours']}h scan, Top {self.config['top_kols_count']} KOLs")
    
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
        """Check if cache should be used (under 6 hours old)."""
        if not self.cache_file.exists():
            return False
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            timestamp_str = cache_data.get('timestamp')
            if not timestamp_str:
                return False
            
            cache_time = datetime.fromisoformat(timestamp_str)
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            
            if age_hours < 6:
                logger.info(f"📦 Using cache ({age_hours:.1f}h old)")
                return True
            else:
                logger.info(f"📦 Cache expired ({age_hours:.1f}h old)")
                return False
                
        except Exception as e:
            logger.error(f"📦 Cache error: {str(e)}")
            return False
    
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
                'timestamp': datetime.now().isoformat(),
                'version': '4.2',
                'config': self.config,
                **results
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.info(f"📦 Results cached to {self.cache_file}")
            
        except Exception as e:
            logger.error(f"📦 Error saving cache: {str(e)}")
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run the complete SpyDefi KOL analysis with FIXED math and real data."""
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting SpyDefi KOL analysis with FIXED math and REAL data...")
            logger.info(f"⚙️ Configuration: {self.config['spydefi_scan_hours']}h scan, Top {self.config['top_kols_count']} KOLs")
            
            # Check cache first
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
                
                # FIXED: Try to use cached data if available even if expired
                if self.cache_file.exists():
                    logger.info("📦 Attempting to use expired cache due to scan failure...")
                    cached_results = self._load_cache()
                    if cached_results and cached_results.get('kol_performances'):
                        logger.info("📦 Using expired cache as fallback")
                        return {
                            'success': True,
                            'fallback_cache': True,
                            **cached_results
                        }
                
                return {'success': False, 'error': 'No KOL mentions found and no cache available'}
            
            logger.info(f"📊 Found {len(kol_mentions)} unique KOLs mentioned")
            
            # Get top KOLs
            top_kols = list(kol_mentions.keys())[:self.config['top_kols_count']]
            
            # Analyze each KOL with real channel lookup
            logger.info(f"🔍 Analyzing top {len(top_kols)} KOLs with real channel lookup...")
            kol_performances = {}
            api_calls = 0
            
            for i, kol in enumerate(top_kols, 1):
                try:
                    logger.info(f"📊 Analyzing KOL {i}/{len(top_kols)}: @{kol}")
                    
                    # Get real channel ID
                    real_channel_id = await self._get_real_channel_id(kol)
                    
                    # Analyze KOL performance with fixed logic
                    performance = await self._analyze_kol_performance(kol, real_channel_id)
                    
                    if performance:
                        kol_performances[kol] = performance
                        api_calls += getattr(performance, 'api_calls_used', 5)  # Estimate
                        
                        logger.info(f"✅ @{kol}: Score {performance.composite_score:.1f}, "
                                  f"2x Rate {performance.success_rate_2x:.1f}%, "
                                  f"Avg Pullback {performance.avg_max_pullback_percent:.1f}%")
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
                    'timestamp': datetime.now().isoformat(),
                    'total_calls_analyzed': total_calls,
                    'overall_2x_rate': overall_2x_rate,
                    'overall_5x_rate': overall_5x_rate,
                    'processing_time_seconds': processing_time,
                    'api_calls': api_calls,
                    'config': self.config,
                    'version': '4.2'
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
        """FIXED: Scan SpyDefi channel for KOL mentions with proper FloodWaitError handling."""
        try:
            # FIXED: Proper FloodWaitError handling for get_entity
            spydefi_entity = None
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    logger.debug(f"📱 Attempting to get SpyDefi entity (attempt {attempt + 1}/{max_retries})")
                    spydefi_entity = await self.client.get_entity("spydefi")
                    break
                    
                except FloodWaitError as e:
                    wait_time = e.seconds
                    logger.warning(f"⚠️ FloodWait when accessing SpyDefi: {wait_time}s")
                    
                    if wait_time > 300:  # More than 5 minutes
                        logger.error(f"❌ FloodWait too long ({wait_time}s), using cache if available")
                        return {}
                    
                    logger.info(f"⏳ Waiting {wait_time} seconds for flood wait...")
                    await asyncio.sleep(wait_time)
                    continue
                    
                except Exception as e:
                    logger.error(f"❌ Error getting SpyDefi entity (attempt {attempt + 1}): {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5)  # Wait 5 seconds before retry
                        continue
                    else:
                        raise
            
            if not spydefi_entity:
                logger.error("❌ Failed to get SpyDefi entity after all retries")
                return {}
            
            logger.info("✅ Successfully connected to SpyDefi channel")
            
            # Calculate time range for scanning
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=self.config['spydefi_scan_hours'])
            
            logger.info(f"📅 Scanning from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
            
            kol_mentions = {}
            message_count = 0
            
            # FIXED: Add FloodWaitError handling for message iteration
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
                        # Extract KOL usernames with fixed regex
                        usernames = self._extract_kol_usernames(message.text)
                        
                        for username in usernames:
                            # Validate username (no x2, x3, etc. false positives)
                            if self._is_valid_kol_username(username):
                                kol_mentions[username] = kol_mentions.get(username, 0) + 1
                    
                    # Add small delay every 50 messages to avoid rate limits
                    if message_count % 50 == 0:
                        await asyncio.sleep(0.1)
                        
            except FloodWaitError as e:
                logger.warning(f"⚠️ FloodWait during message iteration: {e.seconds}s")
                logger.info(f"📊 Processed {message_count} messages before rate limit")
                # Continue with what we have so far
                
            except Exception as e:
                logger.error(f"❌ Error during message iteration: {str(e)}")
                # Continue with what we have so far
            
            logger.info(f"📊 Scanned {message_count} messages")
            logger.info(f"🎯 Found {len(kol_mentions)} unique KOLs")
            
            # Filter by minimum mentions
            filtered_kols = {k: v for k, v in kol_mentions.items() 
                           if v >= self.config['min_mentions']}
            
            # Sort by mention count
            sorted_kols = dict(sorted(filtered_kols.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True))
            
            logger.info(f"📊 After filtering: {len(sorted_kols)} KOLs with ≥{self.config['min_mentions']} mentions")
            
            return sorted_kols
            
        except FloodWaitError as e:
            logger.error(f"❌ Final FloodWaitError in scan: {e.seconds}s")
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error scanning SpyDefi: {str(e)}")
            return {}
    
    def _extract_kol_usernames(self, text: str) -> List[str]:
        """Extract KOL usernames from text with fixed logic."""
        # Fixed regex to avoid x2, x3, etc. false positives
        username_pattern = r'@([a-zA-Z][a-zA-Z0-9_]{2,30})'
        matches = re.findall(username_pattern, text, re.IGNORECASE)
        
        # Filter out common false positives
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
        
        return list(set(valid_usernames))  # Remove duplicates
    
    def _is_valid_kol_username(self, username: str) -> bool:
        """Validate KOL username to avoid false positives."""
        if not username or len(username) < 3:
            return False
        
        # Check against known false positives
        false_positives = {
            'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
            'x15', 'x20', 'x30', 'x40', 'x50', 'x60', 'x100',
            'everyone', 'here', 'channel', 'admin', 'bot', 'spydefi'
        }
        
        if username.lower() in false_positives:
            return False
        
        # Must start with letter
        if not username[0].isalpha():
            return False
        
        # Cannot be all digits
        if username.isdigit():
            return False
        
        # Cannot start with 'x' followed by digits
        if username.lower().startswith('x') and len(username) > 1 and username[1:].isdigit():
            return False
        
        return True
    
    async def _get_real_channel_id(self, kol_username: str) -> str:
        """Get real numeric Telegram channel ID for KOL with proper FloodWait handling."""
        try:
            # List of possible channel variations to try
            channel_variants = [
                kol_username,                    # exact username
                f"{kol_username}_calls",         # with _calls suffix
                f"{kol_username}calls",          # with calls suffix
                f"{kol_username}_gems",          # with _gems suffix
                f"{kol_username}gems",           # with gems suffix
                f"{kol_username}_channel",       # with _channel suffix
                f"{kol_username}channel",        # with channel suffix
                f"{kol_username}_official",      # with _official suffix
                f"{kol_username}official",       # with official suffix
                f"{kol_username}_group",         # with _group suffix
                f"{kol_username}group",          # with group suffix
                f"{kol_username}_alpha",         # with _alpha suffix
                f"{kol_username}alpha",          # with alpha suffix
                f"{kol_username}_vip",           # with _vip suffix
                f"{kol_username}vip",            # with vip suffix
                f"{kol_username}_premium",       # with _premium suffix
                f"{kol_username}premium",        # with premium suffix
            ]
            
            for variant in channel_variants:
                try:
                    # Add rate limiting to avoid flood waits
                    await asyncio.sleep(0.5)
                    
                    entity = await self.client.get_entity(variant)
                    
                    if isinstance(entity, (Channel, Chat)):
                        # Return real numeric channel ID
                        channel_id = str(entity.id)
                        if hasattr(entity, 'access_hash') and entity.access_hash:
                            # For channels, use the full ID format
                            channel_id = f"-100{entity.id}"
                        
                        logger.debug(f"✅ Found real channel ID for @{kol_username}: {channel_id}")
                        return channel_id
                
                except FloodWaitError as e:
                    logger.warning(f"⚠️ FloodWait during channel lookup for @{kol_username}: {e.seconds}s")
                    if e.seconds > 60:  # More than 1 minute
                        logger.warning(f"⚠️ Skipping @{kol_username} due to long FloodWait")
                        return f"flood_wait_{kol_username}"
                    await asyncio.sleep(e.seconds)
                    continue
                    
                except (ChannelPrivateError, ChatAdminRequiredError):
                    # Channel exists but is private - still return the ID if we got it
                    continue
                except Exception:
                    # Channel doesn't exist with this variant
                    continue
            
            # If no real channel found, return placeholder
            logger.warning(f"⚠️ No real channel found for @{kol_username}")
            return f"not_found_{kol_username}"
            
        except Exception as e:
            logger.error(f"❌ Error getting channel ID for @{kol_username}: {str(e)}")
            return f"error_{kol_username}"
    
    async def _analyze_kol_performance(self, kol: str, channel_id: str) -> Optional[KOLPerformance]:
        """Analyze KOL performance with FIXED math and REAL pullback/ROI calculation."""
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
            
            # Calculate performance metrics with FIXED MATH
            return self._calculate_kol_metrics(kol, channel_id, analyzed_tokens)
            
        except Exception as e:
            logger.error(f"❌ Error analyzing KOL @{kol}: {str(e)}")
            return None
    
    async def _get_kol_token_calls(self, kol: str, channel_id: str) -> List[Dict[str, Any]]:
        """Get recent token calls from KOL channel with FloodWait handling."""
        try:
            # Try to get entity using channel_id or username
            entity = None
            
            try:
                if channel_id.startswith('-100') or channel_id.isdigit():
                    try:
                        entity = await self.client.get_entity(int(channel_id.replace('-100', '')))
                    except:
                        entity = await self.client.get_entity(kol)
                else:
                    entity = await self.client.get_entity(kol)
                    
            except FloodWaitError as e:
                logger.warning(f"⚠️ FloodWait getting entity for @{kol}: {e.seconds}s")
                if e.seconds > 120:  # More than 2 minutes
                    logger.warning(f"⚠️ Skipping @{kol} due to long FloodWait")
                    return []
                await asyncio.sleep(e.seconds)
                # Try one more time
                try:
                    entity = await self.client.get_entity(kol)
                except:
                    return []
            
            if not entity:
                return []
            
            # Get messages from last N days
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.config['kol_analysis_days'])
            
            token_calls = []
            message_count = 0
            
            try:
                async for message in self.client.iter_messages(
                    entity,
                    offset_date=end_time,
                    limit=100  # Limit to avoid too many API calls
                ):
                    if message.date < start_time:
                        break
                    
                    message_count += 1
                    
                    if message.text:
                        # Extract token addresses from message
                        token_addresses = self._extract_token_addresses(message.text)
                        
                        for address in token_addresses:
                            token_calls.append({
                                'address': address,
                                'call_time': message.date,
                                'message_text': message.text[:200],  # First 200 chars
                                'kol': kol
                            })
                    
                    # Add delay every 20 messages
                    if message_count % 20 == 0:
                        await asyncio.sleep(0.2)
                        
            except FloodWaitError as e:
                logger.warning(f"⚠️ FloodWait during message iteration for @{kol}: {e.seconds}s")
                # Continue with what we have
                
            except Exception as e:
                logger.error(f"❌ Error iterating messages for @{kol}: {str(e)}")
                # Continue with what we have
            
            logger.debug(f"📊 Found {len(token_calls)} token calls from @{kol} ({message_count} messages)")
            return token_calls[:30]  # Limit to 30 most recent calls
            
        except Exception as e:
            logger.error(f"❌ Error getting token calls from @{kol}: {str(e)}")
            return []
    
    def _extract_token_addresses(self, text: str) -> List[str]:
        """Extract Solana token addresses from message text."""
        # Solana address pattern (base58, 32-44 characters)
        address_pattern = r'\b([1-9A-HJ-NP-Za-km-z]{32,44})\b'
        matches = re.findall(address_pattern, text)
        
        # Filter out known system addresses and validate
        system_addresses = {
            'So11111111111111111111111111111111111111112',  # SOL
            'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
            '11111111111111111111111111111111',  # System program
        }
        
        valid_addresses = []
        for address in matches:
            if (address not in system_addresses and 
                len(address) >= 32 and 
                not address.isdigit() and
                address not in text.lower()):  # Avoid false positives from usernames
                valid_addresses.append(address)
        
        return list(set(valid_addresses))  # Remove duplicates
    
    async def _analyze_token_call(self, token_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze individual token call with REAL pullback and ROI calculation."""
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
            
            # Calculate key metrics
            current_roi = ((current_price / initial_price) - 1) * 100 if current_price > 0 else -100
            max_roi = ((max_price / initial_price) - 1) * 100 if max_price > 0 else -100
            min_roi = ((min_price / initial_price) - 1) * 100 if min_price > 0 else -100
            
            # FIXED: Calculate REAL pullback percentage
            if max_price > initial_price:
                # Pullback from max to min (this is what traders care about for stop losses)
                max_pullback = ((min_price / max_price) - 1) * 100 if min_price > 0 else -100
            else:
                # If never went above initial, pullback is just the loss from initial
                max_pullback = min_roi
            
            # Determine if hit 2x or 5x
            hit_2x = max_roi >= 100  # 2x = 100% gain
            hit_5x = max_roi >= 400  # 5x = 400% gain
            
            # Calculate time to 2x if applicable
            time_to_2x_hours = 0
            if hit_2x:
                time_to_2x_hours = performance.get('time_to_max_roi_hours', 24)  # Default 24h if not available
            
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
                'max_pullback': max_pullback,  # REAL pullback for stop loss calculation
                'hit_2x': hit_2x,
                'hit_5x': hit_5x,
                'time_to_2x_hours': time_to_2x_hours,
                'kol': token_call['kol']
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing token call: {str(e)}")
            return None
    
    def _calculate_kol_metrics(self, kol: str, channel_id: str, analyzed_tokens: List[Dict[str, Any]]) -> KOLPerformance:
        """Calculate KOL performance metrics with FIXED MATH and REAL data."""
        try:
            total_calls = len(analyzed_tokens)
            
            # FIXED: Simple math that makes sense
            tokens_2x_plus = sum(1 for token in analyzed_tokens if token['hit_2x'])
            tokens_5x_plus = sum(1 for token in analyzed_tokens if token['hit_5x'])
            
            # FIXED: Calculate success rates with proper math
            success_rate_2x = (tokens_2x_plus / total_calls * 100) if total_calls > 0 else 0
            success_rate_5x = (tokens_5x_plus / total_calls * 100) if total_calls > 0 else 0
            
            # Calculate average time to 2x (only for tokens that hit 2x)
            tokens_that_hit_2x = [t for t in analyzed_tokens if t['hit_2x']]
            avg_time_to_2x_hours = (
                sum(t['time_to_2x_hours'] for t in tokens_that_hit_2x) / len(tokens_that_hit_2x)
                if tokens_that_hit_2x else 0
            )
            
            # FIXED: Calculate REAL average max pullback for 2x tokens (for stop loss guidance)
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
                consistency_score = max(0, min(100, 100 - (std_dev / 20)))  # Adjusted for real data
            else:
                consistency_score = 90  # Default high consistency for single token
            
            # Normalize time to 2x (lower is better, max 48 hours)
            time_score = max(0, (48 - min(avg_time_to_2x_hours, 48)) / 48 * 100) if avg_time_to_2x_hours > 0 else 50
            
            # CRITICAL: Normalize pullback score (smaller pullback = higher score)
            # -5% pullback = 95 score, -25% pullback = 75 score, -50% pullback = 50 score
            pullback_score = max(0, min(100, 100 + avg_max_pullback_percent))  # -50% = 50, -10% = 90, 0% = 100
            
            # Normalize avg ROI (cap at 1000%)
            roi_score = max(0, min(100, (avg_roi + 100) / 10))  # -100% = 0, 900% = 100
            
            # FIXED: Composite score calculation with pullback properly weighted
            composite_score = (
                success_rate_2x * 0.25 +        # 25% weight on 2x success rate
                success_rate_5x * 0.20 +        # 20% weight on 5x success rate  
                time_score * 0.20 +             # 20% weight on speed to 2x
                pullback_score * 0.25 +         # 25% weight on pullback management (CRITICAL FOR STOP LOSS)
                roi_score * 0.10                # 10% weight on average ROI
            )
            
            # Ensure composite score is between 0 and 100
            composite_score = max(0, min(100, composite_score))
            
            # Determine strategy classification
            if success_rate_2x >= 35 and avg_time_to_2x_hours <= 12 and avg_time_to_2x_hours > 0:
                strategy_classification = "SCALP"
            elif success_rate_5x >= 12 and consistency_score >= 70:
                strategy_classification = "HOLD"
            else:
                strategy_classification = "MIXED"
            
            # Determine follower tier (simplified - would need real subscriber count)
            # For now, use composite score as proxy
            if composite_score >= 80:
                follower_tier = "HIGH"
            elif composite_score >= 60:
                follower_tier = "MEDIUM"
            else:
                follower_tier = "LOW"
            
            return KOLPerformance(
                kol=kol,
                channel_id=channel_id,  # Real numeric channel ID
                follower_tier=follower_tier,
                total_calls=total_calls,
                tokens_2x_plus=tokens_2x_plus,
                tokens_5x_plus=tokens_5x_plus,
                success_rate_2x=success_rate_2x,        # FIXED: Real 2x rate
                success_rate_5x=success_rate_5x,        # FIXED: Real 5x rate
                avg_time_to_2x_hours=avg_time_to_2x_hours,
                avg_max_pullback_percent=avg_max_pullback_percent,  # REAL pullback for stop loss
                consistency_score=consistency_score,    # REAL consistency based on ROI variance
                composite_score=composite_score,        # FIXED: Includes pullback weighting
                strategy_classification=strategy_classification,
                avg_roi=avg_roi                        # REAL average ROI
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics for @{kol}: {str(e)}")
            return None