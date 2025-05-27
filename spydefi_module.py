"""
SpyDefi KOL Analysis Module - Phoenix Project (FINAL FIXED VERSION)

CRITICAL FIXES IMPLEMENTED:
1. TOKEN-BASED ANALYSIS: Analyze fixed number of tokens (5 initial, +5 for ties)
2. WIN/LOSS TRACKING: -50% from initial = loss, everything else categorized properly
3. CHAIN FILTERING: Only warn for Solana tokens, silently skip ETH/other chains
4. TIE-BREAKING SYSTEM: Continue analyzing more tokens until clear ranking
5. DETAILED WIN METRICS: 2x rate, time to 2x, pullback analysis for wins only
6. PERFORMANCE OPTIMIZATION: Predictable token count, no time-based scanning
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
    """FIXED KOL performance metrics with win/loss tracking and tie-breaking support."""
    kol: str
    channel_id: str  # REAL numeric Telegram channel ID like -1001234567890
    follower_tier: str  # HIGH/MEDIUM/LOW
    total_calls: int
    wins: int  # Tokens that didn't dip -50% from initial
    losses: int  # Tokens that dipped -50% or more from initial
    tokens_2x_plus: int  # From wins only
    tokens_5x_plus: int  # From wins only
    win_rate: float  # wins / total_calls * 100
    success_rate_2x: float  # tokens_2x_plus / wins * 100 (from wins only)
    success_rate_5x: float  # tokens_5x_plus / wins * 100 (from wins only)
    avg_time_to_2x_hours: float  # Average time to reach 2x for winning 2x tokens
    avg_max_pullback_percent: float  # Average max pullback for 2x winners before hitting 2x
    consistency_score: float  # Based on ROI variance of wins
    composite_score: float  # Overall performance score
    strategy_classification: str  # SCALP/HOLD/MIXED
    avg_roi: float  # Average ROI of all tokens (wins and losses)
    tokens_analyzed: int  # Track how many tokens were actually analyzed for tie-breaking

class SpyDefiAnalyzer:
    """FIXED SpyDefi analyzer with token-based analysis and tie-breaking system."""
    
    def __init__(self, api_id: str, api_hash: str, session_name: str = "phoenix_spydefi"):
        if not TELETHON_AVAILABLE:
            raise ImportError("Telethon is required for SpyDefi analysis")
        
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = None
        self.api_manager = None
        
        # Fixed configuration for token-based analysis
        self.config = {
            'spydefi_scan_hours': 8,
            'base_tokens_per_kol': 5,  # Start with 5 tokens per KOL
            'tie_break_tokens': 5,  # Add 5 more tokens for ties
            'max_tokens_per_kol': 25,  # Maximum tokens to analyze per KOL
            'top_kols_count': 50,
            'min_mentions': 1,
            'max_market_cap_usd': 10_000_000,
            'loss_threshold_percent': -50,  # -50% = loss
            'tie_threshold_points': 3,  # Score difference for tie
            'timeout_minutes': 30,
        }
        
        # Cache setup
        self.cache_dir = Path.home() / ".phoenix_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "spydefi_kol_analysis.json"
        
        logger.info("🎯 SpyDefi Analyzer initialized with TOKEN-BASED analysis and tie-breaking")
    
    def set_api_manager(self, api_manager):
        """Set the API manager for token analysis."""
        self.api_manager = api_manager
        logger.info("✅ API manager configured for TOKEN-BASED analysis")
    
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
        """FIXED: Force clear old cache - always run fresh token-based analysis."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check version to force refresh for new token-based system
                version = cache_data.get('version', '')
                if version != '5.0_TOKEN_BASED':
                    logger.info(f"🧹 CLEARING OLD CACHE: Version {version} != 5.0_TOKEN_BASED")
                    self.cache_file.unlink()
                    return False
                
                # Check timestamp age
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
                
                logger.info("📦 Valid token-based cache found")
                return True
                
            except Exception as e:
                logger.error(f"📦 Cache error, clearing: {str(e)}")
                if self.cache_file.exists():
                    self.cache_file.unlink()
                return False
        
        return False
    
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load cached analysis results."""
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            kol_count = len(cache_data.get('kol_performances', {}))
            logger.info(f"📦 Loaded token-based cache with {kol_count} KOLs")
            return cache_data
            
        except Exception as e:
            logger.error(f"📦 Error loading cache: {str(e)}")
            return None
    
    def _save_cache(self, results: Dict[str, Any]):
        """Save analysis results to cache."""
        try:
            cache_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'version': '5.0_TOKEN_BASED',
                'config': self.config,
                **results
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            
            logger.info(f"📦 Token-based results cached to {self.cache_file}")
            
        except Exception as e:
            logger.error(f"📦 Error saving cache: {str(e)}")
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run the complete SpyDefi KOL analysis with TOKEN-BASED system and tie-breaking."""
        start_time = time.time()
        
        try:
            logger.info("🚀 Starting SpyDefi TOKEN-BASED KOL analysis with tie-breaking system...")
            
            # Check cache first
            if self._should_use_cache():
                cached_results = self._load_cache()
                if cached_results:
                    logger.info("📦 Using cached token-based analysis")
                    return {
                        'success': True,
                        **cached_results
                    }
            
            # Scan SpyDefi for KOL mentions
            logger.info("📱 Scanning SpyDefi channel for KOL mentions...")
            kol_mentions = await self._scan_spydefi_for_kols()
            
            if not kol_mentions:
                logger.error("❌ No KOL mentions found in SpyDefi")
                return {'success': False, 'error': 'No KOL mentions found'}
            
            logger.info(f"📊 Found {len(kol_mentions)} unique KOLs mentioned")
            
            # Get top KOLs
            top_kols = list(kol_mentions.keys())[:self.config['top_kols_count']]
            
            # PHASE 1: Initial analysis with base token count
            logger.info(f"🎯 PHASE 1: Analyzing {len(top_kols)} KOLs with {self.config['base_tokens_per_kol']} tokens each...")
            kol_performances = await self._analyze_kols_with_token_count(top_kols, self.config['base_tokens_per_kol'])
            
            if not kol_performances:
                logger.error("❌ No KOL performances generated in Phase 1")
                return {'success': False, 'error': 'No valid KOL performances'}
            
            # PHASE 2: Tie-breaking system
            final_performances = await self._resolve_ties(kol_performances)
            
            # Sort by composite score
            sorted_kols = sorted(final_performances.items(), 
                               key=lambda x: x[1].composite_score, 
                               reverse=True)
            final_performances = dict(sorted_kols)
            
            # Calculate overall statistics
            total_calls = sum(p.total_calls for p in final_performances.values())
            total_wins = sum(p.wins for p in final_performances.values())
            total_losses = sum(p.losses for p in final_performances.values())
            total_2x = sum(p.tokens_2x_plus for p in final_performances.values())
            total_5x = sum(p.tokens_5x_plus for p in final_performances.values())
            
            overall_win_rate = (total_wins / total_calls * 100) if total_calls > 0 else 0
            overall_2x_rate = (total_2x / total_wins * 100) if total_wins > 0 else 0
            overall_5x_rate = (total_5x / total_wins * 100) if total_wins > 0 else 0
            
            processing_time = time.time() - start_time
            
            results = {
                'success': True,
                'kol_performances': {k: asdict(v) for k, v in final_performances.items()},
                'kol_mentions': kol_mentions,
                'metadata': {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'total_calls_analyzed': total_calls,
                    'total_wins': total_wins,
                    'total_losses': total_losses,
                    'overall_win_rate': overall_win_rate,
                    'overall_2x_rate': overall_2x_rate,
                    'overall_5x_rate': overall_5x_rate,
                    'processing_time_seconds': processing_time,
                    'config': self.config,
                    'version': '5.0_TOKEN_BASED'
                }
            }
            
            # Cache results
            self._save_cache(results)
            
            logger.info(f"✅ TOKEN-BASED analysis complete: {len(final_performances)} KOLs analyzed")
            logger.info(f"📊 Overall win rate: {overall_win_rate:.1f}%")
            logger.info(f"📊 Overall 2x rate (from wins): {overall_2x_rate:.1f}%")
            logger.info(f"📊 Overall 5x rate (from wins): {overall_5x_rate:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ SpyDefi TOKEN-BASED analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _analyze_kols_with_token_count(self, kols: List[str], token_count: int) -> Dict[str, KOLPerformance]:
        """Analyze KOLs with specific token count."""
        performances = {}
        
        for i, kol in enumerate(kols, 1):
            try:
                logger.info(f"📊 Analyzing KOL {i}/{len(kols)}: @{kol} ({token_count} tokens)")
                
                # Get REAL channel ID
                real_channel_id = await self._get_real_channel_id(kol)
                
                # Analyze KOL performance with specific token count
                performance = await self._analyze_kol_performance(kol, real_channel_id, token_count)
                
                if performance:
                    performances[kol] = performance
                    logger.info(f"✅ @{kol}: Score {performance.composite_score:.1f}, "
                              f"Win Rate {performance.win_rate:.1f}%, "
                              f"2x Rate {performance.success_rate_2x:.1f}%")
                else:
                    logger.warning(f"⚠️ Failed to analyze @{kol}")
                    
            except Exception as e:
                logger.error(f"❌ Error analyzing @{kol}: {str(e)}")
                continue
        
        return performances
    
    async def _resolve_ties(self, initial_performances: Dict[str, KOLPerformance]) -> Dict[str, KOLPerformance]:
        """Resolve ties by analyzing more tokens for tied KOLs."""
        logger.info("🔍 PHASE 2: Checking for ties and resolving...")
        
        # Sort by composite score to identify ties
        sorted_performances = sorted(initial_performances.items(), 
                                   key=lambda x: x[1].composite_score, 
                                   reverse=True)
        
        # Find groups of tied KOLs
        tie_groups = []
        current_group = [sorted_performances[0]]
        
        for i in range(1, len(sorted_performances)):
            current_score = sorted_performances[i][1].composite_score
            previous_score = current_group[-1][1].composite_score
            
            if abs(current_score - previous_score) <= self.config['tie_threshold_points']:
                current_group.append(sorted_performances[i])
            else:
                if len(current_group) > 1:
                    tie_groups.append(current_group)
                current_group = [sorted_performances[i]]
        
        # Don't forget the last group
        if len(current_group) > 1:
            tie_groups.append(current_group)
        
        if not tie_groups:
            logger.info("✅ No ties found, rankings are clear")
            return initial_performances
        
        logger.info(f"🎯 Found {len(tie_groups)} tie groups, resolving with additional tokens...")
        
        final_performances = initial_performances.copy()
        
        for group_i, tie_group in enumerate(tie_groups, 1):
            tied_kols = [kol for kol, _ in tie_group]
            logger.info(f"🔄 Resolving tie group {group_i}: {', '.join(f'@{k}' for k in tied_kols)}")
            
            # Analyze more tokens for tied KOLs
            current_token_count = tie_group[0][1].tokens_analyzed
            
            while current_token_count < self.config['max_tokens_per_kol']:
                current_token_count += self.config['tie_break_tokens']
                logger.info(f"   📊 Analyzing {current_token_count} total tokens for tied KOLs...")
                
                # Re-analyze tied KOLs with more tokens
                updated_performances = await self._analyze_kols_with_token_count(tied_kols, current_token_count)
                
                # Update final performances
                for kol, performance in updated_performances.items():
                    if performance:
                        final_performances[kol] = performance
                
                # Check if tie is resolved
                tied_scores = [final_performances[kol].composite_score for kol in tied_kols if kol in final_performances]
                if len(tied_scores) > 1:
                    max_diff = max(tied_scores) - min(tied_scores)
                    if max_diff > self.config['tie_threshold_points']:
                        logger.info(f"   ✅ Tie resolved after {current_token_count} tokens (score spread: {max_diff:.1f})")
                        break
                
                if current_token_count >= self.config['max_tokens_per_kol']:
                    logger.info(f"   ⚠️ Max tokens reached ({current_token_count}), tie remains")
                    break
        
        return final_performances
    
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
            
            logger.info(f"✅ Successfully connected to SpyDefi channel")
            logger.info(f"📊 Channel type: {type(spydefi_entity).__name__}")
            if hasattr(spydefi_entity, 'title'):
                logger.info(f"📊 Channel title: {spydefi_entity.title}")
            if hasattr(spydefi_entity, 'participants_count'):
                logger.info(f"📊 Participants: {spydefi_entity.participants_count}")
            
            # Calculate time range for scanning
            current_time = datetime.now(timezone.utc)
            start_time = current_time - timedelta(hours=self.config['spydefi_scan_hours'])
            
            logger.info(f"📅 Scanning from {start_time.strftime('%Y-%m-%d %H:%M UTC')} to {current_time.strftime('%Y-%m-%d %H:%M UTC')}")
            
            # Test channel access
            logger.info("🔍 Testing channel access with recent messages...")
            test_count = 0
            try:
                async for message in self.client.iter_messages(spydefi_entity, limit=10):
                    test_count += 1
                    if message.text:
                        logger.debug(f"📝 Sample message: {message.text[:100]}...")
                logger.info(f"✅ Channel access works - found {test_count} recent messages")
            except Exception as e:
                logger.error(f"❌ Channel access test failed: {str(e)}")
                return {}
            
            kol_mentions = {}
            message_count = 0
            
            try:
                async for message in self.client.iter_messages(
                    spydefi_entity,
                    limit=2000
                ):
                    # Check if message is within our time range
                    if message.date < start_time:
                        break
                    
                    message_count += 1
                    
                    if message.text:
                        usernames = self._extract_kol_usernames(message.text)
                        
                        for username in usernames:
                            if self._is_valid_kol_username(username):
                                kol_mentions[username] = kol_mentions.get(username, 0) + 1
                    
                    # Log progress every 100 messages
                    if message_count % 100 == 0:
                        logger.info(f"📊 Processed {message_count} messages, found {len(kol_mentions)} KOLs so far")
                        await asyncio.sleep(0.1)
                        
            except FloodWaitError as e:
                logger.warning(f"⚠️ FloodWait during message iteration: {e.seconds}s")
                logger.info(f"📊 Processed {message_count} messages before rate limit")
                
            except Exception as e:
                logger.error(f"❌ Error during message iteration: {str(e)}")
                logger.info(f"📊 Processed {message_count} messages before error")
            
            logger.info(f"📊 Scanned {message_count} messages, found {len(kol_mentions)} unique KOLs")
            
            # Filter by minimum mentions
            filtered_kols = {k: v for k, v in kol_mentions.items() 
                           if v >= self.config['min_mentions']}
            
            # Sort by mention count
            sorted_kols = dict(sorted(filtered_kols.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True))
            
            logger.info(f"📊 Final results: {len(sorted_kols)} KOLs with ≥{self.config['min_mentions']} mentions")
            
            # Log top KOLs found
            if sorted_kols:
                logger.info("🏆 Top KOLs found:")
                for i, (kol, mentions) in enumerate(list(sorted_kols.items())[:10], 1):
                    logger.info(f"   {i}. @{kol} ({mentions} mentions)")
            
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
                        # Return REAL numeric channel ID
                        if isinstance(entity, Channel):
                            real_id = f"-100{entity.id}"
                        else:
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
            
            # If no real channel found, return username format
            logger.warning(f"⚠️ No accessible channel found for @{kol_username}")
            return f"@{kol_username}"
            
        except Exception as e:
            logger.error(f"❌ Error getting channel ID for @{kol_username}: {str(e)}")
            return f"@{kol_username}"
    
    async def _analyze_kol_performance(self, kol: str, channel_id: str, token_count: int) -> Optional[KOLPerformance]:
        """FIXED: Analyze KOL performance with specific token count and win/loss tracking."""
        try:
            if not self.api_manager:
                logger.error("❌ API manager not configured")
                return None
            
            logger.info(f"🔍 Getting exactly {token_count} tokens for @{kol}...")
            
            # Get exact number of token calls from the KOL
            token_calls = await self._get_kol_token_calls(kol, channel_id, token_count)
            
            if not token_calls or len(token_calls) < 3:
                logger.warning(f"⚠️ Insufficient tokens for @{kol} ({len(token_calls) if token_calls else 0}/{token_count})")
                return None
            
            logger.info(f"📊 Analyzing {len(token_calls)} tokens for @{kol}...")
            
            # Analyze each token call with win/loss tracking
            analyzed_tokens = []
            
            for i, token_call in enumerate(token_calls, 1):
                try:
                    logger.info(f"   Token {i}/{len(token_calls)}: {token_call['address'][:8]}...")
                    
                    token_analysis = await self._analyze_token_call_with_winloss(token_call)
                    if token_analysis:
                        analyzed_tokens.append(token_analysis)
                        
                        # Log result
                        if token_analysis['is_loss']:
                            logger.info(f"   ❌ LOSS: {token_analysis['current_roi']:.1f}% (< -50%)")
                        elif token_analysis['hit_2x']:
                            logger.info(f"   🚀 WIN 2x+: {token_analysis['current_roi']:.1f}%")
                        else:
                            logger.info(f"   ✅ WIN: {token_analysis['current_roi']:.1f}%")
                    else:
                        logger.warning(f"   ⚠️ Token {i} analysis failed")
                        
                except Exception as e:
                    logger.error(f"   ❌ Error analyzing token {i}: {str(e)}")
                    continue
            
            if not analyzed_tokens:
                logger.warning(f"⚠️ No valid token analyses for @{kol}")
                return None
            
            # Calculate performance metrics with win/loss tracking
            performance = self._calculate_kol_metrics_with_winloss(kol, channel_id, analyzed_tokens, len(token_calls))
            
            logger.info(f"✅ @{kol} complete: {performance.wins}W/{performance.losses}L, Score: {performance.composite_score:.1f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"❌ Error analyzing KOL @{kol}: {str(e)}")
            return None
    
    async def _get_kol_token_calls(self, kol: str, channel_id: str, token_count: int) -> List[Dict[str, Any]]:
        """Get exact number of recent token calls from KOL channel."""
        try:
            entity = None
            
            try:
                # Try to get entity by channel_id first, then fallback to username
                if channel_id.startswith('-100'):
                    entity_id = int(channel_id.replace('-100', ''))
                    entity = await self.client.get_entity(entity_id)
                elif channel_id.startswith('@'):
                    entity = await self.client.get_entity(channel_id[1:])
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
            
            token_calls = []
            message_count = 0
            
            try:
                async for message in self.client.iter_messages(entity, limit=500):  # Scan more messages to find tokens
                    message_count += 1
                    
                    if message.text:
                        token_addresses = self._extract_token_addresses(message.text)
                        
                        for address in token_addresses:
                            # Only add Solana tokens
                            if self._is_solana_token(address):
                                token_calls.append({
                                    'address': address,
                                    'call_time': message.date,
                                    'message_text': message.text[:200],
                                    'kol': kol
                                })
                                
                                # Stop when we have enough tokens
                                if len(token_calls) >= token_count:
                                    break
                    
                    # Stop when we have enough tokens
                    if len(token_calls) >= token_count:
                        break
                    
                    if message_count % 50 == 0:
                        await asyncio.sleep(0.2)
                        
            except FloodWaitError as e:
                logger.warning(f"⚠️ FloodWait during message iteration for @{kol}: {e.seconds}s")
                
            except Exception as e:
                logger.error(f"❌ Error iterating messages for @{kol}: {str(e)}")
            
            # Return exactly the requested number of tokens (or fewer if not available)
            result_tokens = token_calls[:token_count]
            logger.debug(f"📊 Found {len(result_tokens)}/{token_count} requested tokens from @{kol}")
            
            return result_tokens
            
        except Exception as e:
            logger.error(f"❌ Error getting token calls from @{kol}: {str(e)}")
            return []
    
    def _is_solana_token(self, address: str) -> bool:
        """Check if address is a Solana token (not ETH or other chains)."""
        if not address or len(address) < 32:
            return False
        
        # Solana addresses are 32-44 characters, base58 encoded
        if len(address) < 32 or len(address) > 44:
            return False
        
        # ETH addresses start with 0x and are 42 characters
        if address.startswith('0x') and len(address) == 42:
            return False
        
        # Check for valid base58 characters (Solana)
        base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        if not all(c in base58_chars for c in address):
            return False
        
        return True
    
    def _extract_token_addresses(self, text: str) -> List[str]:
        """Extract Solana token addresses from message text."""
        # Pattern for Solana addresses (base58, 32-44 chars)
        address_pattern = r'\b([1-9A-HJ-NP-Za-km-z]{32,44})\b'
        matches = re.findall(address_pattern, text)
        
        system_addresses = {
            'So11111111111111111111111111111111111111112',  # SOL
            'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
            '11111111111111111111111111111111',  # System program
        }
        
        valid_addresses = []
        for address in matches:
            if (address not in system_addresses and 
                self._is_solana_token(address) and
                not address.isdigit()):
                valid_addresses.append(address)
        
        return list(set(valid_addresses))
    
    async def _analyze_token_call_with_winloss(self, token_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """FIXED: Analyze token call with win/loss tracking (-50% threshold)."""
        try:
            token_address = token_call['address']
            call_time = token_call['call_time']
            
            # Skip if not a Solana token
            if not self._is_solana_token(token_address):
                logger.debug(f"⚠️ Skipping non-Solana token: {token_address}")
                return None
            
            # Get token performance from call time to now
            performance = self.api_manager.calculate_token_performance(
                token_address, 
                call_time
            )
            
            if not performance.get('success'):
                # Only log warning for Solana tokens that should have data
                logger.debug(f"⚠️ No performance data for Solana token {token_address[:8]}...")
                return None
            
            # Extract key metrics
            initial_price = performance.get('initial_price', 0)
            current_price = performance.get('current_price', 0)
            max_price = performance.get('max_price', 0)
            min_price = performance.get('min_price', 0)
            
            if not initial_price or initial_price <= 0:
                return None
            
            # Calculate ROI metrics
            current_roi = ((current_price / initial_price) - 1) * 100 if current_price > 0 else -100
            max_roi = ((max_price / initial_price) - 1) * 100 if max_price > 0 else -100
            min_roi = ((min_price / initial_price) - 1) * 100 if min_price > 0 else -100
            
            # WIN/LOSS LOGIC: -50% or worse from initial = LOSS
            is_loss = current_roi <= self.config['loss_threshold_percent']
            
            # Calculate pullback only if it hit a peak above initial
            max_pullback = 0
            if max_price > initial_price and max_price > 0 and min_price > 0:
                max_pullback = ((min_price / max_price) - 1) * 100
            else:
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
                'max_pullback': max_pullback,
                'is_loss': is_loss,  # NEW: Win/Loss tracking
                'hit_2x': hit_2x,
                'hit_5x': hit_5x,
                'time_to_2x_hours': time_to_2x_hours,
                'kol': token_call['kol']
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing token call: {str(e)}")
            return None
    
    def _calculate_kol_metrics_with_winloss(self, kol: str, channel_id: str, analyzed_tokens: List[Dict[str, Any]], tokens_analyzed: int) -> KOLPerformance:
        """FIXED: Calculate KOL metrics with win/loss tracking and proper statistics."""
        try:
            total_calls = len(analyzed_tokens)
            
            # Separate wins and losses
            wins = [token for token in analyzed_tokens if not token['is_loss']]
            losses = [token for token in analyzed_tokens if token['is_loss']]
            
            win_count = len(wins)
            loss_count = len(losses)
            
            # Calculate win rate
            win_rate = (win_count / total_calls * 100) if total_calls > 0 else 0
            
            # Calculate 2x and 5x rates FROM WINS ONLY
            tokens_2x_plus = sum(1 for token in wins if token['hit_2x'])
            tokens_5x_plus = sum(1 for token in wins if token['hit_5x'])
            
            success_rate_2x = (tokens_2x_plus / win_count * 100) if win_count > 0 else 0
            success_rate_5x = (tokens_5x_plus / win_count * 100) if win_count > 0 else 0
            
            # Calculate average time to 2x (only for winning 2x tokens)
            winning_2x_tokens = [t for t in wins if t['hit_2x']]
            avg_time_to_2x_hours = (
                sum(t['time_to_2x_hours'] for t in winning_2x_tokens) / len(winning_2x_tokens)
                if winning_2x_tokens else 0
            )
            
            # Calculate average max pullback for 2x winners BEFORE they hit 2x
            if winning_2x_tokens:
                avg_max_pullback_percent = sum(t['max_pullback'] for t in winning_2x_tokens) / len(winning_2x_tokens)
            else:
                # If no 2x winners, use overall winning tokens pullback
                avg_max_pullback_percent = sum(t['max_pullback'] for t in wins) / len(wins) if wins else 0
            
            # Calculate average ROI of ALL tokens (wins and losses)
            avg_roi = sum(t['current_roi'] for t in analyzed_tokens) / len(analyzed_tokens)
            
            # Calculate consistency score based on ROI variance of WINS only
            if len(wins) > 1:
                win_roi_values = [t['current_roi'] for t in wins]
                mean_roi = sum(win_roi_values) / len(win_roi_values)
                variance = sum((x - mean_roi) ** 2 for x in win_roi_values) / len(win_roi_values)
                std_dev = variance ** 0.5
                # Convert to consistency score (0-100, higher is better)
                consistency_score = max(0, min(100, 100 - (std_dev / 50)))
            else:
                consistency_score = 50  # Neutral score for insufficient data
            
            # Calculate composite score components
            win_rate_score = win_rate  # 0-100
            success_2x_score = success_rate_2x  # 0-100
            success_5x_score = success_rate_5x  # 0-100
            
            # Time score (faster = better)
            time_score = max(0, (48 - min(avg_time_to_2x_hours, 48)) / 48 * 100) if avg_time_to_2x_hours > 0 else 50
            
            # Pullback score (smaller pullback = better)
            pullback_score = max(0, min(100, 100 + avg_max_pullback_percent))
            
            # ROI score (higher average ROI = better)
            roi_score = max(0, min(100, (avg_roi + 100) / 10))
            
            # FIXED: Composite score with new weighting
            composite_score = (
                win_rate_score * 0.25 +          # 25% weight on win rate
                success_2x_score * 0.20 +       # 20% weight on 2x success rate from wins
                success_5x_score * 0.15 +       # 15% weight on 5x success rate from wins
                time_score * 0.15 +             # 15% weight on speed to 2x
                pullback_score * 0.15 +         # 15% weight on pullback management
                roi_score * 0.10                # 10% weight on average ROI
            )
            
            composite_score = max(0, min(100, composite_score))
            
            # Determine strategy classification
            if success_rate_2x >= 40 and avg_time_to_2x_hours <= 12 and avg_time_to_2x_hours > 0:
                strategy_classification = "SCALP"
            elif success_rate_5x >= 15 and consistency_score >= 70:
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
                channel_id=channel_id,
                follower_tier=follower_tier,
                total_calls=total_calls,
                wins=win_count,
                losses=loss_count,
                tokens_2x_plus=tokens_2x_plus,
                tokens_5x_plus=tokens_5x_plus,
                win_rate=win_rate,
                success_rate_2x=success_rate_2x,
                success_rate_5x=success_rate_5x,
                avg_time_to_2x_hours=avg_time_to_2x_hours,
                avg_max_pullback_percent=avg_max_pullback_percent,
                consistency_score=consistency_score,
                composite_score=composite_score,
                strategy_classification=strategy_classification,
                avg_roi=avg_roi,
                tokens_analyzed=tokens_analyzed
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics for @{kol}: {str(e)}")
            return None