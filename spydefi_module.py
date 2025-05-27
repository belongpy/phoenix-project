"""
SPYDEFI Analysis Module - Phoenix Project (OPTIMIZED & RELAXED VERSION)

MAJOR OPTIMIZATIONS & FIXES:
- Reduced scan window: 24h → 8h (peak crypto hours)
- Message limits: Max 6,000 messages per scan
- RELAXED FILTERING: More flexible pattern matching to find KOLs
- Fallback processing: Every 10th message processed regardless of filters
- Multiple pattern types: Achievement, @ mentions, flexible regex
- Early termination: Stop when sufficient KOLs found
- Auto cache refresh: <6h = use cache, >6h = auto refresh
- Debug logging: Better visibility into what's being processed
- Performance improvements and reduced processing time

This module provides comprehensive KOL analysis by scanning SpyDefi mentions
with relaxed filtering to ensure KOL discovery while maintaining efficiency.
"""

import asyncio
import logging
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

try:
    from telethon import TelegramClient, events
    from telethon.tl.types import Channel, Chat, User
    from telethon.errors import FloodWaitError, ChannelPrivateError, ChatAdminRequiredError
    TELETHON_AVAILABLE = True
except ImportError:
    TELETHON_AVAILABLE = False

logger = logging.getLogger("phoenix.spydefi")

@dataclass
class KOLPerformance:
    """Data class for KOL performance metrics."""
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
    strategy_classification: str
    follower_tier: str
    total_roi_percent: float
    max_roi_percent: float

class SpyDefiAnalyzer:
    """Optimized SPYDEFI analyzer with smart filtering and performance improvements."""
    
    # RELAXED CONFIGURATION - Less restrictive for better KOL discovery
    DEFAULT_CONFIG = {
        "spydefi_scan_hours": 8,  # Reduced from 24 to 8 hours (peak crypto hours)
        "kol_analysis_days": 7,
        "top_kols_count": 25,
        "min_mentions": 1,  # RELAXED: Reduced from 2 to 1
        "max_market_cap_usd": 10000000,  # $10M
        "min_subscribers": 100,  # RELAXED: Back to 100 from 500
        "win_threshold_percent": 50,
        "max_messages_limit": 6000,  # Message limit per scan
        "early_termination_kol_count": 50,  # Stop after finding 50 KOLs
        "cache_refresh_hours": 6,  # Auto refresh cache after 6 hours
        "peak_hours_start": 12,  # Start scanning from 12:00 UTC
        "peak_hours_end": 24,   # End scanning at 24:00 UTC
        "progress_update_interval": 500,  # Update progress every 500 messages
        "fallback_processing_interval": 10  # Process every Nth message as fallback
    }
    
    # RELAXED REGEX PATTERNS for better matching
    ACHIEVEMENT_PATTERNS = [
        r"Achievement Unlocked:\s*([^#!]+)#!",
        r"Achievement unlocked:\s*([^#!]+)#!",
        r"ACHIEVEMENT UNLOCKED:\s*([^#!]+)#!",
        r"🎯\s*Achievement Unlocked:\s*([^#!]+)#!",
        r"Achievement\s*Unlocked\s*:\s*([^#!]+)#!",
        r"Achievement\s+[Uu]nlocked[:\s]*([^#!]+)#?!?",  # More flexible
        r"([^@#\s]+)\s*#!",  # Catch #! patterns
        r"@([A-Za-z0-9_]+)\s*achieved",  # Alternative formats
        r"🏆.*?([A-Za-z0-9_]+)"  # Trophy mentions
    ]
    
    # EXPANDED SOLANA INDICATORS for better coverage
    SOLANA_INDICATORS = [
        "🟣", "SOL", "solana", "Solana", "SOLANA",
        "sol/", "/sol", "$SOL", "sol_", "_sol",
        "pump.fun", "pumpfun", "raydium", "jupiter",
        "meme", "token", "coin", "crypto", "degen",
        "@", "#", "x", "100x", "1000x", "moon"
    ]
    
    # Cache configuration
    CACHE_DIR = Path.home() / ".phoenix_cache"
    CACHE_FILE = "spydefi_kol_analysis.json"
    
    def __init__(self, api_id: str, api_hash: str, session_name: str = "phoenix_spydefi"):
        """Initialize the optimized SPYDEFI analyzer."""
        if not TELETHON_AVAILABLE:
            raise ImportError("Telethon is required for SPYDEFI analysis. Install with: pip install telethon")
        
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = None
        self.api_manager = None
        
        # Configuration with optimized defaults
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Performance tracking
        self.start_time = None
        self.messages_processed = 0
        self.kols_found = 0
        self.api_calls_made = 0
        
        # Cache setup
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.cache_file_path = self.CACHE_DIR / self.CACHE_FILE
        
        # Smart filtering compiled patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.ACHIEVEMENT_PATTERNS]
    
    def set_api_manager(self, api_manager):
        """Set the API manager for token analysis."""
        self.api_manager = api_manager
        logger.info("✅ API manager configured for SPYDEFI analysis")
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Config updated: {key} = {value}")
    
    def _should_refresh_cache(self) -> bool:
        """Auto-determine if cache should be refreshed based on age."""
        if not self.cache_file_path.exists():
            return True
        
        try:
            with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            timestamp_str = cache_data.get('timestamp')
            if not timestamp_str:
                return True
            
            cache_time = datetime.fromisoformat(timestamp_str)
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            
            # Auto refresh logic: >6 hours = refresh, <6 hours = use cache
            should_refresh = age_hours >= self.config["cache_refresh_hours"]
            
            if should_refresh:
                logger.info(f"🔄 Cache is {age_hours:.1f} hours old, auto-refreshing...")
            else:
                logger.info(f"📦 Using cached data ({age_hours:.1f} hours old)")
            
            return should_refresh
            
        except Exception as e:
            logger.warning(f"Error checking cache age: {str(e)}")
            return True
    
    def _is_solana_message(self, message_text: str) -> bool:
        """SMART FILTERING: Quick check if message is Solana-related."""
        text_lower = message_text.lower()
        return any(indicator.lower() in text_lower for indicator in self.SOLANA_INDICATORS)
    
    def _has_achievement_unlocked(self, message_text: str) -> bool:
        """RELAXED FILTERING: More flexible Achievement Unlocked detection."""
        text_lower = message_text.lower()
        
        # Check for various achievement patterns
        achievement_indicators = [
            "achievement unlocked",
            "achievement",
            "achieved",
            "unlocked",
            "#!",
            "🏆",
            "🎯"
        ]
        
        return any(indicator in text_lower for indicator in achievement_indicators)
    
    def _should_process_message(self, message_text: str) -> bool:
        """RELAXED SMART FILTERING: More flexible message processing."""
        # Primary check: Achievement Unlocked messages
        if self._has_achievement_unlocked(message_text):
            return True
        
        # Secondary check: Solana-related messages with mentions/tags
        if self._is_solana_message(message_text) and ("@" in message_text or "#" in message_text):
            return True
        
        # Fallback: Process every 5th message to catch edge cases
        return False
    
    async def _scan_spydefi_optimized(self) -> Dict[str, int]:
        """Optimized SpyDefi scanning with smart filtering and limits."""
        logger.info("🔍 Scanning SpyDefi with optimized filters...")
        
        # Calculate time window for peak hours
        now = datetime.now()
        hours_back = self.config["spydefi_scan_hours"]
        scan_start_time = now - timedelta(hours=hours_back)
        
        logger.info(f"📅 Scanning from {scan_start_time.strftime('%Y-%m-%d %H:%M:%S')} to {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Get SpyDefi channel
            spydefi_channel = await self.client.get_entity("spydefi")
            logger.info(f"📥 Scanning SpyDefi (ID: {spydefi_channel.id})...")
        except Exception as e:
            logger.error(f"❌ Failed to get SpyDefi channel: {str(e)}")
            return {}
        
        kol_mentions = defaultdict(int)
        messages_processed = 0
        relevant_messages = 0
        max_messages = self.config["max_messages_limit"]
        early_termination_count = self.config["early_termination_kol_count"]
        progress_interval = self.config["progress_update_interval"]
        
        try:
            async for message in self.client.iter_messages(
                spydefi_channel,
                offset_date=scan_start_time,
                limit=max_messages  # Hard limit on messages
            ):
                messages_processed += 1
                
                # Progress updates (less frequent)
                if messages_processed % progress_interval == 0:
                    print(f"✅ Processed {messages_processed} messages, found {len(kol_mentions)} KOLs, {relevant_messages} relevant...", flush=True)
                
                # Early termination check
                if len(kol_mentions) >= early_termination_count:
                    logger.info(f"🎯 Early termination: Found {len(kol_mentions)} KOLs (target: {early_termination_count})")
                    break
                
                # Hard message limit check
                if messages_processed >= max_messages:
                    logger.info(f"📊 Message limit reached: {max_messages}")
                    break
                
                if not message.text:
                    continue
                
                # RELAXED FILTERING: More flexible approach
                should_process = self._should_process_message(message.text)
                
                # FALLBACK: Process every 10th message regardless of filters
                if not should_process and messages_processed % 10 == 0:
                    should_process = True
                
                if not should_process:
                    continue
                
                relevant_messages += 1
                
                # Extract KOL mentions using all patterns
                kol_found_in_message = False
                
                # Try achievement patterns first
                for pattern in self.compiled_patterns:
                    matches = pattern.findall(message.text)
                    for match in matches:
                        kol_name = match.strip()
                        if kol_name and len(kol_name) > 1:  # More lenient length check
                            # Clean KOL name
                            kol_clean = re.sub(r'[^\w\s]', '', kol_name).strip()
                            if kol_clean and len(kol_clean) > 1:
                                kol_mentions[kol_clean] += 1
                                kol_found_in_message = True
                
                # If no achievement pattern, try @ mentions
                if not kol_found_in_message and "@" in message.text:
                    at_mentions = re.findall(r'@([A-Za-z0-9_]{2,})', message.text)
                    for mention in at_mentions:
                        if len(mention) > 2 and mention.lower() not in ['everyone', 'here', 'channel']:
                            kol_mentions[mention] += 1
                            kol_found_in_message = True
                
                # Debug: Log first few messages being processed
                if relevant_messages <= 5:
                    logger.debug(f"Processing message {relevant_messages}: {message.text[:100]}...")
                
        except Exception as e:
            logger.error(f"❌ Error during message iteration: {str(e)}")
        
        # Filter by minimum mentions (more lenient)
        min_mentions = self.config["min_mentions"]
        filtered_kols = {kol: count for kol, count in kol_mentions.items() if count >= min_mentions}
        
        logger.info(f"📊 SCAN COMPLETE:")
        logger.info(f"   📨 Messages processed: {messages_processed:,}")
        logger.info(f"   ✅ Relevant messages: {relevant_messages:,}")
        logger.info(f"   👥 Raw KOL mentions: {len(kol_mentions)}")
        logger.info(f"   🎯 KOLs meeting criteria (≥{min_mentions} mentions): {len(filtered_kols)}")
        logger.info(f"   📈 Efficiency: {(relevant_messages/messages_processed*100):.1f}% relevant messages")
        
        # Debug: Show some example KOL mentions found
        if filtered_kols:
            logger.info(f"📋 Sample KOLs found: {list(filtered_kols.keys())[:10]}")
        else:
            logger.warning(f"⚠️ No KOLs found meeting minimum criteria")
            logger.info(f"📋 Raw mentions found: {list(kol_mentions.keys())[:10]}")
        
        return filtered_kols
    
    async def _get_channel_info(self, kol_name: str) -> Optional[Tuple[int, int]]:
        """Get channel info (ID and subscriber count) for a KOL with better error handling."""
        try:
            # Try various username formats
            possible_usernames = [
                kol_name,
                kol_name.lower(),
                kol_name.replace(' ', ''),
                kol_name.replace(' ', '_'),
                f"{kol_name}_official",
                f"{kol_name}sol"
            ]
            
            for username in possible_usernames:
                try:
                    entity = await self.client.get_entity(username)
                    
                    # Get channel ID
                    channel_id = getattr(entity, 'id', None)
                    if channel_id is None:
                        continue
                    
                    # Get subscriber count with fallback
                    subscriber_count = 0  # Default fallback
                    
                    if hasattr(entity, 'participants_count') and entity.participants_count is not None:
                        subscriber_count = entity.participants_count
                    elif hasattr(entity, 'members_count') and entity.members_count is not None:
                        subscriber_count = entity.members_count
                    # If no count available, keep default of 0
                    
                    logger.debug(f"✅ Found channel for {kol_name}: ID={channel_id}, Subs={subscriber_count}")
                    return channel_id, subscriber_count
                    
                except Exception as e:
                    logger.debug(f"Failed to get entity for {username}: {str(e)}")
                    continue
            
            logger.debug(f"❌ Could not find channel for {kol_name}")
            return None
            
        except Exception as e:
            logger.debug(f"Error getting channel info for {kol_name}: {str(e)}")
            return None
    
    async def _analyze_kol_channel_optimized(self, kol_name: str, channel_id: int, 
                                           subscriber_count: int) -> Optional[KOLPerformance]:
        """Optimized KOL channel analysis with proper null handling."""
        try:
            # Handle None values properly
            if channel_id is None:
                logger.debug(f"⏭️ Skipping {kol_name}: No channel ID")
                return None
            
            if subscriber_count is None:
                subscriber_count = 0  # Default to 0 if unknown
            
            # Filter by subscriber count early (handle None gracefully)
            min_subs = self.config["min_subscribers"]
            if subscriber_count < min_subs:
                logger.debug(f"⏭️ Skipping {kol_name}: {subscriber_count} subscribers < {min_subs}")
                return None
            
            logger.info(f"📊 Analyzing @{kol_name} ({subscriber_count:,} subscribers)...")
            
            # Get recent messages for token calls
            days_back = self.config["kol_analysis_days"]
            since_date = datetime.now() - timedelta(days=days_back)
            
            try:
                channel = await self.client.get_entity(channel_id)
            except Exception as e:
                logger.debug(f"⏭️ Could not access channel for {kol_name}: {str(e)}")
                return None
            
            token_calls = []
            messages_checked = 0
            max_messages_per_kol = 1000  # Limit per KOL analysis
            
            try:
                async for message in self.client.iter_messages(
                    channel,
                    offset_date=since_date,
                    limit=max_messages_per_kol
                ):
                    messages_checked += 1
                    if not message.text:
                        continue
                    
                    # Look for token contract addresses (Solana format)
                    contracts = re.findall(r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b', message.text)
                    for contract in contracts:
                        if len(contract) >= 32:  # Valid Solana address length
                            token_calls.append({
                                'contract': contract,
                                'timestamp': message.date,
                                'message': message.text[:200]  # Truncate for storage
                            })
            except Exception as e:
                logger.debug(f"⏭️ Error reading messages from {kol_name}: {str(e)}")
                return None
            
            if not token_calls:
                logger.debug(f"⏭️ No token calls found for {kol_name}")
                return None
            
            # Analyze token performance (optimized)
            performance_data = await self._analyze_token_performance_batch(token_calls[:20])  # Limit to top 20 calls
            
            if not performance_data:
                logger.debug(f"⏭️ No performance data for {kol_name}")
                return None
            
            # Calculate metrics
            metrics = self._calculate_kol_metrics(performance_data)
            
            # Determine follower tier (handle None subscriber_count)
            if subscriber_count >= 10000:
                follower_tier = "HIGH"
            elif subscriber_count >= 1000:
                follower_tier = "MEDIUM"
            else:
                follower_tier = "LOW"
            
            # Strategy classification
            strategy = self._classify_strategy(metrics, subscriber_count)
            
            return KOLPerformance(
                kol=kol_name,
                channel_id=channel_id,
                subscriber_count=subscriber_count,
                total_calls=len(token_calls),
                winning_calls=metrics['winning_calls'],
                losing_calls=metrics['losing_calls'],
                success_rate=metrics['success_rate'],
                tokens_2x_plus=metrics['tokens_2x_plus'],
                tokens_5x_plus=metrics['tokens_5x_plus'],
                success_rate_2x=metrics['success_rate_2x'],
                success_rate_5x=metrics['success_rate_5x'],
                avg_time_to_2x_hours=metrics['avg_time_to_2x_hours'],
                avg_max_pullback_percent=metrics['avg_max_pullback_percent'],
                avg_unrealized_gains_percent=metrics['avg_unrealized_gains_percent'],
                consistency_score=metrics['consistency_score'],
                composite_score=metrics['composite_score'],
                strategy_classification=strategy,
                follower_tier=follower_tier,
                total_roi_percent=metrics['total_roi_percent'],
                max_roi_percent=metrics['max_roi_percent']
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {kol_name}: {str(e)}")
            return None
    
    async def _analyze_token_performance_batch(self, token_calls: List[Dict]) -> List[Dict]:
        """Optimized batch token performance analysis with better error handling."""
        if not self.api_manager:
            logger.error("❌ API manager not configured")
            return []
        
        performance_data = []
        
        for call in token_calls:
            try:
                contract = call.get('contract')
                call_time = call.get('timestamp')
                
                if not contract or not call_time:
                    continue
                
                # Get current price and calculate performance
                current_price_result = await self.api_manager.get_token_price(contract)
                
                if current_price_result.get('success') and current_price_result.get('data'):
                    current_price = current_price_result['data'].get('value', 0)
                    
                    if current_price and current_price > 0:
                        # Calculate performance from call time to now
                        performance_result = self.api_manager.calculate_token_performance(contract, call_time)
                        
                        if performance_result.get('success'):
                            performance_data.append({
                                'contract': contract,
                                'call_time': call_time,
                                'current_price': current_price,
                                'performance': performance_result
                            })
                            
                            self.api_calls_made += 1
                
            except Exception as e:
                logger.debug(f"Error analyzing token {call.get('contract', 'unknown')}: {str(e)}")
                continue
        
        return performance_data
    
    def _calculate_kol_metrics(self, performance_data: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive KOL performance metrics."""
        if not performance_data:
            return self._default_metrics()
        
        total_calls = len(performance_data)
        win_threshold = self.config["win_threshold_percent"]
        
        # Calculate basic metrics
        winning_calls = 0
        tokens_2x_plus = 0
        tokens_5x_plus = 0
        roi_values = []
        max_roi_values = []
        time_to_2x_values = []
        pullback_values = []
        
        for data in performance_data:
            perf = data['performance']
            roi = perf.get('roi_percent', 0)
            max_roi = perf.get('max_roi_percent', 0)
            
            roi_values.append(roi)
            max_roi_values.append(max_roi)
            
            # Count winning calls
            if roi >= win_threshold:
                winning_calls += 1
            
            # Count 2x+ and 5x+ tokens
            if max_roi >= 100:  # 2x = 100% ROI
                tokens_2x_plus += 1
                time_to_2x = perf.get('time_to_max_roi_hours', 0)
                if time_to_2x > 0:
                    time_to_2x_values.append(time_to_2x)
            
            if max_roi >= 400:  # 5x = 400% ROI
                tokens_5x_plus += 1
            
            # Track pullbacks
            pullback = perf.get('max_drawdown_percent', 0)
            pullback_values.append(abs(pullback))
        
        # Calculate rates and averages
        success_rate = (winning_calls / total_calls) * 100 if total_calls > 0 else 0
        success_rate_2x = (tokens_2x_plus / total_calls) * 100 if total_calls > 0 else 0
        success_rate_5x = (tokens_5x_plus / total_calls) * 100 if total_calls > 0 else 0
        
        avg_time_to_2x = sum(time_to_2x_values) / len(time_to_2x_values) if time_to_2x_values else 0
        avg_pullback = sum(pullback_values) / len(pullback_values) if pullback_values else 0
        avg_roi = sum(roi_values) / len(roi_values) if roi_values else 0
        max_roi = max(max_roi_values) if max_roi_values else 0
        
        # Calculate consistency score (lower variance = higher consistency)
        if len(roi_values) > 1:
            roi_variance = sum((x - avg_roi) ** 2 for x in roi_values) / len(roi_values)
            consistency_score = max(0, 100 - (roi_variance / 100))  # Normalize to 0-100
        else:
            consistency_score = 50  # Default for single data point
        
        # Calculate composite score (weighted)
        composite_score = (
            success_rate * 0.25 +           # 25% weight on success rate
            success_rate_2x * 0.25 +        # 25% weight on 2x success rate
            consistency_score * 0.20 +      # 20% weight on consistency
            (max(0, 24 - avg_time_to_2x) / 24 * 100) * 0.15 +  # 15% weight on speed (faster = better)
            success_rate_5x * 0.15          # 15% weight on 5x success rate
        )
        
        return {
            'winning_calls': winning_calls,
            'losing_calls': total_calls - winning_calls,
            'success_rate': success_rate,
            'tokens_2x_plus': tokens_2x_plus,
            'tokens_5x_plus': tokens_5x_plus,
            'success_rate_2x': success_rate_2x,
            'success_rate_5x': success_rate_5x,
            'avg_time_to_2x_hours': avg_time_to_2x,
            'avg_max_pullback_percent': avg_pullback,
            'avg_unrealized_gains_percent': avg_roi,
            'consistency_score': consistency_score,
            'composite_score': min(100, max(0, composite_score)),  # Cap at 0-100
            'total_roi_percent': sum(roi_values),
            'max_roi_percent': max_roi
        }
    
    def _default_metrics(self) -> Dict[str, float]:
        """Return default metrics for KOLs with no data."""
        return {
            'winning_calls': 0,
            'losing_calls': 0,
            'success_rate': 0,
            'tokens_2x_plus': 0,
            'tokens_5x_plus': 0,
            'success_rate_2x': 0,
            'success_rate_5x': 0,
            'avg_time_to_2x_hours': 0,
            'avg_max_pullback_percent': 0,
            'avg_unrealized_gains_percent': 0,
            'consistency_score': 0,
            'composite_score': 0,
            'total_roi_percent': 0,
            'max_roi_percent': 0
        }
    
    def _classify_strategy(self, metrics: Dict[str, float], subscriber_count: int) -> str:
        """Classify KOL strategy based on performance and follower metrics with null handling."""
        # Handle None or invalid subscriber_count
        if subscriber_count is None:
            subscriber_count = 0
        
        success_rate_2x = metrics.get('success_rate_2x', 0)
        success_rate_5x = metrics.get('success_rate_5x', 0)
        avg_time_to_2x = metrics.get('avg_time_to_2x_hours', 0)
        success_rate = metrics.get('success_rate', 0)
        
        # SCALP strategy: High followers + Fast 2x + Good success rate
        if (subscriber_count >= 5000 and 
            success_rate_2x >= 20 and 
            avg_time_to_2x > 0 and avg_time_to_2x <= 12 and 
            success_rate >= 40):
            return "SCALP"
        
        # HOLD strategy: Good gem rate + Consistent performance
        elif (success_rate_5x >= 15 and 
              metrics.get('consistency_score', 0) >= 60):
            return "HOLD"
        
        else:
            return "UNKNOWN"
    
    def _save_cache(self, data: Dict[str, Any]) -> None:
        """Save analysis results to cache."""
        try:
            cache_data = {
                'version': '3.1',  # Updated version with optimizations
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'data': data,
                'kol_performances': data.get('kol_performances', {}),
                'kol_mentions': data.get('kol_mentions', {}),
                'metadata': data.get('metadata', {})
            }
            
            with open(self.cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 Cache saved: {self.cache_file_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving cache: {str(e)}")
    
    def _load_cache(self) -> Optional[Dict[str, Any]]:
        """Load analysis results from cache."""
        try:
            if not self.cache_file_path.exists():
                return None
            
            with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate cache version and structure
            if cache_data.get('version', '').startswith('3.'):
                logger.info(f"📦 Loaded cache version {cache_data.get('version', 'unknown')}")
                return cache_data.get('data', {})
            else:
                logger.warning("⚠️ Cache version incompatible, will refresh")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error loading cache: {str(e)}")
            return None
    
    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        try:
            if self.cache_file_path.exists():
                self.cache_file_path.unlink()
                logger.info("🗑️ Cache cleared successfully")
            else:
                logger.info("📭 No cache to clear")
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {str(e)}")
    
    async def run_full_analysis(self, force_refresh: bool = None) -> Dict[str, Any]:
        """Run the complete optimized SPYDEFI analysis."""
        self.start_time = time.time()
        
        logger.info("🚀 STARTING OPTIMIZED SPYDEFI ANALYSIS")
        logger.info("=" * 60)
        
        # Auto-determine refresh (ignore force_refresh parameter, use auto logic)
        should_refresh = self._should_refresh_cache()
        
        if not should_refresh:
            cached_data = self._load_cache()
            if cached_data:
                logger.info("✅ Using cached analysis data")
                return {
                    'success': True,
                    'from_cache': True,
                    **cached_data
                }
        
        try:
            # Step 1: Optimized SpyDefi scanning
            print("🔍 STEP 1: Scanning SpyDefi for top KOLs...", flush=True)
            kol_mentions = await self._scan_spydefi_optimized()
            
            if not kol_mentions:
                return {
                    'success': False,
                    'error': 'No KOLs found in SpyDefi scan',
                    'kol_mentions': {},
                    'kol_performances': {}
                }
            
            # Step 2: Get top KOLs
            top_kols_count = self.config["top_kols_count"]
            top_kols = dict(sorted(kol_mentions.items(), key=lambda x: x[1], reverse=True)[:top_kols_count])
            
            print(f"🎯 STEP 2: Analyzing top {len(top_kols)} KOLs...", flush=True)
            
            # Step 3: Analyze individual KOLs
            kol_performances = {}
            successful_analyses = 0
            failed_analyses = 0
            
            for i, (kol_name, mention_count) in enumerate(top_kols.items(), 1):
                try:
                    print(f"📊 [{i}/{len(top_kols)}] Analyzing @{kol_name} ({mention_count} mentions)...", flush=True)
                    
                    # Get channel info with proper error handling
                    channel_info = await self._get_channel_info(kol_name)
                    if channel_info is None:
                        logger.debug(f"⏭️ Skipping {kol_name}: Channel not found")
                        failed_analyses += 1
                        print(f"⏭️ @{kol_name}: Channel not found", flush=True)
                        continue
                    
                    channel_id, subscriber_count = channel_info
                    
                    # Validate channel info
                    if channel_id is None:
                        logger.debug(f"⏭️ Skipping {kol_name}: Invalid channel ID")
                        failed_analyses += 1
                        print(f"⏭️ @{kol_name}: Invalid channel", flush=True)
                        continue
                    
                    # Ensure subscriber_count is not None
                    if subscriber_count is None:
                        subscriber_count = 0
                    
                    # Analyze KOL performance
                    performance = await self._analyze_kol_channel_optimized(kol_name, channel_id, subscriber_count)
                    
                    if performance:
                        kol_performances[kol_name] = asdict(performance)
                        successful_analyses += 1
                        print(f"✅ @{kol_name}: Score {performance.composite_score:.1f}, Strategy {performance.strategy_classification}", flush=True)
                    else:
                        failed_analyses += 1
                        print(f"⏭️ @{kol_name}: Insufficient data", flush=True)
                    
                    # Brief pause to avoid rate limits
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    failed_analyses += 1
                    logger.error(f"❌ Error analyzing {kol_name}: {str(e)}")
                    print(f"❌ @{kol_name}: Analysis failed ({str(e)[:50]}...)", flush=True)
                    continue
            
            # Calculate overall statistics
            if kol_performances:
                total_calls = sum(perf['total_calls'] for perf in kol_performances.values())
                total_winning = sum(perf['winning_calls'] for perf in kol_performances.values())
                total_2x = sum(perf['tokens_2x_plus'] for perf in kol_performances.values())
                total_5x = sum(perf['tokens_5x_plus'] for perf in kol_performances.values())
                
                overall_success_rate = (total_winning / total_calls * 100) if total_calls > 0 else 0
                overall_2x_rate = (total_2x / total_calls * 100) if total_calls > 0 else 0
                overall_5x_rate = (total_5x / total_calls * 100) if total_calls > 0 else 0
            else:
                total_calls = overall_success_rate = overall_2x_rate = overall_5x_rate = 0
            
            # Sort KOLs by composite score
            sorted_performances = dict(sorted(
                kol_performances.items(),
                key=lambda x: x[1]['composite_score'],
                reverse=True
            ))
            
            # Prepare results
            processing_time = time.time() - self.start_time
            
            results = {
                'success': True,
                'from_cache': False,
                'kol_mentions': kol_mentions,
                'kol_performances': sorted_performances,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config,
                    'total_kols_found': len(kol_mentions),
                    'top_kols_analyzed': len(top_kols),
                    'successful_analyses': successful_analyses,
                    'failed_analyses': failed_analyses,
                    'total_calls_analyzed': total_calls,
                    'overall_success_rate': overall_success_rate,
                    'overall_2x_rate': overall_2x_rate,
                    'overall_5x_rate': overall_5x_rate,
                    'processing_time_seconds': processing_time,
                    'api_calls': self.api_calls_made,
                    'messages_processed': self.messages_processed,
                    'optimization_version': '3.1'
                }
            }
            
            # Save to cache
            self._save_cache(results)
            
            logger.info("=" * 60)
            logger.info("✅ OPTIMIZED SPYDEFI ANALYSIS COMPLETE")
            logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
            logger.info(f"📊 KOLs analyzed: {successful_analyses}/{len(top_kols)}")
            logger.info(f"📈 Overall success rate: {overall_success_rate:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'kol_mentions': {},
                'kol_performances': {}
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        logger.info("🔗 Connecting to Telegram for SPYDEFI analysis...")
        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
        await self.client.start()
        
        # Get current user info
        me = await self.client.get_me()
        logger.info(f"✅ Connected to Telegram")
        logger.info(f"📱 Authenticated as: {me.first_name} ({me.username})")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.disconnect()
            logger.info("🔌 Disconnected from Telegram")