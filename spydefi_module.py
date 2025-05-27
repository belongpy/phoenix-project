"""
SpyDefi Module - Phoenix Project (COMPLETE FIXED VERSION)

🔧 CRITICAL FIXES APPLIED:
✅ FIXED: KOL extraction now gets real @usernames instead of multipliers (x2, x3)
✅ FIXED: DateTime timezone comparison issue resolved
✅ ADDED: Validation to filter out false positives
✅ IMPROVED: Better message parsing and error handling

ISSUES RESOLVED:
- Before: Extracted "x2", "x3", "Pump it up" (wrong data)
- After: Extracts "@cryptoking123", "@memecoin_master" (actual usernames)
- Fixed: "can't compare offset-naive and offset-aware datetimes" error

PRESERVED: All other functionality intact for wallet_module compatibility
"""

import asyncio
import logging
import re
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone  # FIXED: Added timezone import
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from telethon import TelegramClient
from telethon.errors import ChannelPrivateError, UsernameNotOccupiedError, FloodWaitError
from telethon.tl.types import User, Channel
import time

logger = logging.getLogger("phoenix.spydefi")

@dataclass
class KOLPerformance:
    """Enhanced KOL performance metrics."""
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
    """Complete SPYDEFI KOL Analysis System with FIXED username extraction and timezone handling."""
    
    def __init__(self, api_id: str, api_hash: str, session_name: str = "phoenix_spydefi"):
        """Initialize the SpyDefi analyzer."""
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = None
        self.api_manager = None
        
        # Configuration with optimized defaults
        self.config = {
            'spydefi_scan_hours': 24,
            'kol_analysis_days': 7,
            'top_kols_count': 25,
            'min_mentions': 2,
            'max_market_cap_usd': 10000000,
            'min_subscribers': 500,
            'win_threshold_percent': 50,
            'max_messages_limit': 6000,
            'early_termination_kol_count': 50,
            'cache_refresh_hours': 6
        }
        
        # Cache settings
        self.cache_dir = Path.home() / ".phoenix_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "spydefi_kol_analysis.json"
        
        logger.info("✅ SpyDefi analyzer initialized")
    
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
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self):
        """Connect to Telegram."""
        logger.info("🔗 Connecting to Telegram for SPYDEFI analysis...")
        
        self.client = TelegramClient(
            self.session_name,
            self.api_id,
            self.api_hash,
            device_model="Phoenix SpyDefi Analyzer",
            system_version="3.1.1"
        )
        
        await self.client.start()
        
        # Get user info
        me = await self.client.get_me()
        logger.info("✅ Connected to Telegram")
        logger.info(f"📱 Authenticated as: {me.first_name} ({me.username})")
    
    async def disconnect(self):
        """Disconnect from Telegram."""
        if self.client:
            await self.client.disconnect()
            logger.info("🔌 Disconnected from Telegram")
    
    def extract_kol_from_spydefi_message(self, message_text: str) -> str:
        """
        🔧 FIXED: Extract actual KOL username from SpyDefi achievement message.
        
        PREVIOUS ISSUE: Was extracting multipliers (x2, x3) instead of @usernames
        FIXED: Now properly extracts @usernames from Achievement Unlocked messages
        
        Args:
            message_text (str): SpyDefi achievement message
            
        Returns:
            str: Extracted KOL username (without @) or empty string
        """
        
        # FIXED PATTERNS - Look for actual @usernames in Achievement messages
        patterns = [
            # Pattern 1: "Achievement Unlocked: @username just called..." (most common)
            r'Achievement\s+Unlocked:?\s*@([a-zA-Z0-9_]+)\s+(?:just\s+)?(?:called|found|hit|achieved)',
            
            # Pattern 2: "@username achieved..." or "@username got..."
            r'@([a-zA-Z0-9_]+)\s+(?:achieved|got|hit|called|found|just)',
            
            # Pattern 3: Direct @username mention with multiplier context
            r'@([a-zA-Z0-9_]+).*?(?:\d+x|\d+X)(?:\s|!|$)',
            
            # Pattern 4: Achievement context with @username
            r'(?:Achievement|achievement).*?@([a-zA-Z0-9_]+)',
            
            # Pattern 5: Broader @username extraction with validation
            r'@([a-zA-Z0-9_]{3,32})',  # 3-32 chars, valid username format
        ]
        
        for i, pattern in enumerate(patterns, 1):
            matches = re.findall(pattern, message_text, re.IGNORECASE)
            if matches:
                for username in matches:
                    # Validate username format
                    if self.is_valid_telegram_username(username):
                        logger.debug(f"✅ Pattern {i} extracted KOL: @{username}")
                        return username
        
        logger.debug(f"❌ No valid KOL found in: {message_text[:100]}...")
        return ""
    
    def is_valid_telegram_username(self, username: str) -> bool:
        """
        ✅ ADDED: Validate if extracted text is a valid Telegram username.
        
        Args:
            username (str): Username to validate
            
        Returns:
            bool: True if valid username format
        """
        
        # Basic format validation
        if not username or len(username) < 3 or len(username) > 32:
            return False
        
        # Must be alphanumeric + underscore only
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False
        
        # Filter out common false positives
        false_positives = [
            # Multipliers and numbers
            'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
            'x15', 'x20', 'x30', 'x40', 'x50', 'x60', 'x100',
            '2x', '3x', '4x', '5x', '6x', '7x', '8x', '9x', '10x',
            
            # Common phrases/words that might get extracted
            'achievement', 'unlocked', 'called', 'found', 'just',
            'token', 'gem', 'coin', 'pump', 'moon', 'bot', 'channel',
            'group', 'admin', 'mod', 'user', 'member', 'community',
            
            # Common false positive patterns
            'pump_it_up', 'alpha_caller', 'heavy_hitter', 'back_with_bang',
            'keep_faith', 'consistency_key', 'achievement_unlocked'
        ]
        
        # Check against false positives
        if username.lower() in [fp.lower() for fp in false_positives]:
            logger.debug(f"❌ False positive filtered: {username}")
            return False
        
        # Additional pattern checks for false positives
        false_positive_patterns = [
            r'^x\d+$',        # x2, x3, x5, etc.
            r'^\d+x$',        # 2x, 3x, 5x, etc.
            r'^\d+$',         # Pure numbers
            r'^[xX]+$',       # Just x's
            r'^.{1,2}$',      # Too short
        ]
        
        for pattern in false_positive_patterns:
            if re.match(pattern, username, re.IGNORECASE):
                logger.debug(f"❌ Pattern false positive: {username}")
                return False
        
        # Must contain at least one letter (not just numbers/underscores)
        if not re.search(r'[a-zA-Z]', username):
            logger.debug(f"❌ No letters in username: {username}")
            return False
        
        logger.debug(f"✅ Valid username: {username}")
        return True
    
    def validate_extracted_kols(self, kol_mentions: dict) -> dict:
        """
        🔧 ADDED: Validate extracted KOL names to filter out false positives.
        
        Args:
            kol_mentions (dict): Dictionary of KOL -> mention count
            
        Returns:
            dict: Cleaned KOL mentions with false positives removed
        """
        
        cleaned_kols = {}
        
        for kol, count in kol_mentions.items():
            if self.is_valid_telegram_username(kol):
                cleaned_kols[kol] = count
                logger.debug(f"✅ Valid KOL kept: @{kol} ({count} mentions)")
            else:
                logger.debug(f"❌ Invalid KOL filtered: {kol}")
        
        logger.info(f"🧹 KOL validation: {len(kol_mentions)} → {len(cleaned_kols)} valid KOLs")
        return cleaned_kols
    
    async def scan_spydefi_for_kols(self) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        🔧 FIXED: Scan SpyDefi for KOL mentions with proper timezone handling and username extraction.
        
        Returns:
            Tuple[Dict[str, int], Dict[str, Any]]: KOL mentions and metadata
        """
        logger.info("🔍 Scanning SpyDefi with optimized filters...")
        
        # FIXED: Calculate time range with timezone awareness
        end_time = datetime.now(timezone.utc)  # Make timezone-aware
        start_time = end_time - timedelta(hours=self.config['spydefi_scan_hours'])
        
        logger.info(f"📅 Scanning from {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        try:
            # Get SpyDefi channel
            spydefi_channel = await self.client.get_entity('spydefi')
            logger.info(f"📥 Scanning SpyDefi (ID: {spydefi_channel.id})...")
            
            kol_mentions = {}
            processed_messages = 0
            relevant_messages = 0
            max_messages = self.config['max_messages_limit']
            early_termination_target = self.config['early_termination_kol_count']
            
            # Scan messages with optimized filtering
            async for message in self.client.iter_messages(
                spydefi_channel,
                offset_date=end_time,
                reverse=True,
                limit=max_messages
            ):
                # FIXED: Stop if message is too old (both are now timezone-aware)
                if message.date < start_time:
                    break
                
                processed_messages += 1
                
                # Process every message initially, then every 10th for fallback
                should_process = (
                    processed_messages <= 1000 or  # Process first 1000 fully
                    processed_messages % 10 == 0   # Then every 10th message
                )
                
                if not should_process:
                    continue
                
                if not message.text:
                    continue
                
                message_text = message.text.strip()
                
                # FIXED: Look for Achievement Unlocked messages with Solana context
                is_relevant = (
                    ('achievement unlocked' in message_text.lower() or 
                     '🎉' in message_text) and
                    ('🧡' in message_text or '🟠' in message_text or 
                     'solana' in message_text.lower() or
                     'sol' in message_text.lower() or
                     '@' in message_text)
                )
                
                if is_relevant:
                    relevant_messages += 1
                    
                    # FIXED: Extract KOL username properly
                    kol_username = self.extract_kol_from_spydefi_message(message_text)
                    
                    if kol_username:
                        kol_mentions[kol_username] = kol_mentions.get(kol_username, 0) + 1
                        logger.debug(f"📊 @{kol_username}: {kol_mentions[kol_username]} mentions")
                
                # Early termination if we found enough KOLs
                if len(kol_mentions) >= early_termination_target:
                    logger.info(f"🎯 Early termination: Found {len(kol_mentions)} KOLs (target: {early_termination_target})")
                    break
            
            # FIXED: Validate extracted KOLs to remove false positives
            kol_mentions = self.validate_extracted_kols(kol_mentions)
            
            # Filter by minimum mentions
            min_mentions = self.config['min_mentions']
            filtered_kols = {
                kol: count for kol, count in kol_mentions.items() 
                if count >= min_mentions
            }
            
            # Sort by mention count
            sorted_kols = dict(sorted(filtered_kols.items(), key=lambda x: x[1], reverse=True))
            
            # Metadata
            metadata = {
                'scan_period_hours': self.config['spydefi_scan_hours'],
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'messages_processed': processed_messages,
                'relevant_messages': relevant_messages,
                'raw_kol_mentions': len(kol_mentions),
                'filtered_kol_mentions': len(sorted_kols),
                'min_mentions_threshold': min_mentions,
                'efficiency_percent': round((relevant_messages / processed_messages * 100) if processed_messages > 0 else 0, 1),
                'optimization_version': '3.1.1'
            }
            
            logger.info("📊 SCAN COMPLETE:")
            logger.info(f"   📨 Messages processed: {processed_messages}")
            logger.info(f"   ✅ Relevant messages: {relevant_messages}")
            logger.info(f"   👥 Raw KOL mentions: {len(kol_mentions)}")
            logger.info(f"   🎯 KOLs meeting criteria (≥{min_mentions} mentions): {len(sorted_kols)}")
            logger.info(f"   📈 Efficiency: {metadata['efficiency_percent']}% relevant messages")
            
            # Log sample of found KOLs
            sample_kols = list(sorted_kols.keys())[:10]
            logger.info(f"📋 Sample KOLs found: {sample_kols}")
            
            return sorted_kols, metadata
            
        except Exception as e:
            logger.error(f"❌ Error scanning SpyDefi: {str(e)}")
            return {}, {'error': str(e)}
    
    async def analyze_kol_performance(self, kol_username: str, kol_mentions: int) -> Optional[KOLPerformance]:
        """
        Analyze individual KOL performance (unchanged - working correctly).
        
        Args:
            kol_username (str): KOL username without @
            kol_mentions (int): Number of SpyDefi mentions
            
        Returns:
            Optional[KOLPerformance]: KOL performance data or None
        """
        try:
            logger.debug(f"📊 Analyzing @{kol_username}...")
            
            # Get KOL channel
            try:
                kol_entity = await self.client.get_entity(kol_username)
            except (ChannelPrivateError, UsernameNotOccupiedError):
                logger.debug(f"⏭️ @{kol_username}: Channel not found or private")
                return None
            except Exception as e:
                logger.debug(f"⏭️ @{kol_username}: Error accessing channel - {str(e)}")
                return None
            
            # Get subscriber count
            if hasattr(kol_entity, 'participants_count'):
                subscriber_count = kol_entity.participants_count or 0
            else:
                subscriber_count = 0
            
            # Skip if below minimum subscribers
            if subscriber_count < self.config['min_subscribers']:
                logger.debug(f"⏭️ @{kol_username}: Too few subscribers ({subscriber_count})")
                return None
            
            # Analyze recent messages for token calls
            analysis_days = self.config['kol_analysis_days']
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=analysis_days)  # FIXED: timezone-aware
            
            token_calls = []
            message_count = 0
            
            try:
                async for message in self.client.iter_messages(
                    kol_entity,
                    limit=200,
                    offset_date=datetime.now(timezone.utc)  # FIXED: timezone-aware
                ):
                    if message.date < cutoff_date:
                        break
                    
                    if not message.text:
                        continue
                    
                    message_count += 1
                    
                    # Extract token addresses from message
                    token_addresses = self.extract_token_addresses(message.text)
                    
                    for token_address in token_addresses:
                        token_calls.append({
                            'token_address': token_address,
                            'call_time': message.date,
                            'message_text': message.text[:200]
                        })
                    
                    # Limit processing for performance
                    if len(token_calls) >= 50:
                        break
            
            except Exception as e:
                logger.debug(f"⏭️ @{kol_username}: Error reading messages - {str(e)}")
                return None
            
            if len(token_calls) < 3:
                logger.debug(f"⏭️ @{kol_username}: Insufficient data ({len(token_calls)} calls)")
                return None
            
            # Analyze token performance
            performance_metrics = await self.analyze_token_calls(token_calls)
            
            if not performance_metrics:
                logger.debug(f"⏭️ @{kol_username}: Performance analysis failed")
                return None
            
            # Calculate composite metrics
            composite_score = self.calculate_composite_score(performance_metrics)
            strategy_classification = self.classify_strategy(performance_metrics, subscriber_count)
            follower_tier = self.classify_follower_tier(subscriber_count)
            
            # Create KOL performance object
            kol_performance = KOLPerformance(
                kol=kol_username,
                channel_id=kol_entity.id,
                subscriber_count=subscriber_count,
                total_calls=performance_metrics['total_calls'],
                winning_calls=performance_metrics['winning_calls'],
                losing_calls=performance_metrics['losing_calls'],
                success_rate=performance_metrics['success_rate'],
                tokens_2x_plus=performance_metrics['tokens_2x_plus'],
                tokens_5x_plus=performance_metrics['tokens_5x_plus'],
                success_rate_2x=performance_metrics['success_rate_2x'],
                success_rate_5x=performance_metrics['success_rate_5x'],
                avg_time_to_2x_hours=performance_metrics['avg_time_to_2x_hours'],
                avg_max_pullback_percent=performance_metrics['avg_max_pullback_percent'],
                avg_unrealized_gains_percent=performance_metrics['avg_unrealized_gains_percent'],
                consistency_score=performance_metrics['consistency_score'],
                composite_score=composite_score,
                strategy_classification=strategy_classification,
                follower_tier=follower_tier,
                total_roi_percent=performance_metrics['total_roi_percent'],
                max_roi_percent=performance_metrics['max_roi_percent']
            )
            
            logger.info(f"✅ @{kol_username}: Score {composite_score:.1f}, Strategy {strategy_classification}")
            return kol_performance
            
        except FloodWaitError as e:
            logger.warning(f"⏭️ @{kol_username}: Rate limited, waiting {e.seconds}s")
            await asyncio.sleep(e.seconds)
            return None
        except Exception as e:
            logger.debug(f"⏭️ @{kol_username}: Analysis error - {str(e)}")
            return None
    
    def extract_token_addresses(self, message_text: str) -> List[str]:
        """Extract Solana token addresses from message text."""
        # Solana address pattern (32-44 chars, base58)
        pattern = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
        
        potential_addresses = re.findall(pattern, message_text)
        
        # Filter out common false positives
        valid_addresses = []
        for addr in potential_addresses:
            # Skip if it looks like a Telegram message ID or other non-token data
            if not self.is_likely_token_address(addr):
                continue
            valid_addresses.append(addr)
        
        return valid_addresses
    
    def is_likely_token_address(self, address: str) -> bool:
        """Check if an address is likely a token address."""
        # Basic validation
        if len(address) < 32 or len(address) > 44:
            return False
        
        # Common system program addresses to exclude
        system_addresses = {
            '11111111111111111111111111111111',  # System Program
            'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA',  # Token Program
            'So11111111111111111111111111111111111111112',  # Wrapped SOL
        }
        
        if address in system_addresses:
            return False
        
        # Should contain mixed case (typical of token addresses)
        has_upper = any(c.isupper() for c in address)
        has_lower = any(c.islower() for c in address)
        has_numbers = any(c.isdigit() for c in address)
        
        return has_upper and has_lower and has_numbers
    
    async def analyze_token_calls(self, token_calls: List[Dict]) -> Optional[Dict]:
        """Analyze performance of token calls."""
        if not self.api_manager:
            logger.error("❌ API manager not configured")
            return None
        
        total_calls = len(token_calls)
        successful_analyses = 0
        winning_calls = 0
        losing_calls = 0
        tokens_2x_plus = 0
        tokens_5x_plus = 0
        time_to_2x_list = []
        max_pullbacks = []
        unrealized_gains = []
        roi_values = []
        
        win_threshold = self.config['win_threshold_percent'] / 100
        
        for call in token_calls:
            try:
                # Get token performance
                performance = self.api_manager.calculate_token_performance(
                    call['token_address'],
                    call['call_time']
                )
                
                if not performance.get('success'):
                    continue
                
                successful_analyses += 1
                
                # Extract metrics
                roi_percent = performance.get('roi_percent', 0)
                max_roi = performance.get('max_roi_percent', 0)
                max_drawdown = abs(performance.get('max_drawdown_percent', 0))
                time_to_max_roi = performance.get('time_to_max_roi_hours', 0)
                
                roi_values.append(roi_percent)
                max_pullbacks.append(max_drawdown)
                
                # Calculate unrealized gains
                unrealized_gain = max_roi - roi_percent if max_roi > roi_percent else 0
                unrealized_gains.append(unrealized_gain)
                
                # Classify performance
                if roi_percent >= win_threshold * 100:
                    winning_calls += 1
                else:
                    losing_calls += 1
                
                # Count 2x+ and 5x+ tokens
                if max_roi >= 200:  # 2x
                    tokens_2x_plus += 1
                    if time_to_max_roi > 0:
                        time_to_2x_list.append(time_to_max_roi)
                
                if max_roi >= 500:  # 5x
                    tokens_5x_plus += 1
                
                # Prevent too many API calls
                if successful_analyses >= 30:
                    break
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.debug(f"Token analysis error: {str(e)}")
                continue
        
        if successful_analyses < 3:
            return None
        
        # Calculate metrics
        success_rate = (winning_calls / successful_analyses) * 100
        success_rate_2x = (tokens_2x_plus / successful_analyses) * 100
        success_rate_5x = (tokens_5x_plus / successful_analyses) * 100
        
        avg_time_to_2x = sum(time_to_2x_list) / len(time_to_2x_list) if time_to_2x_list else 0
        avg_max_pullback = sum(max_pullbacks) / len(max_pullbacks) if max_pullbacks else 0
        avg_unrealized_gains = sum(unrealized_gains) / len(unrealized_gains) if unrealized_gains else 0
        
        # Calculate consistency score
        roi_variance = self.calculate_variance(roi_values) if len(roi_values) > 1 else 0
        consistency_score = max(0, 100 - (roi_variance / 10))  # Simple consistency metric
        
        total_roi = sum(roi_values)
        max_roi = max(roi_values) if roi_values else 0
        
        return {
            'total_calls': total_calls,
            'analyzed_calls': successful_analyses,
            'winning_calls': winning_calls,
            'losing_calls': losing_calls,
            'success_rate': success_rate,
            'tokens_2x_plus': tokens_2x_plus,
            'tokens_5x_plus': tokens_5x_plus,
            'success_rate_2x': success_rate_2x,
            'success_rate_5x': success_rate_5x,
            'avg_time_to_2x_hours': avg_time_to_2x,
            'avg_max_pullback_percent': avg_max_pullback,
            'avg_unrealized_gains_percent': avg_unrealized_gains,
            'consistency_score': consistency_score,
            'total_roi_percent': total_roi,
            'max_roi_percent': max_roi
        }
    
    def calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if len(values) < 2:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def calculate_composite_score(self, metrics: Dict) -> float:
        """Calculate weighted composite score (0-100)."""
        weights = {
            'success_rate': 0.25,
            'success_rate_2x': 0.25,
            'consistency_score': 0.20,
            'time_to_2x': 0.15,
            'success_rate_5x': 0.15
        }
        
        # Normalize time to 2x (faster is better)
        time_score = max(0, 100 - (metrics['avg_time_to_2x_hours'] / 2)) if metrics['avg_time_to_2x_hours'] > 0 else 50
        
        score = (
            metrics['success_rate'] * weights['success_rate'] +
            metrics['success_rate_2x'] * weights['success_rate_2x'] +
            metrics['consistency_score'] * weights['consistency_score'] +
            time_score * weights['time_to_2x'] +
            metrics['success_rate_5x'] * weights['success_rate_5x']
        )
        
        return min(100, max(0, score))
    
    def classify_strategy(self, metrics: Dict, subscriber_count: int) -> str:
        """Classify KOL strategy as SCALP or HOLD."""
        # SCALP indicators: High followers + Fast 2x + Good success rate
        is_scalp = (
            subscriber_count >= 5000 and
            metrics['avg_time_to_2x_hours'] <= 12 and
            metrics['avg_time_to_2x_hours'] > 0 and
            metrics['success_rate'] >= 40
        )
        
        # HOLD indicators: High gem rate + Consistent performance
        is_hold = (
            metrics['success_rate_5x'] >= 15 and
            metrics['consistency_score'] >= 60
        )
        
        if is_scalp:
            return "SCALP"
        elif is_hold:
            return "HOLD"
        else:
            return "UNKNOWN"
    
    def classify_follower_tier(self, subscriber_count: int) -> str:
        """Classify follower tier."""
        if subscriber_count >= 10000:
            return "HIGH"
        elif subscriber_count >= 1000:
            return "MEDIUM"
        else:
            return "LOW"
    
    def should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed."""
        if not self.cache_file.exists():
            return True
        
        try:
            cache_age = datetime.now() - datetime.fromtimestamp(self.cache_file.stat().st_mtime)
            refresh_threshold = timedelta(hours=self.config['cache_refresh_hours'])
            
            if cache_age > refresh_threshold:
                logger.info(f"🔄 Cache expired ({cache_age.total_seconds() / 3600:.1f}h old)")
                return True
            else:
                logger.info(f"✅ Using fresh cache ({cache_age.total_seconds() / 3600:.1f}h old)")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking cache age: {str(e)}")
            return True
    
    def load_cache(self) -> Optional[Dict]:
        """Load cached analysis results."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                logger.info("📦 Loaded cached SPYDEFI analysis")
                return cache_data
        except Exception as e:
            logger.warning(f"Error loading cache: {str(e)}")
        
        return None
    
    def save_cache(self, results: Dict):
        """Save analysis results to cache."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"💾 Cache saved: {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    async def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete SPYDEFI analysis with FIXED KOL extraction and timezone handling.
        
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        start_time = time.time()
        
        logger.info("🚀 STARTING OPTIMIZED SPYDEFI ANALYSIS")
        logger.info("=" * 60)
        
        # Check cache first (auto-refresh based on age)
        if not self.should_refresh_cache():
            cached_results = self.load_cache()
            if cached_results:
                # ADDED: Check if cached results have corrupted KOL names
                kol_performances = cached_results.get('kol_performances', {})
                corrupted_names = {'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'}
                has_corrupted_data = any(kol in corrupted_names for kol in kol_performances.keys())
                
                if has_corrupted_data:
                    logger.info("🧹 Cache contains corrupted KOL names - refreshing with fixed extraction")
                else:
                    logger.info("✅ Using cached results (fresh enough)")
                    return cached_results
        
        try:
            # Step 1: Scan SpyDefi for KOL mentions
            print("🔍 STEP 1: Scanning SpyDefi for top KOLs...")
            kol_mentions, scan_metadata = await self.scan_spydefi_for_kols()
            
            if not kol_mentions:
                return {
                    'success': False,
                    'error': 'No KOLs found in SpyDefi scan',
                    'scan_metadata': scan_metadata
                }
            
            # Step 2: Analyze top KOLs
            top_kols_count = min(self.config['top_kols_count'], len(kol_mentions))
            print(f"🎯 STEP 2: Analyzing top {top_kols_count} KOLs...")
            
            kol_performances = {}
            analyzed_count = 0
            
            for i, (kol_username, mention_count) in enumerate(list(kol_mentions.items())[:top_kols_count], 1):
                print(f"📊 [{i}/{top_kols_count}] Analyzing @{kol_username} ({mention_count} mentions)...")
                
                try:
                    # Rate limiting
                    if i > 1:
                        await asyncio.sleep(1)
                    
                    performance = await self.analyze_kol_performance(kol_username, mention_count)
                    
                    if performance:
                        kol_performances[kol_username] = asdict(performance)
                        analyzed_count += 1
                        logger.info(f"✅ @{kol_username}: Analysis complete")
                    else:
                        print(f"⏭️ @{kol_username}: Insufficient data")
                    
                except Exception as e:
                    print(f"⏭️ @{kol_username}: {str(e)}")
                    logger.debug(f"KOL analysis error for @{kol_username}: {str(e)}")
            
            # Calculate overall statistics
            if kol_performances:
                all_success_rates = [p['success_rate'] for p in kol_performances.values()]
                all_2x_rates = [p['success_rate_2x'] for p in kol_performances.values()]
                all_5x_rates = [p['success_rate_5x'] for p in kol_performances.values()]
                all_calls = [p['total_calls'] for p in kol_performances.values()]
                
                overall_stats = {
                    'overall_success_rate': sum(all_success_rates) / len(all_success_rates),
                    'overall_2x_rate': sum(all_2x_rates) / len(all_2x_rates),
                    'overall_5x_rate': sum(all_5x_rates) / len(all_5x_rates),
                    'total_calls_analyzed': sum(all_calls)
                }
            else:
                overall_stats = {
                    'overall_success_rate': 0,
                    'overall_2x_rate': 0,
                    'overall_5x_rate': 0,
                    'total_calls_analyzed': 0
                }
            
            # Create final results
            processing_time = time.time() - start_time
            
            results = {
                'success': True,
                'version': '3.1.1',
                'timestamp': datetime.now().isoformat(),
                'kol_performances': kol_performances,
                'kol_mentions': kol_mentions,
                'metadata': {
                    **scan_metadata,
                    **overall_stats,
                    'processing_time_seconds': processing_time,
                    'kols_analyzed': analyzed_count,
                    'api_calls': getattr(self.api_manager, 'api_stats', {}),
                    'config': self.config.copy(),
                    'optimization_version': '3.1.1'
                }
            }
            
            # Save to cache
            self.save_cache(results)
            
            logger.info("=" * 60)
            logger.info("✅ OPTIMIZED SPYDEFI ANALYSIS COMPLETE")
            logger.info(f"⏱️ Processing time: {processing_time:.1f} seconds")
            logger.info(f"📊 KOLs analyzed: {analyzed_count}/{top_kols_count}")
            logger.info(f"📈 Overall success rate: {overall_stats['overall_success_rate']:.1f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ SPYDEFI analysis failed: {str(e)}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'processing_time_seconds': time.time() - start_time,
                'metadata': {
                    'config': self.config.copy(),
                    'optimization_version': '3.1.1'
                }
            }