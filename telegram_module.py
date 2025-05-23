"""
Telegram Module - Phoenix Project (REDESIGNED for Real Enhanced Metrics)

This module implements the complete redesigned SpyDefi analysis process:
1. SpyDefi Discovery (24h) -> Find active KOLs
2. Individual KOL Analysis (24h each) -> Real performance metrics  
3. Enhanced Metrics Calculation -> Time-to-2x, pullback data
4. Consistent TOP 10 Ranking -> Channel ID collection
"""

import re
import csv
import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel, User, Channel
from collections import defaultdict, Counter

logger = logging.getLogger("phoenix.telegram")

# Enhanced contract patterns for better detection
SOLANA_ADDRESS_PATTERN = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
CONTRACT_PATTERNS = [
    r'(?i)contract(?:\s*address)?[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)token(?:\s*address)?[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)CA\s*(?:is|:)?\s*([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)address[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)Ca:?\s*([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)Pump:?\s*([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)Token:?\s*([1-9A-HJ-NP-Za-km-z]{32,44})'
]

# KOL detection patterns
KOL_USERNAME_PATTERN = r'@([A-Za-z0-9_]+)'
KOL_CALL_PATTERNS = [
    r'(?i)made a x(\d+)\+ call on',
    r'(?i)Achievement Unlocked: x(\d+)',
    r'(?i)(\d+)x\s+gem'
]

class TelegramScraper:
    """Redesigned class for SpyDefi analysis with real enhanced metrics."""
    
    def __init__(self, api_id: str, api_hash: str, session_name: str = "phoenix", max_days: int = 14):
        """
        Initialize the Telegram scraper.
        
        Args:
            api_id (str): Telegram API ID
            api_hash (str): Telegram API hash
            session_name (str): Session name for Telethon
            max_days (int): Maximum number of days to scrape back
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.max_days = max_days
        self.client = None
        self.message_limit = 2000
        self.spydefi_message_limit = 1000
        self.kol_channel_cache = {}
        self.birdeye_api = None  # Will be set externally
    
    async def connect(self) -> None:
        """Connect to Telegram API."""
        if not self.client:
            logger.info("Connecting to Telegram...")
            self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
            await self.client.start()
            logger.info("Connected to Telegram")
    
    async def disconnect(self) -> None:
        """Disconnect from Telegram API."""
        if self.client:
            logger.info("Disconnecting from Telegram...")
            await self.client.disconnect()
            self.client = None
            logger.info("Disconnected from Telegram")
    
    def extract_contract_addresses(self, text: str) -> List[str]:
        """Extract potential contract addresses from text."""
        addresses = set()
        
        # Try specific contract patterns first
        for pattern in CONTRACT_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) > 0:
                    addresses.add(match.group(1))
        
        # Look for URLs that might contain contract addresses
        url_pattern = r'https?://(?:www\.)?(?:dexscreener\.com|birdeye\.so|solscan\.io|explorer\.solana\.com)/[^"\s]+?([1-9A-HJ-NP-Za-km-z]{32,44})'
        url_matches = re.finditer(url_pattern, text)
        for match in url_matches:
            if len(match.groups()) > 0:
                addresses.add(match.group(1))
        
        # If no matches found with specific patterns, try generic Solana address pattern
        if not addresses:
            matches = re.finditer(SOLANA_ADDRESS_PATTERN, text)
            for match in matches:
                addresses.add(match.group(0))
        
        return list(addresses)
    
    def extract_kol_usernames(self, text: str) -> List[str]:
        """Extract KOL usernames from text with improved patterns."""
        kols = set()
        
        # Enhanced KOL patterns for SpyDefi format
        patterns = [
            r'@(\w+)\s+made\s+a\s+x\d+\+?\s+call\s+on',
            r'first\s+posted\s+by\s+@(\w+)',
            r'Achievement\s+Unlocked:.*?@(\w+)',
            r'@(\w+)(?:\s+made\s+a)',
            r'@([A-Za-z0-9_]+)(?=\s+(?:made|call|posted))',
            r'^[^@]*@(\w+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match.groups():
                    username = match.group(1).lower().strip('@')
                    # Filter out common words and system accounts
                    if username not in ['spydefi', 'everyone', 'here', 'channel', 'group', 'unlocked', 'achievement', 'view', 'stats', 'call']:
                        kols.add(username)
        
        return list(kols)
    
    def extract_token_names(self, text: str) -> List[str]:
        """Extract token names from SpyDefi messages."""
        tokens = set()
        
        # Token name patterns from SpyDefi format
        patterns = [
            r'made\s+a\s+x\d+\+?\s+call\s+on\s+([A-Za-z0-9\s\.\-_]+?)(?:\s+on\s+\w+|\s*\.|$)',
            r'^([A-Za-z0-9\s\.\-_]+?)\s+first\s+posted\s+by\s+@\w+',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match.groups():
                    token_name = match.group(1).strip()
                    # Filter out very short or common words, but keep things like "Bitcoin 2.0"
                    if len(token_name) >= 2 and token_name.lower() not in ['the', 'and', 'for', 'with', 'has', 'been']:
                        tokens.add(token_name)
        
        return list(tokens)
    
    async def get_kol_channel_id(self, kol_username: str) -> str:
        """Get channel ID for a KOL username with rate limiting protection."""
        # Check cache first
        if kol_username in self.kol_channel_cache:
            return self.kol_channel_cache[kol_username]
        
        # Try different variations of the KOL channel name
        possible_names = [
            f"@{kol_username}",
            kol_username,
            f"{kol_username}calls",
            f"@{kol_username}calls",
            f"{kol_username}_calls",
            f"@{kol_username}_calls"
        ]
        
        channel_id = ""
        for name in possible_names:
            try:
                entity = await self.client.get_entity(name)
                if hasattr(entity, 'id'):
                    channel_id = str(entity.id)
                    logger.debug(f"Found channel {name} -> {channel_id}")
                    break
            except Exception as e:
                logger.debug(f"Failed to find channel {name}: {str(e)}")
                continue
        
        # Cache the result (even if empty) to avoid repeated lookups
        self.kol_channel_cache[kol_username] = channel_id
        return channel_id
    
    def is_likely_token_call(self, text: str) -> bool:
        """Determine if a message is likely a token call."""
        # Common phrases used in token calls
        call_indicators = [
            r'(?i)(\bnew\s+call\b|\btoken\s+call\b)',
            r'(?i)(\bbuy\s+now\b|\bentry\s+now\b)',
            r'(?i)(\bcontract\s+address\b|\btoken\s+address\b|\bCA\b|\bCa\b)',
            r'(?i)(\btarget\s+\d+x\b|\bpotential\s+\d+x\b)',
            r'(?i)(\bmoon\s+shot\b|\bpump\b|\bgem\b|\bearly\b)',
            r'(?i)(buy\s*&?\s*sell)',
            r'(?i)(dexscreener\.com|birdeye\.so)'
        ]
        
        # Check if it contains a contract address
        has_address = bool(self.extract_contract_addresses(text))
        
        # Check for call indicators
        call_score = sum(1 for pattern in call_indicators if re.search(pattern, text))
        
        # Check for x5+, x6+ achievement patterns
        achievement_match = False
        for pattern in KOL_CALL_PATTERNS:
            if re.search(pattern, text):
                achievement_match = True
                break
        
        # If it has an address and at least 1 call indicator, or is an achievement post, it's likely a call
        return (has_address and call_score >= 1) or achievement_match
    
    async def get_channel_messages(self, channel_id: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get messages from a Telegram channel."""
        if not self.client:
            await self.connect()
        
        # Apply the maximum days limit and check for SpyDefi
        is_spydefi = channel_id.lower() == "spydefi"
        
        # For SpyDefi, use a lower limit to avoid resource issues
        if is_spydefi:
            days_to_scrape = min(days_back, 7)
            message_limit = self.spydefi_message_limit
            logger.info(f"SpyDefi channel detected, limiting to {days_to_scrape} days and {message_limit} messages")
        else:
            days_to_scrape = min(days_back, self.max_days)
            message_limit = self.message_limit
        
        logger.info(f"Scraping {days_to_scrape} days of messages from {channel_id}")
        
        try:
            # Handle both username and channel ID formats
            if channel_id.lower() == "spydefi":
                channel_id = "SpyDefi"
            elif channel_id.startswith("@"):
                channel_id = channel_id[1:]
            elif channel_id.isdigit():
                # Convert numeric channel ID string to integer
                channel_id = int(channel_id)
                logger.debug(f"Converted channel ID to integer: {channel_id}")
                
            entity = await self.client.get_entity(channel_id)
            
            # Calculate the date limit (timezone-aware to match Telegram message dates)
            date_limit = datetime.now(timezone.utc) - timedelta(days=days_to_scrape)
            
            # Get messages
            messages = []
            offset_id = 0
            limit = 100
            total_messages = 0
            
            while total_messages < message_limit:
                history = await self.client(GetHistoryRequest(
                    peer=entity,
                    offset_id=offset_id,
                    offset_date=None,
                    add_offset=0,
                    limit=limit,
                    max_id=0,
                    min_id=0,
                    hash=0
                ))
                
                if not history.messages:
                    break
                
                for message in history.messages:
                    # Stop if we've reached the date limit
                    if message.date < date_limit:
                        break
                    
                    # Extract message data
                    if hasattr(message, 'message') and message.message:
                        # Extract the KOL username from the message sender if possible
                        sender_username = ""
                        if hasattr(message, 'sender') and message.sender:
                            if isinstance(message.sender, User) and message.sender.username:
                                sender_username = message.sender.username
                            elif isinstance(message.sender, Channel) and message.sender.username:
                                sender_username = message.sender.username
                        
                        # Extract additional usernames from the message text
                        kol_usernames = self.extract_kol_usernames(message.message)
                        
                        message_data = {
                            "id": message.id,
                            "date": message.date.isoformat(),
                            "timestamp": int(message.date.timestamp()),
                            "text": message.message,
                            "is_call": self.is_likely_token_call(message.message),
                            "contract_addresses": self.extract_contract_addresses(message.message),
                            "token_names": self.extract_token_names(message.message),
                            "sender_username": sender_username,
                            "mentioned_usernames": kol_usernames
                        }
                        
                        if message_data["is_call"] or message_data["contract_addresses"] or kol_usernames or message_data["token_names"]:
                            messages.append(message_data)
                
                # Break if we've reached the date limit
                if history.messages and history.messages[-1].date < date_limit:
                    break
                
                # Update offset for next batch
                offset_id = history.messages[-1].id
                total_messages += len(history.messages)
                logger.info(f"Retrieved {total_messages} messages from {channel_id}")
                
                # Break if no more messages
                if len(history.messages) < limit:
                    break
            
            logger.info(f"Finished retrieving messages from {channel_id}. Found {len(messages)} relevant messages.")
            return messages
            
        except Exception as e:
            logger.error(f"Error retrieving messages from {channel_id}: {str(e)}")
            return []
    
    def calculate_enhanced_metrics(self, performance_data: Dict[str, Any], call_timestamp: int) -> Dict[str, Any]:
        """Calculate enhanced metrics including time-to-2x and max pullback."""
        if not performance_data.get("success") or not performance_data.get("data"):
            return {
                'reached_2x': False,
                'reached_5x': False,
                'time_to_2x_seconds': None,
                'time_to_5x_seconds': None,
                'max_pullback_percent': 0,
                'max_roi_percent': 0,
                'current_roi_percent': 0,
                'analysis_success': False
            }
        
        try:
            price_history = performance_data.get("data", {}).get("items", [])
            if not price_history:
                return {
                    'reached_2x': False,
                    'reached_5x': False,
                    'time_to_2x_seconds': None,
                    'time_to_5x_seconds': None,
                    'max_pullback_percent': 0,
                    'max_roi_percent': 0,
                    'current_roi_percent': 0,
                    'analysis_success': False
                }
            
            # Get initial price (first price point after call)
            initial_price = price_history[0].get("value", 0)
            if initial_price <= 0:
                return {
                    'reached_2x': False,
                    'reached_5x': False,
                    'time_to_2x_seconds': None,
                    'time_to_5x_seconds': None,
                    'max_pullback_percent': 0,
                    'max_roi_percent': 0,
                    'current_roi_percent': 0,
                    'analysis_success': False
                }
            
            # Track milestones and metrics
            hit_2x_time = None
            hit_5x_time = None
            max_roi = 0
            max_pullback = 0
            current_price = price_history[-1].get("value", initial_price)
            
            # Analyze each price point
            for i, price_point in enumerate(price_history):
                price = price_point.get("value", 0)
                timestamp = price_point.get("unixTime", call_timestamp)
                
                if price <= 0:
                    continue
                
                # Calculate ROI from initial price
                roi = (price / initial_price - 1) * 100
                max_roi = max(max_roi, roi)
                
                # Check for milestone achievements
                if roi >= 100 and hit_2x_time is None:
                    hit_2x_time = timestamp
                if roi >= 400 and hit_5x_time is None:
                    hit_5x_time = timestamp
                
                # Calculate pullback from peak
                if i > 0:
                    # Find peak price up to this point
                    peak_price = max(p.get("value", 0) for p in price_history[:i+1])
                    if peak_price > 0:
                        pullback = ((peak_price - price) / peak_price) * 100
                        max_pullback = max(max_pullback, pullback)
            
            # Calculate final metrics
            current_roi = (current_price / initial_price - 1) * 100
            
            return {
                'reached_2x': hit_2x_time is not None,
                'reached_5x': hit_5x_time is not None,
                'time_to_2x_seconds': (hit_2x_time - call_timestamp) if hit_2x_time else None,
                'time_to_5x_seconds': (hit_5x_time - call_timestamp) if hit_5x_time else None,
                'max_pullback_percent': round(max_pullback, 2),
                'max_roi_percent': round(max_roi, 2),
                'current_roi_percent': round(current_roi, 2),
                'analysis_success': True,
                'initial_price': initial_price,
                'current_price': current_price,
                'price_points_analyzed': len(price_history)
            }
            
        except Exception as e:
            logger.error(f"Error calculating enhanced metrics: {str(e)}")
            return {
                'reached_2x': False,
                'reached_5x': False,
                'time_to_2x_seconds': None,
                'time_to_5x_seconds': None,
                'max_pullback_percent': 0,
                'max_roi_percent': 0,
                'current_roi_percent': 0,
                'analysis_success': False
            }
    
    def format_duration(self, seconds: Optional[int]) -> str:
        """Format duration in seconds to human readable format."""
        if seconds is None or seconds <= 0:
            return "N/A"
        
        try:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            
            if hours > 0:
                return f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                return f"{minutes}m {secs}s"
            else:
                return f"{secs}s"
        except:
            return "N/A"
    
    async def analyze_individual_kol(self, kol_username: str, channel_id: str = None) -> Dict[str, Any]:
        """Analyze individual KOL's channel for real performance metrics."""
        logger.info(f"🔍 Analyzing individual KOL: @{kol_username}")
        
        # Get channel ID if not provided
        if not channel_id:
            channel_id = await self.get_kol_channel_id(kol_username)
            if not channel_id:
                logger.warning(f"❌ No channel found for @{kol_username}")
                return self._get_empty_kol_analysis(kol_username)
        
        # Scrape KOL's channel for past 24 hours (reduced from 7 days to save API usage)
        try:
            kol_messages = await self.get_channel_messages(channel_id, days_back=1)
            logger.info(f"📨 Found {len(kol_messages)} messages in @{kol_username}'s channel")
            
            if not kol_messages:
                return self._get_empty_kol_analysis(kol_username, channel_id)
            
            # Find token calls in their channel
            token_calls = []
            for message in kol_messages:
                if message["is_call"] and (message["contract_addresses"] or message["token_names"]):
                    # Prefer contract addresses, fallback to token names
                    contracts = message["contract_addresses"]
                    tokens = message["token_names"]
                    
                    # Create call entry
                    call_entry = {
                        'message_id': message["id"],
                        'call_timestamp': message["timestamp"],
                        'call_date': message["date"],
                        'message_text': message["text"],
                        'contract_addresses': contracts,
                        'token_names': tokens,
                        'has_contract': len(contracts) > 0
                    }
                    token_calls.append(call_entry)
            
            logger.info(f"🎯 Found {len(token_calls)} token calls for @{kol_username}")
            
            # Analyze each token call with Birdeye API
            analyzed_calls = []
            successful_2x = 0
            successful_5x = 0
            
            for i, call in enumerate(token_calls):
                if not call['has_contract']:
                    # Skip calls without contract addresses
                    continue
                
                logger.info(f"📊 Analyzing call {i+1}/{len(token_calls)} for @{kol_username}")
                
                # Analyze each contract in the call
                for contract_address in call['contract_addresses']:
                    try:
                        # Get price history from call time to now
                        call_time = datetime.fromtimestamp(call['call_timestamp'])
                        
                        if self.birdeye_api:
                            performance = self.birdeye_api.get_token_price_history(
                                contract_address,
                                start_time=call['call_timestamp'],
                                end_time=int(datetime.now().timestamp()),
                                resolution="15m"
                            )
                            
                            # Calculate enhanced metrics
                            enhanced_metrics = self.calculate_enhanced_metrics(
                                performance, 
                                call['call_timestamp']
                            )
                            
                            if enhanced_metrics['analysis_success']:
                                call_analysis = call.copy()
                                call_analysis.update(enhanced_metrics)
                                call_analysis['contract_address'] = contract_address
                                analyzed_calls.append(call_analysis)
                                
                                # Track successes
                                if enhanced_metrics['reached_2x']:
                                    successful_2x += 1
                                if enhanced_metrics['reached_5x']:
                                    successful_5x += 1
                            
                            # Rate limiting
                            await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"❌ Error analyzing contract {contract_address}: {str(e)}")
                        continue
            
            # Calculate KOL performance metrics
            return self._calculate_kol_performance(
                kol_username, channel_id, analyzed_calls, successful_2x, successful_5x
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing KOL @{kol_username}: {str(e)}")
            return self._get_empty_kol_analysis(kol_username, channel_id)
    
    def _calculate_kol_performance(self, kol_username: str, channel_id: str, 
                                 analyzed_calls: List[Dict[str, Any]], 
                                 successful_2x: int, successful_5x: int) -> Dict[str, Any]:
        """Calculate comprehensive KOL performance metrics."""
        total_calls = len(analyzed_calls)
        
        if total_calls == 0:
            return self._get_empty_kol_analysis(kol_username, channel_id)
        
        # Calculate averages
        time_to_2x_values = [call['time_to_2x_seconds'] for call in analyzed_calls 
                           if call['time_to_2x_seconds'] is not None]
        pullback_values = [call['max_pullback_percent'] for call in analyzed_calls 
                         if call['max_pullback_percent'] > 0]
        roi_values = [call['max_roi_percent'] for call in analyzed_calls]
        
        avg_time_to_2x = sum(time_to_2x_values) / len(time_to_2x_values) if time_to_2x_values else None
        avg_pullback = sum(pullback_values) / len(pullback_values) if pullback_values else 0
        avg_max_roi = sum(roi_values) / len(roi_values) if roi_values else 0
        
        # Calculate success rates
        success_rate_2x = (successful_2x / total_calls * 100) if total_calls > 0 else 0
        success_rate_5x = (successful_5x / total_calls * 100) if total_calls > 0 else 0
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(
            total_calls, success_rate_2x, success_rate_5x, avg_time_to_2x, avg_pullback, avg_max_roi
        )
        
        return {
            'kol': kol_username,
            'channel_id': channel_id,
            'tokens_mentioned': total_calls,
            'tokens_2x_plus': successful_2x,
            'tokens_5x_plus': successful_5x,
            'success_rate_2x': round(success_rate_2x, 2),
            'success_rate_5x': round(success_rate_5x, 2),
            'avg_ath_roi': round(avg_max_roi, 2),
            'composite_score': round(composite_score, 2),
            'avg_max_pullback_percent': round(avg_pullback, 2) if avg_pullback > 0 else 0,
            'avg_time_to_2x_seconds': int(avg_time_to_2x) if avg_time_to_2x else None,
            'avg_time_to_2x_formatted': self.format_duration(int(avg_time_to_2x)) if avg_time_to_2x else "N/A",
            'detailed_analysis_count': total_calls,
            'pullback_data_available': avg_pullback > 0,
            'time_to_2x_data_available': avg_time_to_2x is not None,
            'analyzed_calls': analyzed_calls
        }
    
    def _calculate_composite_score(self, total_calls: int, success_rate_2x: float, 
                                 success_rate_5x: float, avg_time_to_2x: Optional[int], 
                                 avg_pullback: float, avg_max_roi: float) -> float:
        """Calculate composite score for ranking KOLs."""
        # Base score from sample size (more calls = more reliable)
        sample_score = min(total_calls * 10, 100)
        
        # Success rate bonus (weighted towards 5x)
        success_bonus = (success_rate_2x * 0.5) + (success_rate_5x * 1.5)
        
        # ROI bonus
        roi_bonus = min(avg_max_roi / 10, 50)
        
        # Time bonus (faster to 2x is better)
        time_bonus = 0
        if avg_time_to_2x:
            # Bonus for reaching 2x quickly (max 24 hours = 86400 seconds)
            time_bonus = max(0, 25 - (avg_time_to_2x / 3600))  # 25 points max, decreases with time
        
        # Risk bonus (lower pullback is better)
        risk_bonus = max(0, 25 - avg_pullback)  # 25 points max, decreases with pullback
        
        # Sample size multiplier
        reliability_multiplier = 1.0
        if total_calls >= 20:
            reliability_multiplier = 1.5
        elif total_calls >= 10:
            reliability_multiplier = 1.3
        elif total_calls >= 5:
            reliability_multiplier = 1.1
        
        final_score = (sample_score + success_bonus + roi_bonus + time_bonus + risk_bonus) * reliability_multiplier
        return final_score
    
    def _get_empty_kol_analysis(self, kol_username: str, channel_id: str = "") -> Dict[str, Any]:
        """Return empty KOL analysis structure."""
        return {
            'kol': kol_username,
            'channel_id': channel_id,
            'tokens_mentioned': 0,
            'tokens_2x_plus': 0,
            'tokens_5x_plus': 0,
            'success_rate_2x': 0,
            'success_rate_5x': 0,
            'avg_ath_roi': 0,
            'composite_score': 0,
            'avg_max_pullback_percent': 0,
            'avg_time_to_2x_seconds': None,
            'avg_time_to_2x_formatted': "N/A",
            'detailed_analysis_count': 0,
            'pullback_data_available': False,
            'time_to_2x_data_available': False,
            'analyzed_calls': []
        }
    
    async def redesigned_spydefi_analysis(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        REDESIGNED SpyDefi analysis with real enhanced metrics.
        
        Phase 1: Discover active KOLs from SpyDefi (24h)
        Phase 2: Analyze each KOL's individual channel (24h)
        Phase 3: Calculate enhanced metrics and rank
        Phase 4: Get channel IDs for TOP 10
        """
        logger.info("🚀 STARTING REDESIGNED SPYDEFI ANALYSIS")
        logger.info("🎯 Phase 1: Discovering active KOLs from SpyDefi...")
        
        try:
            # Phase 1: Discover active KOLs from SpyDefi channel
            active_kols = await self._discover_active_kols(hours_back)
            logger.info(f"✅ Found {len(active_kols)} active KOLs from SpyDefi")
            
            if not active_kols:
                return {
                    'success': False,
                    'error': 'No active KOLs found in SpyDefi',
                    'scan_period_hours': hours_back
                }
            
            # Phase 2: Analyze individual KOL channels
            logger.info("🎯 Phase 2: Analyzing individual KOL channels (24 hours each)...")
            kol_analyses = {}
            
            # Configuration: How many KOLs to analyze (top X by mentions)
            MAX_KOLS_TO_ANALYZE = 50  # Configurable limit
            MIN_MENTIONS_REQUIRED = 2  # Only analyze KOLs with 2+ mentions
            
            # Filter and sort KOLs
            filtered_kols = {k: v for k, v in active_kols.items() if v >= MIN_MENTIONS_REQUIRED}
            sorted_kols = sorted(filtered_kols.items(), key=lambda x: x[1], reverse=True)[:MAX_KOLS_TO_ANALYZE]
            
            logger.info(f"📊 Found {len(active_kols)} total KOLs, filtering to {len(sorted_kols)} "
                       f"(top {MAX_KOLS_TO_ANALYZE} with {MIN_MENTIONS_REQUIRED}+ mentions)")
            
            for i, (kol_username, mention_count) in enumerate(sorted_kols):
                logger.info(f"📊 Analyzing KOL {i+1}/{len(sorted_kols)}: @{kol_username} ({mention_count} mentions)")
                
                kol_analysis = await self.analyze_individual_kol(kol_username)
                
                if kol_analysis['tokens_mentioned'] > 0:
                    kol_analyses[kol_username] = kol_analysis
                    logger.info(f"✅ @{kol_username}: {kol_analysis['tokens_mentioned']} calls, "
                              f"{kol_analysis['success_rate_2x']:.1f}% 2x rate")
                else:
                    logger.info(f"⚠️ @{kol_username}: No analyzable calls found")
                
                # Rate limiting between KOL analyses
                await asyncio.sleep(1)
            
            # Phase 3: Rank KOLs and calculate summary statistics
            logger.info("🎯 Phase 3: Ranking KOLs and calculating final metrics...")
            ranked_kols = self._rank_kols_consistently(kol_analyses)
            
            # Phase 4: Get channel IDs for TOP 10 only
            logger.info("🎯 Phase 4: Getting channel IDs for TOP 10 KOLs...")
            top_10_with_channels = await self._get_top_10_channel_ids(ranked_kols)
            
            # Calculate summary statistics
            total_calls = sum(kol['tokens_mentioned'] for kol in ranked_kols.values())
            total_2x = sum(kol['tokens_2x_plus'] for kol in ranked_kols.values())
            total_5x = sum(kol['tokens_5x_plus'] for kol in ranked_kols.values())
            
            success_rate_2x = (total_2x / total_calls * 100) if total_calls > 0 else 0
            success_rate_5x = (total_5x / total_calls * 100) if total_calls > 0 else 0
            
            logger.info("🎉 REDESIGNED ANALYSIS COMPLETE!")
            logger.info(f"📊 Total KOLs analyzed: {len(ranked_kols)}")
            logger.info(f"📊 Total calls analyzed: {total_calls}")
            logger.info(f"📊 2x success rate: {success_rate_2x:.1f}%")
            logger.info(f"📊 5x success rate: {success_rate_5x:.1f}%")
            
            return {
                'success': True,
                'scan_period_hours': hours_back,
                'total_kols_analyzed': len(ranked_kols),
                'total_calls': total_calls,
                'successful_2x': total_2x,
                'successful_5x': total_5x,
                'success_rate_2x': round(success_rate_2x, 2),
                'success_rate_5x': round(success_rate_5x, 2),
                'ranked_kols': ranked_kols,
                'top_10_with_channels': top_10_with_channels,
                'summary': {
                    'kols_analyzed': len(ranked_kols),
                    'total_calls': total_calls,
                    'tokens_that_made_x2': total_2x,
                    'tokens_that_made_x5': total_5x,
                    'success_rate_2x_percent': f"{success_rate_2x:.2f}%",
                    'success_rate_5x_percent': f"{success_rate_5x:.2f}%",
                    'enhanced_data_coverage': f"{sum(1 for k in ranked_kols.values() if k['pullback_data_available'])}/{len(ranked_kols)} KOLs"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in redesigned SpyDefi analysis: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'scan_period_hours': hours_back,
                'total_calls': 0,
                'successful_2x': 0,
                'successful_5x': 0,
                'success_rate_2x': 0,
                'success_rate_5x': 0
            }
    
    async def _discover_active_kols(self, hours_back: int) -> Dict[str, int]:
        """Phase 1: Discover active KOLs from SpyDefi channel."""
        try:
            # Get SpyDefi messages
            entity = await self.client.get_entity("SpyDefi")
            time_limit = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            kol_mentions = defaultdict(int)
            message_count = 0
            max_messages = 500
            
            logger.info("📨 Scanning SpyDefi for active KOLs...")
            
            async for message in self.client.iter_messages(entity, limit=max_messages):
                message_count += 1
                
                if not message.message or message.date < time_limit:
                    if message.date < time_limit:
                        logger.info(f"Reached time limit at message {message_count}")
                        break
                    continue
                
                # Extract KOL usernames from achievement messages
                kols = self.extract_kol_usernames(message.message)
                for kol in kols:
                    kol_mentions[kol] += 1
            
            logger.info(f"📊 Processed {message_count} SpyDefi messages")
            logger.info(f"🔍 Found {len(kol_mentions)} unique KOLs")
            
            return dict(kol_mentions)
            
        except Exception as e:
            logger.error(f"❌ Error discovering active KOLs: {str(e)}")
            return {}
    
    def _rank_kols_consistently(self, kol_analyses: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Phase 3: Rank KOLs consistently by composite score."""
        # Convert to list for sorting
        kol_list = list(kol_analyses.items())
        
        # Sort by composite score (descending)
        kol_list.sort(key=lambda x: x[1]['composite_score'], reverse=True)
        
        # Convert back to ordered dict to maintain ranking
        ranked_kols = {}
        for kol_username, kol_data in kol_list:
            ranked_kols[kol_username] = kol_data
        
        logger.info("🏆 TOP 10 KOLs by composite score:")
        for i, (kol, data) in enumerate(list(ranked_kols.items())[:10]):
            logger.info(f"   {i+1}. @{kol}: {data['composite_score']:.1f} score, "
                       f"{data['success_rate_2x']:.1f}% 2x rate, "
                       f"{data['tokens_mentioned']} calls")
        
        return ranked_kols
    
    async def _get_top_10_channel_ids(self, ranked_kols: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Phase 4: Get channel IDs for TOP 10 KOLs only."""
        top_10_kols = list(ranked_kols.items())[:10]
        
        logger.info("📞 Getting channel IDs for TOP 10 KOLs...")
        
        for i, (kol_username, kol_data) in enumerate(top_10_kols):
            try:
                logger.info(f"Getting channel ID for TOP 10 KOL {i+1}/10: @{kol_username}")
                
                # Get channel ID (use cached if available)
                if not kol_data.get('channel_id'):
                    channel_id = await self.get_kol_channel_id(kol_username)
                    kol_data['channel_id'] = channel_id
                
                if kol_data['channel_id']:
                    logger.info(f"✅ Found channel for @{kol_username}: {kol_data['channel_id']}")
                else:
                    logger.info(f"❌ No channel found for @{kol_username}")
                
                # Rate limiting
                if i < len(top_10_kols) - 1:
                    await asyncio.sleep(2.0)
                    
            except Exception as e:
                logger.warning(f"❌ Error getting channel ID for @{kol_username}: {str(e)}")
                kol_data['channel_id'] = ""
                continue
        
        # Return updated ranked_kols dict
        return ranked_kols
    
    # Legacy compatibility methods (keep existing interface)
    async def scan_spydefi_channel(self, hours_back: int = 24, get_channel_ids: bool = True) -> Dict[str, Any]:
        """Legacy method - redirects to redesigned analysis."""
        return await self.redesigned_spydefi_analysis(hours_back)
    
    async def scrape_spydefi(self, channel_id: str, days_back: int = 7, birdeye_api: Any = None) -> Dict[str, Any]:
        """Legacy method - redirects to redesigned analysis (uses 24h per KOL regardless of days_back)."""
        self.birdeye_api = birdeye_api
        hours_back = days_back * 24
        return await self.redesigned_spydefi_analysis(hours_back)
    
    async def export_spydefi_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Export redesigned analysis results to CSV with enhanced metrics."""
        if not analysis.get("success") or not analysis.get("ranked_kols"):
            logger.warning("No analysis data to export")
            return
        
        try:
            # Ensure output directories exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            
            # Export main KOL performance CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'kol', 'channel_id', 'tokens_mentioned', 'tokens_2x_plus', 'tokens_5x_plus',
                    'success_rate_2x', 'success_rate_5x', 'avg_ath_roi', 'composite_score',
                    'avg_max_pullback_percent', 'avg_time_to_2x_seconds', 'avg_time_to_2x_formatted',
                    'detailed_analysis_count', 'pullback_data_available', 'time_to_2x_data_available'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for kol_username, kol_data in analysis['ranked_kols'].items():
                    row = {field: kol_data.get(field, '') for field in fieldnames}
                    writer.writerow(row)
            
            logger.info(f"✅ Exported {len(analysis['ranked_kols'])} KOLs to {output_file}")
            
            # Export detailed calls if available
            detail_file = output_file.replace('.csv', '_detailed_calls.csv')
            self._export_detailed_calls(analysis['ranked_kols'], detail_file)
            
            # Export summary
            summary_file = output_file.replace('.csv', '_summary.txt')
            self._export_summary(analysis, summary_file)
            
        except Exception as e:
            logger.error(f"❌ Error exporting analysis: {str(e)}")
    
    def _export_detailed_calls(self, ranked_kols: Dict[str, Dict[str, Any]], output_file: str) -> None:
        """Export detailed call analysis."""
        try:
            all_calls = []
            for kol_username, kol_data in ranked_kols.items():
                for call in kol_data.get('analyzed_calls', []):
                    call_row = {
                        'kol': kol_username,
                        'contract_address': call.get('contract_address', ''),
                        'call_date': call.get('call_date', ''),
                        'reached_2x': call.get('reached_2x', False),
                        'reached_5x': call.get('reached_5x', False),
                        'max_roi_percent': call.get('max_roi_percent', 0),
                        'current_roi_percent': call.get('current_roi_percent', 0),
                        'max_pullback_percent': call.get('max_pullback_percent', 0),
                        'time_to_2x_formatted': self.format_duration(call.get('time_to_2x_seconds')),
                        'time_to_5x_formatted': self.format_duration(call.get('time_to_5x_seconds')),
                        'analysis_success': call.get('analysis_success', False)
                    }
                    all_calls.append(call_row)
            
            if all_calls:
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = list(all_calls[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_calls)
                
                logger.info(f"✅ Exported {len(all_calls)} detailed calls to {output_file}")
        
        except Exception as e:
            logger.error(f"❌ Error exporting detailed calls: {str(e)}")
    
    def _export_summary(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Export analysis summary."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== REDESIGNED SPYDEFI PERFORMANCE ANALYSIS ===\n\n")
                f.write(f"Analysis period: {analysis.get('scan_period_hours', 24)} hours (SpyDefi discovery)\n")
                f.write(f"Individual KOL analysis: 24 hours per channel\n\n")
                
                f.write("=== OVERALL STATISTICS ===\n")
                f.write(f"KOLs analyzed: {analysis.get('total_kols_analyzed', 0)}\n")
                f.write(f"Total calls analyzed: {analysis.get('total_calls', 0)}\n")
                f.write(f"Tokens that made x2: {analysis.get('successful_2x', 0)}\n")
                f.write(f"Tokens that made x5: {analysis.get('successful_5x', 0)}\n")
                f.write(f"Success rate (2x): {analysis.get('success_rate_2x', 0):.2f}%\n")
                f.write(f"Success rate (5x): {analysis.get('success_rate_5x', 0):.2f}%\n\n")
                
                f.write("=== TOP 10 KOLs (with enhanced metrics) ===\n")
                ranked_kols = analysis.get('ranked_kols', {})
                for i, (kol, data) in enumerate(list(ranked_kols.items())[:10]):
                    f.write(f"{i+1}. @{kol}\n")
                    f.write(f"   Channel ID: {data.get('channel_id', 'Not found')}\n")
                    f.write(f"   Calls analyzed: {data.get('tokens_mentioned', 0)}\n")
                    f.write(f"   2x success rate: {data.get('success_rate_2x', 0):.1f}%\n")
                    f.write(f"   5x success rate: {data.get('success_rate_5x', 0):.1f}%\n")
                    f.write(f"   Avg max pullback: {data.get('avg_max_pullback_percent', 0):.1f}%\n")
                    f.write(f"   Avg time to 2x: {data.get('avg_time_to_2x_formatted', 'N/A')}\n")
                    f.write(f"   Composite score: {data.get('composite_score', 0):.1f}\n\n")
            
            logger.info(f"✅ Exported summary to {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error exporting summary: {str(e)}")