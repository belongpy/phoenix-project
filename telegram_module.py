"""
Telegram Module - Phoenix Project (Enhanced)

This module handles all Telegram-related functionality, including scraping and analysis
of KOL channels and token calls.

Enhanced Features:
- Max average pullback % calculation for stop loss setting
- Average time to reach 2x calculation for holding strategy
- Detailed price analysis using Birdeye API
- Enhanced contract address detection
- Real-time performance tracking
"""

import re
import csv
import os
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel, User, Channel
from collections import defaultdict

logger = logging.getLogger("phoenix.telegram")

# Enhanced regex patterns for detecting Solana addresses and contract mentions
SOLANA_ADDRESS_PATTERN = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
CONTRACT_PATTERNS = [
    r'(?i)contract(?:\s*address)?[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)token(?:\s*address)?[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)CA\s*(?:is|:)?\s*([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)address[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)Ca:?\s*([1-9A-HJ-NP-Za-km-z]{32,44})',
    # Enhanced patterns for better extraction
    r'(?i)(?:contract|token|ca|address)[:=\s]*([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'([1-9A-HJ-NP-Za-km-z]{32,44})(?=\s*(?:\n|$|[^\w]))',  # Standalone addresses
    r'(?:dexscreener\.com|birdeye\.so|solscan\.io).*?([1-9A-HJ-NP-Za-km-z]{32,44})',  # URL extraction
]

# KOL detection patterns (enhanced)
KOL_USERNAME_PATTERN = r'@([A-Za-z0-9_]+)'
KOL_CALL_PATTERNS = [
    r'(?i)made a x(\d+)\+ call on',
    r'(?i)Achievement Unlocked: x(\d+)',
    r'(?i)(\d+)x\s+gem'
]

class TelegramScraper:
    """Enhanced class for scraping and analyzing Telegram channels."""
    
    def __init__(self, api_id: str, api_hash: str, session_name: str = "phoenix", max_days: int = 14):
        """
        Initialize the enhanced Telegram scraper.
        
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
        self.message_limit = 2000  # Maximum number of messages to retrieve per channel
        self.spydefi_message_limit = 1000  # Lower limit for SpyDefi to avoid resource issues
        self.kol_channel_cache = {}  # Cache for KOL channel IDs
    
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
        """
        Extract potential contract addresses from text (enhanced).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of extracted contract addresses
        """
        addresses = set()
        
        # Try all enhanced contract patterns
        for pattern in CONTRACT_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) > 0:
                    address = match.group(1)
                else:
                    address = match.group(0)
                
                # Validate address length and format
                if 32 <= len(address) <= 44 and self._is_valid_solana_address(address):
                    addresses.add(address)
        
        # Additional URL-based extraction
        url_patterns = [
            r'https?://(?:www\.)?(?:dexscreener\.com|birdeye\.so|solscan\.io|explorer\.solana\.com)/[^"\s]*?([1-9A-HJ-NP-Za-km-z]{32,44})',
            r'https?://(?:www\.)?raydium\.io/[^"\s]*?([1-9A-HJ-NP-Za-km-z]{32,44})',
            r'https?://(?:www\.)?jupiter-terminal\.com/[^"\s]*?([1-9A-HJ-NP-Za-km-z]{32,44})'
        ]
        
        for pattern in url_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) > 0:
                    address = match.group(1)
                    if 32 <= len(address) <= 44 and self._is_valid_solana_address(address):
                        addresses.add(address)
        
        return list(addresses)
    
    def _is_valid_solana_address(self, address: str) -> bool:
        """
        Basic validation for Solana addresses.
        
        Args:
            address (str): Address to validate
            
        Returns:
            bool: True if valid format, False otherwise
        """
        try:
            # Check length
            if not (32 <= len(address) <= 44):
                return False
            
            # Check for valid base58 characters
            valid_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            if not all(c in valid_chars for c in address):
                return False
            
            # Additional checks can be added here
            return True
            
        except Exception:
            return False
    
    def extract_kol_usernames(self, text: str) -> List[str]:
        """
        Extract KOL usernames from text with enhanced patterns.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of extracted KOL usernames
        """
        kols = set()
        
        # Enhanced KOL patterns for SpyDefi format
        patterns = [
            # "@username made a x2+ call on TOKEN"
            r'@(\w+)\s+made\s+a\s+x\d+\+?\s+call\s+on',
            # "TOKEN first posted by @username"
            r'first\s+posted\s+by\s+@(\w+)',
            # Achievement patterns: "Achievement Unlocked: x2! @username"
            r'Achievement\s+Unlocked:.*?@(\w+)',
            # Direct @mentions anywhere in achievement messages
            r'@(\w+)(?:\s+made\s+a)',
            # General @mentions (but more selective)
            r'@([A-Za-z0-9_]+)(?=\s+(?:made|call|posted))',
            # Catch @username at start of achievement lines
            r'^[^@]*@(\w+)',
            # New patterns for enhanced detection
            r'(?:by|from|via)\s+@(\w+)',
            r'@(\w+)\s+(?:called|shared|posted)',
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
        """Extract token names from SpyDefi messages (enhanced)."""
        tokens = set()
        
        # Enhanced token name patterns from SpyDefi format
        patterns = [
            # "@username made a x2+ call on TOKEN_NAME"
            r'made\s+a\s+x\d+\+?\s+call\s+on\s+([A-Za-z0-9\s\.\-_]+?)(?:\s+on\s+\w+|\s*\.|$)',
            # "TOKEN_NAME first posted by @username"
            r'^([A-Za-z0-9\s\.\-_]+?)\s+first\s+posted\s+by\s+@\w+',
            # "TOKEN_NAME reached x2"
            r'^([A-Za-z0-9\s\.\-_]+?)\s+reached\s+x\d+',
            # Enhanced patterns
            r'(?:token|coin)\s+([A-Za-z0-9\s\.\-_]+?)\s+(?:hit|reached|made)',
            r'([A-Za-z0-9\s\.\-_]+?)\s+(?:mooning|pumping|going\s+up)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match.groups():
                    token_name = match.group(1).strip()
                    # Filter out very short or common words, but keep things like "Bitcoin 2.0"
                    if len(token_name) >= 2 and token_name.lower() not in ['the', 'and', 'for', 'with', 'has', 'been', 'this', 'that']:
                        tokens.add(token_name)
        
        return list(tokens)
    
    async def get_kol_channel_id(self, kol_username: str) -> str:
        """
        Get channel ID for a KOL username with rate limiting protection.
        
        Args:
            kol_username (str): KOL username without @
            
        Returns:
            str: Channel ID or empty string if not found
        """
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
            f"@{kol_username}_calls",
            f"{kol_username}signal",
            f"@{kol_username}signal",
            f"{kol_username}gems",
            f"@{kol_username}gems",
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
        """
        Determine if a message is likely a token call (enhanced).
        
        Args:
            text (str): Message text
            
        Returns:
            bool: True if likely a token call, False otherwise
        """
        # Common phrases used in token calls (enhanced)
        call_indicators = [
            r'(?i)(\bnew\s+call\b|\btoken\s+call\b)',
            r'(?i)(\bbuy\s+now\b|\bentry\s+now\b)',
            r'(?i)(\bcontract\s+address\b|\btoken\s+address\b|\bCA\b|\bCa\b)',
            r'(?i)(\btarget\s+\d+x\b|\bpotential\s+\d+x\b)',
            r'(?i)(\bmoon\s+shot\b|\bpump\b|\bgem\b|\bearly\b)',
            r'(?i)(buy\s*&?\s*sell)',
            r'(?i)(dexscreener\.com|birdeye\.so)',
            # Enhanced indicators
            r'(?i)(\blong\s+term\b|\bhold\b|\bhodl\b)',
            r'(?i)(\bentry\s+point\b|\btake\s+profit\b|\bstop\s+loss\b)',
            r'(?i)(\bjust\s+launched\b|\bfresh\s+launch\b)',
            r'(?i)(\bmarket\s+cap\b|\bmcap\b|\bliquidity\b)',
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
        
        # Enhanced scoring
        if has_address and call_score >= 1:
            return True
        if achievement_match:
            return True
        if call_score >= 2:  # Multiple indicators without address
            return True
        
        return False
    
    async def get_channel_messages(self, channel_id: str, 
                                days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Get messages from a Telegram channel (enhanced).
        
        Args:
            channel_id (str): Channel ID or username
            days_back (int): Number of days to scrape back
            
        Returns:
            List[Dict[str, Any]]: List of message data
        """
        if not self.client:
            await self.connect()
        
        # Apply the maximum days limit and check for SpyDefi
        is_spydefi = channel_id.lower() == "spydefi"
        
        # For SpyDefi, use a lower limit to avoid resource issues
        if is_spydefi:
            days_to_scrape = min(days_back, 7)  # Always limit SpyDefi to 7 days max
            message_limit = self.spydefi_message_limit
            logger.info(f"SpyDefi channel detected, limiting to {days_to_scrape} days and {message_limit} messages")
        else:
            days_to_scrape = min(days_back, self.max_days)
            message_limit = self.message_limit
        
        logger.info(f"Scraping {days_to_scrape} days of messages from {channel_id}")
        
        try:
            # Handle both username and channel ID formats
            if channel_id.lower() == "spydefi":
                channel_id = "SpyDefi"  # Ensure proper capitalization
            
            if channel_id.startswith("@"):
                channel_id = channel_id[1:]  # Remove @ if present
                
            entity = await self.client.get_entity(channel_id)
            
            # Calculate the date limit
            date_limit = datetime.now() - timedelta(days=days_to_scrape)
            
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
                            "text": message.message,
                            "is_call": self.is_likely_token_call(message.message),
                            "contract_addresses": self.extract_contract_addresses(message.message),
                            "sender_username": sender_username,
                            "mentioned_usernames": kol_usernames
                        }
                        
                        if message_data["is_call"] or message_data["contract_addresses"] or kol_usernames:
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
    
    async def analyze_token_detailed_performance(self, token_address: str, call_timestamp: int, 
                                               birdeye_api: Any) -> Dict[str, Any]:
        """
        Analyze detailed token performance including pullback and time-to-2x.
        
        Args:
            token_address (str): Token contract address
            call_timestamp (int): Unix timestamp of the initial call
            birdeye_api: Birdeye API instance
            
        Returns:
            Dict[str, Any]: Detailed performance metrics including pullback and time-to-2x
        """
        try:
            call_datetime = datetime.fromtimestamp(call_timestamp)
            current_time = datetime.now()
            
            # Get price history from call time to now
            price_history = birdeye_api.get_token_price_history(
                token_address,
                call_timestamp,
                int(current_time.timestamp()),
                "5m"  # 5-minute intervals for detailed analysis
            )
            
            if not price_history.get("success") or not price_history.get("data", {}).get("items"):
                logger.warning(f"No price history available for detailed analysis: {token_address}")
                return {
                    "success": False,
                    "error": "No price history available",
                    "has_detailed_analysis": False
                }
            
            prices = price_history["data"]["items"]
            if not prices:
                return {
                    "success": False,
                    "error": "No price data points",
                    "has_detailed_analysis": False
                }
            
            # Sort prices by timestamp
            prices = sorted(prices, key=lambda x: x.get("unixTime", 0))
            
            initial_price = prices[0].get("value", 0)
            if not initial_price or initial_price <= 0:
                return {
                    "success": False,
                    "error": "Invalid initial price",
                    "has_detailed_analysis": False
                }
            
            # Calculate detailed metrics
            min_price = initial_price
            max_price = initial_price
            current_price = prices[-1].get("value", initial_price)
            max_pullback_percent = 0
            time_to_2x_seconds = None
            time_to_2x_formatted = "Never reached"
            
            # Process each price point
            for i, price_data in enumerate(prices):
                price = price_data.get("value", 0)
                timestamp = price_data.get("unixTime", call_timestamp)
                
                if price <= 0:
                    continue
                
                # Update min/max prices
                if price < min_price:
                    min_price = price
                if price > max_price:
                    max_price = price
                
                # Calculate pullback from initial price
                pullback_percent = ((price / initial_price) - 1) * 100
                if pullback_percent < max_pullback_percent:
                    max_pullback_percent = pullback_percent
                
                # Check for 2x achievement (100% ROI)
                if time_to_2x_seconds is None and price >= (initial_price * 2):
                    time_to_2x_seconds = timestamp - call_timestamp
                    time_to_2x_formatted = self._format_duration(time_to_2x_seconds)
            
            # Calculate final metrics
            current_roi_percent = ((current_price / initial_price) - 1) * 100
            max_roi_percent = ((max_price / initial_price) - 1) * 100
            
            return {
                "success": True,
                "has_detailed_analysis": True,
                "token_address": token_address,
                "call_timestamp": call_timestamp,
                "call_date": call_datetime.isoformat(),
                "initial_price": initial_price,
                "current_price": current_price,
                "min_price": min_price,
                "max_price": max_price,
                "current_roi_percent": round(current_roi_percent, 2),
                "max_roi_percent": round(max_roi_percent, 2),
                "max_pullback_percent": round(abs(max_pullback_percent), 2),  # Make positive for easier reading
                "time_to_2x_seconds": time_to_2x_seconds,
                "time_to_2x_formatted": time_to_2x_formatted,
                "analysis_duration_hours": (current_time.timestamp() - call_timestamp) / 3600,
                "data_points_analyzed": len(prices)
            }
            
        except Exception as e:
            logger.error(f"Error in detailed token performance analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "has_detailed_analysis": False
            }
    
    def _format_duration(self, seconds: int) -> str:
        """
        Format duration in seconds to human-readable format.
        
        Args:
            seconds (int): Duration in seconds
            
        Returns:
            str: Formatted duration (e.g., "2h 30m 15s")
        """
        if seconds is None:
            return "N/A"
        
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            if hours >= 24:
                days = hours // 24
                hours = hours % 24
                return f"{days}d {hours}h {minutes}m"
            else:
                return f"{hours}h {minutes}m {secs}s"
    
    async def scan_spydefi_channel_enhanced(self, hours_back: int = 24, 
                                          get_channel_ids: bool = True,
                                          birdeye_api: Any = None) -> Dict[str, Any]:
        """
        ENHANCED SpyDefi channel scanning with pullback and time-to-2x analysis.
        
        Args:
            hours_back (int): Hours to analyze back
            get_channel_ids (bool): Whether to fetch KOL channel IDs
            birdeye_api: Birdeye API instance for detailed analysis
            
        Returns:
            Dict[str, Any]: Enhanced analysis results with new metrics
        """
        logger.info(f"🎯 Starting ENHANCED SpyDefi scanning (past {hours_back}h) with detailed metrics...")
        
        if not self.client:
            await self.connect()
        
        try:
            # Try different SpyDefi entity variations
            entity = None
            possible_names = ["@spydefi", "spydefi", "SpyDefi", "@SpyDefi"]
            
            for name in possible_names:
                try:
                    logger.info(f"Trying to get entity: {name}")
                    entity = await self.client.get_entity(name)
                    logger.info(f"Successfully found entity: {name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to get entity {name}: {str(e)}")
                    continue
            
            if not entity:
                logger.error("Could not find SpyDefi entity with any variation")
                return {
                    'error': 'SpyDefi channel not found',
                    'enhanced_analysis': False
                }
            
            logger.info(f"Entity found: {entity.title if hasattr(entity, 'title') else 'Unknown'}")
            
            # Calculate time limit for proper filtering
            time_limit = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            # Collect all token calls with enhanced analysis
            token_calls = []
            kol_mentions = defaultdict(list)
            message_count = 0
            max_messages = 500
            detailed_analysis_count = 0
            
            logger.info("Starting enhanced message iteration...")
            
            # Enhanced SpyDefi message processing loop
            async for message in self.client.iter_messages(entity, limit=max_messages):
                message_count += 1
                
                if not message.message:
                    continue
                
                # Apply time filter - stop if message is too old
                if message.date < time_limit:
                    logger.info(f"Reached time limit at message {message_count}")
                    break
                
                text = message.message
                unix_time = int(message.date.timestamp())
                message_date = message.date
                
                # Show first few messages for debugging
                if message_count <= 5:
                    logger.info(f"Message {message_count} ({message_date}): {text[:150]}...")
                
                # Enhanced extraction
                contracts = self.extract_contract_addresses(text)
                kols = self.extract_kol_usernames(text)
                token_names = self.extract_token_names(text)
                
                # Always log extraction results for first 5 messages
                if message_count <= 5:
                    logger.info(f"  EXTRACTION - Contracts: {contracts}, KOLs: {kols}, Tokens: {token_names}")
                
                # Skip messages without tokens or KOLs
                if not (contracts or kols or token_names):
                    continue
                
                # Filter out non-Solana chains
                if re.search(r'\bon\s+(bsc|eth|ethereum|polygon|arbitrum|base)\b', text, re.IGNORECASE):
                    logger.info(f"  Filtered out non-Solana chain in message {message_count}")
                    continue
                
                # Use token names if no contracts found
                tokens_to_process = contracts if contracts else token_names
                
                if tokens_to_process:
                    logger.info(f"  Processing {len(tokens_to_process)} tokens from message {message_count}")
                    
                    for token in tokens_to_process:
                        call_entry = {
                            'contract': token if token in contracts else '',
                            'token_name': token if token in token_names else '',
                            'unix_time': unix_time,
                            'date': message_date.replace(tzinfo=timezone.utc).isoformat(),
                            'text': text[:200],
                            'kols': kols,
                            'has_detailed_analysis': False,
                            'max_pullback_percent': 0,
                            'time_to_2x_seconds': None,
                            'time_to_2x_formatted': "N/A"
                        }
                        
                        # Enhanced analysis for tokens with contract addresses
                        if token in contracts and birdeye_api:
                            logger.info(f"  🎯 Running detailed analysis for {token}")
                            detailed_performance = await self.analyze_token_detailed_performance(
                                token, unix_time, birdeye_api
                            )
                            
                            if detailed_performance.get("success"):
                                call_entry.update({
                                    'has_detailed_analysis': True,
                                    'initial_price': detailed_performance.get('initial_price', 0),
                                    'current_price': detailed_performance.get('current_price', 0),
                                    'min_price': detailed_performance.get('min_price', 0),
                                    'max_price': detailed_performance.get('max_price', 0),
                                    'current_roi_percent': detailed_performance.get('current_roi_percent', 0),
                                    'max_roi_percent': detailed_performance.get('max_roi_percent', 0),
                                    'max_pullback_percent': detailed_performance.get('max_pullback_percent', 0),
                                    'time_to_2x_seconds': detailed_performance.get('time_to_2x_seconds'),
                                    'time_to_2x_formatted': detailed_performance.get('time_to_2x_formatted', 'N/A'),
                                })
                                detailed_analysis_count += 1
                                logger.info(f"    ✅ Detailed analysis complete - Pullback: {call_entry['max_pullback_percent']}%, Time to 2x: {call_entry['time_to_2x_formatted']}")
                            else:
                                logger.info(f"    ⚠️ Detailed analysis failed: {detailed_performance.get('error', 'Unknown error')}")
                        
                        # Fallback to ROI extraction from message text if no detailed analysis
                        if not call_entry['has_detailed_analysis']:
                            roi_data = self._extract_roi_from_message(text)
                            if roi_data:
                                call_entry.update(roi_data)
                        
                        token_calls.append(call_entry)
                        
                        # Track KOL mentions
                        for kol in kols:
                            kol_mentions[kol].append(call_entry)
            
            logger.info(f"Processed {message_count} total messages")
            logger.info(f"Found {len(token_calls)} token calls from {len(kol_mentions)} KOLs")
            logger.info(f"🎯 Enhanced analysis completed for {detailed_analysis_count} tokens")
            
            # Calculate enhanced KOL performance metrics
            kol_performance = {}
            successful_2x = 0
            successful_5x = 0
            
            for kol, kol_calls in kol_mentions.items():
                kol_detailed_calls = [c for c in kol_calls if c.get('has_detailed_analysis', False)]
                kol_2x_calls = [c for c in kol_calls if c.get('current_roi_percent', 0) >= 100]
                kol_5x_calls = [c for c in kol_calls if c.get('current_roi_percent', 0) >= 400]
                
                # Calculate enhanced metrics
                pullback_values = [c.get('max_pullback_percent', 0) for c in kol_detailed_calls if c.get('max_pullback_percent', 0) > 0]
                time_to_2x_values = [c.get('time_to_2x_seconds', 0) for c in kol_detailed_calls if c.get('time_to_2x_seconds') is not None]
                
                avg_pullback = sum(pullback_values) / len(pullback_values) if pullback_values else 0
                avg_time_to_2x = sum(time_to_2x_values) / len(time_to_2x_values) if time_to_2x_values else 0
                
                kol_performance[kol] = {
                    'channel_id': '',  # Will be populated later
                    'tokens_mentioned': len(kol_calls),
                    'tokens_2x_plus': len(kol_2x_calls),
                    'tokens_5x_plus': len(kol_5x_calls),
                    'success_rate_2x': round((len(kol_2x_calls) / len(kol_calls) * 100) if kol_calls else 0, 2),
                    'success_rate_5x': round((len(kol_5x_calls) / len(kol_calls) * 100) if kol_calls else 0, 2),
                    'avg_ath_roi': round(sum(c.get('max_roi_percent', 0) for c in kol_calls) / len(kol_calls) if kol_calls else 0, 2),
                    'composite_score': self.calculate_kol_score(len(kol_calls), len(kol_2x_calls), len(kol_5x_calls), kol_calls),
                    # NEW ENHANCED METRICS
                    'avg_max_pullback_percent': round(avg_pullback, 2),
                    'avg_time_to_2x_seconds': int(avg_time_to_2x) if avg_time_to_2x > 0 else None,
                    'avg_time_to_2x_formatted': self._format_duration(int(avg_time_to_2x)) if avg_time_to_2x > 0 else "N/A",
                    'detailed_analysis_count': len(kol_detailed_calls),
                    'pullback_data_available': len(pullback_values) > 0,
                    'time_to_2x_data_available': len(time_to_2x_values) > 0
                }
                
                successful_2x += len(kol_2x_calls)
                successful_5x += len(kol_5x_calls)
            
            # Get channel IDs for top KOLs (rate-limited)
            if get_channel_ids:
                logger.info(f"Getting channel IDs for top KOLs...")
                sorted_kols_by_score = sorted(kol_performance.items(), 
                                            key=lambda x: x[1]['composite_score'], 
                                            reverse=True)
                
                # Only get channel IDs for top 10 KOLs to avoid rate limits
                top_kols = sorted_kols_by_score[:10]
                
                for i, (kol, kol_data) in enumerate(top_kols):
                    try:
                        logger.info(f"Getting channel ID for KOL {i+1}/{len(top_kols)}: @{kol}")
                        channel_id = await self.get_kol_channel_id(kol)
                        kol_performance[kol]['channel_id'] = channel_id
                        
                        if channel_id:
                            logger.info(f"Found channel for @{kol}: {channel_id}")
                        else:
                            logger.info(f"No channel found for @{kol}")
                        
                        # Rate limiting
                        if i < len(top_kols) - 1:
                            await asyncio.sleep(2.0)
                            
                    except Exception as e:
                        logger.warning(f"Error getting channel ID for @{kol}: {str(e)}")
                        continue
            
            # Sort KOLs by composite score
            sorted_kols = sorted(kol_performance.items(), 
                               key=lambda x: x[1]['composite_score'], 
                               reverse=True)
            
            total_calls = len(token_calls)
            success_rate_2x = (successful_2x / total_calls * 100) if total_calls > 0 else 0
            success_rate_5x = (successful_5x / total_calls * 100) if total_calls > 0 else 0
            
            logger.info("🎯 Enhanced Analysis Complete!")
            logger.info(f"Tokens mentioned: {total_calls}")
            logger.info(f"Tokens that made x2: {successful_2x}")
            logger.info(f"Tokens that made x5: {successful_5x}")
            logger.info(f"Success rate (2x): {success_rate_2x:.2f}%")
            logger.info(f"Success rate (5x): {success_rate_5x:.2f}%")
            logger.info(f"Enhanced analysis coverage: {detailed_analysis_count}/{total_calls} tokens")
            
            return {
                'enhanced_analysis': True,
                'scan_period_hours': hours_back,
                'total_calls': total_calls,
                'successful_2x': successful_2x,
                'successful_5x': successful_5x,
                'success_rate_2x': round(success_rate_2x, 2),
                'success_rate_5x': round(success_rate_5x, 2),
                'analyzed_calls': token_calls,
                'kol_performance': dict(sorted_kols),
                'ranked_kols': dict(sorted_kols),
                'enhanced_metrics': {
                    'detailed_analysis_count': detailed_analysis_count,
                    'detailed_analysis_coverage': round((detailed_analysis_count / total_calls * 100) if total_calls > 0 else 0, 2),
                    'pullback_data_available': len([c for c in token_calls if c.get('max_pullback_percent', 0) > 0]),
                    'time_to_2x_data_available': len([c for c in token_calls if c.get('time_to_2x_seconds') is not None])
                },
                'summary': {
                    'tokens_mentioned': total_calls,
                    'tokens_that_made_x2': successful_2x,
                    'tokens_that_made_x5': successful_5x,
                    'success_rate_2x_percent': f"{success_rate_2x:.2f}%",
                    'success_rate_5x_percent': f"{success_rate_5x:.2f}%",
                    'enhanced_analysis_coverage': f"{detailed_analysis_count}/{total_calls} tokens"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced SpyDefi scanning: {str(e)}")
            return {
                'error': str(e),
                'enhanced_analysis': False,
                'total_calls': 0,
                'successful_2x': 0,
                'successful_5x': 0,
                'success_rate_2x': 0,
                'success_rate_5x': 0
            }
    
    def _extract_roi_from_message(self, text: str) -> Dict[str, Any]:
        """
        Extract ROI information from SpyDefi message text (fallback method).
        
        Args:
            text (str): Message text
            
        Returns:
            Dict[str, Any]: ROI data if found
        """
        roi_patterns = [
            r'\$(\d+(?:\.\d+)?)([KM]?)\s*->\s*\$(\d+(?:\.\d+)?)([KM]?)',
            r'\$(\d+(?:\.\d+)?)([KM]?)\s*->\s*\$(\d+(?:\.\d+)?)([KM]?)\.{0,3}',
            r'\$(\d+(?:\.\d+)?)([KM]?)\s*→\s*\$(\d+(?:\.\d+)?)([KM]?)',
        ]
        
        for pattern in roi_patterns:
            roi_match = re.search(pattern, text)
            if roi_match:
                try:
                    # Extract values and units
                    start_val = float(roi_match.group(1))
                    start_unit = roi_match.group(2) if len(roi_match.groups()) >= 2 else ''
                    end_val = float(roi_match.group(3))
                    end_unit = roi_match.group(4) if len(roi_match.groups()) >= 4 else ''
                    
                    # Handle K and M notation
                    if start_unit == 'K':
                        start_val *= 1000
                    elif start_unit == 'M':
                        start_val *= 1000000
                        
                    if end_unit == 'K':
                        end_val *= 1000
                    elif end_unit == 'M':
                        end_val *= 1000000
                    
                    # Calculate ROI
                    roi_percent = ((end_val / start_val) - 1) * 100 if start_val > 0 else 0
                    
                    return {
                        'call_price': start_val,
                        'ath_price': end_val,
                        'current_price': end_val,
                        'current_roi_percent': round(roi_percent, 2),
                        'max_roi_percent': round(roi_percent, 2),
                        'is_2x_plus': roi_percent >= 100,
                        'is_5x_plus': roi_percent >= 400,
                        'has_detailed_analysis': False
                    }
                    
                except (ValueError, ZeroDivisionError) as e:
                    logger.debug(f"Error parsing ROI from text: {str(e)}")
                    continue
        
        return {}
    
    def calculate_kol_score(self, total_calls: int, calls_2x: int, calls_5x: int, analyzed_calls: List[Dict]) -> float:
        """
        Calculate a composite KOL score that properly weighs sample size.
        
        Args:
            total_calls (int): Total number of calls
            calls_2x (int): Number of 2x+ calls
            calls_5x (int): Number of 5x+ calls  
            analyzed_calls (List[Dict]): List of analyzed call data
            
        Returns:
            float: Composite score (higher = better)
        """
        if total_calls == 0:
            return 0
        
        # Base score from sample size (heavily weighted)
        sample_score = min(total_calls * 10, 100)  # Max 100 points for sample size
        
        # Success rate bonus
        success_rate_2x = (calls_2x / total_calls) * 100
        success_rate_5x = (calls_5x / total_calls) * 100
        
        success_bonus = (success_rate_2x * 0.5) + (success_rate_5x * 1.0)  # 5x calls worth more
        
        # ROI bonus
        roi_values = [c.get('current_roi_percent', 0) for c in analyzed_calls if 'current_roi_percent' in c]
        avg_roi = sum(roi_values) / len(roi_values) if roi_values else 0
        roi_bonus = min(avg_roi / 10, 50)  # Max 50 points for ROI
        
        # Enhanced metrics bonus
        detailed_calls = [c for c in analyzed_calls if c.get('has_detailed_analysis', False)]
        enhanced_bonus = len(detailed_calls) * 5  # Bonus for detailed analysis availability
        
        # Sample size multiplier (more calls = more reliable)
        reliability_multiplier = 1.0
        if total_calls >= 10:
            reliability_multiplier = 1.5
        elif total_calls >= 5:
            reliability_multiplier = 1.2
        elif total_calls >= 3:
            reliability_multiplier = 1.1
        
        final_score = (sample_score + success_bonus + roi_bonus + enhanced_bonus) * reliability_multiplier
        
        return round(final_score, 2)
    
    async def export_enhanced_spydefi_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """
        Export ENHANCED Spydefi analysis to CSV with new metrics.
        
        Args:
            analysis (Dict[str, Any]): Enhanced Spydefi analysis data
            output_file (str): Output file path
        """
        if not analysis or not analysis.get("analyzed_calls"):
            logger.warning("No analyzed calls to export")
            return
        
        try:
            # Ensure output directories exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            
            # Export detailed calls with NEW COLUMNS
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'contract', 'token_name', 'date', 'kols', 'call_price', 
                    'ath_price', 'current_roi_percent', 'max_roi_percent',
                    'is_2x_plus', 'is_5x_plus', 'text',
                    # NEW ENHANCED COLUMNS
                    'has_detailed_analysis', 'max_pullback_percent', 
                    'time_to_2x_seconds', 'time_to_2x_formatted',
                    'min_price', 'initial_price', 'current_price'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for call in analysis['analyzed_calls']:
                    row = {
                        'contract': call.get('contract', ''),
                        'token_name': call.get('token_name', ''),
                        'date': call.get('date', ''),
                        'kols': ', '.join(call.get('kols', [])),
                        'call_price': call.get('call_price', ''),
                        'ath_price': call.get('ath_price', ''),
                        'current_roi_percent': call.get('current_roi_percent', ''),
                        'max_roi_percent': call.get('max_roi_percent', ''),
                        'is_2x_plus': call.get('is_2x_plus', False),
                        'is_5x_plus': call.get('is_5x_plus', False),
                        'text': call.get('text', ''),
                        # NEW ENHANCED FIELDS
                        'has_detailed_analysis': call.get('has_detailed_analysis', False),
                        'max_pullback_percent': call.get('max_pullback_percent', ''),
                        'time_to_2x_seconds': call.get('time_to_2x_seconds', ''),
                        'time_to_2x_formatted': call.get('time_to_2x_formatted', ''),
                        'min_price': call.get('min_price', ''),
                        'initial_price': call.get('initial_price', ''),
                        'current_price': call.get('current_price', '')
                    }
                    writer.writerow(row)
            
            logger.info(f"✅ Exported {len(analysis['analyzed_calls'])} enhanced calls to {output_file}")
            
            # Export ENHANCED KOL performance summary with NEW METRICS
            kol_file = output_file.replace('.csv', '_kol_performance_enhanced.csv')
            with open(kol_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'kol', 'channel_id', 'tokens_mentioned', 'tokens_2x_plus', 'tokens_5x_plus',
                    'success_rate_2x', 'success_rate_5x', 'avg_ath_roi', 'composite_score',
                    # NEW ENHANCED METRICS FOR TRADING STRATEGY
                    'avg_max_pullback_percent', 'avg_time_to_2x_seconds', 'avg_time_to_2x_formatted',
                    'detailed_analysis_count', 'pullback_data_available', 'time_to_2x_data_available'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for kol, performance in analysis.get('kol_performance', {}).items():
                    row = {
                        'kol': kol,
                        'channel_id': performance.get('channel_id', ''),
                        'tokens_mentioned': performance.get('tokens_mentioned', 0),
                        'tokens_2x_plus': performance.get('tokens_2x_plus', 0),
                        'tokens_5x_plus': performance.get('tokens_5x_plus', 0),
                        'success_rate_2x': performance.get('success_rate_2x', 0),
                        'success_rate_5x': performance.get('success_rate_5x', 0),
                        'avg_ath_roi': performance.get('avg_ath_roi', 0),
                        'composite_score': performance.get('composite_score', 0),
                        # NEW ENHANCED METRICS
                        'avg_max_pullback_percent': performance.get('avg_max_pullback_percent', 0),
                        'avg_time_to_2x_seconds': performance.get('avg_time_to_2x_seconds', ''),
                        'avg_time_to_2x_formatted': performance.get('avg_time_to_2x_formatted', 'N/A'),
                        'detailed_analysis_count': performance.get('detailed_analysis_count', 0),
                        'pullback_data_available': performance.get('pullback_data_available', False),
                        'time_to_2x_data_available': performance.get('time_to_2x_data_available', False)
                    }
                    writer.writerow(row)
            
            logger.info(f"✅ Exported ENHANCED KOL performance to {kol_file}")
            
            # Export enhanced summary
            summary_file = output_file.replace('.csv', '_enhanced_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                summary = analysis.get('summary', {})
                enhanced_metrics = analysis.get('enhanced_metrics', {})
                
                f.write("=== 🎯 ENHANCED SPYDEFI PERFORMANCE ANALYSIS ===\n\n")
                f.write(f"Analysis period: {analysis.get('scan_period_hours', 24)} hours\n")
                f.write(f"Tokens mentioned: {summary.get('tokens_mentioned', 0)}\n")
                f.write(f"Tokens that made x2: {summary.get('tokens_that_made_x2', 0)}\n")
                f.write(f"Tokens that made x5: {summary.get('tokens_that_made_x5', 0)}\n")
                f.write(f"Success rate (2x): {summary.get('success_rate_2x_percent', '0.00%')}\n")
                f.write(f"Success rate (5x): {summary.get('success_rate_5x_percent', '0.00%')}\n\n")
                
                f.write("=== 🎯 ENHANCED ANALYSIS COVERAGE ===\n")
                f.write(f"Enhanced analysis: {enhanced_metrics.get('enhanced_analysis_coverage', 0)}%\n")
                f.write(f"Detailed analysis count: {enhanced_metrics.get('detailed_analysis_count', 0)}\n")
                f.write(f"Pullback data available: {enhanced_metrics.get('pullback_data_available', 0)} tokens\n")
                f.write(f"Time-to-2x data available: {enhanced_metrics.get('time_to_2x_data_available', 0)} tokens\n\n")
                
                f.write("=== 🎯 TOP KOLs WITH ENHANCED METRICS ===\n")
                for i, (kol, perf) in enumerate(list(analysis.get('kol_performance', {}).items())[:10]):
                    f.write(f"{i+1}. @{kol}\n")
                    f.write(f"   Channel ID: {perf.get('channel_id', 'Not found')}\n")
                    f.write(f"   Tokens mentioned: {perf['tokens_mentioned']}\n")
                    f.write(f"   Success rate (2x): {perf['success_rate_2x']}%\n")
                    f.write(f"   🎯 Avg Max Pullback: {perf.get('avg_max_pullback_percent', 0)}% (for SL setting)\n")
                    f.write(f"   ⏱️  Avg Time to 2x: {perf.get('avg_time_to_2x_formatted', 'N/A')} (for holding strategy)\n")
                    f.write(f"   📊 Enhanced data: {perf.get('detailed_analysis_count', 0)} tokens\n\n")
                
                f.write("=== 💡 TRADING STRATEGY NOTES ===\n")
                f.write("• avg_max_pullback_percent: Use this + 5-10% buffer for stop loss\n")
                f.write("• avg_time_to_2x_formatted: Minimum holding time before taking profits\n")
                f.write("• Higher pullback % = Higher risk/reward ratio\n")
                f.write("• Faster time-to-2x = Quick scalping opportunities\n")
                f.write("• More enhanced data = More reliable metrics\n")
            
            logger.info(f"✅ Exported enhanced summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting enhanced Spydefi analysis: {str(e)}")
    
    # Keep all existing methods for backward compatibility
    async def scan_spydefi_channel(self, hours_back: int = 24, get_channel_ids: bool = True) -> Dict[str, Any]:
        """Scan SpyDefi channel (backward compatibility - calls enhanced version)."""
        return await self.scan_spydefi_channel_enhanced(hours_back, get_channel_ids)
    
    async def scrape_spydefi(self, channel_id: str, days_back: int = 7, birdeye_api: Any = None) -> Dict[str, Any]:
        """Scrape the Spydefi channel (backward compatibility)."""
        hours_back = days_back * 24
        return await self.scan_spydefi_channel_enhanced(hours_back, get_channel_ids=True, birdeye_api=birdeye_api)
    
    async def export_spydefi_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Export Spydefi analysis (backward compatibility - calls enhanced version)."""
        await self.export_enhanced_spydefi_analysis(analysis, output_file)
    
    async def export_analysis_results(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Export analysis results (backward compatibility)."""
        await self.export_enhanced_spydefi_analysis(analysis, output_file)
    
    # All other existing methods remain unchanged for full backward compatibility...
    def extract_token_names(self, text: str) -> List[str]:
        """Extract token names from SpyDefi messages."""
        tokens = set()
        
        # Token name patterns from SpyDefi format
        patterns = [
            # "@username made a x2+ call on TOKEN_NAME"
            r'made\s+a\s+x\d+\+?\s+call\s+on\s+([A-Za-z0-9\s\.\-_]+?)(?:\s+on\s+\w+|\s*\.|$)',
            # "TOKEN_NAME first posted by @username"
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
    
    async def analyze_channel(self, channel_id: str, 
                            days_back: int = 7,
                            birdeye_api: Any = None) -> Dict[str, Any]:
        """
        Analyze a Telegram channel for token calls.
        
        Args:
            channel_id (str): Channel ID or username
            days_back (int): Number of days to analyze
            birdeye_api (BirdeyeAPI): API client for token data
            
        Returns:
            Dict[str, Any]: Channel analysis results
        """
        logger.info(f"Analyzing channel {channel_id} for the past {days_back} days")
        
        # Get messages
        messages = await self.get_channel_messages(channel_id, days_back)
        
        # Extract token calls
        token_calls = []
        for message in messages:
            if message["is_call"] and message["contract_addresses"]:
                for contract in message["contract_addresses"]:
                    token_call = {
                        "channel_id": channel_id,
                        "message_id": message["id"],
                        "date": message["date"],
                        "contract_address": contract,
                        "text": message["text"]
                    }
                    
                    # Add token data if Birdeye API is provided
                    if birdeye_api:
                        try:
                            # Get token info
                            token_info = birdeye_api.get_token_info(contract)
                            
                            if token_info.get("success"):
                                token_data = token_info.get("data", {})
                                token_call["token_name"] = token_data.get("name")
                                token_call["token_symbol"] = token_data.get("symbol")
                                
                                # Get market cap if available
                                if "marketCap" in token_data:
                                    token_call["market_cap_usd"] = token_data.get("marketCap")
                            
                            # Calculate performance since call
                            call_date = datetime.fromisoformat(message["date"])
                            performance = birdeye_api.calculate_token_performance(contract, call_date)
                            
                            if performance.get("success"):
                                token_call.update({
                                    "initial_price": performance.get("initial_price"),
                                    "current_price": performance.get("current_price"),
                                    "max_price": performance.get("max_price"),
                                    "roi_percent": performance.get("roi_percent"),
                                    "max_roi_percent": performance.get("max_roi_percent"),
                                    "max_drawdown_percent": performance.get("max_drawdown_percent")
                                })
                            
                            # Check which platform the token is on
                            platform = self.identify_platform(contract, birdeye_api)
                            if platform:
                                token_call["platform"] = platform
                            
                        except Exception as e:
                            logger.error(f"Error getting token data for {contract}: {str(e)}")
                    
                    token_calls.append(token_call)
        
        # Calculate channel metrics
        total_calls = len(token_calls)
        if total_calls == 0:
            success_rate = 0
        else:
            # Count calls with positive ROI
            successful_calls = sum(1 for call in token_calls if call.get("roi_percent", 0) > 0)
            success_rate = (successful_calls / total_calls) * 100
        
        # Calculate average ROI and max ROI
        roi_values = [call.get("roi_percent", 0) for call in token_calls if "roi_percent" in call]
        max_roi_values = [call.get("max_roi_percent", 0) for call in token_calls if "max_roi_percent" in call]
        
        avg_roi = sum(roi_values) / len(roi_values) if roi_values else 0
        avg_max_roi = sum(max_roi_values) / len(max_roi_values) if max_roi_values else 0
        
        # Generate channel analysis results
        analysis = {
            "channel_id": channel_id,
            "analysis_period_days": days_back,
            "total_calls": total_calls,
            "success_rate": success_rate,
            "avg_roi": avg_roi,
            "avg_max_roi": avg_max_roi,
            "token_calls": token_calls,
            "confidence_level": min(success_rate, 100)  # Use success rate as confidence level
        }
        
        # Generate strategy recommendations if confidence level is high enough
        if analysis["confidence_level"] >= 60:
            # Strategy for high confidence channels
            if avg_max_roi >= 500:  # 5x or more
                strategy = {
                    "recommendation": "HOLD_MOON",
                    "entry_type": "IMMEDIATE",
                    "entry": "IMMEDIATE",
                    "take_profit_1": 100,  # 100% ROI
                    "take_profit_2": 200,  # 200% ROI
                    "take_profit_3": 500,  # 500% ROI
                    "stop_loss": -30,  # 30% loss
                    "trailing_stop": {
                        "activation": 100,  # Activate at 100% profit
                        "trailing_percent": 25  # 25% trailing stop
                    },
                    "notes": "This channel finds potential moonshots. Take 30% at TP1, 20% at TP2, and hold 50% for major gains."
                }
            elif avg_max_roi >= 200:  # 2x or more
                strategy = {
                    "recommendation": "SCALP_AND_HOLD",
                    "entry_type": "IMMEDIATE",
                    "entry": "IMMEDIATE",
                    "take_profit_1": 50,  # 50% ROI
                    "take_profit_2": 100,  # 100% ROI
                    "take_profit_3": 200,  # 200% ROI
                    "stop_loss": -30,  # 30% loss
                    "trailing_stop": {
                        "activation": 50,  # Activate at 50% profit
                        "trailing_percent": 20  # 20% trailing stop
                    },
                    "notes": "Take 50% profit at TP1, 25% at TP2, and trail remaining with specified stop"
                }
            elif avg_max_roi >= 100:  # 1x or more
                strategy = {
                    "recommendation": "SCALP",
                    "entry_type": "IMMEDIATE",
                    "entry": "IMMEDIATE",
                    "take_profit_1": 30,  # 30% ROI
                    "take_profit_2": 50,  # 50% ROI
                    "take_profit_3": 100,  # 100% ROI
                    "stop_loss": -20,  # 20% loss
                    "trailing_stop": {
                        "activation": 30,  # Activate at 30% profit
                        "trailing_percent": 15  # 15% trailing stop
                    },
                    "notes": "Take 50% profit at TP1, 25% at TP2, and trail remaining with specified stop"
                }
            else:
                strategy = {
                    "recommendation": "CAUTIOUS",
                    "entry_type": "WAIT_FOR_CONFIRMATION",
                    "entry": "WAIT_FOR_CONFIRMATION",
                    "take_profit_1": 20,  # 20% ROI
                    "take_profit_2": 40,  # 40% ROI
                    "stop_loss": -15,  # 15% loss
                    "trailing_stop": {
                        "activation": 20,  # Activate at 20% profit
                        "trailing_percent": 10  # 10% trailing stop
                    },
                    "notes": "Wait for initial price movement confirmation before entering"
                }
            
            analysis["strategy"] = strategy
        else:
            # Default cautious strategy for low confidence
            strategy = {
                "recommendation": "CAUTIOUS",
                "entry_type": "WAIT_FOR_CONFIRMATION",
                "entry": "WAIT_FOR_CONFIRMATION",
                "take_profit_1": 20,  # 20% ROI
                "take_profit_2": 40,  # 40% ROI
                "stop_loss": -15,  # 15% loss
                "trailing_stop": {
                    "activation": 20,  # Activate at 20% profit
                    "trailing_percent": 10  # 10% trailing stop
                },
                "notes": "Low confidence signal. Wait for confirmation before entering."
            }
            analysis["strategy"] = strategy
        
        return analysis
    
    def identify_platform(self, contract_address: str, birdeye_api: Any = None) -> str:
        """
        Identify which platform a token contract is associated with.
        
        Args:
            contract_address (str): Token contract address
            birdeye_api (BirdeyeAPI): API client for token data
            
        Returns:
            str: Platform name or empty string if not identified
        """
        # Known platform contract prefixes or identifiers
        platforms = {
            "letsbonk": ["BONK", "BK"],
            "raydium": ["RAY"],
            "pumpfun": ["PUMP", "PF"],
            "pumpswap": ["PUMP", "PS"],
            "meteora": ["MTR"],
            "launchpad": ["LP", "LAUNCH"]
        }
        
        try:
            if birdeye_api:
                # Get token metadata from Birdeye
                token_info = birdeye_api.get_token_info(contract_address)
                if token_info.get("success"):
                    token_data = token_info.get("data", {})
                    
                    # Check token symbol against known platforms
                    symbol = token_data.get("symbol", "")
                    for platform, identifiers in platforms.items():
                        for identifier in identifiers:
                            if identifier in symbol:
                                return platform
                    
                    # Check token name
                    name = token_data.get("name", "")
                    for platform, identifiers in platforms.items():
                        if platform.lower() in name.lower():
                            return platform
                    
                    # Check token tags if available
                    tags = token_data.get("tags", [])
                    for tag in tags:
                        for platform in platforms.keys():
                            if platform.lower() in tag.lower():
                                return platform
                
                # Try to identify based on DEX trades
                dex_trades = birdeye_api.get_dex_trades(contract_address, limit=5)
                if dex_trades.get("success"):
                    for trade in dex_trades.get("data", []):
                        source = trade.get("source", "").lower()
                        for platform in platforms.keys():
                            if platform.lower() in source:
                                return platform
        
        except Exception as e:
            logger.error(f"Error identifying platform for {contract_address}: {str(e)}")
        
        return ""
    
    async def export_channel_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """
        Export channel analysis to CSV.
        
        Args:
            analysis (Dict[str, Any]): Channel analysis data
            output_file (str): Output file path
        """
        if not analysis or not analysis.get("token_calls"):
            logger.warning("No token calls to export")
            return
        
        try:
            # Ensure output directories exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            
            with open(output_file, 'w', newline='') as f:
                # Prepare CSV writer
                fieldnames = [
                    "channel_id", "message_id", "date", "contract_address", 
                    "token_name", "token_symbol", "initial_price", "current_price", 
                    "max_price", "roi_percent", "max_roi_percent", "max_drawdown_percent",
                    "market_cap_usd", "platform"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write token calls
                for call in analysis["token_calls"]:
                    row = {field: call.get(field, "") for field in fieldnames}
                    writer.writerow(row)
                
                logger.info(f"Exported {len(analysis['token_calls'])} token calls to {output_file}")
            
            # Export strategy to a separate file if available
            if "strategy" in analysis:
                strategy_file = output_file.replace(".csv", "_strategy.csv")
                with open(strategy_file, 'w', newline='') as f:
                    # Prepare CSV writer for strategy
                    strategy = analysis["strategy"]
                    fieldnames = ["parameter", "value"]
                    
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Write strategy parameters
                    for key, value in strategy.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                writer.writerow({
                                    "parameter": f"{key}_{sub_key}",
                                    "value": sub_value
                                })
                        else:
                            writer.writerow({
                                "parameter": key,
                                "value": value
                            })
                    
                    logger.info(f"Exported strategy to {strategy_file}")
        
        except Exception as e:
            logger.error(f"Error exporting channel analysis: {str(e)}")