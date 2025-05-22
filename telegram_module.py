"""
Enhanced Telegram Module - Phoenix Project (COMPLETE FIXED VERSION)

This module handles all Telegram-related functionality, including scraping and analysis
of KOL channels and token calls.

🎯 ENHANCED FEATURES:
- Max average pullback % calculation for stop loss setting
- Average time to reach 2x calculation for holding strategy
- Enhanced contract address detection from SpyDefi messages
- Detailed price analysis using Birdeye API
- Multiple resolution attempts for better data coverage
- Complete error handling and logging
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
    # Enhanced patterns for SpyDefi format
    r'(?i)mint[:\s]*([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)token[:\s]*([1-9A-HJ-NP-Za-km-z]{32,44})',
    # URL extraction patterns
    r'(?:dexscreener\.com|birdeye\.so|solscan\.io|explorer\.solana\.com)/[^"\s]*?([1-9A-HJ-NP-Za-km-z]{32,44})',
    # Standalone addresses (more aggressive)
    r'\b([1-9A-HJ-NP-Za-km-z]{43,44})\b(?!\w)',  # 43-44 char addresses
    r'\b([1-9A-HJ-NP-Za-km-z]{32,42})\b(?=\s|$|\n|\.)',  # 32-42 char addresses
]

# KOL detection patterns
KOL_USERNAME_PATTERN = r'@([A-Za-z0-9_]+)'
KOL_CALL_PATTERNS = [
    r'(?i)made a x(\d+)\+ call on',
    r'(?i)Achievement Unlocked: x(\d+)',
    r'(?i)(\d+)x\s+gem'
]

class TelegramScraper:
    """Enhanced class for scraping and analyzing Telegram channels with pullback analysis."""
    
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
        Enhanced extraction of potential contract addresses from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of extracted contract addresses
        """
        addresses = set()
        
        # Try specific contract patterns first
        for pattern in CONTRACT_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) > 0:
                    addr = match.group(1).strip()
                    if self._is_valid_solana_address(addr):
                        addresses.add(addr)
                        logger.debug(f"🎯 Found contract address with pattern: {addr}")
        
        # Look for URLs that might contain contract addresses
        url_pattern = r'https?://(?:www\.)?(?:dexscreener\.com|birdeye\.so|solscan\.io|explorer\.solana\.com)/[^"\s]+?([1-9A-HJ-NP-Za-km-z]{32,44})'
        url_matches = re.finditer(url_pattern, text)
        for match in url_matches:
            if len(match.groups()) > 0:
                addr = match.group(1).strip()
                if self._is_valid_solana_address(addr):
                    addresses.add(addr)
                    logger.debug(f"🔗 Found contract address from URL: {addr}")
        
        # Enhanced standalone address detection
        if not addresses:
            # Try different length ranges for Solana addresses
            standalone_patterns = [
                r'\b([1-9A-HJ-NP-Za-km-z]{43,44})\b',  # Most common length
                r'\b([1-9A-HJ-NP-Za-km-z]{38,42})\b',  # Alternative lengths
                r'\b([1-9A-HJ-NP-Za-km-z]{32,37})\b',  # Shorter variants
            ]
            
            for pattern in standalone_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    addr = match.group(1).strip()
                    if self._is_valid_solana_address(addr):
                        addresses.add(addr)
                        logger.debug(f"📍 Found standalone contract address: {addr}")
                        break  # Stop after finding addresses to avoid duplicates
                
                if addresses:  # If we found addresses, stop trying other patterns
                    break
        
        result = list(addresses)
        if result:
            logger.info(f"✅ Extracted {len(result)} contract addresses from message")
        else:
            logger.debug("❌ No contract addresses found in message")
        
        return result
    
    def _is_valid_solana_address(self, address: str) -> bool:
        """
        Validate if an address looks like a valid Solana address.
        
        Args:
            address (str): Address to validate
            
        Returns:
            bool: True if valid-looking, False otherwise
        """
        if not address or len(address) < 32 or len(address) > 44:
            return False
        
        # Check for valid base58 characters
        valid_chars = set('123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz')
        if not all(c in valid_chars for c in address):
            return False
        
        # Exclude common false positives
        exclude_patterns = [
            r'^[0-9]+$',  # All numbers
            r'^[A-Z]+$',  # All uppercase letters
            r'^[a-z]+$',  # All lowercase letters
            r'test|example|dummy|fake',  # Test strings
        ]
        
        for pattern in exclude_patterns:
            if re.search(pattern, address, re.IGNORECASE):
                return False
        
        return True
    
    def extract_kol_usernames(self, text: str) -> List[str]:
        """
        Extract KOL usernames from text with improved patterns.
        
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
        """
        Determine if a message is likely a token call.
        
        Args:
            text (str): Message text
            
        Returns:
            bool: True if likely a token call, False otherwise
        """
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
    
    async def get_channel_messages(self, channel_id: str, 
                                days_back: int = 7) -> List[Dict[str, Any]]:
        """
        Get messages from a Telegram channel.
        
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
    
    async def perform_detailed_token_analysis(self, contract_address: str, call_timestamp: int, 
                                            birdeye_api: Any) -> Dict[str, Any]:
        """
        Perform detailed token analysis including pullback % and time to 2x.
        
        Args:
            contract_address (str): Token contract address
            call_timestamp (int): Unix timestamp when token was called
            birdeye_api: Birdeye API instance
            
        Returns:
            Dict[str, Any]: Detailed analysis results
        """
        logger.debug(f"🎯 Running detailed analysis for {contract_address}")
        
        try:
            call_date = datetime.fromtimestamp(call_timestamp)
            
            # Try multiple resolutions for better data coverage
            resolutions = ["5m", "15m", "1h"]
            price_history = None
            resolution_used = None
            
            for resolution in resolutions:
                logger.debug(f"📊 Trying {resolution} resolution for {contract_address}")
                
                # Get price history from call time to now (max 30 days)
                end_time = min(int(datetime.now().timestamp()), call_timestamp + (30 * 24 * 3600))
                
                price_result = birdeye_api.get_token_price_history(
                    contract_address,
                    call_timestamp,
                    end_time,
                    resolution
                )
                
                if price_result.get("success") and price_result.get("data", {}).get("items"):
                    price_history = price_result["data"]["items"]
                    resolution_used = resolution
                    logger.debug(f"✅ Got price data with {resolution} resolution")
                    break
                else:
                    logger.debug(f"❌ No price data with {resolution} resolution")
            
            if not price_history:
                logger.warning(f"⚠️ No price history available for {contract_address}")
                return {
                    "success": False,
                    "error": "No price history available",
                    "has_detailed_analysis": False
                }
            
            # Sort price data by timestamp
            price_history.sort(key=lambda x: x.get("unixTime", 0))
            
            if len(price_history) < 2:
                logger.warning(f"⚠️ Insufficient price data for {contract_address}")
                return {
                    "success": False,
                    "error": "Insufficient price data",
                    "has_detailed_analysis": False
                }
            
            # Extract price information
            initial_price = price_history[0].get("value", 0)
            if initial_price <= 0:
                logger.warning(f"⚠️ Invalid initial price for {contract_address}")
                return {
                    "success": False,
                    "error": "Invalid initial price",
                    "has_detailed_analysis": False
                }
            
            # Calculate metrics
            current_price = price_history[-1].get("value", initial_price)
            max_price = max(p.get("value", 0) for p in price_history)
            min_price = min(p.get("value", 0) for p in price_history if p.get("value", 0) > 0)
            
            # Calculate ROI
            current_roi = ((current_price / initial_price) - 1) * 100 if initial_price > 0 else 0
            max_roi = ((max_price / initial_price) - 1) * 100 if initial_price > 0 else 0
            
            # Calculate maximum pullback %
            max_pullback = 0
            running_peak = initial_price
            
            for price_point in price_history:
                price = price_point.get("value", 0)
                if price > running_peak:
                    running_peak = price
                elif running_peak > 0:
                    pullback = ((running_peak - price) / running_peak) * 100
                    max_pullback = max(max_pullback, pullback)
            
            # Calculate time to reach 2x (100% ROI)
            time_to_2x_seconds = None
            time_to_2x_formatted = "N/A"
            target_price = initial_price * 2
            
            for price_point in price_history:
                if price_point.get("value", 0) >= target_price:
                    time_to_2x_seconds = price_point.get("unixTime", 0) - call_timestamp
                    if time_to_2x_seconds > 0:
                        time_to_2x_formatted = self._format_time_duration(time_to_2x_seconds)
                    break
            
            analysis_result = {
                "success": True,
                "has_detailed_analysis": True,
                "initial_price": initial_price,
                "current_price": current_price,
                "max_price": max_price,
                "min_price": min_price,
                "current_roi_percent": round(current_roi, 2),
                "max_roi_percent": round(max_roi, 2),
                "max_pullback_percent": round(max_pullback, 2),
                "time_to_2x_seconds": time_to_2x_seconds,
                "time_to_2x_formatted": time_to_2x_formatted,
                "resolution_used": resolution_used,
                "data_points": len(price_history)
            }
            
            logger.info(f"✅ Analysis complete - ROI: {current_roi:.1f}%, Max ROI: {max_roi:.1f}%, Pullback: {max_pullback:.1f}%")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Error in detailed analysis for {contract_address}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "has_detailed_analysis": False
            }
    
    async def lookup_contract_from_token_name(self, token_name: str) -> Optional[str]:
        """
        Lookup contract address from token name using various methods.
        
        Args:
            token_name (str): Token name to lookup
            
        Returns:
            Optional[str]: Contract address if found, None otherwise
        """
        if not token_name or len(token_name) < 2:
            return None
        
        try:
            # Method 1: Try Birdeye search API (if available)
            if hasattr(self, '_birdeye_api') and self._birdeye_api:
                try:
                    # Try to search for token by name
                    # Note: This would require a search endpoint in Birdeye API
                    # For now, we'll use the most common tokens mapping
                    pass
                except Exception as e:
                    logger.debug(f"Birdeye search failed for {token_name}: {str(e)}")
            
            # Method 2: Common token mappings (you can expand this)
            common_tokens = {
                # Popular tokens that might appear in SpyDefi
                "solana": "So11111111111111111111111111111111111111112",
                "wrapped sol": "So11111111111111111111111111111111111111112",
                "wsol": "So11111111111111111111111111111111111111112",
                "usdc": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "usdt": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                "bonk": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
                "wif": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
                "pepe": "BzLXKqRFqqakaUZWyZWPNr2HMXRu7TKb7WmMqM2PLj3v",
                # Add more popular tokens here
            }
            
            # Try exact match first
            token_lower = token_name.lower().strip()
            if token_lower in common_tokens:
                logger.info(f"  🎯 Token name lookup: {token_name} -> {common_tokens[token_lower]}")
                return common_tokens[token_lower]
            
            # Try partial matches
            for known_token, contract in common_tokens.items():
                if known_token in token_lower or token_lower in known_token:
                    logger.info(f"  🔍 Partial token name match: {token_name} ~= {known_token} -> {contract}")
                    return contract
            
            # Method 3: Try to extract contract from any URLs in the original message
            # This would be implemented if we had access to the original message
            
            logger.debug(f"  ❌ No contract found for token name: {token_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error in token name lookup for {token_name}: {str(e)}")
            return None
    
    def _format_time_duration(self, seconds: int) -> str:
        """
        Format time duration in human-readable format.
        
        Args:
            seconds (int): Duration in seconds
            
        Returns:
            str: Formatted duration (e.g., "2h 15m 30s")
        """
        if seconds <= 0:
            return "N/A"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:  # Always show seconds if no other parts
            parts.append(f"{secs}s")
        
        return " ".join(parts)
    
    async def scan_spydefi_channel(self, hours_back: int = 24, get_channel_ids: bool = True) -> Dict[str, Any]:
        """Enhanced SpyDefi scanning with pullback and time-to-2x analysis."""
        logger.info(f"🚀 Starting enhanced SpyDefi analysis (past {hours_back}h)...")
        
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
                    'total_calls': 0,
                    'successful_2x': 0,
                    'successful_5x': 0,
                    'success_rate_2x': 0,
                    'success_rate_5x': 0
                }
            
            logger.info(f"Entity found: {entity.title if hasattr(entity, 'title') else 'Unknown'}")
            
            # Calculate time limit for proper filtering
            time_limit = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            # Collect all token calls
            token_calls = []
            kol_mentions = defaultdict(list)
            message_count = 0
            max_messages = 500
            
            logger.info("Starting message iteration...")
            
            # SpyDefi message processing loop with proper time filtering
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
                
                # Extract data with enhanced detection
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
                            'has_contract': bool(token in contracts)
                        }
                        token_calls.append(call_entry)
                        
                        # Track KOL mentions
                        for kol in kols:
                            kol_mentions[kol].append(call_entry)
            
            logger.info(f"Processed {message_count} total messages")
            logger.info(f"Found {len(token_calls)} token calls from {len(kol_mentions)} KOLs")
            
            # Enhanced analysis with Birdeye API
            analyzed_calls = []
            successful_2x = 0
            successful_5x = 0
            enhanced_analysis_count = 0
            
            logger.info("🎯 Starting enhanced analysis with pullback & time-to-2x calculations...")
            
            for i, call in enumerate(token_calls):
                token_identifier = call.get('contract') or call.get('token_name', 'Unknown')
                logger.info(f"Analyzing call {i+1}/{len(token_calls)}: {token_identifier}")
                
                # Initialize analysis data
                analyzed_call = call.copy()
                analyzed_call.update({
                    'has_detailed_analysis': False,
                    'call_price': 0,
                    'ath_price': 0,
                    'current_price': 0,
                    'ath_roi_percent': 0,
                    'current_roi_percent': 0,
                    'max_drawdown_percent': 0,
                    'max_pullback_percent': 0,
                    'time_to_2x_seconds': None,
                    'time_to_2x_formatted': 'N/A',
                    'is_2x_plus': False,
                    'is_5x_plus': False
                })
                
                # Try enhanced analysis - first with contract address, then try token name lookup
                contract_to_analyze = None
                
                if call.get('has_contract') and call.get('contract'):
                    contract_to_analyze = call['contract']
                    logger.info(f"  🎯 Using direct contract address: {contract_to_analyze}")
                elif call.get('token_name'):
                    # Try to find contract address using token name lookup
                    contract_to_analyze = await self.lookup_contract_from_token_name(call['token_name'])
                    if contract_to_analyze:
                        logger.info(f"  🔍 Found contract via token name lookup: {call['token_name']} -> {contract_to_analyze}")
                    else:
                        logger.info(f"  ❌ No contract found for token name: {call['token_name']}")
                
                if contract_to_analyze:
                    try:
                        detailed_analysis = await self.perform_detailed_token_analysis(
                            contract_to_analyze, 
                            call['unix_time'], 
                            getattr(self, '_birdeye_api', None)
                        )
                        
                        if detailed_analysis.get('success'):
                            enhanced_analysis_count += 1
                            analyzed_call.update(detailed_analysis)
                            analyzed_call['call_price'] = detailed_analysis['initial_price']
                            analyzed_call['ath_price'] = detailed_analysis['max_price']
                            analyzed_call['current_price'] = detailed_analysis['current_price']
                            analyzed_call['ath_roi_percent'] = detailed_analysis['max_roi_percent']
                            analyzed_call['current_roi_percent'] = detailed_analysis['current_roi_percent']
                            analyzed_call['is_2x_plus'] = detailed_analysis['max_roi_percent'] >= 100
                            analyzed_call['is_5x_plus'] = detailed_analysis['max_roi_percent'] >= 400
                            
                            logger.info(f"  ✅ Enhanced analysis - Pullback: {detailed_analysis['max_pullback_percent']:.1f}%, Time to 2x: {detailed_analysis['time_to_2x_formatted']}")
                        else:
                            logger.info(f"  ⚠️ Enhanced analysis failed: {detailed_analysis.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        logger.warning(f"  ❌ Enhanced analysis error: {str(e)}")
                else:
                    logger.info(f"  ⚠️ No contract address available for enhanced analysis")
                
                # Fallback to SpyDefi message parsing if enhanced analysis failed
                if not analyzed_call.get('has_detailed_analysis'):
                    # Extract ROI from the message text (existing logic)
                    text = call['text']
                    roi_patterns = [
                        r'\$(\d+(?:\.\d+)?)([KM]?)\s*->\s*\$(\d+(?:\.\d+)?)([KM]?)',
                        r'\$(\d+(?:\.\d+)?)([KM]?)\s*->\s*\$(\d+(?:\.\d+)?)([KM]?)\.{0,3}',
                        r'\$(\d+(?:\.\d+)?)([KM]?)\s*→\s*\$(\d+(?:\.\d+)?)([KM]?)',
                    ]
                    
                    roi_match = None
                    for pattern in roi_patterns:
                        roi_match = re.search(pattern, text)
                        if roi_match:
                            break
                    
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
                            
                            analyzed_call.update({
                                'call_price': start_val,
                                'ath_price': end_val,
                                'current_price': end_val,
                                'ath_roi_percent': round(roi_percent, 2),
                                'current_roi_percent': round(roi_percent, 2),
                                'is_2x_plus': roi_percent >= 100,
                                'is_5x_plus': roi_percent >= 400
                            })
                            
                        except (ValueError, ZeroDivisionError) as e:
                            logger.warning(f"Error calculating ROI for call {i+1}: {str(e)}")
                
                # Count successful calls
                if analyzed_call.get('is_2x_plus'):
                    successful_2x += 1
                if analyzed_call.get('is_5x_plus'):
                    successful_5x += 1
                
                analyzed_calls.append(analyzed_call)
                
                # Rate limiting for API calls
                if analyzed_call.get('has_detailed_analysis'):
                    await asyncio.sleep(0.1)  # Small delay between API calls
            
            logger.info(f"🎯 Enhanced analysis coverage: {enhanced_analysis_count}/{len(token_calls)} tokens")
            
            # Calculate overall performance
            total_calls = len(analyzed_calls)
            success_rate_2x = (successful_2x / total_calls * 100) if total_calls > 0 else 0
            success_rate_5x = (successful_5x / total_calls * 100) if total_calls > 0 else 0
            
            # Generate enhanced KOL performance breakdown
            kol_performance = {}
            for kol, kol_calls in kol_mentions.items():
                kol_analyzed = [c for c in analyzed_calls if kol in c.get('kols', [])]
                if kol_analyzed:
                    kol_2x = sum(1 for c in kol_analyzed if c.get('is_2x_plus', False))
                    kol_5x = sum(1 for c in kol_analyzed if c.get('is_5x_plus', False))
                    kol_total = len(kol_analyzed)
                    
                    # Calculate enhanced metrics
                    detailed_analyses = [c for c in kol_analyzed if c.get('has_detailed_analysis')]
                    
                    avg_pullback = 0
                    avg_time_to_2x_seconds = None
                    avg_time_to_2x_formatted = "N/A"
                    
                    if detailed_analyses:
                        pullbacks = [c.get('max_pullback_percent', 0) for c in detailed_analyses]
                        avg_pullback = sum(pullbacks) / len(pullbacks) if pullbacks else 0
                        
                        times_to_2x = [c.get('time_to_2x_seconds') for c in detailed_analyses if c.get('time_to_2x_seconds') is not None]
                        if times_to_2x:
                            avg_time_to_2x_seconds = sum(times_to_2x) / len(times_to_2x)
                            avg_time_to_2x_formatted = self._format_time_duration(int(avg_time_to_2x_seconds))
                    
                    kol_performance[kol] = {
                        'channel_id': '',  # Will be populated later
                        'tokens_mentioned': kol_total,
                        'tokens_2x_plus': kol_2x,
                        'tokens_5x_plus': kol_5x,
                        'success_rate_2x': round((kol_2x / kol_total * 100) if kol_total > 0 else 0, 2),
                        'success_rate_5x': round((kol_5x / kol_total * 100) if kol_total > 0 else 0, 2),
                        'avg_ath_roi': round(sum(c.get('ath_roi_percent', 0) for c in kol_analyzed) / kol_total if kol_total > 0 else 0, 2),
                        'composite_score': self.calculate_kol_score(kol_total, kol_2x, kol_5x, kol_analyzed),
                        # Enhanced metrics
                        'avg_max_pullback_percent': round(avg_pullback, 2),
                        'avg_time_to_2x_seconds': avg_time_to_2x_seconds,
                        'avg_time_to_2x_formatted': avg_time_to_2x_formatted,
                        'detailed_analysis_count': len(detailed_analyses),
                        'pullback_data_available': len(detailed_analyses) > 0,
                        'time_to_2x_data_available': bool(times_to_2x) if 'times_to_2x' in locals() else False
                    }
            
            # Get channel IDs for top KOLs only if requested (rate-limited approach)
            if get_channel_ids:
                logger.info(f"Getting channel IDs for top KOLs (this may take a while due to rate limits)...")
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
                        
                        # Longer delay to avoid rate limiting
                        if i < len(top_kols) - 1:  # Don't sleep after the last one
                            await asyncio.sleep(2.0)  # 2 second delay between requests
                            
                    except Exception as e:
                        logger.warning(f"Error getting channel ID for @{kol}: {str(e)}")
                        # Continue with next KOL even if this one fails
                        continue
            else:
                logger.info("Skipping channel ID lookup to avoid rate limits")
            
            # Sort KOLs by composite score (preserve this order in final results)
            sorted_kols = sorted(kol_performance.items(), 
                               key=lambda x: x[1]['composite_score'], 
                               reverse=True)
            
            logger.info("🎉 Enhanced Analysis Complete!")
            logger.info(f"Tokens mentioned: {total_calls}")
            logger.info(f"Enhanced analysis coverage: {enhanced_analysis_count}/{total_calls}")
            logger.info(f"Tokens that made x2: {successful_2x}")
            logger.info(f"Tokens that made x5: {successful_5x}")
            logger.info(f"Success rate (2x): {success_rate_2x:.2f}%")
            logger.info(f"Success rate (5x): {success_rate_5x:.2f}%")
            
            return {
                'scan_period_hours': hours_back,
                'total_calls': total_calls,
                'successful_2x': successful_2x,
                'successful_5x': successful_5x,
                'success_rate_2x': round(success_rate_2x, 2),
                'success_rate_5x': round(success_rate_5x, 2),
                'analyzed_calls': analyzed_calls,
                'kol_performance': dict(sorted_kols),  # Keep sorted order
                'ranked_kols': dict(sorted_kols),      # Keep sorted order
                'enhanced_analysis_count': enhanced_analysis_count,
                'enhanced_analysis_coverage_percent': round((enhanced_analysis_count / total_calls * 100) if total_calls > 0 else 0, 2),
                'summary': {
                    'tokens_mentioned': total_calls,
                    'tokens_that_made_x2': successful_2x,
                    'tokens_that_made_x5': successful_5x,
                    'success_rate_2x_percent': f"{success_rate_2x:.2f}%",
                    'success_rate_5x_percent': f"{success_rate_5x:.2f}%",
                    'enhanced_analysis_count': enhanced_analysis_count,
                    'enhanced_coverage_percent': f"{(enhanced_analysis_count / total_calls * 100) if total_calls > 0 else 0:.1f}%"
                }
            }
            
        except Exception as e:
            logger.error(f"Error scanning SpyDefi: {str(e)}")
            return {
                'error': str(e),
                'total_calls': 0,
                'successful_2x': 0,
                'successful_5x': 0,
                'success_rate_2x': 0,
                'success_rate_5x': 0
            }
    
    async def scrape_spydefi(self, channel_id: str, 
                          days_back: int = 7,
                          birdeye_api: Any = None) -> Dict[str, Any]:
        """
        Enhanced SpyDefi scraping with pullback and time-to-2x analysis.
        
        Args:
            channel_id (str): Spydefi channel ID
            days_back (int): Number of days to analyze
            birdeye_api (BirdeyeAPI): API client for token data
            
        Returns:
            Dict[str, Any]: Enhanced Spydefi analysis results with KOL channels
        """
        logger.info(f"🚀 Enhanced SpyDefi analysis for KOL mentions with pullback & time-to-2x")
        
        # Store birdeye_api for use in detailed analysis
        self._birdeye_api = birdeye_api
        
        # Convert days to hours for the new method
        hours_back = days_back * 24
        
        # Use the enhanced scan method with limited channel ID lookup to avoid rate limits
        return await self.scan_spydefi_channel(hours_back, get_channel_ids=True)
    
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
        avg_roi = sum(c.get('ath_roi_percent', 0) for c in analyzed_calls) / total_calls if analyzed_calls else 0
        roi_bonus = min(avg_roi / 10, 50)  # Max 50 points for ROI
        
        # Sample size multiplier (more calls = more reliable)
        reliability_multiplier = 1.0
        if total_calls >= 10:
            reliability_multiplier = 1.5
        elif total_calls >= 5:
            reliability_multiplier = 1.2
        elif total_calls >= 3:
            reliability_multiplier = 1.1
        
        final_score = (sample_score + success_bonus + roi_bonus) * reliability_multiplier
        
        return round(final_score, 2)
    
    async def export_enhanced_spydefi_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """
        Export enhanced Spydefi analysis to CSV with pullback and time-to-2x data.
        
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
            
            # Export detailed calls with enhanced columns
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'contract', 'token_name', 'date', 'kols', 'call_price', 
                    'ath_price', 'current_price', 'ath_roi_percent', 'current_roi_percent',
                    'is_2x_plus', 'is_5x_plus', 'has_detailed_analysis',
                    'max_pullback_percent', 'time_to_2x_seconds', 'time_to_2x_formatted',
                    'min_price', 'initial_price', 'text'
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
                        'current_price': call.get('current_price', ''),
                        'ath_roi_percent': call.get('ath_roi_percent', ''),
                        'current_roi_percent': call.get('current_roi_percent', ''),
                        'is_2x_plus': call.get('is_2x_plus', False),
                        'is_5x_plus': call.get('is_5x_plus', False),
                        'has_detailed_analysis': call.get('has_detailed_analysis', False),
                        'max_pullback_percent': call.get('max_pullback_percent', ''),
                        'time_to_2x_seconds': call.get('time_to_2x_seconds', ''),
                        'time_to_2x_formatted': call.get('time_to_2x_formatted', 'N/A'),
                        'min_price': call.get('min_price', ''),
                        'initial_price': call.get('initial_price', ''),
                        'text': call.get('text', '')
                    }
                    writer.writerow(row)
            
            logger.info(f"Exported {len(analysis['analyzed_calls'])} calls to {output_file}")
            
            # Export enhanced KOL performance summary with pullback and time-to-2x data
            kol_file = output_file.replace('.csv', '_kol_performance_enhanced.csv')
            with open(kol_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'kol', 'channel_id', 'tokens_mentioned', 'tokens_2x_plus', 'tokens_5x_plus',
                    'success_rate_2x', 'success_rate_5x', 'avg_ath_roi', 'composite_score',
                    # Enhanced columns
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
                        # Enhanced columns
                        'avg_max_pullback_percent': performance.get('avg_max_pullback_percent', 0),
                        'avg_time_to_2x_seconds': performance.get('avg_time_to_2x_seconds', ''),
                        'avg_time_to_2x_formatted': performance.get('avg_time_to_2x_formatted', 'N/A'),
                        'detailed_analysis_count': performance.get('detailed_analysis_count', 0),
                        'pullback_data_available': performance.get('pullback_data_available', False),
                        'time_to_2x_data_available': performance.get('time_to_2x_data_available', False)
                    }
                    writer.writerow(row)
            
            logger.info(f"Exported enhanced KOL performance to {kol_file}")
            
            # Export enhanced summary
            summary_file = output_file.replace('.csv', '_enhanced_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                summary = analysis.get('summary', {})
                f.write("=== ENHANCED SPYDEFI PERFORMANCE ANALYSIS ===\n\n")
                f.write(f"Analysis period: {analysis.get('scan_period_hours', 24)} hours\n")
                f.write(f"Tokens mentioned: {summary.get('tokens_mentioned', 0)}\n")
                f.write(f"Enhanced analysis coverage: {summary.get('enhanced_analysis_count', 0)}/{summary.get('tokens_mentioned', 0)} ({summary.get('enhanced_coverage_percent', '0%')})\n")
                f.write(f"Tokens that made x2: {summary.get('tokens_that_made_x2', 0)}\n")
                f.write(f"Tokens that made x5: {summary.get('tokens_that_made_x5', 0)}\n")
                f.write(f"Success rate (2x): {summary.get('success_rate_2x_percent', '0.00%')}\n")
                f.write(f"Success rate (5x): {summary.get('success_rate_5x_percent', '0.00%')}\n\n")
                
                f.write("=== TOP KOLs WITH ENHANCED METRICS ===\n")
                for i, (kol, perf) in enumerate(list(analysis.get('kol_performance', {}).items())[:10]):
                    f.write(f"{i+1}. @{kol}\n")
                    f.write(f"   Channel ID: {perf.get('channel_id', 'Not found')}\n")
                    f.write(f"   Tokens mentioned: {perf['tokens_mentioned']}\n")
                    f.write(f"   Success rate (2x): {perf['success_rate_2x']}%\n")
                    f.write(f"   Avg Pullback: {perf.get('avg_max_pullback_percent', 0)}% (use for SL)\n")
                    f.write(f"   Avg Time to 2x: {perf.get('avg_time_to_2x_formatted', 'N/A')}\n")
                    f.write(f"   Enhanced analysis: {perf.get('detailed_analysis_count', 0)} tokens\n\n")
            
            logger.info(f"Exported enhanced summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting enhanced Spydefi analysis: {str(e)}")
    
    # Alias methods for compatibility
    async def export_analysis_results(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Export analysis results to CSV files (enhanced version)."""
        await self.export_enhanced_spydefi_analysis(analysis, output_file)

    async def export_spydefi_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Export Spydefi analysis to CSV (enhanced version)."""
        await self.export_enhanced_spydefi_analysis(analysis, output_file)