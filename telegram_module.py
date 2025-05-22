"""
Telegram Module - Phoenix Project

This module handles all Telegram-related functionality, including scraping and analysis
of KOL channels and token calls.
"""

import re
import csv
import os
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import PeerChannel, User, Channel

logger = logging.getLogger("phoenix.telegram")

# Regex patterns for detecting Solana addresses and contract mentions
SOLANA_ADDRESS_PATTERN = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
CONTRACT_PATTERNS = [
    r'(?i)contract(?:\s*address)?[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)token(?:\s*address)?[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)CA\s*(?:is|:)?\s*([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)address[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
    r'(?i)Ca:?\s*([1-9A-HJ-NP-Za-km-z]{32,44})'  # Format seen in the images
]

# KOL detection patterns - Updated to match real SpyDefi format
KOL_USERNAME_PATTERN = r'@([A-Za-z0-9_]+)'
KOL_CALL_PATTERNS = [
    r'(?i)made a x(\d+)\+ call on',  # "@cortezgems made a x2+ call on"
    r'(?i)Achievement Unlocked: x(\d+)!',  # "Achievement Unlocked: x2! 🔥"
    r'(?i)(\d+)x\s+gem',
    r'(?i)x(\d+)!',  # "x2! 🔥"
    r'(?i)x(\d+)\+',  # "x2+"
    r'(?i)(\d+)x\s+call',
    r'(?i)(\d+)x\s+gain',
    r'(?i)(\d+)x\s+win',
]

class TelegramScraper:
    """Class for scraping and analyzing Telegram channels."""
    
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
        self.message_limit = 2000  # Maximum number of messages to retrieve per channel
        self.spydefi_message_limit = 1000  # Lower limit for SpyDefi to avoid resource issues
    
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
        Extract potential contract addresses from text - Made much smarter.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of extracted contract addresses
        """
        addresses = set()
        
        # First try specific contract patterns (original approach)
        for pattern in CONTRACT_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) > 0:
                    addresses.add(match.group(1))
        
        # Look for URLs that might contain contract addresses
        url_pattern = r'https?://(?:www\.)?(?:dexscreener\.com|birdeye\.so|solscan\.io|explorer\.solana\.com)/[^"\s]*?([1-9A-HJ-NP-Za-km-z]{32,44})'
        url_matches = re.finditer(url_pattern, text)
        for match in url_matches:
            if len(match.groups()) > 0:
                addresses.add(match.group(1))
        
        # NEW: Smart detection - Find any Solana address in the text
        # Look for potential Solana addresses (base58, 32-44 chars)
        potential_addresses = re.finditer(SOLANA_ADDRESS_PATTERN, text)
        for match in potential_addresses:
            address = match.group(0)
            
            # Filter out false positives
            if self._is_likely_contract_address(address, text):
                addresses.add(address)
        
        return list(addresses)
    
    def _is_likely_contract_address(self, address: str, context: str) -> bool:
        """
        Determine if a string is likely a contract address based on context.
        
        Args:
            address (str): Potential contract address
            context (str): Full message text for context
            
        Returns:
            bool: True if likely a contract address
        """
        # Length check - Solana addresses are typically 32-44 characters
        if len(address) < 32 or len(address) > 44:
            return False
        
        # Must start with valid base58 characters (not 0, O, I, l)
        if address[0] in '0OIl':
            return False
        
        # Context clues that suggest it's a contract address
        context_lower = context.lower()
        positive_indicators = [
            'token', 'contract', 'address', 'ca', 'mint',
            'pump', 'moon', 'gem', 'call', 'buy', 'sell',
            'mc', 'market cap', 'marketcap', '
    
    def extract_kol_usernames(self, text: str) -> List[str]:
        """
        Extract KOL usernames from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of extracted KOL usernames
        """
        usernames = set()
        
        # Find @username mentions
        matches = re.finditer(KOL_USERNAME_PATTERN, text)
        for match in matches:
            if len(match.groups()) > 0:
                usernames.add(match.group(1))
        
        return list(usernames)
    
    def is_likely_token_call(self, text: str) -> bool:
        """
        Determine if a message is likely a token call - Made much smarter.
        
        Args:
            text (str): Message text
            
        Returns:
            bool: True if likely a token call, False otherwise
        """
        text_lower = text.lower()
        
        # Common phrases used in token calls
        call_indicators = [
            r'(?i)(\bnew\s+call\b|\btoken\s+call\b)',
            r'(?i)(\bbuy\s+now\b|\bentry\s+now\b)',
            r'(?i)(\bcontract\s+address\b|\btoken\s+address\b|\bCA\b|\bCa\b)',
            r'(?i)(\btarget\s+\d+x\b|\bpotential\s+\d+x\b)',
            r'(?i)(\bmoon\s+shot\b|\bpump\b|\bgem\b|\bearly\b)',
            r'(?i)(buy\s*&?\s*sell)',
            r'(?i)(dexscreener\.com|birdeye\.so)',
            # NEW: More flexible indicators
            r'(?i)(\d+k\s+mc|\d+m\s+mc|market\s+cap)',  # Market cap mentions
            r'(?i)(going\s+to\s+\d+|\bgonna\s+\d+)',  # Price predictions
            r'(?i)(x\d+\s+potential|potential\s+x\d+)',  # Potential gains
            r'(?i)(looks?\s+good|seems?\s+good)',  # Positive sentiment
            r'(?i)(def\s+gonna|definitely\s+gonna)',  # Strong predictions
            r'(?i)(\d+\%\s+up|\d+\%\s+gain)',  # Percentage gains
            r'(?i)(keep\s+an?\s+eye|watch\s+this)',  # Attention calls
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
        
        # NEW: Additional context-based scoring
        context_score = 0
        
        # Price/value mentions
        if any(word in text_lower for word in ['
    
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
    
    async def scrape_spydefi(self, channel_id: str, 
                          days_back: int = 7,
                          birdeye_api: Any = None) -> Dict[str, Any]:
        """
        Scrape the Spydefi channel for KOL mentions.
        
        Args:
            channel_id (str): Spydefi channel ID
            days_back (int): Number of days to analyze
            birdeye_api (BirdeyeAPI): API client for token data
            
        Returns:
            Dict[str, Any]: Spydefi analysis results with KOL channels
        """
        logger.info(f"Scraping Spydefi channel {channel_id} for KOL mentions")
        
        # Get messages from Spydefi
        messages = await self.get_channel_messages(channel_id, days_back)
        
        # Extract KOL mentions
        kol_data = {}
        
        # Find KOLs in the messages
        for message in messages:
            # Look for KOL mentions in the message text
            for username in message.get("mentioned_usernames", []):
                if username not in kol_data:
                    kol_data[username] = {
                        "username": username,
                        "mentions": 0,
                        "calls": 0,
                        "achievements": []
                    }
                
                kol_data[username]["mentions"] += 1
                
                # Check if it's an achievement or call mention
                for pattern in KOL_CALL_PATTERNS:
                    match = re.search(pattern, message["text"])
                    if match:
                        if len(match.groups()) > 0 and match.group(1).isdigit():
                            multiplier = int(match.group(1))
                            kol_data[username]["achievements"].append({
                                "multiplier": multiplier,
                                "message_id": message["id"],
                                "date": message["date"],
                                "text": message["text"]
                            })
                        kol_data[username]["calls"] += 1
                
                # Check for contract addresses in the message
                if message["contract_addresses"]:
                    for address in message["contract_addresses"]:
                        if "contracts" not in kol_data[username]:
                            kol_data[username]["contracts"] = []
                        
                        kol_data[username]["contracts"].append({
                            "address": address,
                            "message_id": message["id"],
                            "date": message["date"]
                        })
        
        # Also check for direct sender usernames
        for message in messages:
            sender = message.get("sender_username")
            if sender and sender not in kol_data and sender.lower() != "spydefi":
                kol_data[sender] = {
                    "username": sender,
                    "mentions": 1,
                    "calls": 0,
                    "achievements": []
                }
        
        # Extract channel links too
        channel_pattern = r'(?:https?://)?(?:t|telegram)\.(?:me|dog)/(?:joinchat/)?([a-zA-Z0-9_-]+)'
        for message in messages:
            matches = re.finditer(channel_pattern, message["text"])
            for match in matches:
                if len(match.groups()) > 0:
                    channel = match.group(1)
                    if channel not in kol_data and channel.lower() != "spydefi":
                        kol_data[channel] = {
                            "username": channel,
                            "mentions": 1,
                            "calls": 0,
                            "achievements": []
                        }
        
        logger.info(f"Found {len(kol_data)} potential KOL channels/usernames")
        
        # Analyze each KOL channel
        kol_analyses = []
        max_kols_to_analyze = 20  # Limit to top KOLs to avoid excessive API calls
        
        # Sort KOLs by mentions and achievements
        sorted_kols = sorted(
            kol_data.values(),
            key=lambda x: (len(x.get("achievements", [])), x.get("mentions", 0), x.get("calls", 0)),
            reverse=True
        )
        
        # Take top KOLs for analysis
        top_kols = sorted_kols[:max_kols_to_analyze]
        
        for kol in top_kols:
            try:
                logger.info(f"Analyzing KOL: {kol['username']}")
                analysis = await self.analyze_channel(kol['username'], days_back, birdeye_api)
                kol_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing KOL channel {kol['username']}: {str(e)}")
        
        # Rank KOL channels by confidence level and performance
        ranked_kols = sorted(
            [kol for kol in kol_analyses if kol.get("total_calls", 0) > 0],
            key=lambda x: (x.get("confidence_level", 0), x.get("avg_roi", 0)),
            reverse=True
        )
        
        return {
            "spydefi_channel": channel_id,
            "analysis_period_days": days_back,
            "total_kols_found": len(kol_data),
            "total_kols_analyzed": len(kol_analyses),
            "ranked_kols": ranked_kols,
            "kol_data": list(kol_data.values())  # Include the raw KOL data
        }
    
    async def export_spydefi_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """
        Export Spydefi analysis to CSV.
        
        Args:
            analysis (Dict[str, Any]): Spydefi analysis data
            output_file (str): Output file path
        """
        if not analysis or not analysis.get("ranked_kols"):
            logger.warning("No KOL analyses to export")
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
                    "channel_id", "total_calls", "success_rate", "avg_roi", 
                    "avg_max_roi", "confidence_level", "recommendation", "entry_type"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write KOL analyses
                for kol in analysis["ranked_kols"]:
                    strategy = kol.get("strategy", {})
                    row = {
                        "channel_id": kol["channel_id"],
                        "total_calls": kol["total_calls"],
                        "success_rate": kol["success_rate"],
                        "avg_roi": kol["avg_roi"],
                        "avg_max_roi": kol["avg_max_roi"],
                        "confidence_level": kol["confidence_level"],
                        "recommendation": strategy.get("recommendation", "N/A"),
                        "entry_type": strategy.get("entry_type", "WAIT_FOR_CONFIRMATION")
                    }
                    writer.writerow(row)
                
                logger.info(f"Exported {len(analysis['ranked_kols'])} KOL analyses to {output_file}")
            
            # Export detailed token calls for each KOL
            for kol in analysis["ranked_kols"]:
                kol_file = output_file.replace(".csv", f"_{kol['channel_id']}.csv")
                await self.export_channel_analysis(kol, kol_file)
        
        except Exception as e:
            logger.error(f"Error exporting Spydefi analysis: {str(e)}")
, 'mil', 'million',
            'dex', 'raydium', 'jupiter', 'orca', 'meteora',
            'solana', 'sol', 'usdc', 'chart', 'price',
            'x2', 'x3', 'x5', 'x10', '2x', '3x', '5x', '10x',
            'going to', 'gonna', 'potential', 'looks good',
            'trading', 'volume', 'liquidity', 'pair'
        ]
        
        # Negative indicators (probably not a contract address)
        negative_indicators = [
            'user', 'username', 'password', 'email', 'phone',
            'wallet address', 'my address', 'send to', 'transfer to'
        ]
        
        # Check for negative indicators first
        for indicator in negative_indicators:
            if indicator in context_lower:
                return False
        
        # Check for positive indicators
        positive_score = sum(1 for indicator in positive_indicators if indicator in context_lower)
        
        # If we have market cap info nearby, it's very likely a contract
        if any(mc_word in context_lower for mc_word in ['mc', 'market cap', 'marketcap', '200k', '1m', '5m', '10m']):
            return True
        
        # If the address is mentioned with trading/price context
        if positive_score >= 2:
            return True
        
        # If it's a standalone address in a crypto channel context, probably valid
        if positive_score >= 1 and len(context.strip()) < 200:  # Short message with some context
            return True
        
        # Check if it looks like other known Solana program addresses
        known_prefixes = [
            '11111111111111111111111111111112',  # System Program
            'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9',  # Token Program
            'So11111111111111111111111111111111',  # Wrapped SOL
        ]
        
        # Don't flag system addresses as contract addresses
        if any(address.startswith(prefix[:10]) for prefix in known_prefixes):
            return False
        
        # If we have some context clues, it's probably valid
        return positive_score > 0
    
    def extract_kol_usernames(self, text: str) -> List[str]:
        """
        Extract KOL usernames from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of extracted KOL usernames
        """
        usernames = set()
        
        # Find @username mentions
        matches = re.finditer(KOL_USERNAME_PATTERN, text)
        for match in matches:
            if len(match.groups()) > 0:
                usernames.add(match.group(1))
        
        return list(usernames)
    
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
    
    async def scrape_spydefi(self, channel_id: str, 
                          days_back: int = 7,
                          birdeye_api: Any = None) -> Dict[str, Any]:
        """
        Scrape the Spydefi channel for KOL mentions.
        
        Args:
            channel_id (str): Spydefi channel ID
            days_back (int): Number of days to analyze
            birdeye_api (BirdeyeAPI): API client for token data
            
        Returns:
            Dict[str, Any]: Spydefi analysis results with KOL channels
        """
        logger.info(f"Scraping Spydefi channel {channel_id} for KOL mentions")
        
        # Get messages from Spydefi
        messages = await self.get_channel_messages(channel_id, days_back)
        
        # Extract KOL mentions
        kol_data = {}
        
        # Find KOLs in the messages
        for message in messages:
            # Look for KOL mentions in the message text
            for username in message.get("mentioned_usernames", []):
                if username not in kol_data:
                    kol_data[username] = {
                        "username": username,
                        "mentions": 0,
                        "calls": 0,
                        "achievements": []
                    }
                
                kol_data[username]["mentions"] += 1
                
                # Check if it's an achievement or call mention
                for pattern in KOL_CALL_PATTERNS:
                    match = re.search(pattern, message["text"])
                    if match:
                        if len(match.groups()) > 0 and match.group(1).isdigit():
                            multiplier = int(match.group(1))
                            kol_data[username]["achievements"].append({
                                "multiplier": multiplier,
                                "message_id": message["id"],
                                "date": message["date"],
                                "text": message["text"]
                            })
                        kol_data[username]["calls"] += 1
                
                # Check for contract addresses in the message
                if message["contract_addresses"]:
                    for address in message["contract_addresses"]:
                        if "contracts" not in kol_data[username]:
                            kol_data[username]["contracts"] = []
                        
                        kol_data[username]["contracts"].append({
                            "address": address,
                            "message_id": message["id"],
                            "date": message["date"]
                        })
        
        # Also check for direct sender usernames
        for message in messages:
            sender = message.get("sender_username")
            if sender and sender not in kol_data and sender.lower() != "spydefi":
                kol_data[sender] = {
                    "username": sender,
                    "mentions": 1,
                    "calls": 0,
                    "achievements": []
                }
        
        # Extract channel links too
        channel_pattern = r'(?:https?://)?(?:t|telegram)\.(?:me|dog)/(?:joinchat/)?([a-zA-Z0-9_-]+)'
        for message in messages:
            matches = re.finditer(channel_pattern, message["text"])
            for match in matches:
                if len(match.groups()) > 0:
                    channel = match.group(1)
                    if channel not in kol_data and channel.lower() != "spydefi":
                        kol_data[channel] = {
                            "username": channel,
                            "mentions": 1,
                            "calls": 0,
                            "achievements": []
                        }
        
        logger.info(f"Found {len(kol_data)} potential KOL channels/usernames")
        
        # Analyze each KOL channel
        kol_analyses = []
        max_kols_to_analyze = 20  # Limit to top KOLs to avoid excessive API calls
        
        # Sort KOLs by mentions and achievements
        sorted_kols = sorted(
            kol_data.values(),
            key=lambda x: (len(x.get("achievements", [])), x.get("mentions", 0), x.get("calls", 0)),
            reverse=True
        )
        
        # Take top KOLs for analysis
        top_kols = sorted_kols[:max_kols_to_analyze]
        
        for kol in top_kols:
            try:
                logger.info(f"Analyzing KOL: {kol['username']}")
                analysis = await self.analyze_channel(kol['username'], days_back, birdeye_api)
                kol_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing KOL channel {kol['username']}: {str(e)}")
        
        # Rank KOL channels by confidence level and performance
        ranked_kols = sorted(
            [kol for kol in kol_analyses if kol.get("total_calls", 0) > 0],
            key=lambda x: (x.get("confidence_level", 0), x.get("avg_roi", 0)),
            reverse=True
        )
        
        return {
            "spydefi_channel": channel_id,
            "analysis_period_days": days_back,
            "total_kols_found": len(kol_data),
            "total_kols_analyzed": len(kol_analyses),
            "ranked_kols": ranked_kols,
            "kol_data": list(kol_data.values())  # Include the raw KOL data
        }
    
    async def export_spydefi_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """
        Export Spydefi analysis to CSV.
        
        Args:
            analysis (Dict[str, Any]): Spydefi analysis data
            output_file (str): Output file path
        """
        if not analysis or not analysis.get("ranked_kols"):
            logger.warning("No KOL analyses to export")
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
                    "channel_id", "total_calls", "success_rate", "avg_roi", 
                    "avg_max_roi", "confidence_level", "recommendation", "entry_type"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write KOL analyses
                for kol in analysis["ranked_kols"]:
                    strategy = kol.get("strategy", {})
                    row = {
                        "channel_id": kol["channel_id"],
                        "total_calls": kol["total_calls"],
                        "success_rate": kol["success_rate"],
                        "avg_roi": kol["avg_roi"],
                        "avg_max_roi": kol["avg_max_roi"],
                        "confidence_level": kol["confidence_level"],
                        "recommendation": strategy.get("recommendation", "N/A"),
                        "entry_type": strategy.get("entry_type", "WAIT_FOR_CONFIRMATION")
                    }
                    writer.writerow(row)
                
                logger.info(f"Exported {len(analysis['ranked_kols'])} KOL analyses to {output_file}")
            
            # Export detailed token calls for each KOL
            for kol in analysis["ranked_kols"]:
                kol_file = output_file.replace(".csv", f"_{kol['channel_id']}.csv")
                await self.export_channel_analysis(kol, kol_file)
        
        except Exception as e:
            logger.error(f"Error exporting Spydefi analysis: {str(e)}")
, 'mil', 'million', 'billion', 'k mc', 'm mc']):
            context_score += 1
        
        # Trading platforms
        if any(platform in text_lower for platform in ['raydium', 'jupiter', 'orca', 'meteora', 'dex']):
            context_score += 1
        
        # Excitement/urgency indicators
        if any(indicator in text_lower for indicator in ['!', '🔥', '🚀', '📈', '💎', '🌙']):
            context_score += 1
        
        # Numbers suggesting multipliers or gains
        if re.search(r'\d+x|\dx|x\d+', text_lower):
            context_score += 1
        
        # Final decision logic - made more flexible
        if achievement_match:  # Achievement posts are definitely calls
            return True
        
        if has_address and (call_score >= 1 or context_score >= 2):
            return True
        
        if call_score >= 2:  # Strong call indicators even without address
            return True
        
        if has_address and context_score >= 1 and len(text) < 500:  # Short message with address and context
            return True
        
        return False
    
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
    
    async def scrape_spydefi(self, channel_id: str, 
                          days_back: int = 7,
                          birdeye_api: Any = None) -> Dict[str, Any]:
        """
        Scrape the Spydefi channel for KOL mentions.
        
        Args:
            channel_id (str): Spydefi channel ID
            days_back (int): Number of days to analyze
            birdeye_api (BirdeyeAPI): API client for token data
            
        Returns:
            Dict[str, Any]: Spydefi analysis results with KOL channels
        """
        logger.info(f"Scraping Spydefi channel {channel_id} for KOL mentions")
        
        # Get messages from Spydefi
        messages = await self.get_channel_messages(channel_id, days_back)
        
        # Extract KOL mentions
        kol_data = {}
        
        # Find KOLs in the messages
        for message in messages:
            # Look for KOL mentions in the message text
            for username in message.get("mentioned_usernames", []):
                if username not in kol_data:
                    kol_data[username] = {
                        "username": username,
                        "mentions": 0,
                        "calls": 0,
                        "achievements": []
                    }
                
                kol_data[username]["mentions"] += 1
                
                # Check if it's an achievement or call mention
                for pattern in KOL_CALL_PATTERNS:
                    match = re.search(pattern, message["text"])
                    if match:
                        if len(match.groups()) > 0 and match.group(1).isdigit():
                            multiplier = int(match.group(1))
                            kol_data[username]["achievements"].append({
                                "multiplier": multiplier,
                                "message_id": message["id"],
                                "date": message["date"],
                                "text": message["text"]
                            })
                        kol_data[username]["calls"] += 1
                
                # Check for contract addresses in the message
                if message["contract_addresses"]:
                    for address in message["contract_addresses"]:
                        if "contracts" not in kol_data[username]:
                            kol_data[username]["contracts"] = []
                        
                        kol_data[username]["contracts"].append({
                            "address": address,
                            "message_id": message["id"],
                            "date": message["date"]
                        })
        
        # Also check for direct sender usernames
        for message in messages:
            sender = message.get("sender_username")
            if sender and sender not in kol_data and sender.lower() != "spydefi":
                kol_data[sender] = {
                    "username": sender,
                    "mentions": 1,
                    "calls": 0,
                    "achievements": []
                }
        
        # Extract channel links too
        channel_pattern = r'(?:https?://)?(?:t|telegram)\.(?:me|dog)/(?:joinchat/)?([a-zA-Z0-9_-]+)'
        for message in messages:
            matches = re.finditer(channel_pattern, message["text"])
            for match in matches:
                if len(match.groups()) > 0:
                    channel = match.group(1)
                    if channel not in kol_data and channel.lower() != "spydefi":
                        kol_data[channel] = {
                            "username": channel,
                            "mentions": 1,
                            "calls": 0,
                            "achievements": []
                        }
        
        logger.info(f"Found {len(kol_data)} potential KOL channels/usernames")
        
        # Analyze each KOL channel
        kol_analyses = []
        max_kols_to_analyze = 20  # Limit to top KOLs to avoid excessive API calls
        
        # Sort KOLs by mentions and achievements
        sorted_kols = sorted(
            kol_data.values(),
            key=lambda x: (len(x.get("achievements", [])), x.get("mentions", 0), x.get("calls", 0)),
            reverse=True
        )
        
        # Take top KOLs for analysis
        top_kols = sorted_kols[:max_kols_to_analyze]
        
        for kol in top_kols:
            try:
                logger.info(f"Analyzing KOL: {kol['username']}")
                analysis = await self.analyze_channel(kol['username'], days_back, birdeye_api)
                kol_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing KOL channel {kol['username']}: {str(e)}")
        
        # Rank KOL channels by confidence level and performance
        ranked_kols = sorted(
            [kol for kol in kol_analyses if kol.get("total_calls", 0) > 0],
            key=lambda x: (x.get("confidence_level", 0), x.get("avg_roi", 0)),
            reverse=True
        )
        
        return {
            "spydefi_channel": channel_id,
            "analysis_period_days": days_back,
            "total_kols_found": len(kol_data),
            "total_kols_analyzed": len(kol_analyses),
            "ranked_kols": ranked_kols,
            "kol_data": list(kol_data.values())  # Include the raw KOL data
        }
    
    async def export_spydefi_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """
        Export Spydefi analysis to CSV.
        
        Args:
            analysis (Dict[str, Any]): Spydefi analysis data
            output_file (str): Output file path
        """
        if not analysis or not analysis.get("ranked_kols"):
            logger.warning("No KOL analyses to export")
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
                    "channel_id", "total_calls", "success_rate", "avg_roi", 
                    "avg_max_roi", "confidence_level", "recommendation", "entry_type"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write KOL analyses
                for kol in analysis["ranked_kols"]:
                    strategy = kol.get("strategy", {})
                    row = {
                        "channel_id": kol["channel_id"],
                        "total_calls": kol["total_calls"],
                        "success_rate": kol["success_rate"],
                        "avg_roi": kol["avg_roi"],
                        "avg_max_roi": kol["avg_max_roi"],
                        "confidence_level": kol["confidence_level"],
                        "recommendation": strategy.get("recommendation", "N/A"),
                        "entry_type": strategy.get("entry_type", "WAIT_FOR_CONFIRMATION")
                    }
                    writer.writerow(row)
                
                logger.info(f"Exported {len(analysis['ranked_kols'])} KOL analyses to {output_file}")
            
            # Export detailed token calls for each KOL
            for kol in analysis["ranked_kols"]:
                kol_file = output_file.replace(".csv", f"_{kol['channel_id']}.csv")
                await self.export_channel_analysis(kol, kol_file)
        
        except Exception as e:
            logger.error(f"Error exporting Spydefi analysis: {str(e)}")
, 'mil', 'million',
            'dex', 'raydium', 'jupiter', 'orca', 'meteora',
            'solana', 'sol', 'usdc', 'chart', 'price',
            'x2', 'x3', 'x5', 'x10', '2x', '3x', '5x', '10x',
            'going to', 'gonna', 'potential', 'looks good',
            'trading', 'volume', 'liquidity', 'pair'
        ]
        
        # Negative indicators (probably not a contract address)
        negative_indicators = [
            'user', 'username', 'password', 'email', 'phone',
            'wallet address', 'my address', 'send to', 'transfer to'
        ]
        
        # Check for negative indicators first
        for indicator in negative_indicators:
            if indicator in context_lower:
                return False
        
        # Check for positive indicators
        positive_score = sum(1 for indicator in positive_indicators if indicator in context_lower)
        
        # If we have market cap info nearby, it's very likely a contract
        if any(mc_word in context_lower for mc_word in ['mc', 'market cap', 'marketcap', '200k', '1m', '5m', '10m']):
            return True
        
        # If the address is mentioned with trading/price context
        if positive_score >= 2:
            return True
        
        # If it's a standalone address in a crypto channel context, probably valid
        if positive_score >= 1 and len(context.strip()) < 200:  # Short message with some context
            return True
        
        # Check if it looks like other known Solana program addresses
        known_prefixes = [
            '11111111111111111111111111111112',  # System Program
            'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9',  # Token Program
            'So11111111111111111111111111111111',  # Wrapped SOL
        ]
        
        # Don't flag system addresses as contract addresses
        if any(address.startswith(prefix[:10]) for prefix in known_prefixes):
            return False
        
        # If we have some context clues, it's probably valid
        return positive_score > 0
    
    def extract_kol_usernames(self, text: str) -> List[str]:
        """
        Extract KOL usernames from text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of extracted KOL usernames
        """
        usernames = set()
        
        # Find @username mentions
        matches = re.finditer(KOL_USERNAME_PATTERN, text)
        for match in matches:
            if len(match.groups()) > 0:
                usernames.add(match.group(1))
        
        return list(usernames)
    
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
    
    async def scrape_spydefi(self, channel_id: str, 
                          days_back: int = 7,
                          birdeye_api: Any = None) -> Dict[str, Any]:
        """
        Scrape the Spydefi channel for KOL mentions.
        
        Args:
            channel_id (str): Spydefi channel ID
            days_back (int): Number of days to analyze
            birdeye_api (BirdeyeAPI): API client for token data
            
        Returns:
            Dict[str, Any]: Spydefi analysis results with KOL channels
        """
        logger.info(f"Scraping Spydefi channel {channel_id} for KOL mentions")
        
        # Get messages from Spydefi
        messages = await self.get_channel_messages(channel_id, days_back)
        
        # Extract KOL mentions
        kol_data = {}
        
        # Find KOLs in the messages
        for message in messages:
            # Look for KOL mentions in the message text
            for username in message.get("mentioned_usernames", []):
                if username not in kol_data:
                    kol_data[username] = {
                        "username": username,
                        "mentions": 0,
                        "calls": 0,
                        "achievements": []
                    }
                
                kol_data[username]["mentions"] += 1
                
                # Check if it's an achievement or call mention
                for pattern in KOL_CALL_PATTERNS:
                    match = re.search(pattern, message["text"])
                    if match:
                        if len(match.groups()) > 0 and match.group(1).isdigit():
                            multiplier = int(match.group(1))
                            kol_data[username]["achievements"].append({
                                "multiplier": multiplier,
                                "message_id": message["id"],
                                "date": message["date"],
                                "text": message["text"]
                            })
                        kol_data[username]["calls"] += 1
                
                # Check for contract addresses in the message
                if message["contract_addresses"]:
                    for address in message["contract_addresses"]:
                        if "contracts" not in kol_data[username]:
                            kol_data[username]["contracts"] = []
                        
                        kol_data[username]["contracts"].append({
                            "address": address,
                            "message_id": message["id"],
                            "date": message["date"]
                        })
        
        # Also check for direct sender usernames
        for message in messages:
            sender = message.get("sender_username")
            if sender and sender not in kol_data and sender.lower() != "spydefi":
                kol_data[sender] = {
                    "username": sender,
                    "mentions": 1,
                    "calls": 0,
                    "achievements": []
                }
        
        # Extract channel links too
        channel_pattern = r'(?:https?://)?(?:t|telegram)\.(?:me|dog)/(?:joinchat/)?([a-zA-Z0-9_-]+)'
        for message in messages:
            matches = re.finditer(channel_pattern, message["text"])
            for match in matches:
                if len(match.groups()) > 0:
                    channel = match.group(1)
                    if channel not in kol_data and channel.lower() != "spydefi":
                        kol_data[channel] = {
                            "username": channel,
                            "mentions": 1,
                            "calls": 0,
                            "achievements": []
                        }
        
        logger.info(f"Found {len(kol_data)} potential KOL channels/usernames")
        
        # Analyze each KOL channel
        kol_analyses = []
        max_kols_to_analyze = 20  # Limit to top KOLs to avoid excessive API calls
        
        # Sort KOLs by mentions and achievements
        sorted_kols = sorted(
            kol_data.values(),
            key=lambda x: (len(x.get("achievements", [])), x.get("mentions", 0), x.get("calls", 0)),
            reverse=True
        )
        
        # Take top KOLs for analysis
        top_kols = sorted_kols[:max_kols_to_analyze]
        
        for kol in top_kols:
            try:
                logger.info(f"Analyzing KOL: {kol['username']}")
                analysis = await self.analyze_channel(kol['username'], days_back, birdeye_api)
                kol_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing KOL channel {kol['username']}: {str(e)}")
        
        # Rank KOL channels by confidence level and performance
        ranked_kols = sorted(
            [kol for kol in kol_analyses if kol.get("total_calls", 0) > 0],
            key=lambda x: (x.get("confidence_level", 0), x.get("avg_roi", 0)),
            reverse=True
        )
        
        return {
            "spydefi_channel": channel_id,
            "analysis_period_days": days_back,
            "total_kols_found": len(kol_data),
            "total_kols_analyzed": len(kol_analyses),
            "ranked_kols": ranked_kols,
            "kol_data": list(kol_data.values())  # Include the raw KOL data
        }
    
    async def export_spydefi_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """
        Export Spydefi analysis to CSV.
        
        Args:
            analysis (Dict[str, Any]): Spydefi analysis data
            output_file (str): Output file path
        """
        if not analysis or not analysis.get("ranked_kols"):
            logger.warning("No KOL analyses to export")
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
                    "channel_id", "total_calls", "success_rate", "avg_roi", 
                    "avg_max_roi", "confidence_level", "recommendation", "entry_type"
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write KOL analyses
                for kol in analysis["ranked_kols"]:
                    strategy = kol.get("strategy", {})
                    row = {
                        "channel_id": kol["channel_id"],
                        "total_calls": kol["total_calls"],
                        "success_rate": kol["success_rate"],
                        "avg_roi": kol["avg_roi"],
                        "avg_max_roi": kol["avg_max_roi"],
                        "confidence_level": kol["confidence_level"],
                        "recommendation": strategy.get("recommendation", "N/A"),
                        "entry_type": strategy.get("entry_type", "WAIT_FOR_CONFIRMATION")
                    }
                    writer.writerow(row)
                
                logger.info(f"Exported {len(analysis['ranked_kols'])} KOL analyses to {output_file}")
            
            # Export detailed token calls for each KOL
            for kol in analysis["ranked_kols"]:
                kol_file = output_file.replace(".csv", f"_{kol['channel_id']}.csv")
                await self.export_channel_analysis(kol, kol_file)
        
        except Exception as e:
            logger.error(f"Error exporting Spydefi analysis: {str(e)}")