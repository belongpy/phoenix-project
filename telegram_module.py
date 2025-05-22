"""
Telegram Module - Phoenix Project (Fixed)

This module handles all Telegram-related functionality with improved error handling.
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
    r'(?i)Ca:?\s*([1-9A-HJ-NP-Za-km-z]{32,44})'
]

# KOL detection patterns
KOL_USERNAME_PATTERN = r'@([A-Za-z0-9_]+)'
KOL_CALL_PATTERNS = [
    r'(?i)made a x(\d+)\+ call on',
    r'(?i)Achievement Unlocked: x(\d+)',
    r'(?i)(\d+)x\s+gem'
]

class TelegramScraper:
    """Class for scraping and analyzing Telegram channels."""
    
    def __init__(self, api_id: str, api_hash: str, session_name: str = "phoenix", max_days: int = 14):
        """Initialize the Telegram scraper."""
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.max_days = max_days
        self.client = None
        self.message_limit = 2000
        self.spydefi_message_limit = 1000
    
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
        """Extract KOL usernames from text."""
        usernames = set()
        
        # Find @username mentions
        matches = re.finditer(KOL_USERNAME_PATTERN, text)
        for match in matches:
            if len(match.groups()) > 0:
                usernames.add(match.group(1))
        
        return list(usernames)
    
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
            
            if channel_id.startswith("@"):
                channel_id = channel_id[1:]
                
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
    
    async def analyze_channel(self, channel_id: str, days_back: int = 7, api_client: Any = None) -> Dict[str, Any]:
        """Analyze a Telegram channel for token calls with improved error handling."""
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
                    
                    # Add token data if API client is provided
                    if api_client:
                        try:
                            # Get token info with error handling
                            token_info = api_client.get_token_info(contract)
                            
                            if token_info.get("success") and token_info.get("data"):
                                token_data = token_info["data"]
                                token_call["token_name"] = token_data.get("name", "Unknown")
                                token_call["token_symbol"] = token_data.get("symbol", "UNKNOWN")
                                
                                # Get market cap if available
                                market_cap = token_data.get("mc", 0) or token_data.get("marketCap", 0)
                                if market_cap:
                                    token_call["market_cap_usd"] = market_cap
                            else:
                                logger.warning(f"Could not get token info for {contract}: {token_info.get('error', 'Unknown error')}")
                                token_call["token_name"] = "Unknown"
                                token_call["token_symbol"] = "UNKNOWN"
                            
                            # Calculate performance since call with better error handling
                            call_date = datetime.fromisoformat(message["date"])
                            performance = api_client.calculate_token_performance(contract, call_date)
                            
                            if performance.get("success"):
                                token_call.update({
                                    "initial_price": performance.get("initial_price", 0),
                                    "current_price": performance.get("current_price", 0),
                                    "max_price": performance.get("max_price", 0),
                                    "roi_percent": performance.get("roi_percent", 0),
                                    "max_roi_percent": performance.get("max_roi_percent", 0),
                                    "max_drawdown_percent": performance.get("max_drawdown_percent", 0)
                                })
                            else:
                                logger.warning(f"Could not calculate performance for {contract}: {performance.get('error', 'Unknown error')}")
                                # Set default values
                                token_call.update({
                                    "initial_price": 0,
                                    "current_price": 0,
                                    "max_price": 0,
                                    "roi_percent": 0,
                                    "max_roi_percent": 0,
                                    "max_drawdown_percent": 0
                                })
                            
                            # Identify platform with better error handling
                            platform = api_client.identify_platform(contract, token_info)
                            if platform:
                                token_call["platform"] = platform
                            
                        except Exception as e:
                            logger.error(f"Error processing token data for {contract}: {str(e)}")
                            # Set default values to prevent crashes
                            token_call.update({
                                "token_name": "Unknown",
                                "token_symbol": "UNKNOWN",
                                "initial_price": 0,
                                "current_price": 0,
                                "max_price": 0,
                                "roi_percent": 0,
                                "max_roi_percent": 0,
                                "max_drawdown_percent": 0,
                                "platform": "unknown"
                            })
                    
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
            "confidence_level": min(success_rate, 100)
        }
        
        # Generate strategy recommendations
        strategy = self._generate_strategy(analysis)
        analysis["strategy"] = strategy
        
        return analysis
    
    def _generate_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading strategy based on analysis."""
        confidence_level = analysis["confidence_level"]
        avg_max_roi = analysis["avg_max_roi"]
        
        if confidence_level >= 60:
            if avg_max_roi >= 500:  # 5x or more
                return {
                    "recommendation": "HOLD_MOON",
                    "entry_type": "IMMEDIATE",
                    "take_profit_1": 100,
                    "take_profit_2": 200,
                    "take_profit_3": 500,
                    "stop_loss": -30,
                    "trailing_stop": {"activation": 100, "trailing_percent": 25},
                    "notes": "This channel finds potential moonshots. Take 30% at TP1, 20% at TP2, and hold 50% for major gains."
                }
            elif avg_max_roi >= 200:  # 2x or more
                return {
                    "recommendation": "SCALP_AND_HOLD",
                    "entry_type": "IMMEDIATE",
                    "take_profit_1": 50,
                    "take_profit_2": 100,
                    "take_profit_3": 200,
                    "stop_loss": -30,
                    "trailing_stop": {"activation": 50, "trailing_percent": 20},
                    "notes": "Take 50% profit at TP1, 25% at TP2, and trail remaining with specified stop"
                }
            elif avg_max_roi >= 100:  # 1x or more
                return {
                    "recommendation": "SCALP",
                    "entry_type": "IMMEDIATE",
                    "take_profit_1": 30,
                    "take_profit_2": 50,
                    "take_profit_3": 100,
                    "stop_loss": -20,
                    "trailing_stop": {"activation": 30, "trailing_percent": 15},
                    "notes": "Take 50% profit at TP1, 25% at TP2, and trail remaining with specified stop"
                }
            else:
                return {
                    "recommendation": "CAUTIOUS",
                    "entry_type": "WAIT_FOR_CONFIRMATION",
                    "take_profit_1": 20,
                    "take_profit_2": 40,
                    "stop_loss": -15,
                    "trailing_stop": {"activation": 20, "trailing_percent": 10},
                    "notes": "Wait for initial price movement confirmation before entering"
                }
        else:
            return {
                "recommendation": "CAUTIOUS",
                "entry_type": "WAIT_FOR_CONFIRMATION",
                "take_profit_1": 20,
                "take_profit_2": 40,
                "stop_loss": -15,
                "trailing_stop": {"activation": 20, "trailing_percent": 10},
                "notes": "Low confidence signal. Wait for confirmation before entering."
            }
    
    async def export_channel_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Export channel analysis to CSV."""
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
    
    # ... (rest of the methods remain the same but with similar error handling improvements)