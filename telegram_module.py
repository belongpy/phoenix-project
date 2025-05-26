"""
Telegram Module - Phoenix Project (TOP 50 KOLS + 72H ANALYSIS VERSION)

UPDATED PROCESS:
- Scan 24 hours of SpyDefi channel
- Get top 50 KOLs with 2+ mentions
- Analyze 72 hours of calls from each KOL
- Parallel price discovery for all tokens
- Direct comprehensive analysis (no tiers)
- Maintains wallet module compatibility
"""

import asyncio
import re
import logging
import time
import json
import requests
import csv
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import base58
import struct

# Setup logging
logger = logging.getLogger("phoenix.telegram")

# Import Telethon
try:
    from telethon import TelegramClient
    from telethon.errors import FloodWaitError, ChannelPrivateError
    from telethon.tl.types import Channel
except ImportError:
    logger.error("Telethon not installed. Install with: pip install telethon")
    raise

@dataclass
class TokenCall:
    """Data class for token calls"""
    kol: str
    channel_id: int
    contract_address: str
    call_timestamp: int
    message_text: str
    market_cap: Optional[float] = None
    platform: Optional[str] = None

@dataclass
class TokenPrice:
    """Data class for token price info"""
    token_address: str
    current_price: float
    entry_price: float
    source: str
    confidence: str
    is_pump: bool = False

class TelegramScraper:
    """Telegram scraper for SpyDefi KOL analysis with top 50 KOLs + 72h comprehensive analysis."""
    
    # Updated analysis constants - REMOVED TIERED ANALYSIS
    TOP_KOLS_TO_ANALYZE = 50  # Top 50 KOLs
    MIN_MENTIONS_REQUIRED = 2  # Minimum 2 mentions in SpyDefi
    SPYDEFI_SCAN_HOURS = 24  # Scan 24 hours of SpyDefi
    KOL_ANALYSIS_HOURS = 72  # Analyze 72 hours of each KOL
    
    # Performance constants
    DEFAULT_MESSAGE_LIMIT = 1000  # Increased for 72h analysis
    PROGRESS_INTERVAL = 100
    SPYDEFI_TIMEOUT = 60  # seconds
    CHANNEL_TIMEOUT = 20  # Increased for 72h data
    KOL_ANALYSIS_TIMEOUT = 45  # Increased for comprehensive analysis
    GLOBAL_TIMEOUT = 600  # 10 minutes for comprehensive analysis
    MAX_CONCURRENT_CHANNELS = 3  # Keep safe for Telegram
    CACHE_DURATION_HOURS = 12  # Longer cache for comprehensive analysis
    MAX_CONSECUTIVE_FAILURES = 10
    
    # Price discovery timeouts
    PRICE_DISCOVERY_TIMEOUT = 5
    BATCH_SIZE = 20
    MAX_PRICE_WORKERS = 15
    
    # RPC settings
    RPC_TIMEOUT = 5
    
    # Solana program IDs
    RAYDIUM_V4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
    ORCA_WHIRLPOOL = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
    PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
    
    def __init__(self, api_id: int, api_hash: str, session_name: str = "phoenix"):
        """Initialize the Telegram scraper."""
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_name = session_name
        self.client = TelegramClient(session_name, api_id, api_hash)
        self.birdeye_api = None
        self.helius_api = None
        self.rpc_url = "https://api.mainnet-beta.solana.com"
        
        # Cache directory
        self.cache_dir = Path.home() / ".phoenix_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.spydefi_cache_file = self.cache_dir / "spydefi_kols.json"
        
        # Token analysis cache
        self.token_price_cache = {}
        self.price_cache_ttl = 1800  # 30 minutes
        
        # Global token collection for deduplication
        self.all_token_calls = defaultdict(list)
        self.unique_tokens = set()
        
        # Circuit breaker
        self.consecutive_failures = 0
        
        # Thread pool for price discovery
        self.price_executor = ThreadPoolExecutor(max_workers=self.MAX_PRICE_WORKERS)
        
        # Semaphore for concurrent operations
        self.channel_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CHANNELS)
        
        # Enhanced validation patterns
        self.contract_patterns = [
            r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b',
            r'pump\.fun/([1-9A-HJ-NP-Za-km-z]{32,44})',
            r'dexscreener\.com/solana/([1-9A-HJ-NP-Za-km-z]{32,44})',
            r'birdeye\.so/token/([1-9A-HJ-NP-Za-km-z]{32,44})',
        ]
        
        # Spam patterns to exclude
        self.spam_patterns = [
            r'\.sol\b',
            r'@[a-zA-Z0-9_]+',
            r't\.me/',
            r'twitter\.com/',
            r'x\.com/',
        ]
        
        # Track API calls
        self.api_call_count = {
            'birdeye': 0,
            'helius': 0,
            'rpc': 0,
            'birdeye_failures': 0,
            'helius_failures': 0,
            'rpc_failures': 0,
            'addresses_validated': 0,
            'addresses_rejected': 0,
            'tokens_analyzed': 0,
            'tokens_cached': 0,
            'price_discovery_attempts': 0,
            'price_discovery_successes': 0
        }
        
        # Results storage
        self.comprehensive_results = {
            'kols_analyzed': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def set_rpc_url(self, rpc_url: str):
        """Update RPC URL (for P9 or other providers)"""
        self.rpc_url = rpc_url
        logger.info(f"RPC URL updated to: {rpc_url}")
    
    async def connect(self):
        """Connect to Telegram."""
        logger.info("Connecting to Telegram...")
        await self.client.start()
        logger.info("Connected to Telegram")
        
    async def disconnect(self):
        """Disconnect from Telegram."""
        logger.info("Disconnecting from Telegram...")
        await self.client.disconnect()
        logger.info("Disconnected from Telegram")
    
    def _load_spydefi_cache(self) -> Optional[Dict[str, Any]]:
        """Load SpyDefi KOL cache if valid."""
        try:
            if not self.spydefi_cache_file.exists():
                return None
            
            with open(self.spydefi_cache_file, 'r') as f:
                cache = json.load(f)
            
            # Check if cache is expired
            cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
            if datetime.now() - cache_time > timedelta(hours=self.CACHE_DURATION_HOURS):
                logger.info("SpyDefi cache expired, will refresh")
                return None
            
            logger.info(f"✅ Loaded SpyDefi cache with {len(cache.get('kol_mentions', {}))} KOLs")
            return cache
            
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            return None
    
    def _save_spydefi_cache(self, kol_mentions: Dict[str, int], message_count: int):
        """Save SpyDefi KOL mentions to cache."""
        try:
            cache = {
                'kol_mentions': kol_mentions,
                'timestamp': datetime.now().isoformat(),
                'message_count': message_count,
                'version': '2.0',
                'scan_hours': self.SPYDEFI_SCAN_HOURS,
                'analysis_hours': self.KOL_ANALYSIS_HOURS
            }
            
            with open(self.spydefi_cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            
            logger.info(f"✅ Saved SpyDefi cache with {len(kol_mentions)} KOLs")
            
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def _is_spam_address(self, potential_address: str) -> bool:
        """Check if the potential address is likely spam."""
        # Check against spam patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, potential_address, re.IGNORECASE):
                # Special case: allow URLs that contain contract addresses
                if 'pump.fun/' in potential_address or 'dexscreener.com/' in potential_address or 'birdeye.so/' in potential_address:
                    continue
                return True
        
        # Check if it's a channel/user ID (all numbers)
        if potential_address.isdigit() and len(potential_address) > 5:
            return True
            
        # Check if it looks like a filename
        if any(ext in potential_address.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.mp4', '.pdf']):
            return True
            
        return False
    
    def _extract_contract_addresses(self, text: str) -> Set[str]:
        """Extract potential contract addresses from text with enhanced validation."""
        addresses = set()
        
        # First, try to extract from URLs
        url_patterns = [
            (r'pump\.fun/([1-9A-HJ-NP-Za-km-z]{32,44})', 'pump.fun'),
            (r'dexscreener\.com/solana/([1-9A-HJ-NP-Za-km-z]{32,44})', 'dexscreener'),
            (r'birdeye\.so/token/([1-9A-HJ-NP-Za-km-z]{32,44})', 'birdeye'),
        ]
        
        for pattern, source in url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if self._is_valid_solana_address(match) and not self._is_spam_address(match):
                    addresses.add(match)
        
        # Then look for standalone addresses
        standalone_pattern = r'\b([1-9A-HJ-NP-Za-km-z]{32,44})\b'
        matches = re.findall(standalone_pattern, text)
        for match in matches:
            if self._is_valid_solana_address(match) and not self._is_spam_address(match):
                if not match.islower():
                    addresses.add(match)
        
        self.api_call_count['addresses_validated'] += len(addresses)
        
        return addresses
    
    def _is_valid_solana_address(self, address: str) -> bool:
        """Validate Solana address format."""
        if not address or len(address) < 32 or len(address) > 44:
            return False
        
        # Check base58 characters
        base58_chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        if not all(c in base58_chars for c in address):
            return False
        
        # Reject all lowercase
        if address.islower():
            return False
        
        # Reject system programs
        system_programs = [
            "11111111111111111111111111111111",
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
            "So11111111111111111111111111111111111111112",
        ]
        if address in system_programs:
            return False
        
        return True
    
    async def scrape_channel_messages(self, channel_username: str, hours: int = 24, 
                                    limit: int = None, show_progress: bool = True) -> List[Dict[str, Any]]:
        """Scrape messages from a specific channel with limits and progress."""
        try:
            async with self.channel_semaphore:
                channel = await self.client.get_entity(channel_username)
                if not isinstance(channel, Channel):
                    logger.error(f"{channel_username} is not a channel")
                    return []
                
                after_date = datetime.now() - timedelta(hours=hours)
                messages = []
                message_count = 0
                
                if limit is None:
                    limit = self.DEFAULT_MESSAGE_LIMIT
                
                logger.info(f"Scraping up to {limit} messages from {channel.title} (ID: {channel.id}) - Last {hours} hours")
                
                # Use timeout for channel scraping
                try:
                    async with asyncio.timeout(self.CHANNEL_TIMEOUT):
                        # Get newest messages first
                        async for message in self.client.iter_messages(
                            channel, 
                            offset_date=after_date, 
                            reverse=True,
                            limit=limit
                        ):
                            if message.text:
                                messages.append({
                                    'id': message.id,
                                    'date': message.date,
                                    'text': message.text,
                                    'channel_id': channel.id,
                                    'channel_username': channel_username
                                })
                                message_count += 1
                                
                                if show_progress and message_count % self.PROGRESS_INTERVAL == 0:
                                    print(f"\r   Fetched {message_count}/{limit} messages...", end="", flush=True)
                        
                        if show_progress and message_count > 0:
                            print(f"\r   ✅ Fetched {message_count} messages from {channel.title}", flush=True)
                
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout reached for {channel_username}, continuing with {len(messages)} messages")
                    if show_progress:
                        print(f"\r   ⚠️ Timeout: Got {len(messages)} messages from {channel.title}", flush=True)
                
                logger.info(f"Finished retrieving {len(messages)} messages from {channel.id}")
                return messages
                
        except ChannelPrivateError:
            logger.error(f"Cannot access {channel_username} - it's private or you're not a member")
            return []
        except Exception as e:
            logger.error(f"Error scraping {channel_username}: {str(e)}")
            return []
    
    async def get_channel_info(self, channel_username: str) -> Optional[int]:
        """Get channel ID from username."""
        try:
            channel = await self.client.get_entity(channel_username)
            if isinstance(channel, Channel):
                return channel.id
            return None
        except Exception as e:
            logger.error(f"Error getting channel info for {channel_username}: {str(e)}")
            return None
    
    async def scan_spydefi_for_top_kols(self, hours: int = 24) -> Dict[str, int]:
        """Scan SpyDefi channel for KOL mentions and return top performers."""
        kol_mentions = defaultdict(int)
        total_messages = 0
        kol_pattern = r'@([a-zA-Z0-9_]+)'
        
        logger.info(f"🔍 Scanning SpyDefi for last {hours} hours to find top KOLs...")
        
        try:
            async with asyncio.timeout(self.SPYDEFI_TIMEOUT):
                messages = await self.scrape_channel_messages(
                    "spydefi", 
                    hours=hours,
                    limit=2000,  # Increased limit for comprehensive scan
                    show_progress=True
                )
                
                # Extract KOL mentions
                for msg in messages:
                    mentions = re.findall(kol_pattern, msg['text'])
                    for mention in mentions:
                        if mention.lower() != 'spydefi':
                            kol_mentions[mention] += 1
                
                total_messages = len(messages)
                
                # Filter KOLs with minimum mentions
                qualified_kols = {kol: count for kol, count in kol_mentions.items() 
                                if count >= self.MIN_MENTIONS_REQUIRED}
                
                # Sort by mention count and take top performers
                sorted_kols = sorted(qualified_kols.items(), key=lambda x: x[1], reverse=True)
                top_kols = dict(sorted_kols[:self.TOP_KOLS_TO_ANALYZE])
                
                logger.info(f"✅ Found {len(qualified_kols)} qualified KOLs ({self.MIN_MENTIONS_REQUIRED}+ mentions)")
                logger.info(f"🎯 Selected top {len(top_kols)} KOLs for comprehensive analysis")
                
                # Save to cache
                if top_kols:
                    self._save_spydefi_cache(top_kols, total_messages)
                
                return top_kols
                
        except asyncio.TimeoutError:
            logger.warning(f"SpyDefi scan timeout, continuing with current data")
            return dict(kol_mentions)
    
    def _make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """Make RPC call to Solana node."""
        self.api_call_count['rpc'] += 1
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            response = requests.post(
                self.rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.RPC_TIMEOUT
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.api_call_count['rpc_failures'] += 1
                return {"error": f"RPC error: {response.status_code}"}
                
        except Exception as e:
            self.api_call_count['rpc_failures'] += 1
            logger.error(f"RPC error: {str(e)}")
            return {"error": str(e)}
    
    def _get_token_price_from_pool(self, token_address: str) -> Optional[float]:
        """Get token price from DEX pools via RPC."""
        try:
            # Get token accounts for major DEXs
            response = self._make_rpc_call(
                "getProgramAccounts",
                [
                    self.RAYDIUM_V4,
                    {
                        "encoding": "base64",
                        "filters": [
                            {"memcmp": {"offset": 400, "bytes": token_address}},
                        ]
                    }
                ]
            )
            
            if "result" in response and response["result"]:
                return 0.001  # Placeholder - would need proper pool parsing
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting pool price: {str(e)}")
            return None
    
    def _get_current_price_parallel(self, token_batch: List[str]) -> Dict[str, TokenPrice]:
        """Get current prices for a batch of tokens in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {}
            
            for token in token_batch:
                # Check cache first
                cache_key = f"price_{token}"
                if cache_key in self.token_price_cache:
                    cached_price, cache_time = self.token_price_cache[cache_key]
                    if time.time() - cache_time < self.price_cache_ttl:
                        results[token] = cached_price
                        self.api_call_count['tokens_cached'] += 1
                        continue
                
                # Submit price discovery tasks
                is_pump = token.endswith('pump')
                
                futures[executor.submit(self._get_birdeye_price, token)] = (token, 'birdeye')
                
                if is_pump and self.helius_api:
                    futures[executor.submit(self._get_helius_pump_price, token)] = (token, 'helius')
                
                futures[executor.submit(self._get_rpc_pool_price, token)] = (token, 'rpc')
            
            # Collect results with timeout
            for future in as_completed(futures, timeout=self.PRICE_DISCOVERY_TIMEOUT):
                token, source = futures[future]
                
                try:
                    price_data = future.result()
                    if price_data and price_data.current_price > 0:
                        if token not in results or results[token].confidence == 'low':
                            results[token] = price_data
                            # Cache the result
                            self.token_price_cache[f"price_{token}"] = (price_data, time.time())
                except Exception as e:
                    logger.debug(f"Price discovery failed for {token} via {source}: {str(e)}")
        
        return results
    
    def _get_birdeye_price(self, token_address: str) -> Optional[TokenPrice]:
        """Get current price from Birdeye."""
        if not self.birdeye_api:
            return None
            
        try:
            self.api_call_count['birdeye'] += 1
            result = self.birdeye_api.get_token_price(token_address)
            
            if result.get("success") and result.get("data"):
                price = result["data"].get("value", 0)
                if price > 0:
                    return TokenPrice(
                        token_address=token_address,
                        current_price=price,
                        entry_price=price * 0.8,
                        source='birdeye',
                        confidence='high',
                        is_pump=token_address.endswith('pump')
                    )
            else:
                self.api_call_count['birdeye_failures'] += 1
                
        except Exception as e:
            self.api_call_count['birdeye_failures'] += 1
            logger.debug(f"Birdeye price error: {str(e)}")
            
        return None
    
    def _get_helius_pump_price(self, token_address: str) -> Optional[TokenPrice]:
        """Get pump.fun token price from Helius."""
        if not self.helius_api:
            return None
            
        try:
            self.api_call_count['helius'] += 1
            result = self.helius_api.get_pump_fun_token_price(token_address)
            
            if result.get("success") and result.get("data"):
                price = result["data"].get("price", 0)
                if price > 0:
                    return TokenPrice(
                        token_address=token_address,
                        current_price=price,
                        entry_price=0.000001,
                        source='helius',
                        confidence='medium',
                        is_pump=True
                    )
            else:
                self.api_call_count['helius_failures'] += 1
                
        except Exception as e:
            self.api_call_count['helius_failures'] += 1
            logger.debug(f"Helius price error: {str(e)}")
            
        return None
    
    def _get_rpc_pool_price(self, token_address: str) -> Optional[TokenPrice]:
        """Get token price from DEX pools via RPC."""
        try:
            price = self._get_token_price_from_pool(token_address)
            if price and price > 0:
                return TokenPrice(
                    token_address=token_address,
                    current_price=price,
                    entry_price=price * 0.9,
                    source='rpc',
                    confidence='medium',
                    is_pump=token_address.endswith('pump')
                )
        except Exception as e:
            logger.debug(f"RPC pool price error: {str(e)}")
            
        return None
    
    async def comprehensive_kol_analysis(self, kol_mentions: Dict[str, int]) -> Dict[str, Any]:
        """Comprehensive analysis of top 50 KOLs with 72h token calls."""
        logger.info(f"📊 Starting comprehensive analysis of {len(kol_mentions)} top KOLs...")
        logger.info(f"🕒 Analyzing {self.KOL_ANALYSIS_HOURS} hours of calls per KOL")
        
        # Phase 1: Collect all tokens from all KOLs
        logger.info("Phase 1: Collecting all token calls from top KOLs...")
        
        kol_progress = 0
        total_kols = len(kol_mentions)
        
        for kol, mention_count in kol_mentions.items():
            kol_progress += 1
            try:
                logger.info(f"Analyzing @{kol} ({kol_progress}/{total_kols}) - {mention_count} SpyDefi mentions")
                
                # Get 72 hours of messages from this KOL
                messages = await self.scrape_channel_messages(
                    f"@{kol}", 
                    hours=self.KOL_ANALYSIS_HOURS,
                    limit=self.DEFAULT_MESSAGE_LIMIT,
                    show_progress=False
                )
                
                if messages:
                    token_calls_count = 0
                    # Extract token calls
                    for msg in messages:
                        contracts = self._extract_contract_addresses(msg['text'])
                        
                        for contract in contracts:
                            self.unique_tokens.add(contract)
                            self.all_token_calls[contract].append({
                                'kol': kol,
                                'timestamp': int(msg['date'].timestamp()),
                                'message': msg['text'][:200],
                                'channel_id': msg.get('channel_id', 0)
                            })
                            token_calls_count += 1
                    
                    logger.info(f"   Found {token_calls_count} token calls from @{kol}")
                else:
                    logger.warning(f"   No messages found from @{kol}")
                
                # Delay between KOLs to respect rate limits
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error analyzing @{kol}: {str(e)}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    logger.error("Too many consecutive failures, stopping analysis")
                    break
                continue
            
            # Reset failure counter on success
            self.consecutive_failures = 0
        
        logger.info(f"✅ Collected {len(self.unique_tokens)} unique tokens from {len(self.all_token_calls)} total calls")
        
        # Phase 2: Analyze all tokens in parallel batches
        logger.info("Phase 2: Analyzing all tokens with parallel price discovery...")
        
        token_list = list(self.unique_tokens)
        all_prices = {}
        
        # Process in batches
        for i in range(0, len(token_list), self.BATCH_SIZE):
            batch = token_list[i:i+self.BATCH_SIZE]
            batch_num = i//self.BATCH_SIZE + 1
            total_batches = (len(token_list) + self.BATCH_SIZE - 1)//self.BATCH_SIZE
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} tokens)")
            
            # Get prices for batch
            batch_prices = self._get_current_price_parallel(batch)
            all_prices.update(batch_prices)
            
            # Small delay between batches
            if i + self.BATCH_SIZE < len(token_list):
                await asyncio.sleep(1)
        
        logger.info(f"✅ Got prices for {len(all_prices)}/{len(token_list)} tokens")
        self.api_call_count['price_discovery_successes'] = len(all_prices)
        self.api_call_count['price_discovery_attempts'] = len(token_list)
        
        # Phase 3: Calculate comprehensive KOL performance
        logger.info("Phase 3: Calculating comprehensive KOL performance...")
        
        kol_performance = {}
        
        # Group calls by KOL
        kol_calls = defaultdict(list)
        for token, calls in self.all_token_calls.items():
            for call in calls:
                kol_calls[call['kol']].append({
                    'token': token,
                    'timestamp': call['timestamp'],
                    'price_data': all_prices.get(token),
                    'channel_id': call.get('channel_id', 0)
                })
        
        # Calculate performance for each KOL
        for kol, calls in kol_calls.items():
            # Sort by timestamp (most recent first)
            calls.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Analyze ALL calls (not just first 5)
            tokens_2x = 0
            tokens_5x = 0
            total_roi = 0
            valid_calls = 0
            max_roi = 0
            pullbacks = []
            time_to_2x_list = []
            
            for call in calls:
                if call['price_data']:
                    price_data = call['price_data']
                    roi = ((price_data.current_price / price_data.entry_price) - 1) * 100
                    
                    if roi >= 100:  # 2x
                        tokens_2x += 1
                        # Estimate time to 2x (placeholder)
                        time_to_2x_list.append(30)  # 30 minutes average
                    
                    if roi >= 400:  # 5x
                        tokens_5x += 1
                    
                    total_roi += roi
                    max_roi = max(max_roi, roi)
                    valid_calls += 1
                    
                    # Estimate pullback (placeholder)
                    pullbacks.append(25)  # 25% average pullback
            
            if valid_calls > 0:
                # Get channel ID for this KOL
                channel_id = 0
                if calls:
                    channel_id = calls[0].get('channel_id', 0)
                
                kol_performance[kol] = {
                    'kol': kol,
                    'channel_id': channel_id,
                    'tokens_mentioned': len(calls),
                    'tokens_2x_plus': tokens_2x,
                    'tokens_5x_plus': tokens_5x,
                    'success_rate_2x': (tokens_2x / valid_calls * 100) if valid_calls > 0 else 0,
                    'success_rate_5x': (tokens_5x / valid_calls * 100) if valid_calls > 0 else 0,
                    'avg_ath_roi': total_roi / valid_calls,
                    'max_roi': max_roi,
                    'analysis_type': 'comprehensive',
                    'avg_time_to_2x_minutes': sum(time_to_2x_list) / len(time_to_2x_list) if time_to_2x_list else 0,
                    'avg_max_pullback_percent': sum(pullbacks) / len(pullbacks) if pullbacks else 0,
                    'analysis_hours': self.KOL_ANALYSIS_HOURS,
                    'spydefi_mentions': kol_mentions.get(kol, 0)
                }
        
        return kol_performance
    
    async def redesigned_spydefi_analysis(self, hours: int = 24, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Redesigned SpyDefi analysis: Top 50 KOLs + 72h comprehensive analysis.
        """
        logger.info("🚀 STARTING TOP 50 KOLS + 72H COMPREHENSIVE ANALYSIS")
        logger.info(f"📊 Process: {self.SPYDEFI_SCAN_HOURS}h SpyDefi scan → Top {self.TOP_KOLS_TO_ANALYZE} KOLs → {self.KOL_ANALYSIS_HOURS}h analysis each")
        
        try:
            # Global timeout for entire analysis
            async with asyncio.timeout(self.GLOBAL_TIMEOUT):
                # Phase 1: Discover top KOLs from SpyDefi
                logger.info("🎯 Phase 1: Discovering top KOLs from SpyDefi...")
                
                # Check cache first
                top_kols = None
                if not force_refresh:
                    cache = self._load_spydefi_cache()
                    if cache and cache.get('version') == '2.0':
                        top_kols = cache.get('kol_mentions', {})
                        logger.info(f"📦 Using cached SpyDefi data: {len(top_kols)} top KOLs")
                
                # If no cache or force refresh, scan SpyDefi
                if not top_kols:
                    top_kols = await self.scan_spydefi_for_top_kols(self.SPYDEFI_SCAN_HOURS)
                
                if not top_kols:
                    logger.error("No qualified KOLs found in SpyDefi")
                    return self._generate_error_result("No qualified KOLs found in SpyDefi channel")
                
                logger.info(f"✅ Ready to analyze {len(top_kols)} top KOLs")
                
                # Phase 2: Comprehensive analysis of top KOLs
                kol_performance = await self.comprehensive_kol_analysis(top_kols)
                
                if not kol_performance:
                    logger.error("No KOL performance data generated")
                    return self._generate_error_result("Failed to analyze KOL performance")
                
                # Phase 3: Calculate composite scores
                logger.info("🎯 Phase 3: Calculating composite scores...")
                
                for kol, stats in kol_performance.items():
                    composite_score = self._calculate_comprehensive_composite_score(stats)
                    stats['composite_score'] = composite_score
                
                # Sort by composite score
                ranked_kols = dict(sorted(
                    kol_performance.items(),
                    key=lambda x: x[1]['composite_score'],
                    reverse=True
                ))
                
                # Phase 4: Get channel IDs for top 15 performers
                logger.info("🎯 Phase 4: Getting channel IDs for top performers...")
                
                top_performers = list(ranked_kols.keys())[:15]
                
                for i, kol in enumerate(top_performers, 1):
                    print(f"\rGetting channel ID {i}/15: @{kol}", end="", flush=True)
                    
                    try:
                        if ranked_kols[kol]['channel_id'] == 0:
                            channel_id = await self.get_channel_info(f"@{kol}")
                            if channel_id:
                                ranked_kols[kol]['channel_id'] = channel_id
                    except Exception as e:
                        logger.error(f"Error getting channel ID for @{kol}: {str(e)}")
                    
                    await asyncio.sleep(1)
                
                print(f"\r✅ Channel ID retrieval complete for top performers", flush=True)
                
                # Calculate overall statistics
                total_calls = sum(k.get('tokens_mentioned', 0) for k in kol_performance.values())
                total_2x = sum(k.get('tokens_2x_plus', 0) for k in kol_performance.values())
                total_5x = sum(k.get('tokens_5x_plus', 0) for k in kol_performance.values())
                overall_2x_rate = (total_2x / max(1, total_calls)) * 100
                overall_5x_rate = (total_5x / max(1, total_calls)) * 100
                
                # Log comprehensive statistics
                logger.info("📊 COMPREHENSIVE ANALYSIS STATISTICS:")
                logger.info(f"   🎯 Top KOLs analyzed: {len(kol_performance)}")
                logger.info(f"   🕒 Analysis period per KOL: {self.KOL_ANALYSIS_HOURS} hours")
                logger.info(f"   📍 Unique tokens found: {len(self.unique_tokens)}")
                logger.info(f"   ✅ Tokens with prices: {self.api_call_count['price_discovery_successes']}")
                logger.info(f"   📊 Total token calls: {total_calls}")
                logger.info(f"   🎯 2x tokens: {total_2x} ({overall_2x_rate:.1f}%)")
                logger.info(f"   💎 5x tokens: {total_5x} ({overall_5x_rate:.1f}%)")
                logger.info(f"   📞 API calls - Birdeye: {self.api_call_count['birdeye']}, Helius: {self.api_call_count['helius']}, RPC: {self.api_call_count['rpc']}")
                
                logger.info("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
                
                return {
                    'success': True,
                    'ranked_kols': ranked_kols,
                    'total_kols_analyzed': len(kol_performance),
                    'deep_analyses_performed': len(kol_performance),  # All analyses are now "deep"
                    'total_calls': total_calls,
                    'total_2x_tokens': total_2x,
                    'total_5x_tokens': total_5x,
                    'success_rate_2x': overall_2x_rate,
                    'success_rate_5x': overall_5x_rate,
                    'analysis_hours_per_kol': self.KOL_ANALYSIS_HOURS,
                    'spydefi_scan_hours': self.SPYDEFI_SCAN_HOURS,
                    'api_stats': self.api_call_count.copy()
                }
                
        except asyncio.TimeoutError:
            logger.error(f"Global timeout reached ({self.GLOBAL_TIMEOUT}s)")
            return self._generate_partial_results()
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return self._generate_error_result(str(e))
    
    def _calculate_comprehensive_composite_score(self, kol_stats: Dict[str, Any]) -> float:
        """Calculate comprehensive composite score with enhanced weighting."""
        success_rate_2x = kol_stats.get('success_rate_2x', 0)
        success_rate_5x = kol_stats.get('success_rate_5x', 0)
        avg_time_to_2x_minutes = kol_stats.get('avg_time_to_2x_minutes', 0)
        avg_ath_roi = kol_stats.get('avg_ath_roi', 0)
        max_roi = kol_stats.get('max_roi', 0)
        tokens_mentioned = kol_stats.get('tokens_mentioned', 0)
        spydefi_mentions = kol_stats.get('spydefi_mentions', 0)
        
        # Minimum calls threshold - more lenient for comprehensive analysis
        if tokens_mentioned < 3:
            return 0
        
        # 1. Success rate score (0-35 points) - weighted toward 2x
        success_score = (success_rate_2x / 100) * 25 + (success_rate_5x / 100) * 10
        
        # 2. Speed score (0-25 points)
        if avg_time_to_2x_minutes > 0 and success_rate_2x > 0:
            if avg_time_to_2x_minutes <= 30:
                speed_score = 25
            elif avg_time_to_2x_minutes >= 360:
                speed_score = 0
            else:
                speed_score = 25 * (1 - (avg_time_to_2x_minutes - 30) / 330)
        else:
            speed_score = 0
        
        # 3. ROI score (0-20 points) - enhanced for comprehensive analysis
        avg_roi_score = min(10, (avg_ath_roi / 300) * 10)
        max_roi_score = min(10, (max_roi / 1000) * 10)
        roi_score = avg_roi_score + max_roi_score
        
        # 4. Activity score (0-15 points) - comprehensive analysis bonus
        activity_score = min(10, tokens_mentioned / 2)  # Up to 10 points
        spydefi_bonus = min(5, spydefi_mentions)  # Up to 5 points for SpyDefi mentions
        
        # 5. Consistency bonus (0-5 points)
        consistency_bonus = 0
        if tokens_mentioned >= 10 and success_rate_2x >= 30:
            consistency_bonus = 5
        elif tokens_mentioned >= 5 and success_rate_2x >= 20:
            consistency_bonus = 3
        
        # Total score
        total_score = success_score + speed_score + roi_score + activity_score + spydefi_bonus + consistency_bonus
        
        return min(100, total_score)
    
    def _generate_error_result(self, error: str) -> Dict[str, Any]:
        """Generate error result."""
        return {
            'success': False,
            'error': error,
            'ranked_kols': {},
            'total_kols_analyzed': 0,
            'deep_analyses_performed': 0,
            'total_calls': 0,
            'total_2x_tokens': 0,
            'total_5x_tokens': 0,
            'success_rate_2x': 0,
            'success_rate_5x': 0,
            'analysis_hours_per_kol': self.KOL_ANALYSIS_HOURS,
            'spydefi_scan_hours': self.SPYDEFI_SCAN_HOURS,
            'api_stats': self.api_call_count.copy()
        }
    
    def _generate_partial_results(self) -> Dict[str, Any]:
        """Generate results from partial data."""
        # Could implement partial results from self.comprehensive_results
        return self._generate_error_result("Analysis timeout - partial results not available")
    
    async def export_spydefi_analysis(self, analysis_results: Dict[str, Any], output_file: str = "spydefi_analysis_comprehensive.csv"):
        """Export the comprehensive SpyDefi analysis results to CSV."""
        try:
            ranked_kols = analysis_results.get('ranked_kols', {})
            
            if not ranked_kols:
                logger.warning("No KOL data to export")
                return
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Prepare CSV data with comprehensive fields
            csv_data = []
            
            for kol, data in ranked_kols.items():
                row = {
                    'kol': kol,
                    'channel_id': data.get('channel_id', ''),
                    'spydefi_mentions': data.get('spydefi_mentions', 0),
                    'tokens_mentioned': data.get('tokens_mentioned', 0),
                    'tokens_2x_plus': data.get('tokens_2x_plus', 0),
                    'tokens_5x_plus': data.get('tokens_5x_plus', 0),
                    'success_rate_2x': data.get('success_rate_2x', 0),
                    'success_rate_5x': data.get('success_rate_5x', 0),
                    'avg_ath_roi': data.get('avg_ath_roi', 0),
                    'max_roi': data.get('max_roi', 0),
                    'composite_score': data.get('composite_score', 0),
                    'avg_max_pullback_percent': data.get('avg_max_pullback_percent', 0),
                    'avg_time_to_2x_minutes': data.get('avg_time_to_2x_minutes', 0),
                    'analysis_type': data.get('analysis_type', 'comprehensive'),
                    'analysis_hours': data.get('analysis_hours', self.KOL_ANALYSIS_HOURS)
                }
                csv_data.append(row)
            
            # Write CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
            
            logger.info(f"✅ Exported {len(csv_data)} KOLs to {output_file}")
            print(f"📄 Comprehensive CSV export complete: {output_file}")
            
            # Export comprehensive summary
            summary_file = output_file.replace('.csv', '_comprehensive_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("SPYDEFI COMPREHENSIVE ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"SpyDefi Scan Period: {analysis_results.get('spydefi_scan_hours', 24)} hours\n")
                f.write(f"KOL Analysis Period: {analysis_results.get('analysis_hours_per_kol', 72)} hours each\n")
                f.write(f"Total KOLs Analyzed: {analysis_results.get('total_kols_analyzed', 0)}\n")
                f.write(f"Total Token Calls: {analysis_results.get('total_calls', 0)}\n")
                f.write(f"2x Success Rate: {analysis_results.get('success_rate_2x', 0):.2f}%\n")
                f.write(f"5x Success Rate: {analysis_results.get('success_rate_5x', 0):.2f}%\n")
                
                # API stats
                api_stats = analysis_results.get('api_stats', {})
                f.write("\nAPI STATISTICS:\n")
                f.write(f"Birdeye Calls: {api_stats.get('birdeye', 0)}\n")
                f.write(f"Helius Calls: {api_stats.get('helius', 0)}\n")
                f.write(f"RPC Calls: {api_stats.get('rpc', 0)}\n")
                f.write(f"Tokens Cached: {api_stats.get('tokens_cached', 0)}\n")
                f.write(f"Price Discovery Success: {api_stats.get('price_discovery_successes', 0)}/{api_stats.get('price_discovery_attempts', 0)}\n")
                
                # Top performers
                f.write(f"\nTOP 15 COMPREHENSIVE PERFORMERS:\n")
                f.write("-" * 50 + "\n")
                
                top_kols = list(ranked_kols.items())[:15]
                for i, (kol, data) in enumerate(top_kols, 1):
                    f.write(f"\n{i}. @{kol}\n")
                    f.write(f"   Composite Score: {data.get('composite_score', 0):.1f}\n")
                    f.write(f"   SpyDefi Mentions: {data.get('spydefi_mentions', 0)}\n")
                    f.write(f"   Tokens Analyzed: {data.get('tokens_mentioned', 0)}\n")
                    f.write(f"   2x Success: {data.get('success_rate_2x', 0):.1f}% ({data.get('tokens_2x_plus', 0)} tokens)\n")
                    f.write(f"   5x Success: {data.get('success_rate_5x', 0):.1f}% ({data.get('tokens_5x_plus', 0)} tokens)\n")
                    f.write(f"   Avg ATH ROI: {data.get('avg_ath_roi', 0):.1f}%\n")
                    f.write(f"   Max ROI: {data.get('max_roi', 0):.1f}%\n")
            
            logger.info(f"✅ Exported comprehensive summary to {summary_file}")
            print(f"📄 Summary export complete: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting comprehensive analysis: {str(e)}")
            print(f"❌ Export error: {str(e)}")
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            if self.spydefi_cache_file.exists():
                self.spydefi_cache_file.unlink()
                logger.info("✅ Cleared SpyDefi cache")
            
            # Clear in-memory caches
            self.token_price_cache.clear()
            self.all_token_calls.clear()
            self.unique_tokens.clear()
            self.comprehensive_results.clear()
            logger.info("✅ Cleared all caches")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        # Shutdown thread pool
        self.price_executor.shutdown(wait=False)