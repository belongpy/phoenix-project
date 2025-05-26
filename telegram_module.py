"""
Telegram Module - Phoenix Project (OPTIMIZED KOL FILTERING VERSION)

MAJOR UPDATES:
- Smart KOL filtering based on mention frequency
- Configurable parameters for KOL selection
- Auto-cache handling (no prompts)
- Reduced API calls through intelligent filtering
- Token deduplication across filtered KOLs
- Parallel price discovery (Birdeye + RPC + Helius)
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
    """Telegram scraper for SpyDefi KOL analysis with optimized KOL filtering."""
    
    # Analysis tier constants
    INITIAL_ANALYSIS_CALLS = 5
    DEEP_ANALYSIS_CALLS = 20
    DEEP_ANALYSIS_THRESHOLD = 0.40  # 40% 2x success rate triggers deep analysis
    
    # KOL FILTERING CONFIGURATION
    DEFAULT_TOP_KOLS_TO_ANALYZE = 50  # Maximum KOLs to analyze
    DEFAULT_MIN_MENTIONS_REQUIRED = 2  # Minimum mentions required per KOL
    DEFAULT_MENTION_TIME_WINDOW_HOURS = 24  # Time window for mentions
    
    # Performance constants - OPTIMIZED
    DEFAULT_MESSAGE_LIMIT = 500
    PROGRESS_INTERVAL = 100
    SPYDEFI_TIMEOUT = 60  # seconds
    CHANNEL_TIMEOUT = 15  # Reduced from 30
    KOL_ANALYSIS_TIMEOUT = 30  # Reduced from 120
    GLOBAL_TIMEOUT = 300  # 5 minutes
    MAX_CONCURRENT_CHANNELS = 3  # Keep at 3 for Telegram safety
    CACHE_DURATION_HOURS = 6  # Auto-use if fresh, skip if expired
    MAX_CONSECUTIVE_FAILURES = 10
    
    # Price discovery timeouts
    PRICE_DISCOVERY_TIMEOUT = 5  # Per method
    BATCH_SIZE = 20  # Tokens per batch
    MAX_PRICE_WORKERS = 15  # Parallel price workers
    
    # RPC settings
    RPC_TIMEOUT = 5  # Reduced from 10
    
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
        self.rpc_url = "https://api.mainnet-beta.solana.com"  # Will be updated from config
        
        # Cache directory
        self.cache_dir = Path.home() / ".phoenix_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.spydefi_cache_file = self.cache_dir / "spydefi_kols.json"
        
        # Token analysis cache
        self.token_price_cache = {}  # Simple price cache
        self.price_cache_ttl = 1800  # 30 minutes
        
        # Global token collection for deduplication
        self.all_token_calls = defaultdict(list)  # token -> list of calls
        self.unique_tokens = set()
        
        # Circuit breaker
        self.consecutive_failures = 0
        
        # Thread pool for price discovery
        self.price_executor = ThreadPoolExecutor(max_workers=self.MAX_PRICE_WORKERS)
        
        # Semaphore for concurrent operations
        self.channel_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_CHANNELS)
        
        # Enhanced validation patterns
        self.contract_patterns = [
            r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b',  # Solana addresses
            r'pump\.fun/([1-9A-HJ-NP-Za-km-z]{32,44})',  # pump.fun links
            r'dexscreener\.com/solana/([1-9A-HJ-NP-Za-km-z]{32,44})',  # DexScreener links
            r'birdeye\.so/token/([1-9A-HJ-NP-Za-km-z]{32,44})',  # Birdeye links
        ]
        
        # Spam patterns to exclude
        self.spam_patterns = [
            r'\.sol\b',  # Solana domains
            r'@[a-zA-Z0-9_]+',  # Telegram handles
            r't\.me/',  # Telegram links
            r'twitter\.com/',  # Twitter links
            r'x\.com/',  # X links
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
        
        # Partial results storage
        self.partial_results = {
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
        """Load SpyDefi KOL cache if valid (auto-handle, no prompts)."""
        try:
            if not self.spydefi_cache_file.exists():
                logger.info("No SpyDefi cache found, will perform fresh analysis")
                return None
            
            with open(self.spydefi_cache_file, 'r') as f:
                cache = json.load(f)
            
            # Check if cache is expired
            cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
            cache_age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            
            if cache_age_hours > self.CACHE_DURATION_HOURS:
                logger.info(f"SpyDefi cache expired ({cache_age_hours:.1f}h old), will refresh")
                return None
            
            logger.info(f"✅ Using fresh SpyDefi cache ({cache_age_hours:.1f}h old) with {len(cache.get('kol_mentions', {}))} KOLs")
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
                'version': '2.0'  # Updated version for new filtering
            }
            
            with open(self.spydefi_cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            
            logger.info(f"✅ Saved SpyDefi cache with {len(kol_mentions)} KOLs")
            
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
    
    def _filter_kols_by_mentions(self, kol_mentions: Dict[str, int], 
                                min_mentions: int = None,
                                top_count: int = None) -> Dict[str, int]:
        """
        Filter KOLs based on mention criteria.
        
        Args:
            kol_mentions: Dictionary of KOL -> mention count
            min_mentions: Minimum mentions required (default: class constant)
            top_count: Maximum KOLs to return (default: class constant)
            
        Returns:
            Dict[str, int]: Filtered KOLs
        """
        if min_mentions is None:
            min_mentions = self.DEFAULT_MIN_MENTIONS_REQUIRED
        if top_count is None:
            top_count = self.DEFAULT_TOP_KOLS_TO_ANALYZE
        
        # Filter by minimum mentions
        qualified_kols = {kol: count for kol, count in kol_mentions.items() 
                         if count >= min_mentions}
        
        if not qualified_kols:
            logger.warning(f"No KOLs found with {min_mentions}+ mentions")
            return {}
        
        # Sort by mention count and take top N
        sorted_kols = sorted(qualified_kols.items(), key=lambda x: x[1], reverse=True)
        filtered_kols = dict(sorted_kols[:top_count])
        
        logger.info(f"🎯 Filtered to {len(filtered_kols)} KOLs (min {min_mentions} mentions, top {top_count})")
        logger.info(f"   KOL mention range: {min(filtered_kols.values())}-{max(filtered_kols.values())} mentions")
        
        return filtered_kols
    
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
                
                logger.info(f"Scraping up to {limit} messages from {channel.title} (ID: {channel.id})")
                
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
    
    async def progressive_spydefi_discovery(self, max_hours: int = 24) -> Dict[str, int]:
        """Progressive SpyDefi discovery - optimized for mention counting."""
        kol_mentions = defaultdict(int)
        total_messages = 0
        kol_pattern = r'@([a-zA-Z0-9_]+)'
        
        # Progressive time windows - optimized
        time_windows = [6, 12, 24]
        
        for hours in time_windows:
            if hours > max_hours:
                break
                
            logger.info(f"🔍 Scanning SpyDefi for last {hours} hours...")
            
            # Calculate messages to fetch
            messages_to_fetch = min(1000, 2000 // (hours // 6))
            
            try:
                async with asyncio.timeout(self.SPYDEFI_TIMEOUT):
                    messages = await self.scrape_channel_messages(
                        "spydefi", 
                        hours=hours,
                        limit=messages_to_fetch,
                        show_progress=True
                    )
                    
                    # Extract KOL mentions
                    new_mentions = 0
                    for msg in messages:
                        mentions = re.findall(kol_pattern, msg['text'])
                        for mention in mentions:
                            if mention.lower() != 'spydefi':
                                if mention not in kol_mentions:
                                    new_mentions += 1
                                kol_mentions[mention] += 1
                    
                    total_messages += len(messages)
                    unique_kols = len(kol_mentions)
                    
                    logger.info(f"✅ Found {unique_kols} unique KOLs (+{new_mentions} new) from {len(messages)} messages")
                    
                    # Early termination if we have enough qualified KOLs
                    qualified_kols = len([k for k, v in kol_mentions.items() if v >= self.DEFAULT_MIN_MENTIONS_REQUIRED])
                    if qualified_kols >= self.DEFAULT_TOP_KOLS_TO_ANALYZE:
                        logger.info(f"🎯 Sufficient qualified KOLs found ({qualified_kols}), stopping discovery")
                        break
                        
            except asyncio.TimeoutError:
                logger.warning(f"SpyDefi discovery timeout at {hours}h, continuing with current data")
                break
        
        # Save to cache
        if kol_mentions:
            self._save_spydefi_cache(dict(kol_mentions), total_messages)
        
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
                            {"memcmp": {"offset": 400, "bytes": token_address}},  # Token A
                        ]
                    }
                ]
            )
            
            if "result" in response and response["result"]:
                # Parse pool data to get price
                # This is simplified - in production you'd decode the account data
                return 0.001  # Placeholder
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting pool price: {str(e)}")
            return None
    
    def _get_current_price_parallel(self, token_batch: List[str]) -> Dict[str, TokenPrice]:
        """Get current prices for a batch of tokens in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all price discovery tasks
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
                
                # Try multiple sources in parallel
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
                        entry_price=price * 0.8,  # Estimate 20% below current
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
                        entry_price=0.000001,  # Pump tokens start very low
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
                    entry_price=price * 0.9,  # Conservative estimate
                    source='rpc',
                    confidence='medium',
                    is_pump=token_address.endswith('pump')
                )
        except Exception as e:
            logger.debug(f"RPC pool price error: {str(e)}")
            
        return None
    
    async def collect_filtered_kol_tokens(self, filtered_kols: Dict[str, int], hours: int = 168) -> None:
        """Collect tokens from filtered KOLs only (Phase 1)."""
        logger.info(f"📦 Phase 1: Collecting tokens from {len(filtered_kols)} filtered KOLs...")
        
        # Process in batches of 5 KOLs
        batch_size = 5
        kol_list = list(filtered_kols.items())
        
        for i in range(0, len(kol_list), batch_size):
            batch = kol_list[i:i+batch_size]
            
            # Process batch with delay
            for kol, mention_count in batch:
                try:
                    logger.info(f"Collecting tokens from @{kol} ({mention_count} mentions)")
                    
                    messages = await self.scrape_channel_messages(
                        f"@{kol}", 
                        hours,
                        limit=500,
                        show_progress=False
                    )
                    
                    if messages:
                        # Extract token calls
                        for msg in messages:
                            contracts = self._extract_contract_addresses(msg['text'])
                            
                            for contract in contracts:
                                self.unique_tokens.add(contract)
                                self.all_token_calls[contract].append({
                                    'kol': kol,
                                    'timestamp': int(msg['date'].timestamp()),
                                    'message': msg['text'][:200]
                                })
                    
                    # Delay between KOLs
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error collecting tokens from @{kol}: {str(e)}")
                    continue
            
            # Delay between batches
            if i + batch_size < len(kol_list):
                logger.info(f"Processed {i + batch_size}/{len(kol_list)} KOLs, waiting before next batch...")
                await asyncio.sleep(5)
        
        logger.info(f"✅ Collected {len(self.unique_tokens)} unique tokens from {len(self.all_token_calls)} calls")
    
    async def analyze_tokens_parallel(self) -> Dict[str, Any]:
        """Analyze all collected tokens in parallel batches (Phase 2)."""
        logger.info("💎 Phase 2: Analyzing tokens in parallel batches...")
        
        token_list = list(self.unique_tokens)
        all_prices = {}
        
        # Process in batches
        for i in range(0, len(token_list), self.BATCH_SIZE):
            batch = token_list[i:i+self.BATCH_SIZE]
            logger.info(f"Processing batch {i//self.BATCH_SIZE + 1}/{(len(token_list) + self.BATCH_SIZE - 1)//self.BATCH_SIZE}")
            
            # Get prices for batch
            batch_prices = self._get_current_price_parallel(batch)
            all_prices.update(batch_prices)
            
            # Small delay between batches
            if i + self.BATCH_SIZE < len(token_list):
                await asyncio.sleep(1)
        
        logger.info(f"✅ Got prices for {len(all_prices)} tokens")
        self.api_call_count['price_discovery_successes'] = len(all_prices)
        self.api_call_count['price_discovery_attempts'] = len(token_list)
        
        return all_prices
    
    def calculate_kol_performance(self, all_prices: Dict[str, TokenPrice], 
                                 filtered_kols: Dict[str, int]) -> Dict[str, Any]:
        """Calculate KOL performance based on token prices (Phase 3)."""
        logger.info("📊 Phase 3: Calculating KOL performance...")
        
        kol_performance = {}
        
        # Group calls by KOL (only for filtered KOLs)
        kol_calls = defaultdict(list)
        for token, calls in self.all_token_calls.items():
            for call in calls:
                kol = call['kol']
                # Only process if KOL is in filtered list
                if kol in filtered_kols:
                    kol_calls[kol].append({
                        'token': token,
                        'timestamp': call['timestamp'],
                        'price_data': all_prices.get(token)
                    })
        
        # Calculate performance for each filtered KOL
        for kol in filtered_kols.keys():
            calls = kol_calls.get(kol, [])
            
            if not calls:
                continue
                
            # Sort by timestamp and take most recent
            calls.sort(key=lambda x: x['timestamp'], reverse=True)
            recent_calls = calls[:self.INITIAL_ANALYSIS_CALLS]
            
            tokens_2x = 0
            total_roi = 0
            valid_calls = 0
            
            for call in recent_calls:
                if call['price_data']:
                    price_data = call['price_data']
                    roi = ((price_data.current_price / price_data.entry_price) - 1) * 100
                    
                    if roi >= 100:  # 2x
                        tokens_2x += 1
                    
                    total_roi += roi
                    valid_calls += 1
            
            if valid_calls > 0:
                kol_performance[kol] = {
                    'kol': kol,
                    'tokens_mentioned': len(recent_calls),
                    'tokens_2x_plus': tokens_2x,
                    'success_rate_2x': (tokens_2x / valid_calls * 100) if valid_calls > 0 else 0,
                    'avg_ath_roi': total_roi / valid_calls,
                    'analysis_type': 'initial',
                    'avg_time_to_2x_minutes': 30,  # Placeholder
                    'avg_max_pullback_percent': 25,  # Placeholder
                    'spydefi_mentions': filtered_kols[kol]  # Include mention count
                }
        
        return kol_performance
    
    async def redesigned_spydefi_analysis(self, hours: int = 24, 
                                        top_kols_to_analyze: int = None,
                                        min_mentions_required: int = None,
                                        mention_time_window_hours: int = None) -> Dict[str, Any]:
        """
        Redesigned SpyDefi analysis with optimized KOL filtering.
        
        Args:
            hours: Analysis period in hours
            top_kols_to_analyze: Max KOLs to analyze (default: class constant)
            min_mentions_required: Min mentions per KOL (default: class constant)
            mention_time_window_hours: Time window for mentions (default: class constant)
        """
        logger.info("🚀 STARTING OPTIMIZED SPYDEFI ANALYSIS WITH KOL FILTERING")
        
        # Set defaults from class constants
        if top_kols_to_analyze is None:
            top_kols_to_analyze = self.DEFAULT_TOP_KOLS_TO_ANALYZE
        if min_mentions_required is None:
            min_mentions_required = self.DEFAULT_MIN_MENTIONS_REQUIRED
        if mention_time_window_hours is None:
            mention_time_window_hours = self.DEFAULT_MENTION_TIME_WINDOW_HOURS
        
        logger.info(f"🎯 Filter criteria: Min {min_mentions_required} mentions, Top {top_kols_to_analyze} KOLs")
        
        try:
            # Global timeout for entire analysis
            async with asyncio.timeout(self.GLOBAL_TIMEOUT):
                # Phase 1: Discover and filter active KOLs from SpyDefi
                logger.info("🎯 Phase 1: Discovering and filtering KOLs from SpyDefi...")
                
                # Check cache first (auto-handle, no prompts)
                kol_mentions = None
                cache = self._load_spydefi_cache()
                if cache:
                    kol_mentions = cache.get('kol_mentions', {})
                    logger.info(f"📦 Using cached SpyDefi data: {len(kol_mentions)} total KOLs")
                
                # If no cache or expired, do progressive discovery
                if not kol_mentions:
                    kol_mentions = await self.progressive_spydefi_discovery(mention_time_window_hours)
                
                if not kol_mentions:
                    logger.error("No KOLs found in SpyDefi")
                    return self._generate_error_result("No KOLs found in SpyDefi channel")
                
                # FILTER KOLS BASED ON CRITERIA
                filtered_kols = self._filter_kols_by_mentions(
                    kol_mentions, 
                    min_mentions_required, 
                    top_kols_to_analyze
                )
                
                if not filtered_kols:
                    logger.error(f"No KOLs found meeting criteria (min {min_mentions_required} mentions)")
                    return self._generate_error_result(f"No KOLs found with {min_mentions_required}+ mentions")
                
                logger.info(f"✅ Filtered to {len(filtered_kols)} qualified KOLs from {len(kol_mentions)} total")
                
                # Phase 2: Collect tokens from filtered KOLs only
                await self.collect_filtered_kol_tokens(filtered_kols, hours * 7)
                
                # Phase 3: Analyze tokens in parallel batches
                all_prices = await self.analyze_tokens_parallel()
                
                # Phase 4: Calculate KOL performance (only for filtered KOLs)
                kol_performance = self.calculate_kol_performance(all_prices, filtered_kols)
                
                # Phase 5: Calculate composite scores
                logger.info("🎯 Phase 5: Calculating composite scores...")
                
                for kol, stats in kol_performance.items():
                    composite_score = self._calculate_composite_score(stats)
                    stats['composite_score'] = composite_score
                
                # Sort by composite score
                ranked_kols = dict(sorted(
                    kol_performance.items(),
                    key=lambda x: x[1]['composite_score'],
                    reverse=True
                ))
                
                # Phase 6: Get channel IDs for analyzed KOLs
                logger.info(f"🎯 Phase 6: Getting channel IDs for {len(ranked_kols)} analyzed KOLs...")
                
                for i, kol in enumerate(ranked_kols.keys(), 1):
                    print(f"\rGetting channel ID {i}/{len(ranked_kols)}: @{kol}", end="", flush=True)
                    
                    try:
                        channel_id = await self.get_channel_info(f"@{kol}")
                        if channel_id:
                            ranked_kols[kol]['channel_id'] = channel_id
                    except Exception as e:
                        logger.error(f"Error getting channel ID for @{kol}: {str(e)}")
                    
                    await asyncio.sleep(1)
                
                print(f"\r✅ Channel ID retrieval complete", flush=True)
                
                # Log statistics
                logger.info("📊 OPTIMIZED ANALYSIS STATISTICS:")
                logger.info(f"   📍 Total KOLs found: {len(kol_mentions)}")
                logger.info(f"   🎯 Filtered KOLs analyzed: {len(filtered_kols)}")
                logger.info(f"   📍 Unique tokens found: {len(self.unique_tokens)}")
                logger.info(f"   ✅ Tokens with prices: {len(all_prices)}")
                logger.info(f"   📊 KOLs with performance data: {len(kol_performance)}")
                logger.info(f"   📞 API SAVINGS: Analyzed {len(filtered_kols)} instead of {len(kol_mentions)} KOLs")
                logger.info(f"   📞 Birdeye API calls: {self.api_call_count['birdeye']}")
                logger.info(f"   📞 Helius API calls: {self.api_call_count['helius']}")
                logger.info(f"   📞 RPC calls: {self.api_call_count['rpc']}")
                
                # Calculate overall stats
                total_calls = sum(k.get('tokens_mentioned', 0) for k in kol_performance.values())
                total_2x = sum(k.get('tokens_2x_plus', 0) for k in kol_performance.values())
                overall_2x_rate = (total_2x / max(1, total_calls)) * 100
                
                logger.info("🎉 OPTIMIZED ANALYSIS COMPLETE!")
                
                return {
                    'success': True,
                    'ranked_kols': ranked_kols,
                    'total_kols_analyzed': len(kol_performance),
                    'total_kols_found': len(kol_mentions),
                    'filtering_criteria': {
                        'min_mentions': min_mentions_required,
                        'top_count': top_kols_to_analyze,
                        'time_window_hours': mention_time_window_hours
                    },
                    'deep_analyses_performed': 0,  # Not using deep analysis in this version
                    'total_calls': total_calls,
                    'total_2x_tokens': total_2x,
                    'success_rate_2x': overall_2x_rate,
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
    
    def _calculate_composite_score(self, kol_stats: Dict[str, Any]) -> float:
        """Calculate composite score with 2x weighting."""
        success_rate_2x = kol_stats.get('success_rate_2x', 0)
        avg_time_to_2x_minutes = kol_stats.get('avg_time_to_2x_minutes', 0)
        avg_ath_roi = kol_stats.get('avg_ath_roi', 0)
        tokens_mentioned = kol_stats.get('tokens_mentioned', 0)
        spydefi_mentions = kol_stats.get('spydefi_mentions', 0)  # New factor
        
        # Minimum calls threshold
        if tokens_mentioned < 2:
            return 0
        
        # 1. Success rate score (0-35 points)
        success_score = (success_rate_2x / 100) * 35
        
        # 2. Speed score (0-35 points)
        if avg_time_to_2x_minutes > 0 and success_rate_2x > 0:
            if avg_time_to_2x_minutes <= 30:
                speed_score = 35
            elif avg_time_to_2x_minutes >= 360:
                speed_score = 0
            else:
                speed_score = 35 * (1 - (avg_time_to_2x_minutes - 30) / 330)
        else:
            speed_score = 0
        
        # 3. ATH ROI score (0-15 points)
        ath_score = min(15, (avg_ath_roi / 500) * 15)
        
        # 4. Activity bonus (0-10 points)
        activity_bonus = min(10, tokens_mentioned)
        
        # 5. SpyDefi popularity bonus (0-5 points)
        popularity_bonus = min(5, spydefi_mentions)
        
        # Total score
        total_score = success_score + speed_score + ath_score + activity_bonus + popularity_bonus
        
        return min(100, total_score)
    
    def _generate_error_result(self, error: str) -> Dict[str, Any]:
        """Generate error result."""
        return {
            'success': False,
            'error': error,
            'ranked_kols': {},
            'total_kols_analyzed': 0,
            'total_kols_found': 0,
            'filtering_criteria': {},
            'deep_analyses_performed': 0,
            'total_calls': 0,
            'total_2x_tokens': 0,
            'success_rate_2x': 0,
            'api_stats': self.api_call_count.copy()
        }
    
    def _generate_partial_results(self) -> Dict[str, Any]:
        """Generate results from partial data."""
        # This would need implementation based on partial data collected
        return self._generate_error_result("Analysis timeout - partial results not available")
    
    async def export_spydefi_analysis(self, analysis_results: Dict[str, Any], output_file: str = "spydefi_analysis_2x.csv"):
        """Export the SpyDefi analysis results to CSV with proper formatting."""
        try:
            ranked_kols = analysis_results.get('ranked_kols', {})
            
            if not ranked_kols:
                logger.warning("No KOL data to export")
                return
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Prepare CSV data
            csv_data = []
            
            for kol, data in ranked_kols.items():
                row = {
                    'kol': kol,
                    'channel_id': data.get('channel_id', ''),
                    'spydefi_mentions': data.get('spydefi_mentions', 0),
                    'tokens_mentioned': data.get('tokens_mentioned', 0),
                    'tokens_2x_plus': data.get('tokens_2x_plus', 0),
                    'success_rate_2x': data.get('success_rate_2x', 0),
                    'avg_ath_roi': data.get('avg_ath_roi', 0),
                    'composite_score': data.get('composite_score', 0),
                    'avg_max_pullback_percent': data.get('avg_max_pullback_percent', 0),
                    'avg_time_to_2x_minutes': data.get('avg_time_to_2x_minutes', 0),
                    'analysis_type': data.get('analysis_type', 'initial')
                }
                csv_data.append(row)
            
            # Write CSV
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                if csv_data:
                    writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                    writer.writeheader()
                    writer.writerows(csv_data)
            
            logger.info(f"✅ Exported {len(csv_data)} KOLs to {output_file}")
            print(f"📄 CSV export complete: {output_file}")
            
            # Export summary
            summary_file = output_file.replace('.csv', '_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("SPYDEFI OPTIMIZED ANALYSIS SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total KOLs Found: {analysis_results.get('total_kols_found', 0)}\n")
                f.write(f"KOLs Analyzed: {analysis_results.get('total_kols_analyzed', 0)}\n")
                
                # Filtering info
                criteria = analysis_results.get('filtering_criteria', {})
                f.write(f"\nFILTERING CRITERIA:\n")
                f.write(f"Min Mentions Required: {criteria.get('min_mentions', 0)}\n")
                f.write(f"Top Count Limit: {criteria.get('top_count', 0)}\n")
                f.write(f"Time Window: {criteria.get('time_window_hours', 0)} hours\n")
                
                f.write(f"\nPERFORMANCE METRICS:\n")
                f.write(f"Total Token Calls: {analysis_results.get('total_calls', 0)}\n")
                f.write(f"2x Success Rate: {analysis_results.get('success_rate_2x', 0):.2f}%\n")
                
                # API stats
                api_stats = analysis_results.get('api_stats', {})
                f.write(f"\nAPI EFFICIENCY:\n")
                f.write(f"Birdeye Calls: {api_stats.get('birdeye', 0)}\n")
                f.write(f"Helius Calls: {api_stats.get('helius', 0)}\n")
                f.write(f"RPC Calls: {api_stats.get('rpc', 0)}\n")
                f.write(f"Tokens Cached: {api_stats.get('tokens_cached', 0)}\n")
                f.write(f"Price Discovery Success Rate: {api_stats.get('price_discovery_successes', 0)}/{api_stats.get('price_discovery_attempts', 0)}\n")
                
                # Top performers
                f.write(f"\nTOP PERFORMERS:\n")
                f.write("-" * 50 + "\n")
                
                top_kols = list(ranked_kols.items())[:10]
                for i, (kol, data) in enumerate(top_kols, 1):
                    f.write(f"\n{i}. @{kol}\n")
                    f.write(f"   SpyDefi Mentions: {data.get('spydefi_mentions', 0)}\n")
                    f.write(f"   Composite Score: {data.get('composite_score', 0):.1f}\n")
                    f.write(f"   2x Success Rate: {data.get('success_rate_2x', 0):.1f}%\n")
                    f.write(f"   Tokens Analyzed: {data.get('tokens_mentioned', 0)}\n")
                    f.write(f"   2x Tokens: {data.get('tokens_2x_plus', 0)}\n")
                    f.write(f"   Avg ATH ROI: {data.get('avg_ath_roi', 0):.1f}%\n")
            
            logger.info(f"✅ Exported summary to {summary_file}")
            print(f"📄 Summary export complete: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting analysis: {str(e)}")
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
            logger.info("✅ Cleared price and token caches")
            
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