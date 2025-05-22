"""
Telegram Module - Phoenix Project (Clean Fixed Version)

Focused SpyDefi analyzer with simple flow:
UNIX time → Call price → Historical data → ATH/ATL → Performance
"""

import re
import csv
import os
import logging
import requests
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta, timezone
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from collections import defaultdict

logger = logging.getLogger("phoenix.telegram")

# Simple Solana address pattern
SOLANA_ADDRESS_PATTERN = r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'

# KOL mention patterns
KOL_PATTERNS = [
    r'@(\w+)',
    r'(\w+)\s+made\s+a.*?call',
    r'(\w+)\s+called',
    r'Achievement.*?@?(\w+)',
    r'(\w+)\s+x\d+'
]

class TelegramScraper:
    """SpyDefi-focused Telegram scraper."""
    
    def __init__(self, api_id: str, api_hash: str, birdeye_api_key: str, session_name: str = "phoenix"):
        """
        Initialize the Telegram scraper.
        
        Args:
            api_id (str): Telegram API ID
            api_hash (str): Telegram API hash
            birdeye_api_key (str): Birdeye API key
            session_name (str): Session name for Telethon
        """
        self.api_id = api_id
        self.api_hash = api_hash
        self.birdeye_api_key = birdeye_api_key
        self.session_name = session_name
        self.client = None
        
        # Birdeye API headers
        self.birdeye_headers = {
            "X-API-KEY": birdeye_api_key,
            "Content-Type": "application/json"
        }
        self.birdeye_base = "https://public-api.birdeye.so"
    
    async def connect(self) -> None:
        """Connect to Telegram API."""
        if not self.client:
            logger.info("Connecting to Telegram...")
            self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
            await self.client.start()
            logger.info("Connected to Telegram successfully")
    
    async def disconnect(self) -> None:
        """Disconnect from Telegram API."""
        if self.client:
            logger.info("Disconnecting from Telegram...")
            await self.client.disconnect()
            self.client = None
            logger.info("Disconnected from Telegram")
    
    def extract_contracts(self, text: str) -> List[str]:
        """Extract Solana contract addresses from text."""
        contracts = set()
        
        # Solana-only patterns
        patterns = [
            # Standard Solana address pattern
            r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b',
            # URLs containing Solana addresses
            r'dexscreener\.com/solana/([1-9A-HJ-NP-Za-km-z]{32,44})',
            r'birdeye\.so/token/([1-9A-HJ-NP-Za-km-z]{32,44})',
            r'solscan\.io/token/([1-9A-HJ-NP-Za-km-z]{32,44})',
            # Contract mentions
            r'Contract[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
            r'CA[:\s]+([1-9A-HJ-NP-Za-km-z]{32,44})',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if match.groups():
                    addr = match.group(1)
                else:
                    addr = match.group(0)
                
                # Validate Solana address (no 0x prefix, correct length)
                if not addr.startswith('0x') and 32 <= len(addr) <= 44:
                    # Additional validation: Solana addresses use base58
                    if re.match(r'^[1-9A-HJ-NP-Za-km-z]+$', addr):
                        contracts.add(addr)
        
        return list(contracts)
    
    def extract_kols(self, text: str) -> List[str]:
        """Extract KOL usernames from text."""
        kols = set()
        
        # KOL patterns based on SpyDefi format
        patterns = [
            # Achievement format: "@username made a x2+ call"
            r'@(\w+)\s+made\s+a\s+x\d+\+?\s+call',
            # General @mentions
            r'@(\w+)',
            # Channel mentions
            r'(\w+)\s+made\s+a\s+x\d+\+?\s+call',
            # Achievement unlocked mentions
            r'Achievement\s+Unlocked.*?(@?\w+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    username = match.group(1).lower().strip('@')
                    # Filter out common words
                    if username not in ['spydefi', 'everyone', 'here', 'channel', 'group', 'unlocked', 'achievement']:
                        kols.add(username)
        
        return list(kols)
    
    def get_price_at_unix_time(self, contract: str, unix_timestamp: int) -> Optional[float]:
        """
        Get token price at specific UNIX timestamp using Birdeye.
        
        Args:
            contract (str): Token contract address
            unix_timestamp (int): UNIX timestamp
            
        Returns:
            Optional[float]: Price at that time or None
        """
        try:
            # Convert to milliseconds for Birdeye API
            timestamp_ms = unix_timestamp * 1000
            
            # Get historical price data around that time
            url = f"{self.birdeye_base}/defi/history_price"
            params = {
                'address': contract,
                'address_type': 'token',
                'type': '5m',  # 5-minute intervals
                'time_from': timestamp_ms - (5 * 60 * 1000),  # 5 min before
                'time_to': timestamp_ms + (5 * 60 * 1000)     # 5 min after
            }
            
            response = requests.get(url, headers=self.birdeye_headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('success') and data.get('data', {}).get('items'):
                items = data['data']['items']
                if items:
                    # Find closest price to our timestamp
                    closest_item = min(items, key=lambda x: abs(x['unixTime'] - timestamp_ms))
                    return float(closest_item['value'])
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting price for {contract} at {unix_timestamp}: {str(e)}")
            return None
    
    def get_historical_ath_atl(self, contract: str, from_unix: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Get historical ATH, ATL, and current price since specific UNIX time.
        
        Args:
            contract (str): Token contract address
            from_unix (int): Starting UNIX timestamp
            
        Returns:
            Tuple[Optional[float], Optional[float], Optional[float]]: (ATH, ATL, Current Price)
        """
        try:
            # Get current price first
            current_url = f"{self.birdeye_base}/defi/price"
            current_params = {'address': contract}
            
            current_response = requests.get(current_url, headers=self.birdeye_headers, params=current_params, timeout=10)
            current_response.raise_for_status()
            
            current_data = current_response.json()
            current_price = None
            if current_data.get('success') and current_data.get('data'):
                current_price = float(current_data['data']['value'])
            
            # Get historical data from call time to now
            from_ms = from_unix * 1000
            to_ms = int(datetime.now().timestamp() * 1000)
            
            hist_url = f"{self.birdeye_base}/defi/history_price"
            hist_params = {
                'address': contract,
                'address_type': 'token',
                'type': '15m',  # 15-minute intervals for better coverage
                'time_from': from_ms,
                'time_to': to_ms
            }
            
            hist_response = requests.get(hist_url, headers=self.birdeye_headers, params=hist_params, timeout=15)
            hist_response.raise_for_status()
            
            hist_data = hist_response.json()
            
            ath_price = current_price  # Default to current
            atl_price = current_price  # Default to current
            
            if hist_data.get('success') and hist_data.get('data', {}).get('items'):
                prices = []
                for item in hist_data['data']['items']:
                    if 'value' in item and item['value'] > 0:
                        prices.append(float(item['value']))
                
                if prices:
                    ath_price = max(prices)
                    atl_price = min(prices)
            
            return ath_price, atl_price, current_price
            
        except Exception as e:
            logger.warning(f"Error getting historical data for {contract}: {str(e)}")
            return None, None, None
    
    async def scan_spydefi_channel(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Scan SpyDefi channel and analyze KOL performance.
        
        Args:
            hours_back (int): Hours to scan back
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info(f"Scanning SpyDefi channel (past {hours_back}h)...")
        
        if not self.client:
            await self.connect()
        
        try:
            # Try different SpyDefi entity variations
            entity = None
            possible_names = ["SpyDefi", "spydefi", "@SpyDefi", "@spydefi"]
            
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
            
            # Increase time range for testing
            time_limit = datetime.now(timezone.utc) - timedelta(hours=hours_back * 2)  # Double the time range
            logger.info(f"Time limit: {time_limit} (expanded range for testing)")
            
            # Collect all token calls
            token_calls = []
            kol_mentions = defaultdict(list)
            message_count = 0
            
            logger.info("Starting message iteration...")
            async for message in self.client.iter_messages(entity, offset_date=time_limit):
                message_count += 1
                
                if message.date < time_limit:
                    logger.info(f"Reached time limit at message {message_count}")
                    break
                
                if not message.message:
                    continue
                
                text = message.message
                unix_time = int(message.date.timestamp())
                
                # Debug logging for first few messages
                if message_count <= 5:
                    logger.info(f"Message {message_count}: {text[:100]}...")
                
                # Extract contracts and KOLs
                contracts = self.extract_contracts(text)
                kols = self.extract_kols(text)
                
                # Debug logging for any found items
                if contracts or kols:
                    logger.info(f"Message {message_count} - Contracts: {contracts}, KOLs: {kols}")
                
                # Create call entries
                for contract in contracts:
                    call_entry = {
                        'contract': contract,
                        'unix_time': unix_time,
                        'date': message.date.replace(tzinfo=timezone.utc).isoformat(),
                        'text': text[:200],  # First 200 chars for context
                        'kols': kols
                    }
                    token_calls.append(call_entry)
                    
                    # Track KOL mentions
                    for kol in kols:
                        kol_mentions[kol].append(call_entry)
            
            logger.info(f"Processed {message_count} total messages")
            logger.info(f"Found {len(token_calls)} token calls from {len(kol_mentions)} KOLs")
            
            # Analyze each token call performance
            analyzed_calls = []
            successful_2x = 0
            successful_5x = 0
            
            for i, call in enumerate(token_calls):
                logger.info(f"Analyzing call {i+1}/{len(token_calls)}: {call['contract'][:8]}...")
                
                # Get price at call time
                call_price = self.get_price_at_unix_time(call['contract'], call['unix_time'])
                
                if call_price and call_price > 0:
                    # Get ATH, ATL, and current price since call
                    ath, atl, current = self.get_historical_ath_atl(call['contract'], call['unix_time'])
                    
                    if ath and atl:
                        # Calculate performance metrics
                        ath_roi = ((ath / call_price) - 1) * 100 if call_price > 0 else 0
                        current_roi = ((current / call_price) - 1) * 100 if call_price and current and call_price > 0 else 0
                        max_drawdown = ((atl / call_price) - 1) * 100 if call_price > 0 else 0
                        
                        # Count successful calls
                        if ath_roi >= 100:  # 2x or more
                            successful_2x += 1
                        if ath_roi >= 400:  # 5x or more
                            successful_5x += 1
                        
                        analyzed_call = call.copy()
                        analyzed_call.update({
                            'call_price': call_price,
                            'ath_price': ath,
                            'atl_price': atl,
                            'current_price': current,
                            'ath_roi_percent': round(ath_roi, 2),
                            'current_roi_percent': round(current_roi, 2),
                            'max_drawdown_percent': round(max_drawdown, 2),
                            'is_2x_plus': ath_roi >= 100,
                            'is_5x_plus': ath_roi >= 400
                        })
                        
                        analyzed_calls.append(analyzed_call)
                
                # Small delay to respect API limits
                time.sleep(0.1)
            
            # Calculate overall performance
            total_calls = len(analyzed_calls)
            success_rate_2x = (successful_2x / total_calls * 100) if total_calls > 0 else 0
            success_rate_5x = (successful_5x / total_calls * 100) if total_calls > 0 else 0
            
            # Generate KOL performance breakdown
            kol_performance = {}
            for kol, kol_calls in kol_mentions.items():
                kol_analyzed = [c for c in analyzed_calls if kol in c.get('kols', [])]
                if kol_analyzed:
                    kol_2x = sum(1 for c in kol_analyzed if c.get('is_2x_plus', False))
                    kol_5x = sum(1 for c in kol_analyzed if c.get('is_5x_plus', False))
                    kol_total = len(kol_analyzed)
                    
                    kol_performance[kol] = {
                        'tokens_mentioned': kol_total,
                        'tokens_2x_plus': kol_2x,
                        'tokens_5x_plus': kol_5x,
                        'success_rate_2x': round((kol_2x / kol_total * 100) if kol_total > 0 else 0, 2),
                        'success_rate_5x': round((kol_5x / kol_total * 100) if kol_total > 0 else 0, 2),
                        'avg_ath_roi': round(sum(c.get('ath_roi_percent', 0) for c in kol_analyzed) / kol_total if kol_total > 0 else 0, 2)
                    }
            
            # Sort KOLs by performance
            sorted_kols = sorted(kol_performance.items(), 
                               key=lambda x: (x[1]['success_rate_2x'], x[1]['avg_ath_roi']), 
                               reverse=True)
            
            logger.info("Analysis Complete!")
            logger.info(f"Tokens mentioned: {total_calls}")
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
                'kol_performance': dict(sorted_kols),
                'summary': {
                    'tokens_mentioned': total_calls,
                    'tokens_that_made_x2': successful_2x,
                    'tokens_that_made_x5': successful_5x,
                    'success_rate_2x_percent': f"{success_rate_2x:.2f}%",
                    'success_rate_5x_percent': f"{success_rate_5x:.2f}%"
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
    
    async def export_analysis_results(self, analysis: Dict[str, Any], output_file: str) -> None:
        """
        Export analysis results to CSV files.
        
        Args:
            analysis (Dict[str, Any]): Analysis results
            output_file (str): Output file path
        """
        if not analysis.get('analyzed_calls'):
            logger.warning("No analyzed calls to export")
            return
        
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            
            # Export detailed calls
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'contract', 'date', 'unix_time', 'kols', 'call_price', 
                    'ath_price', 'current_price', 'ath_roi_percent', 
                    'current_roi_percent', 'max_drawdown_percent', 
                    'is_2x_plus', 'is_5x_plus', 'text'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for call in analysis['analyzed_calls']:
                    row = {field: call.get(field, '') for field in fieldnames}
                    row['kols'] = ', '.join(call.get('kols', []))
                    writer.writerow(row)
            
            logger.info(f"Exported {len(analysis['analyzed_calls'])} calls to {output_file}")
            
            # Export KOL performance summary
            kol_file = output_file.replace('.csv', '_kol_performance.csv')
            with open(kol_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'kol', 'tokens_mentioned', 'tokens_2x_plus', 'tokens_5x_plus',
                    'success_rate_2x', 'success_rate_5x', 'avg_ath_roi'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for kol, performance in analysis.get('kol_performance', {}).items():
                    row = {'kol': kol}
                    row.update(performance)
                    writer.writerow(row)
            
            logger.info(f"Exported KOL performance to {kol_file}")
            
            # Export summary
            summary_file = output_file.replace('.csv', '_summary.txt')
            with open(summary_file, 'w', encoding='utf-8') as f:
                summary = analysis.get('summary', {})
                f.write("=== SPYDEFI PERFORMANCE ANALYSIS ===\n\n")
                f.write(f"Analysis period: {analysis.get('scan_period_hours', 24)} hours\n")
                f.write(f"Tokens mentioned: {summary.get('tokens_mentioned', 0)}\n")
                f.write(f"Tokens that made x2: {summary.get('tokens_that_made_x2', 0)}\n")
                f.write(f"Tokens that made x5: {summary.get('tokens_that_made_x5', 0)}\n")
                f.write(f"Success rate (2x): {summary.get('success_rate_2x_percent', '0.00%')}\n")
                f.write(f"Success rate (5x): {summary.get('success_rate_5x_percent', '0.00%')}\n\n")
                
                f.write("=== TOP KOLs ===\n")
                for i, (kol, perf) in enumerate(list(analysis.get('kol_performance', {}).items())[:10]):
                    f.write(f"{i+1}. @{kol}\n")
                    f.write(f"   Tokens mentioned: {perf['tokens_mentioned']}\n")
                    f.write(f"   Tokens that made x2: {perf['tokens_2x_plus']}\n")
                    f.write(f"   Success rate: {perf['success_rate_2x']}%\n\n")
            
            logger.info(f"Exported summary to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")

    # Legacy methods for backward compatibility
    async def get_channel_messages(self, channel_id: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Legacy method - redirects to new scan method."""
        if channel_id.lower() == "spydefi":
            analysis = await self.scan_spydefi_channel(days_back * 24)
            return analysis.get('analyzed_calls', [])
        return []
    
    async def analyze_channel(self, channel_id: str, days_back: int = 7, birdeye_api: Any = None) -> Dict[str, Any]:
        """Legacy method - redirects to new scan method.""" 
        if channel_id.lower() == "spydefi":
            return await self.scan_spydefi_channel(days_back * 24)
        return {'channel_id': channel_id, 'total_calls': 0, 'success_rate': 0}
    
    async def scrape_spydefi(self, channel_id: str, days_back: int = 7, birdeye_api: Any = None) -> Dict[str, Any]:
        """Legacy method - redirects to new scan method."""
        return await self.scan_spydefi_channel(days_back * 24)
    
    async def export_channel_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Legacy method - redirects to new export method."""
        await self.export_analysis_results(analysis, output_file)
    
    async def export_spydefi_analysis(self, analysis: Dict[str, Any], output_file: str) -> None:
        """Legacy method - redirects to new export method."""
        await self.export_analysis_results(analysis, output_file)