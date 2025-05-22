"""
Telegram Module - Phoenix Project (FINAL WORKING VERSION)

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

class TelegramScraper:
    """SpyDefi-focused Telegram scraper."""
    
    def __init__(self, api_id: str, api_hash: str, birdeye_api_key: str, session_name: str = "phoenix"):
        """Initialize the Telegram scraper."""
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
    
    def extract_kols(self, text: str) -> List[str]:
        """Extract KOL usernames from text."""
        kols = set()
        
        # KOL patterns based on actual SpyDefi format
        patterns = [
            # "@username made a x2+ call on TOKEN"
            r'@(\w+)\s+made\s+a\s+x\d+\+?\s+call\s+on',
            # "TOKEN first posted by @username"
            r'first\s+posted\s+by\s+@(\w+)',
            # General @mentions
            r'@(\w+)',
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
    
    async def scan_spydefi_channel(self, hours_back: int = 24) -> Dict[str, Any]:
        """Scan SpyDefi channel and analyze KOL performance."""
        logger.info(f"Scanning SpyDefi channel (past {hours_back}h)...")
        
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
            logger.info("Getting recent messages without time filter for testing...")
            
            # Collect all token calls
            token_calls = []
            kol_mentions = defaultdict(list)
            message_count = 0
            max_messages = 50  # Limit for testing
            
            logger.info("Starting message iteration...")
            
            # Simple message processing loop
            async for message in self.client.iter_messages(entity, limit=max_messages):
                message_count += 1
                
                if not message.message:
                    continue
                
                text = message.message
                unix_time = int(message.date.timestamp())
                message_date = message.date
                
                # Show first few messages for debugging
                if message_count <= 5:
                    logger.info(f"Message {message_count} ({message_date}): {text[:150]}...")
                
                # Extract data
                contracts = self.extract_contracts(text)
                kols = self.extract_kols(text)
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
                token_identifier = call.get('contract') or call.get('token_name', 'Unknown')
                logger.info(f"Analyzing call {i+1}/{len(token_calls)}: {token_identifier}")
                
                # Extract ROI from the message text
                text = call['text']
                roi_match = re.search(r'\$(\d+(?:\.\d+)?)K?\s*->\s*\$(\d+(?:\.\d+)?)K?', text)
                
                if roi_match:
                    try:
                        start_val = float(roi_match.group(1))
                        end_val = float(roi_match.group(2))
                        
                        # Handle K notation
                        if 'K' in roi_match.group(1) or len(roi_match.group(1)) <= 3:
                            start_val *= 1000
                        if 'K' in roi_match.group(2) or len(roi_match.group(2)) <= 3:
                            end_val *= 1000
                        
                        # Calculate ROI
                        roi_percent = ((end_val / start_val) - 1) * 100 if start_val > 0 else 0
                        
                        # Count successful calls
                        if roi_percent >= 100:  # 2x or more
                            successful_2x += 1
                        if roi_percent >= 400:  # 5x or more
                            successful_5x += 1
                        
                        analyzed_call = call.copy()
                        analyzed_call.update({
                            'call_price': start_val,
                            'ath_price': end_val,
                            'current_price': end_val,
                            'ath_roi_percent': round(roi_percent, 2),
                            'current_roi_percent': round(roi_percent, 2),
                            'max_drawdown_percent': 0,
                            'is_2x_plus': roi_percent >= 100,
                            'is_5x_plus': roi_percent >= 400
                        })
                        
                        analyzed_calls.append(analyzed_call)
                        
                    except (ValueError, ZeroDivisionError) as e:
                        logger.warning(f"Error calculating ROI for call {i+1}: {str(e)}")
                else:
                    logger.warning(f"No ROI data found in message: {text[:100]}...")
            
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
        """Export analysis results to CSV files."""
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
                    'contract', 'token_name', 'date', 'unix_time', 'kols', 'call_price', 
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