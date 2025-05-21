#!/usr/bin/env python3
"""
Phoenix Project - Solana Chain Analysis CLI Tool

A command-line interface tool for analyzing Solana blockchain signals and wallet behaviors
using Birdeye API and data from Telegram groups.
"""

import os
import sys
import argparse
import json
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phoenix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("phoenix")

# Configuration
CONFIG_FILE = os.path.expanduser("~/.phoenix_config.json")

def load_config() -> Dict[str, Any]:
    """Load configuration from the config file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "birdeye_api_key": "",
        "telegram_api_id": "",
        "telegram_api_hash": "",
        "telegram_session": "",
        "sources": {
            "telegram_groups": [],
            "wallets": []
        },
        "analysis_period_days": 7
    }

def ensure_output_dir(output_path: str) -> str:
    """
    Ensure the output directory exists and return the full path.
    
    Args:
        output_path (str): Relative or absolute output path
        
    Returns:
        str: Full path with output directory
    """
    # Create outputs directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created outputs directory: {output_dir}")
    
    # If path doesn't include directory, add outputs dir
    if not os.path.dirname(output_path):
        return os.path.join(output_dir, output_path)
    
    return output_path

def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to the config file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

class PhoenixCLI:
    """Main CLI class for the Phoenix Project."""
    
    def __init__(self):
        self.config = load_config()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="Phoenix Project - Solana Chain Analysis CLI Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Command")
        
        # Configure command
        configure_parser = subparsers.add_parser("configure", help="Configure API keys and sources")
        configure_parser.add_argument("--birdeye-api-key", help="Birdeye Solana API key")
        configure_parser.add_argument("--telegram-api-id", help="Telegram API ID")
        configure_parser.add_argument("--telegram-api-hash", help="Telegram API hash")
        
        # Telegram analysis command
        telegram_parser = subparsers.add_parser("telegram", help="Analyze Telegram channels")
        telegram_parser.add_argument("--channels", nargs="+", help="Telegram channel IDs to analyze")
        telegram_parser.add_argument("--days", type=int, default=7, help="Number of days to analyze")
        telegram_parser.add_argument("--output", default="telegram_analysis.csv", help="Output CSV file")
        telegram_parser.add_argument("--excel", action="store_true", help="Export results to Excel format")
        
        # Wallet analysis command
        wallet_parser = subparsers.add_parser("wallet", help="Analyze wallets for copy trading")
        wallet_parser.add_argument("--wallets", nargs="+", help="Wallet addresses to analyze")
        wallet_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
        wallet_parser.add_argument("--min-winrate", type=float, default=45.0, help="Minimum win rate percentage")
        wallet_parser.add_argument("--output", default="wallet_analysis.csv", help="Output CSV file")
        wallet_parser.add_argument("--excel", action="store_true", help="Export results to Excel format")
        
        # Add source command
        add_source_parser = subparsers.add_parser("add-source", help="Add a new data source")
        add_source_parser.add_argument("type", choices=["telegram", "wallet"], help="Source type")
        add_source_parser.add_argument("identifier", help="Channel ID or wallet address")
        
        # Combined analysis command (new)
        combined_parser = subparsers.add_parser("analyze", help="Run combined Telegram and wallet analysis")
        combined_parser.add_argument("--telegram-days", type=int, default=7, help="Number of days for Telegram analysis")
        combined_parser.add_argument("--wallet-days", type=int, default=30, help="Number of days for wallet analysis")
        combined_parser.add_argument("--min-winrate", type=float, default=45.0, help="Minimum win rate percentage for wallets")
        combined_parser.add_argument("--output", default="phoenix_analysis.xlsx", help="Output Excel file")
        
        return parser
    
    def run(self) -> None:
        """Run the CLI application."""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        # Execute the appropriate command
        if args.command == "configure":
            self._handle_configure(args)
        elif args.command == "telegram":
            self._handle_telegram_analysis(args)
        elif args.command == "wallet":
            self._handle_wallet_analysis(args)
        elif args.command == "add-source":
            self._handle_add_source(args)
        elif args.command == "analyze":
            self._handle_combined_analysis(args)
    
    def _handle_configure(self, args: argparse.Namespace) -> None:
        """Handle the configure command."""
        if args.birdeye_api_key:
            self.config["birdeye_api_key"] = args.birdeye_api_key
            logger.info("Birdeye API key configured.")
        
        if args.telegram_api_id:
            self.config["telegram_api_id"] = args.telegram_api_id
            logger.info("Telegram API ID configured.")
        
        if args.telegram_api_hash:
            self.config["telegram_api_hash"] = args.telegram_api_hash
            logger.info("Telegram API hash configured.")
        
        save_config(self.config)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
    
    def _handle_telegram_analysis(self, args: argparse.Namespace) -> None:
        """Handle the telegram analysis command."""
        import asyncio
        from telegram_module import TelegramScraper
        from birdeye_api import BirdeyeAPI
        
        channels = args.channels or self.config["sources"]["telegram_groups"]
        if not channels:
            logger.error("No Telegram channels specified. Use --channels or add sources with add-source command.")
            return
        
        # Check if we have the necessary API keys
        if not self.config.get("birdeye_api_key"):
            logger.error("Birdeye API key not configured. Run 'phoenix configure --birdeye-api-key YOUR_KEY'")
            return
            
        if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
            logger.error("Telegram API credentials not configured. Run 'phoenix configure --telegram-api-id ID --telegram-api-hash HASH'")
            return
        
        # Ensure output directory exists and get full path
        output_file = ensure_output_dir(args.output)
        
        logger.info(f"Analyzing {len(channels)} Telegram channels for the past {args.days} days.")
        logger.info(f"Results will be saved to {output_file}")
        
        # Initialize API clients
        birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
        telegram_scraper = TelegramScraper(
            self.config["telegram_api_id"],
            self.config["telegram_api_hash"],
            self.config.get("telegram_session", "phoenix")
        )
        
        # Store all analysis results
        telegram_analyses = {"ranked_kols": []}
        
        # Handle special case for Spydefi
        if any(ch.lower() == "spydefi" for ch in channels):
            logger.info("Spydefi channel detected. Analyzing KOLs from Spydefi.")
            
            try:
                # Find the Spydefi channel ID
                spydefi_channel = next((ch for ch in channels if ch.lower() == "spydefi"), None)
                
                # Run the analysis asynchronously
                async def run_spydefi_analysis():
                    try:
                        await telegram_scraper.connect()
                        analysis = await telegram_scraper.scrape_spydefi(
                            spydefi_channel,
                            args.days,
                            birdeye_api
                        )
                        await telegram_scraper.export_spydefi_analysis(analysis, output_file)
                        return analysis
                    finally:
                        await telegram_scraper.disconnect()
                
                # Run the async function
                telegram_analyses = asyncio.run(run_spydefi_analysis())
                logger.info(f"Spydefi analysis completed. Results saved to {output_file}")
                
            except Exception as e:
                logger.error(f"Error analyzing Spydefi: {str(e)}", exc_info=True)
                
        else:
            # Regular channel analysis
            channel_analyses = []
            for channel in channels:
                try:
                    # Run the analysis asynchronously
                    async def run_channel_analysis():
                        try:
                            await telegram_scraper.connect()
                            analysis = await telegram_scraper.analyze_channel(
                                channel,
                                args.days,
                                birdeye_api
                            )
                            
                            # Customize output file for each channel
                            channel_output = output_file
                            if len(channels) > 1:
                                base, ext = os.path.splitext(output_file)
                                channel_output = f"{base}_{channel}{ext}"
                            
                            await telegram_scraper.export_channel_analysis(analysis, channel_output)
                            return analysis
                        finally:
                            await telegram_scraper.disconnect()
                    
                    # Run the async function
                    analysis = asyncio.run(run_channel_analysis())
                    channel_analyses.append(analysis)
                    logger.info(f"Analysis for channel {channel} completed")
                    
                except Exception as e:
                    logger.error(f"Error analyzing channel {channel}: {str(e)}", exc_info=True)
            
            telegram_analyses = {
                "ranked_kols": channel_analyses
            }
        
        logger.info(f"Telegram analysis completed. Results saved to {output_file} and related files.")
        
        # Export to Excel if requested
        if args.excel:
            try:
                from export_utils import export_to_excel
                excel_file = output_file.replace(".csv", ".xlsx")
                export_to_excel(telegram_analyses, {}, excel_file)
                logger.info(f"Excel export completed: {excel_file}")
            except Exception as e:
                logger.error(f"Error exporting to Excel: {str(e)}", exc_info=True)
    
    def _handle_wallet_analysis(self, args: argparse.Namespace) -> None:
        """Handle the wallet analysis command."""
        from wallet_module import WalletAnalyzer
        from birdeye_api import BirdeyeAPI
        
        wallets = args.wallets or self.config["sources"]["wallets"]
        if not wallets:
            logger.error("No wallets specified. Use --wallets or add sources with add-source command.")
            return
        
        # Check if we have the necessary API key
        if not self.config.get("birdeye_api_key"):
            logger.error("Birdeye API key not configured. Run 'phoenix configure --birdeye-api-key YOUR_KEY'")
            return
        
        # Ensure output directory exists and get full path
        output_file = ensure_output_dir(args.output)
        
        logger.info(f"Analyzing {len(wallets)} wallets for the past {args.days} days.")
        logger.info(f"Minimum win rate: {args.min_winrate}%")
        logger.info(f"Results will be saved to {output_file}")
        
        # Initialize API client and wallet analyzer
        birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
        wallet_analyzer = WalletAnalyzer(birdeye_api)
        
        try:
            # Batch analyze wallets
            if len(wallets) > 1:
                logger.info("Performing batch wallet analysis...")
                wallet_analyses = wallet_analyzer.batch_analyze_wallets(
                    wallets,
                    args.days,
                    args.min_winrate
                )
                
                wallet_analyzer.export_batch_analysis(wallet_analyses, output_file)
                logger.info(f"Batch analysis completed. Results saved to {output_file} and related files.")
                
                # Print summary
                logger.info(f"Analysis Summary:")
                logger.info(f"- Total wallets: {wallet_analyses['total_wallets']}")
                logger.info(f"- Analyzed wallets: {wallet_analyses['analyzed_wallets']}")
                logger.info(f"- Filtered wallets: {wallet_analyses['filtered_wallets']}")
                logger.info(f"- Gem Finders: {len(wallet_analyses['gem_finders'])}")
                logger.info(f"- Consistent: {len(wallet_analyses['consistent'])}")
                logger.info(f"- Flippers: {len(wallet_analyses['flippers'])}")
                
                # Export to Excel if requested
                if args.excel:
                    try:
                        from export_utils import export_to_excel
                        excel_file = output_file.replace(".csv", ".xlsx")
                        export_to_excel({}, wallet_analyses, excel_file)
                        logger.info(f"Excel export completed: {excel_file}")
                    except Exception as e:
                        logger.error(f"Error exporting to Excel: {str(e)}", exc_info=True)
                
            # Analyze single wallet
            else:
                logger.info(f"Analyzing wallet {wallets[0]}...")
                wallet_analysis = wallet_analyzer.analyze_wallet(wallets[0], args.days)
                
                if wallet_analysis.get("success"):
                    wallet_analyzer.export_wallet_analysis(wallet_analysis, output_file)
                    
                    # Print summary
                    metrics = wallet_analysis["metrics"]
                    logger.info(f"Analysis completed for {wallets[0]}")
                    logger.info(f"- Wallet type: {wallet_analysis['wallet_type']}")
                    logger.info(f"- Win rate: {metrics['win_rate']:.2f}%")
                    logger.info(f"- Total trades: {metrics['total_trades']}")
                    logger.info(f"- ROI stats: median={metrics['median_roi']:.2f}%, max={metrics['max_roi']:.2f}%")
                    logger.info(f"Results saved to {output_file} and related files.")
                else:
                    logger.error(f"Analysis failed: {wallet_analysis.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error during wallet analysis: {str(e)}", exc_info=True)
    
    def _handle_add_source(self, args: argparse.Namespace) -> None:
        """Handle the add source command."""
        if args.type == "telegram":
            if args.identifier not in self.config["sources"]["telegram_groups"]:
                self.config["sources"]["telegram_groups"].append(args.identifier)
                logger.info(f"Added Telegram group/channel: {args.identifier}")
            else:
                logger.info(f"Telegram group/channel already exists: {args.identifier}")
        elif args.type == "wallet":
            if args.identifier not in self.config["sources"]["wallets"]:
                self.config["sources"]["wallets"].append(args.identifier)
                logger.info(f"Added wallet: {args.identifier}")
            else:
                logger.info(f"Wallet already exists: {args.identifier}")
        
        save_config(self.config)
    
    def _handle_combined_analysis(self, args: argparse.Namespace) -> None:
        """Handle the combined analysis command."""
        import asyncio
        from telegram_module import TelegramScraper
        from wallet_module import WalletAnalyzer
        from birdeye_api import BirdeyeAPI
        from export_utils import export_to_excel
        
        # Check if we have the necessary configurations
        if not self.config.get("birdeye_api_key"):
            logger.error("Birdeye API key not configured. Run 'phoenix configure --birdeye-api-key YOUR_KEY'")
            return
            
        if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
            logger.error("Telegram API credentials not configured. Run 'phoenix configure --telegram-api-id ID --telegram-api-hash HASH'")
            return
        
        # Ensure output directory exists and get full path
        output_file = ensure_output_dir(args.output)
        
        telegram_groups = self.config["sources"]["telegram_groups"]
        wallets = self.config["sources"]["wallets"]
        
        if not telegram_groups:
            logger.warning("No Telegram groups configured. Add some with 'phoenix add-source telegram CHANNEL_ID'")
        
        if not wallets:
            logger.warning("No wallets configured. Add some with 'phoenix add-source wallet ADDRESS'")
        
        if not telegram_groups and not wallets:
            logger.error("No sources configured. Cannot perform analysis.")
            return
        
        logger.info(f"Starting combined analysis. Results will be saved to {output_file}")
        
        # Initialize API clients
        birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
        
        # Run Telegram analysis
        telegram_analyses = {"ranked_kols": []}
        if telegram_groups:
            logger.info(f"Analyzing {len(telegram_groups)} Telegram channels for the past {args.telegram_days} days.")
            
            telegram_scraper = TelegramScraper(
                self.config["telegram_api_id"],
                self.config["telegram_api_hash"],
                self.config.get("telegram_session", "phoenix")
            )
            
            # Handle special case for Spydefi
            if any(ch.lower() == "spydefi" for ch in telegram_groups):
                logger.info("Spydefi channel detected. Analyzing KOLs from Spydefi.")
                
                try:
                    # Find the Spydefi channel ID
                    spydefi_channel = next((ch for ch in telegram_groups if ch.lower() == "spydefi"), None)
                    
                    # Run the analysis asynchronously
                    async def run_spydefi_analysis():
                        try:
                            await telegram_scraper.connect()
                            return await telegram_scraper.scrape_spydefi(
                                spydefi_channel,
                                args.telegram_days,
                                birdeye_api
                            )
                        finally:
                            await telegram_scraper.disconnect()
                    
                    # Run the async function
                    telegram_analyses = asyncio.run(run_spydefi_analysis())
                    logger.info(f"Spydefi analysis completed.")
                    
                except Exception as e:
                    logger.error(f"Error analyzing Spydefi: {str(e)}", exc_info=True)
            else:
                # Regular channel analysis
                channel_analyses = []
                for channel in telegram_groups:
                    try:
                        # Run the analysis asynchronously
                        async def run_channel_analysis():
                            try:
                                await telegram_scraper.connect()
                                return await telegram_scraper.analyze_channel(
                                    channel,
                                    args.telegram_days,
                                    birdeye_api
                                )
                            finally:
                                await telegram_scraper.disconnect()
                        
                        # Run the async function
                        analysis = asyncio.run(run_channel_analysis())
                        channel_analyses.append(analysis)
                        logger.info(f"Analysis for channel {channel} completed")
                        
                    except Exception as e:
                        logger.error(f"Error analyzing channel {channel}: {str(e)}", exc_info=True)
                
                telegram_analyses = {
                    "ranked_kols": channel_analyses
                }
            
            logger.info(f"Telegram analysis completed.")
        
        # Run wallet analysis
        wallet_analyses = {}
        if wallets:
            logger.info(f"Analyzing {len(wallets)} wallets for the past {args.wallet_days} days.")
            
            wallet_analyzer = WalletAnalyzer(birdeye_api)
            
            try:
                wallet_analyses = wallet_analyzer.batch_analyze_wallets(
                    wallets,
                    args.wallet_days,
                    args.min_winrate
                )
                
                logger.info(f"Wallet analysis completed.")
                logger.info(f"- Gem Finders: {len(wallet_analyses['gem_finders'])}")
                logger.info(f"- Consistent: {len(wallet_analyses['consistent'])}")
                logger.info(f"- Flippers: {len(wallet_analyses['flippers'])}")
                
            except Exception as e:
                logger.error(f"Error during wallet analysis: {str(e)}", exc_info=True)
        
        # Export combined results to Excel
        try:
            export_to_excel(telegram_analyses, wallet_analyses, output_file)
            logger.info(f"Combined analysis exported to Excel: {output_file}")
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}", exc_info=True)

def main():
    """Main entry point for the Phoenix CLI."""
    try:
        cli = PhoenixCLI()
        cli.run()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()