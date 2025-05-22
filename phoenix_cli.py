#!/usr/bin/env python3
"""
Phoenix Project - Solana Chain Analysis CLI Tool (Updated for Cielo Finance API)

A command-line interface tool for analyzing Solana blockchain signals and wallet behaviors
using Cielo Finance API and data from Telegram groups.
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
    """Load configuration from the config file and text files."""
    # Load from JSON config file
    config = {
        "birdeye_api_key": "",  # Back to Birdeye for Telegram
        "cielo_finance_api_key": "",  # Cielo for wallets only
        "telegram_api_id": "",
        "telegram_api_hash": "",
        "telegram_session": "",
        "sources": {
            "telegram_groups": [],
            "wallets": []
        },
        "analysis_period_days": 7,
        "api_provider": "mixed"  # Mixed: Birdeye + Cielo
    }
    
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config.update(json.load(f))
    
    # Load from text files if they exist
    channels_file = "channels.txt"
    wallets_file = "wallets.txt"
    
    # Load Telegram channels from channels.txt
    if os.path.exists(channels_file):
        try:
            with open(channels_file, 'r') as f:
                channels = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
                if channels:
                    config["sources"]["telegram_groups"] = channels
                    logger.info(f"Loaded {len(channels)} Telegram channels from {channels_file}")
        except Exception as e:
            logger.warning(f"Error reading {channels_file}: {str(e)}")
    
    # Load wallets from wallets.txt
    if os.path.exists(wallets_file):
        try:
            with open(wallets_file, 'r') as f:
                wallets = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
                if wallets:
                    config["sources"]["wallets"] = wallets
                    logger.info(f"Loaded {len(wallets)} wallets from {wallets_file}")
        except Exception as e:
            logger.warning(f"Error reading {wallets_file}: {str(e)}")
    
    return config

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
            description="Phoenix Project - Solana Chain Analysis CLI Tool (Cielo Finance Edition)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False  # We'll handle help manually to show numbered menu
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Command")
        
        # Configure command
        configure_parser = subparsers.add_parser("configure", help="Configure API keys and sources")
        configure_parser.add_argument("--birdeye-api-key", help="Birdeye API key (for Telegram analysis)")
        configure_parser.add_argument("--cielo-api-key", help="Cielo Finance API key (for wallet analysis)")
        configure_parser.add_argument("--telegram-api-id", help="Telegram API ID")
        configure_parser.add_argument("--telegram-api-hash", help="Telegram API hash")
        configure_parser.add_argument("--check", action="store_true", help="Check current configuration")
        
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
        
        # Combined analysis command
        combined_parser = subparsers.add_parser("analyze", help="Run combined Telegram and wallet analysis")
        combined_parser.add_argument("--telegram-days", type=int, default=7, help="Number of days for Telegram analysis")
        combined_parser.add_argument("--wallet-days", type=int, default=30, help="Number of days for wallet analysis")
        combined_parser.add_argument("--min-winrate", type=float, default=45.0, help="Minimum win rate percentage for wallets")
        combined_parser.add_argument("--output", default="phoenix_analysis.xlsx", help="Output Excel file")
        
        # Test API command
        test_parser = subparsers.add_parser("test", help="Test API connectivity")
        test_parser.add_argument("--api", choices=["cielo", "telegram", "all"], default="all", help="Which API to test")
        
        return parser
    
    def _show_numbered_menu(self) -> None:
        """Show the numbered menu interface."""
        print("\n" + "="*60)
        print("🔥 Phoenix Project - Solana Chain Analysis Tool 🔥")
        print("         (Cielo Finance Edition)")
        print("="*60)
        print("\nSelect an option:")
        print("\n📋 CONFIGURATION:")
        print("  1. Configure API Keys")
        print("  2. Check Configuration")
        print("  3. Test API Connectivity")
        print("  4. Add Data Sources")
        
        print("\n📊 ANALYSIS:")
        print("  5. Analyze Telegram Channels")
        print("  6. Analyze Wallets")
        print("  7. Combined Analysis (Telegram + Wallets)")
        
        print("\n🔧 UTILITIES:")
        print("  8. View Current Sources")
        print("  9. Help & Examples")
        print("  0. Exit")
        
        print("\n" + "="*60)
    
    def _handle_numbered_menu(self) -> None:
        """Handle the numbered menu interface."""
        while True:
            self._show_numbered_menu()
            
            try:
                choice = input("\nEnter your choice (0-9): ").strip()
                
                if choice == "0":
                    print("\n👋 Thanks for using Phoenix Project!")
                    sys.exit(0)
                elif choice == "1":
                    self._interactive_configure()
                elif choice == "2":
                    self._handle_configure(argparse.Namespace(check=True, cielo_api_key=None, telegram_api_id=None, telegram_api_hash=None))
                elif choice == "3":
                    self._handle_test_api(argparse.Namespace(api="all"))
                elif choice == "4":
                    self._interactive_add_source()
                elif choice == "5":
                    self._interactive_telegram_analysis()
                elif choice == "6":
                    self._interactive_wallet_analysis()
                elif choice == "7":
                    self._interactive_combined_analysis()
                elif choice == "8":
                    self._show_current_sources()
                elif choice == "9":
                    self._show_help_examples()
                else:
                    print("\n[ERROR] Invalid choice. Please enter a number between 0-9.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Thanks for using Phoenix Project!")
                sys.exit(0)
            except Exception as e:
                print(f"\n[ERROR] Error: {str(e)}")
                
            input("\nPress Enter to continue...")
    
    def _interactive_configure(self) -> None:
        """Interactive configuration setup."""
        print("\n🔧 Configuration Setup")
        print("-" * 30)
        
        # Birdeye API Key
        current_birdeye = "Set" if self.config.get("birdeye_api_key") else "Not set"
        print(f"\nBirdeye API Key (for Telegram): {current_birdeye}")
        birdeye_key = input("Enter new Birdeye API key (or press Enter to skip): ").strip()
        
        # Cielo Finance API Key
        current_cielo = "Set" if self.config.get("cielo_finance_api_key") else "Not set"
        print(f"\nCielo Finance API Key (for Wallets): {current_cielo}")
        cielo_key = input("Enter new Cielo Finance API key (or press Enter to skip): ").strip()
        
        # Telegram API credentials
        current_tg_id = "Set" if self.config.get("telegram_api_id") else "Not set"
        current_tg_hash = "Set" if self.config.get("telegram_api_hash") else "Not set"
        print(f"\nTelegram API ID: {current_tg_id}")
        print(f"Telegram API Hash: {current_tg_hash}")
        
        tg_id = input("Enter Telegram API ID (or press Enter to skip): ").strip()
        tg_hash = input("Enter Telegram API Hash (or press Enter to skip): ").strip()
        
        # Apply changes
        if birdeye_key:
            self.config["birdeye_api_key"] = birdeye_key
            print("[SUCCESS] Birdeye API key updated")
        
        if cielo_key:
            self.config["cielo_finance_api_key"] = cielo_key
            print("[SUCCESS] Cielo Finance API key updated")
        
        if tg_id:
            self.config["telegram_api_id"] = tg_id
            print("[SUCCESS] Telegram API ID updated")
        
        if tg_hash:
            self.config["telegram_api_hash"] = tg_hash
            print("[SUCCESS] Telegram API Hash updated")
        
        if birdeye_key or cielo_key or tg_id or tg_hash:
            save_config(self.config)
            print(f"\n[SUCCESS] Configuration saved to {CONFIG_FILE}")
        else:
            print("\n[INFO] No changes made")
    
    def _interactive_add_source(self) -> None:
        """Interactive source addition."""
        print("\n📝 Add Data Source")
        print("-" * 20)
        print("1. Telegram Channel/Group")
        print("2. Wallet Address")
        print("0. Back to main menu")
        
        choice = input("\nSelect source type: ").strip()
        
        if choice == "1":
            identifier = input("Enter Telegram channel ID or username: ").strip()
            if identifier:
                self._handle_add_source(argparse.Namespace(type="telegram", identifier=identifier))
        elif choice == "2":
            identifier = input("Enter wallet address: ").strip()
            if identifier:
                self._handle_add_source(argparse.Namespace(type="wallet", identifier=identifier))
        elif choice == "0":
            return
        else:
            print("[ERROR] Invalid choice")
    
    def _interactive_telegram_analysis(self) -> None:
        """Interactive Telegram analysis."""
        print("\n📱 Telegram Analysis")
        print("-" * 20)
        
        channels = self.config["sources"]["telegram_groups"]
        if not channels:
            print("[ERROR] No Telegram channels configured.")
            print("Create a 'channels.txt' file with one channel per line, or use option 4 to add sources.")
            return
        
        print(f"Configured channels: {', '.join(channels)}")
        
        # Get analysis parameters
        days = input(f"Days to analyze (default 7): ").strip()
        days = int(days) if days.isdigit() else 7
        
        output = input("Output filename (default: telegram_analysis.csv): ").strip()
        output = output if output else "telegram_analysis.csv"
        
        excel = input("Export to Excel? (y/N): ").strip().lower() == 'y'
        
        # Run analysis
        args = argparse.Namespace(
            channels=channels,
            days=days,
            output=output,
            excel=excel
        )
        self._handle_telegram_analysis(args)
    
    def _interactive_wallet_analysis(self) -> None:
        """Interactive wallet analysis."""
        print("\n💼 Wallet Analysis")
        print("-" * 18)
        
        wallets = self.config["sources"]["wallets"]
        if not wallets:
            print("[ERROR] No wallets configured.")
            print("Create a 'wallets.txt' file with one wallet address per line, or use option 4 to add sources.")
            return
        
        print(f"Configured wallets: {len(wallets)} total")
        
        # Get analysis parameters
        days = input(f"Days to analyze (default 30): ").strip()
        days = int(days) if days.isdigit() else 30
        
        min_winrate = input("Minimum win rate % (default 45): ").strip()
        min_winrate = float(min_winrate) if min_winrate else 45.0
        
        output = input("Output filename (default: wallet_analysis.csv): ").strip()
        output = output if output else "wallet_analysis.csv"
        
        excel = input("Export to Excel? (y/N): ").strip().lower() == 'y'
        
        # Run analysis
        args = argparse.Namespace(
            wallets=wallets,
            days=days,
            min_winrate=min_winrate,
            output=output,
            excel=excel
        )
        self._handle_wallet_analysis(args)
    
    def _interactive_combined_analysis(self) -> None:
        """Interactive combined analysis."""
        print("\n🔄 Combined Analysis")
        print("-" * 20)
        
        telegram_groups = self.config["sources"]["telegram_groups"]
        wallets = self.config["sources"]["wallets"]
        
        if not telegram_groups and not wallets:
            print("[ERROR] No sources configured.")
            print("Create 'channels.txt' and/or 'wallets.txt' files, or use option 4 to add sources.")
            return
        
        print(f"Sources: {len(telegram_groups)} Telegram, {len(wallets)} Wallets")
        
        # Get analysis parameters
        tg_days = input(f"Telegram analysis days (default 7): ").strip()
        tg_days = int(tg_days) if tg_days.isdigit() else 7
        
        wallet_days = input(f"Wallet analysis days (default 30): ").strip()
        wallet_days = int(wallet_days) if wallet_days.isdigit() else 30
        
        min_winrate = input("Minimum win rate % (default 45): ").strip()
        min_winrate = float(min_winrate) if min_winrate else 45.0
        
        output = input("Output filename (default: combined_analysis.xlsx): ").strip()
        output = output if output else "combined_analysis.xlsx"
        
        # Run analysis
        args = argparse.Namespace(
            telegram_days=tg_days,
            wallet_days=wallet_days,
            min_winrate=min_winrate,
            output=output
        )
        self._handle_combined_analysis(args)
    
    def _show_current_sources(self) -> None:
        """Show currently configured sources."""
        print("\n📋 Current Data Sources")
        print("-" * 25)
        
        telegram_groups = self.config["sources"]["telegram_groups"]
        wallets = self.config["sources"]["wallets"]
        
        print(f"\n📱 Telegram Channels ({len(telegram_groups)}):")
        if telegram_groups:
            for i, channel in enumerate(telegram_groups, 1):
                print(f"  {i}. {channel}")
        else:
            print("  None configured")
        
        print(f"\n💼 Wallets ({len(wallets)}):")
        if wallets:
            for i, wallet in enumerate(wallets, 1):
                print(f"  {i}. {wallet[:8]}...{wallet[-8:]}")
        else:
            print("  None configured")
        
        # Show source files status
        print(f"\n📄 Source Files:")
        channels_file = "channels.txt"
        wallets_file = "wallets.txt"
        
        if os.path.exists(channels_file):
            print(f"  ✓ {channels_file} found")
        else:
            print(f"  ✗ {channels_file} not found")
        
        if os.path.exists(wallets_file):
            print(f"  ✓ {wallets_file} found")
        else:
            print(f"  ✗ {wallets_file} not found")
    
    def _show_help_examples(self) -> None:
        """Show help and examples."""
        print("\n📚 Help & Examples")
        print("-" * 20)
        print("\n🔧 CLI Commands:")
        print("  phoenix configure --cielo-api-key YOUR_KEY")
        print("  phoenix telegram --channels spydefi --days 7")
        print("  phoenix wallet --wallets ADDRESS1 ADDRESS2 --days 30")
        print("  phoenix analyze --output consolidated_analysis.xlsx")
        print("  phoenix test --api all")
        
        print("\n📱 Telegram Analysis:")
        print("  - Analyzes KOL calls and token mentions")
        print("  - Supports Spydefi channel for KOL discovery")
        print("  - Calculates success rates and ROI metrics")
        
        print("\n💼 Wallet Analysis:")
        print("  - Categorizes wallets: Gem Finders, Consistent, Flippers")
        print("  - Calculates win rates, ROI, and trading patterns")
        print("  - Generates trading strategies based on performance")
        
        print("\n🔄 Combined Analysis:")
        print("  - Runs both Telegram and Wallet analysis")
        print("  - Creates comprehensive Excel reports")
        print("  - Includes correlation analysis and clustering")
    
    def run(self) -> None:
        """Run the CLI application."""
        args = self.parser.parse_args()
        
        # If no command provided, show numbered menu
        if not args.command:
            self._handle_numbered_menu()
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
        elif args.command == "test":
            self._handle_test_api(args)
    
    def _handle_configure(self, args: argparse.Namespace) -> None:
        """Handle the configure command."""
        if args.check:
            # Display current configuration
            logger.info("Current Configuration:")
            logger.info(f"- Birdeye API Key: {'[SET]' if self.config.get('birdeye_api_key') else '[NOT SET]'}")
            logger.info(f"- Cielo Finance API Key: {'[SET]' if self.config.get('cielo_finance_api_key') else '[NOT SET]'}")
            logger.info(f"- Telegram API ID: {'[SET]' if self.config.get('telegram_api_id') else '[NOT SET]'}")
            logger.info(f"- Telegram API Hash: {'[SET]' if self.config.get('telegram_api_hash') else '[NOT SET]'}")
            logger.info(f"- Telegram Groups: {len(self.config['sources']['telegram_groups'])} configured")
            logger.info(f"- Wallets: {len(self.config['sources']['wallets'])} configured")
            logger.info(f"- API Provider: {self.config.get('api_provider', 'mixed')}")
            return
        
        if args.birdeye_api_key:
            self.config["birdeye_api_key"] = args.birdeye_api_key
            logger.info("Birdeye API key configured.")
        
        if args.cielo_api_key:
            self.config["cielo_finance_api_key"] = args.cielo_api_key
            logger.info("Cielo Finance API key configured.")
        
        if args.telegram_api_id:
            self.config["telegram_api_id"] = args.telegram_api_id
            logger.info("Telegram API ID configured.")
        
        if args.telegram_api_hash:
            self.config["telegram_api_hash"] = args.telegram_api_hash
            logger.info("Telegram API hash configured.")
        
        save_config(self.config)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
    
    def _handle_test_api(self, args: argparse.Namespace) -> None:
        """Handle the test API command."""
        if args.api in ["cielo", "all"]:
            logger.info("Testing Cielo Finance API connectivity...")
            
            if not self.config.get("cielo_finance_api_key"):
                logger.error("Cielo Finance API key not configured. Run 'phoenix configure --cielo-api-key YOUR_KEY'")
                return
            
            try:
                from cielo_api import CieloFinanceAPI
                
                cielo_api = CieloFinanceAPI(self.config["cielo_finance_api_key"])
                health_check = cielo_api.health_check()
                
                if health_check.get("success", True):
                    logger.info("[SUCCESS] Cielo Finance API connection successful")
                    
                    # Test API usage endpoint
                    try:
                        usage = cielo_api.get_api_usage()
                        if usage.get("success"):
                            logger.info("[SUCCESS] API usage data retrieved successfully")
                        else:
                            logger.warning("[WARNING] API usage data not available")
                    except Exception as e:
                        logger.warning(f"[WARNING] API usage check failed: {str(e)}")
                else:
                    logger.error(f"[ERROR] Cielo Finance API connection failed: {health_check.get('error', 'Unknown error')}")
            
            except Exception as e:
                logger.error(f"[ERROR] Cielo Finance API test failed: {str(e)}")
        
        # Test Birdeye API
        if args.api in ["birdeye", "all"]:
            logger.info("Testing Birdeye API connectivity...")
            
            if not self.config.get("birdeye_api_key"):
                logger.error("Birdeye API key not configured. Run 'phoenix configure --birdeye-api-key YOUR_KEY'")
                return
            
            try:
                from birdeye_api import BirdeyeAPI
                
                birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
                # Simple test - try to get token info for a known token
                test_result = birdeye_api.get_token_info("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")  # USDC
                
                if test_result.get("success"):
                    logger.info("[SUCCESS] Birdeye API connection successful")
                else:
                    logger.error(f"[ERROR] Birdeye API connection failed: {test_result.get('error', 'Unknown error')}")
            
            except Exception as e:
                logger.error(f"[ERROR] Birdeye API test failed: {str(e)}")
        
        if args.api in ["telegram", "all"]:
            logger.info("Testing Telegram API connectivity...")
            
            if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
                logger.error("Telegram API credentials not configured. Run 'phoenix configure --telegram-api-id ID --telegram-api-hash HASH'")
                return
            
            try:
                import asyncio
                from telegram_module import TelegramScraper
                
                telegram_scraper = TelegramScraper(
                    self.config["telegram_api_id"],
                    self.config["telegram_api_hash"],
                    self.config.get("telegram_session", "phoenix")
                )
                
                async def test_telegram():
                    try:
                        await telegram_scraper.connect()
                        logger.info("[SUCCESS] Telegram API connection successful")
                        return True
                    except Exception as e:
                        logger.error(f"[ERROR] Telegram API connection failed: {str(e)}")
                        return False
                    finally:
                        await telegram_scraper.disconnect()
                
                asyncio.run(test_telegram())
            
            except Exception as e:
                logger.error(f"[ERROR] Telegram API test failed: {str(e)}")
    
    def _handle_telegram_analysis(self, args: argparse.Namespace) -> None:
        """Handle the telegram analysis command."""
        import asyncio
        from telegram_module import TelegramScraper
        from birdeye_api import BirdeyeAPI  # Use Birdeye for Telegram analysis
        
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
        
        logger.info(f"Analyzing {len(channels)} Telegram channels for the past {args.days} days using Birdeye API.")
        logger.info(f"Results will be saved to {output_file}")
        
        # Initialize API clients
        try:
            birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])  # Use Birdeye for Telegram
            telegram_scraper = TelegramScraper(
                self.config["telegram_api_id"],
                self.config["telegram_api_hash"],
                self.config.get("telegram_session", "phoenix")
            )
        except Exception as e:
            logger.error(f"Failed to initialize API clients: {str(e)}")
            return
        
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
                            birdeye_api  # Using Birdeye API for token analysis
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
                                birdeye_api  # Using Birdeye API for token analysis
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
        from cielo_api import CieloFinanceAPI  # Use Cielo for wallet analysis
        
        wallets = args.wallets or self.config["sources"]["wallets"]
        if not wallets:
            logger.error("No wallets specified. Use --wallets or add sources with add-source command.")
            return
        
        # Check if we have the necessary API key
        if not self.config.get("cielo_finance_api_key"):
            logger.error("Cielo Finance API key not configured. Run 'phoenix configure --cielo-api-key YOUR_KEY'")
            return
        
        # Ensure output directory exists and get full path
        output_file = ensure_output_dir(args.output)
        
        logger.info(f"Analyzing {len(wallets)} wallets for the past {args.days} days using Cielo Finance API.")
        logger.info(f"Minimum win rate: {args.min_winrate}%")
        logger.info(f"Results will be saved to {output_file}")
        
        # Initialize API client and wallet analyzer
        try:
            cielo_api = CieloFinanceAPI(self.config["cielo_finance_api_key"])
            wallet_analyzer = WalletAnalyzer(cielo_api)  # Using Cielo Finance API for wallet analysis
        except Exception as e:
            logger.error(f"Failed to initialize Cielo Finance API: {str(e)}")
            logger.error("Please check your API key and network connection.")
            return
        
        try:
            # Batch analyze wallets
            if len(wallets) > 1:
                logger.info("Performing batch wallet analysis...")
                wallet_analyses = wallet_analyzer.batch_analyze_wallets(
                    wallets,
                    args.days,
                    args.min_winrate
                )
                
                if not wallet_analyses.get("success"):
                    logger.error(f"Batch analysis failed: {wallet_analyses.get('error', 'Unknown error')}")
                    return
                
                wallet_analyzer.export_batch_analysis(wallet_analyses, output_file)
                logger.info(f"Batch analysis completed. Results saved to {output_file} and related files.")
                
                # Print summary
                logger.info(f"Analysis Summary:")
                logger.info(f"- Total wallets: {wallet_analyses['total_wallets']}")
                logger.info(f"- Analyzed wallets: {wallet_analyses['analyzed_wallets']}")
                logger.info(f"- Failed wallets: {wallet_analyses.get('failed_wallets', 0)}")
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
                    if wallet_analysis.get("error_type") == "API_ERROR":
                        logger.error("This appears to be a Cielo Finance API issue. Please check your API key and try again.")
        
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
        print("🚧 Combined analysis functionality coming soon with Cielo Finance API integration!")
        print("This feature is being updated to work with the new API.")

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