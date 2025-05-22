#!/usr/bin/env python3
"""
Phoenix Project - Solana Chain Analysis CLI Tool (Updated)

Auto-defaults for Telegram analysis - no prompts needed
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
        logging.FileHandler("phoenix.log", encoding='utf-8'),
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
            "telegram_groups": ["spydefi"],
            "wallets": []
        },
        "analysis_period_days": 1  # Changed to 1 day (24 hours)
    }

def ensure_output_dir(output_path: str) -> str:
    """Ensure the output directory exists and return the full path."""
    output_dir = os.path.join(os.getcwd(), "outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created outputs directory: {output_dir}")
    
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
        
        # Telegram analysis command (simplified)
        telegram_parser = subparsers.add_parser("telegram", help="Analyze SpyDefi channel")
        telegram_parser.add_argument("--hours", type=int, default=24, help="Hours to analyze (default: 24)")
        telegram_parser.add_argument("--output", default="spydefi_analysis.csv", help="Output CSV file")
        
        # Wallet analysis command
        wallet_parser = subparsers.add_parser("wallet", help="Analyze wallets for copy trading")
        wallet_parser.add_argument("--wallets", nargs="+", help="Wallet addresses to analyze")
        wallet_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
        wallet_parser.add_argument("--min-winrate", type=float, default=45.0, help="Minimum win rate percentage")
        wallet_parser.add_argument("--output", default="wallet_analysis.csv", help="Output CSV file")
        
        return parser
    
    def _handle_numbered_menu(self):
        """Handle the numbered menu interface."""
        print("\n" + "="*60)
        print("Phoenix Project - Solana Chain Analysis Tool")
        print("(Cielo Finance Edition)")
        print("="*60)
        print("\nSelect an option:")
        print("\n🔧 CONFIGURATION:")
        print("1. Configure API Keys")
        print("2. Check Configuration")
        print("3. Test API Connectivity")
        print("4. Add Data Sources")
        print("\n📊 ANALYSIS:")
        print("5. Analyze Telegram Channels")  # This should be auto-default
        print("6. Analyze Wallets")
        print("7. Combined Analysis (Telegram + Wallets)")
        print("\n🔍 UTILITIES:")
        print("8. View Current Sources")
        print("9. Help & Examples")
        print("0. Exit")
        print("="*60)
        
        try:
            choice = input("\nEnter your choice (0-9): ").strip()
            
            if choice == '0':
                print("\nExiting Phoenix Project. Goodbye! 👋")
                sys.exit(0)
            elif choice == '1':
                self._interactive_configure()
            elif choice == '2':
                self._check_configuration()
            elif choice == '3':
                self._test_api_connectivity()
            elif choice == '4':
                self._interactive_add_sources()
            elif choice == '5':
                self._auto_telegram_analysis()  # Auto-run with defaults
            elif choice == '6':
                self._interactive_wallet_analysis()
            elif choice == '7':
                self._interactive_combined_analysis()
            elif choice == '8':
                self._view_current_sources()
            elif choice == '9':
                self._show_help()
            else:
                print("❌ Invalid choice. Please try again.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error in menu: {str(e)}")
            input("Press Enter to continue...")
    
    def _auto_telegram_analysis(self):
        """Run Telegram analysis automatically with defaults."""
        print("\n" + "="*42)
        print("    SPYDEFI TELEGRAM ANALYSIS")
        print("="*42)
        print("\n🚀 Starting SpyDefi analysis...")
        print("📅 Analysis period: 24 hours")
        print("📁 Output: spydefi_analysis.csv")
        print("📊 Excel export: Enabled")
        print("\nProcessing...")
        
        # Create args object with defaults
        class Args:
            def __init__(self):
                self.channels = ["spydefi"]
                self.days = 1  # 24 hours
                self.output = "spydefi_analysis.csv"
                self.excel = True
        
        args = Args()
        
        try:
            self._handle_telegram_analysis(args)
            print("\n✅ Analysis completed successfully!")
            print("📁 Check the outputs folder for results.")
        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            logger.error(f"Telegram analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _handle_telegram_analysis(self, args) -> None:
        """Handle the telegram analysis command."""
        import asyncio
        from telegram_module import TelegramScraper
        from birdeye_api import BirdeyeAPI
        
        channels = args.channels or self.config["sources"]["telegram_groups"]
        if not channels:
            logger.error("No Telegram channels specified.")
            return
        
        # Check if we have the necessary API keys
        if not self.config.get("birdeye_api_key"):
            logger.error("Birdeye API key not configured. Configure it first.")
            return
            
        if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
            logger.error("Telegram API credentials not configured. Configure them first.")
            return
        
        # Ensure output directory exists and get full path
        output_file = ensure_output_dir(args.output)
        
        logger.info(f"Analyzing SpyDefi channel for the past {args.days} day(s).")
        logger.info(f"Results will be saved to {output_file}")
        
        # Initialize API clients
        birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
        telegram_scraper = TelegramScraper(
            self.config["telegram_api_id"],
            self.config["telegram_api_hash"],
            self.config["birdeye_api_key"],
            self.config.get("telegram_session", "phoenix")
        )
        
        # Store all analysis results
        telegram_analyses = {"ranked_kols": []}
        
        # Handle SpyDefi analysis
        if any(ch.lower() == "spydefi" for ch in channels):
            logger.info("SpyDefi channel detected. Analyzing KOLs from SpyDefi.")
            
            try:
                # Run the analysis asynchronously
                async def run_spydefi_analysis():
                    try:
                        await telegram_scraper.connect()
                        analysis = await telegram_scraper.scan_spydefi_channel(
                            hours_back=args.days * 24  # Convert days to hours
                        )
                        await telegram_scraper.export_analysis_results(analysis, output_file)
                        return analysis
                    finally:
                        await telegram_scraper.disconnect()
                
                # Run the async function
                telegram_analyses = asyncio.run(run_spydefi_analysis())
                logger.info(f"SpyDefi analysis completed.")
                
            except Exception as e:
                logger.error(f"Error analyzing SpyDefi: {str(e)}")
                return
        
        logger.info(f"Telegram analysis completed. Results saved to {output_file}")
        
        # Export to Excel automatically
        if hasattr(args, 'excel') and args.excel:
            try:
                from export_utils import export_to_excel
                excel_file = output_file.replace(".csv", ".xlsx")
                export_to_excel(telegram_analyses, {}, excel_file)
                logger.info(f"Excel export completed: {excel_file}")
            except Exception as e:
                logger.error(f"Error exporting to Excel: {str(e)}")
    
    def _interactive_configure(self):
        """Interactive configuration setup."""
        print("\n" + "="*42)
        print("    CONFIGURATION SETUP")
        print("="*42)
        
        # Birdeye API Key
        current_birdeye = self.config.get("birdeye_api_key", "")
        if current_birdeye:
            print(f"\n🔑 Current Birdeye API Key: {current_birdeye[:8]}...")
            change_birdeye = input("Change Birdeye API key? (y/N): ").lower().strip()
            if change_birdeye == 'y':
                new_key = input("Enter new Birdeye API key: ").strip()
                if new_key:
                    self.config["birdeye_api_key"] = new_key
                    print("✅ Birdeye API key updated")
        else:
            new_key = input("Enter Birdeye API key: ").strip()
            if new_key:
                self.config["birdeye_api_key"] = new_key
                print("✅ Birdeye API key configured")
        
        # Telegram API ID
        current_tg_id = self.config.get("telegram_api_id", "")
        if current_tg_id:
            print(f"\n📱 Current Telegram API ID: {current_tg_id}")
            change_tg = input("Change Telegram API credentials? (y/N): ").lower().strip()
            if change_tg == 'y':
                new_id = input("Enter new Telegram API ID: ").strip()
                new_hash = input("Enter new Telegram API Hash: ").strip()
                if new_id and new_hash:
                    self.config["telegram_api_id"] = new_id
                    self.config["telegram_api_hash"] = new_hash
                    print("✅ Telegram API credentials updated")
        else:
            new_id = input("Enter Telegram API ID: ").strip()
            new_hash = input("Enter Telegram API Hash: ").strip()
            if new_id and new_hash:
                self.config["telegram_api_id"] = new_id
                self.config["telegram_api_hash"] = new_hash
                print("✅ Telegram API credentials configured")
        
        save_config(self.config)
        print("\n✅ Configuration saved successfully!")
        input("Press Enter to continue...")
    
    def _check_configuration(self):
        """Check current configuration."""
        print("\n" + "="*42)
        print("    CURRENT CONFIGURATION")
        print("="*42)
        
        print(f"\n🔑 Birdeye API Key: {'✅ Configured' if self.config.get('birdeye_api_key') else '❌ Not configured'}")
        print(f"📱 Telegram API ID: {'✅ Configured' if self.config.get('telegram_api_id') else '❌ Not configured'}")
        print(f"📱 Telegram API Hash: {'✅ Configured' if self.config.get('telegram_api_hash') else '❌ Not configured'}")
        
        print(f"\n📊 Telegram Channels: {len(self.config.get('sources', {}).get('telegram_groups', []))}")
        for channel in self.config.get('sources', {}).get('telegram_groups', []):
            print(f"   - {channel}")
        
        print(f"\n💰 Wallets: {len(self.config.get('sources', {}).get('wallets', []))}")
        for wallet in self.config.get('sources', {}).get('wallets', [])[:5]:  # Show first 5
            print(f"   - {wallet}")
        
        if len(self.config.get('sources', {}).get('wallets', [])) > 5:
            print(f"   ... and {len(self.config.get('sources', {}).get('wallets', [])) - 5} more")
        
        input("\nPress Enter to continue...")
    
    def run(self) -> None:
        """Run the CLI application."""
        args = self.parser.parse_args()
        
        if not args.command:
            # If no command specified, show the numbered menu
            while True:
                self._handle_numbered_menu()
        else:
            # Execute the appropriate command
            if args.command == "configure":
                self._handle_configure(args)
            elif args.command == "telegram":
                self._handle_telegram_analysis(args)
            elif args.command == "wallet":
                self._handle_wallet_analysis(args)
    
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
    
    def _handle_wallet_analysis(self, args: argparse.Namespace) -> None:
        """Handle the wallet analysis command."""
        from wallet_module import WalletAnalyzer
        from birdeye_api import BirdeyeAPI
        
        wallets = args.wallets or self.config["sources"]["wallets"]
        if not wallets:
            logger.error("No wallets specified.")
            return
        
        # Check if we have the necessary API key
        if not self.config.get("birdeye_api_key"):
            logger.error("Birdeye API key not configured.")
            return
        
        # Ensure output directory exists and get full path
        output_file = ensure_output_dir(args.output)
        
        logger.info(f"Analyzing {len(wallets)} wallets for the past {args.days} days.")
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
                logger.info(f"Batch analysis completed. Results saved to {output_file}")
                
            # Analyze single wallet
            else:
                logger.info(f"Analyzing wallet {wallets[0]}...")
                wallet_analysis = wallet_analyzer.analyze_wallet(wallets[0], args.days)
                
                if wallet_analysis.get("success"):
                    wallet_analyzer.export_wallet_analysis(wallet_analysis, output_file)
                    logger.info(f"Analysis completed for {wallets[0]}")
                else:
                    logger.error(f"Analysis failed: {wallet_analysis.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"Error during wallet analysis: {str(e)}")

def main():
    """Main entry point for the Phoenix CLI."""
    try:
        cli = PhoenixCLI()
        cli.run()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()