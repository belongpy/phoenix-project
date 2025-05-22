#!/usr/bin/env python3
"""
Phoenix Project - Solana Chain Analysis CLI Tool (UPDATED WITH CIELO FINANCE FIX)

Auto-defaults for Telegram analysis - no prompts needed
Enhanced with corrected Cielo Finance API integration and RPC configuration
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
        "cielo_api_key": "",
        "telegram_api_id": "",
        "telegram_api_hash": "",
        "telegram_session": "",
        "solana_rpc_url": "https://api.mainnet-beta.solana.com",  # Default RPC
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

def load_wallets_from_file(file_path: str = "wallets.txt") -> List[str]:
    """Load wallet addresses from wallets.txt file."""
    if not os.path.exists(file_path):
        logger.warning(f"Wallets file {file_path} not found.")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            wallets = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    # Basic validation for Solana address length
                    if 32 <= len(line) <= 44:
                        wallets.append(line)
                    else:
                        logger.warning(f"Line {line_num}: Invalid wallet address format: {line}")
            
            logger.info(f"Loaded {len(wallets)} wallet addresses from {file_path}")
            return wallets
            
    except Exception as e:
        logger.error(f"Error reading wallets file {file_path}: {str(e)}")
        return []

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
        configure_parser.add_argument("--cielo-api-key", help="Cielo Finance API key")
        configure_parser.add_argument("--telegram-api-id", help="Telegram API ID")
        configure_parser.add_argument("--telegram-api-hash", help="Telegram API hash")
        configure_parser.add_argument("--rpc-url", help="Solana RPC URL (P9 or other provider)")
        
        # Telegram analysis command (simplified)
        telegram_parser = subparsers.add_parser("telegram", help="Analyze SpyDefi channel")
        telegram_parser.add_argument("--hours", type=int, default=24, help="Hours to analyze (default: 24)")
        telegram_parser.add_argument("--output", default="spydefi_analysis.csv", help="Output CSV file")
        
        # Wallet analysis command
        wallet_parser = subparsers.add_parser("wallet", help="Analyze wallets for copy trading")
        wallet_parser.add_argument("--wallets-file", default="wallets.txt", help="File containing wallet addresses")
        wallet_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
        wallet_parser.add_argument("--min-winrate", type=float, default=45.0, help="Minimum win rate percentage")
        wallet_parser.add_argument("--output", default="wallet_analysis.xlsx", help="Output file")
        wallet_parser.add_argument("--no-contested", action="store_true", help="Skip contested wallet analysis")
        
        return parser
    
    def _handle_numbered_menu(self):
        """Handle the numbered menu interface."""
        print("\n" + "="*60)
        print("Phoenix Project - Solana Chain Analysis Tool")
        print("(Enhanced with Cielo Finance & Contested Analysis)")
        print("="*60)
        print("\nSelect an option:")
        print("\n🔧 CONFIGURATION:")
        print("1. Configure API Keys")
        print("2. Check Configuration")
        print("3. Test API Connectivity")
        print("4. Add Data Sources")
        print("\n📊 ANALYSIS:")
        print("5. Analyze Telegram Channels")
        print("6. Analyze Wallets (Auto-load from wallets.txt)")
        print("7. Analyze Single Wallet")
        print("\n🔍 UTILITIES:")
        print("8. View Current Sources")
        print("9. View Wallets File")
        print("10. Help & Examples")
        print("0. Exit")
        print("="*60)
        
        try:
            choice = input("\nEnter your choice (0-10): ").strip()
            
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
                self._auto_telegram_analysis()
            elif choice == '6':
                self._auto_wallet_analysis()
            elif choice == '7':
                self._single_wallet_analysis()
            elif choice == '8':
                self._view_current_sources()
            elif choice == '9':
                self._view_wallets_file()
            elif choice == '10':
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
        print("\n" + "="*50)
        print("    SPYDEFI TELEGRAM ANALYSIS")
        print("="*50)
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
    
    def _auto_wallet_analysis(self):
        """Analyze wallets automatically from wallets.txt file."""
        print("\n" + "="*60)
        print("    ENHANCED WALLET ANALYSIS")
        print("    (with Contested Wallet Detection)")
        print("="*60)
        
        # Check API configuration first
        if not self.config.get("cielo_api_key"):
            print("\n❌ Cielo Finance API key not configured!")
            print("Cielo Finance API is REQUIRED for wallet analysis.")
            print("Please configure your Cielo Finance API key first (Option 1).")
            input("Press Enter to continue...")
            return
        
        # Load wallets from file
        wallets_to_analyze = load_wallets_from_file("wallets.txt")
        
        if not wallets_to_analyze:
            print("\n❌ No valid wallet addresses found in wallets.txt!")
            print("Please add wallet addresses to wallets.txt (one per line).")
            print("Example format:")
            print("  # This is a comment")
            print("  DJPKomwbTTsjyc3bZZZayE9mHhwAkJHkpwRvePYjV9VR")
            print("  9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM")
            input("Press Enter to continue...")
            return
        
        # Configuration options
        days = 30
        min_winrate = 45.0
        include_contested = True
        output_file = "phoenix_wallet_analysis.xlsx"
        
        print(f"\n🚀 Starting enhanced wallet analysis...")
        print(f"📁 Wallets file: wallets.txt")
        print(f"📊 Wallets to analyze: {len(wallets_to_analyze)}")
        print(f"📅 Analysis period: ALL-TIME (Cielo Finance data)")
        print(f"📈 Minimum win rate: {min_winrate}%")
        print(f"⚔️  Contested analysis: {'Enabled' if include_contested else 'Disabled'}")  
        print(f"🌐 RPC Endpoint: {self.config.get('solana_rpc_url', 'Default')}")
        print(f"📁 Output file: {output_file}")
        
        # Show which APIs are available
        apis = []
        if self.config.get("cielo_api_key"):
            apis.append("Cielo Finance")
        if self.config.get("birdeye_api_key"):
            apis.append("Birdeye")
        print(f"🔧 APIs: {' + '.join(apis) if apis else 'None configured'}")
        
        print("\nProcessing...")
        
        try:
            self._run_enhanced_wallet_analysis(
                wallets_to_analyze, 
                days, 
                min_winrate, 
                output_file, 
                include_contested
            )
            print("\n✅ Enhanced wallet analysis completed successfully!")
            print("📁 Check the outputs folder for results.")
            print("📊 Results include:")
            print("   • Comprehensive wallet metrics")
            print("   • Trading strategy recommendations") 
            print("   • Contested wallet analysis (copy trader detection)")
            print("   • Excel export with multiple sheets")
            
        except Exception as e:
            print(f"\n❌ Wallet analysis failed: {str(e)}")
            logger.error(f"Enhanced wallet analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _single_wallet_analysis(self):
        """Analyze a single wallet address."""
        print("\n" + "="*50)
        print("    SINGLE WALLET ANALYSIS")
        print("="*50)
        
        # Check API configuration
        if not self.config.get("cielo_api_key"):
            print("\n❌ Cielo Finance API key not configured!")
            print("Cielo Finance API is REQUIRED for wallet analysis.")
            print("Please configure your Cielo Finance API key first (Option 1).")
            input("Press Enter to continue...")
            return
        
        # Get wallet address from user
        wallet_address = input("\nEnter wallet address: ").strip()
        
        if not wallet_address:
            print("❌ No wallet address provided.")
            input("Press Enter to continue...")
            return
        
        # Basic validation
        if not (32 <= len(wallet_address) <= 44):
            print("❌ Invalid wallet address format.")
            input("Press Enter to continue...")
            return
        
        # Configuration
        days = 30
        include_contested = True
        output_file = f"single_wallet_{wallet_address[:8]}.xlsx"
        
        print(f"\n🚀 Analyzing wallet: {wallet_address}")
        print(f"📅 Analysis period: ALL-TIME (Cielo Finance data)")
        print(f"⚔️  Contested analysis: {'Enabled' if include_contested else 'Disabled'}")
        print(f"🌐 RPC Endpoint: {self.config.get('solana_rpc_url', 'Default')}")
        print(f"📁 Output file: {output_file}")
        print("\nProcessing...")
        
        try:
            self._run_enhanced_wallet_analysis(
                [wallet_address], 
                days, 
                0.0,  # No min win rate filter for single wallet
                output_file, 
                include_contested
            )
            print("\n✅ Single wallet analysis completed!")
            print(f"📁 Results saved to outputs/{output_file}")
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            logger.error(f"Single wallet analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _run_enhanced_wallet_analysis(self, wallets: List[str], days: int, 
                                    min_winrate: float, output_file: str, 
                                    include_contested: bool = True):
        """Run the enhanced wallet analysis with Cielo Finance API and contested detection."""
        # Ensure output directory exists
        full_output_path = ensure_output_dir(output_file)
        
        # Initialize Cielo Finance API (REQUIRED)
        cielo_api = None
        if self.config.get("cielo_api_key"):
            try:
                from cielo_api import CieloFinanceAPI
                cielo_api = CieloFinanceAPI(self.config["cielo_api_key"])
                logger.info("✅ Cielo Finance API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Cielo Finance API: {str(e)}")
                raise Exception(f"Cielo Finance API initialization failed: {str(e)}")
        else:
            raise Exception("Cielo Finance API key is REQUIRED for wallet analysis")
        
        # Initialize Birdeye API (OPTIONAL, for token metadata only)
        birdeye_api = None
        if self.config.get("birdeye_api_key"):
            try:
                from birdeye_api import BirdeyeAPI
                birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
                logger.info("✅ Birdeye API initialized (for token metadata)")
            except Exception as e:
                logger.warning(f"Could not initialize Birdeye API: {str(e)}")
        
        # Initialize enhanced wallet analyzer
        from wallet_module import WalletAnalyzer
        
        analyzer = WalletAnalyzer(
            cielo_api=cielo_api,
            birdeye_api=birdeye_api,
            rpc_url=self.config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
        )
        
        logger.info(f"Using RPC: {self.config.get('solana_rpc_url', 'Default')}")
        
        # Run batch analysis with contested detection
        result = analyzer.batch_analyze_wallets(
            wallets,
            days,
            min_winrate,
            include_contested
        )
        
        if result.get("success"):
            analyzer.export_batch_analysis(result, full_output_path)
            
            # Log summary
            logger.info(f"\n📊 ANALYSIS SUMMARY:")
            logger.info(f"   Total wallets: {result['total_wallets']}")
            logger.info(f"   Successfully analyzed: {result['analyzed_wallets']}")
            logger.info(f"   Failed: {result.get('failed_wallets', 0)}")
            logger.info(f"   Gem Finders: {len(result['gem_finders'])}")
            logger.info(f"   Consistent: {len(result['consistent'])}")
            logger.info(f"   Flippers: {len(result['flippers'])}")
            logger.info(f"   Others: {len(result['others'])}")
            
            # Log contested analysis stats if included
            if include_contested:
                all_analyses = (result['gem_finders'] + result['consistent'] + 
                              result['flippers'] + result['others'])
                contested_stats = {
                    "highly_contested": 0,
                    "moderately_contested": 0,
                    "lightly_contested": 0,
                    "not_contested": 0
                }
                
                for analysis in all_analyses:
                    if "contested_analysis" in analysis and analysis["contested_analysis"].get("success"):
                        classification = analysis["contested_analysis"].get("classification", "UNKNOWN")
                        if "HIGHLY" in classification:
                            contested_stats["highly_contested"] += 1
                        elif "MODERATELY" in classification:
                            contested_stats["moderately_contested"] += 1
                        elif "LIGHTLY" in classification:
                            contested_stats["lightly_contested"] += 1
                        else:
                            contested_stats["not_contested"] += 1
                
                logger.info(f"\n⚔️  CONTESTED ANALYSIS:")
                logger.info(f"   Highly Contested: {contested_stats['highly_contested']}")
                logger.info(f"   Moderately Contested: {contested_stats['moderately_contested']}")
                logger.info(f"   Lightly Contested: {contested_stats['lightly_contested']}")
                logger.info(f"   Not Contested: {contested_stats['not_contested']}")
        else:
            raise Exception(f"Batch analysis failed: {result.get('error')}")
    
    def _test_api_connectivity(self):
        """Test API connectivity."""
        print("\n" + "="*50)
        print("    API CONNECTIVITY TEST")
        print("="*50)
        
        # Test Cielo Finance API
        if self.config.get("cielo_api_key"):
            print("\n💰 Testing Cielo Finance API...")
            try:
                from cielo_api import CieloFinanceAPI
                cielo_api = CieloFinanceAPI(self.config["cielo_api_key"])
                test_result = cielo_api.health_check()
                if test_result:
                    print("✅ Cielo Finance API: Connected successfully")
                else:
                    print("❌ Cielo Finance API: Connection failed")
            except Exception as e:
                print(f"❌ Cielo Finance API: Error - {str(e)}")
        else:
            print("❌ Cielo Finance API: Not configured")
        
        # Test Birdeye API
        if self.config.get("birdeye_api_key"):
            print("\n🔍 Testing Birdeye API...")
            try:
                from birdeye_api import BirdeyeAPI
                birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
                test_result = birdeye_api.get_token_info("So11111111111111111111111111111111111111112")
                if test_result.get("success"):
                    print("✅ Birdeye API: Connected successfully")
                else:
                    print("❌ Birdeye API: Connection failed")
            except Exception as e:
                print(f"❌ Birdeye API: Error - {str(e)}")
        else:
            print("❌ Birdeye API: Not configured")
        
        # Test Telegram API
        if self.config.get("telegram_api_id") and self.config.get("telegram_api_hash"):
            print("\n📱 Testing Telegram API...")
            try:
                from telegram_module import TelegramScraper
                print("✅ Telegram API: Configuration appears valid")
            except Exception as e:
                print(f"❌ Telegram API: Error - {str(e)}")
        else:
            print("❌ Telegram API: Not configured")
        
        # Test RPC Connection
        print(f"\n🌐 Testing Solana RPC Connection...")
        print(f"   RPC URL: {self.config.get('solana_rpc_url', 'Default')}")
        try:
            import requests
            response = requests.post(
                self.config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com"),
                json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Solana RPC: Connected successfully")
            else:
                print(f"❌ Solana RPC: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Solana RPC: Error - {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _interactive_configure(self):
        """Interactive configuration setup."""
        print("\n" + "="*50)
        print("    CONFIGURATION SETUP")
        print("="*50)
        
        # Cielo Finance API Key (MOST IMPORTANT)
        current_cielo = self.config.get("cielo_api_key", "")
        if current_cielo:
            print(f"\n💰 Current Cielo Finance API Key: {current_cielo[:8]}...")
            change_cielo = input("Change Cielo Finance API key? (y/N): ").lower().strip()
            if change_cielo == 'y':
                new_key = input("Enter new Cielo Finance API key: ").strip()
                if new_key:
                    self.config["cielo_api_key"] = new_key
                    print("✅ Cielo Finance API key updated")
        else:
            print("\n💰 Cielo Finance API Key (REQUIRED for wallet analysis):")
            new_key = input("Enter Cielo Finance API key: ").strip()
            if new_key:
                self.config["cielo_api_key"] = new_key
                print("✅ Cielo Finance API key configured")
            else:
                print("⚠️ Warning: Wallet analysis will NOT work without Cielo Finance API key")
        
        # Birdeye API Key (OPTIONAL)
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
            new_key = input("Enter Birdeye API key (optional, for Telegram analysis): ").strip()
            if new_key:
                self.config["birdeye_api_key"] = new_key
                print("✅ Birdeye API key configured")
        
        # Solana RPC URL
        current_rpc = self.config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
        print(f"\n🌐 Current Solana RPC URL: {current_rpc}")
        change_rpc = input("Change RPC URL? (y/N): ").lower().strip()
        if change_rpc == 'y':
            print("\nCommon RPC Providers:")
            print("1. Solana Mainnet (Free): https://api.mainnet-beta.solana.com")
            print("2. P9 (Your Provider): https://your-p9-endpoint.com")
            print("3. QuickNode: https://your-quicknode-endpoint.solana-mainnet.quiknode.pro")
            print("4. Alchemy: https://solana-mainnet.g.alchemy.com/v2/YOUR_API_KEY")
            print("5. Custom: Enter your own URL")
            
            choice = input("\nSelect option (1-5) or enter custom URL: ").strip()
            
            if choice == '1':
                new_rpc = "https://api.mainnet-beta.solana.com"
            elif choice == '2':
                new_rpc = input("Enter your P9 RPC URL: ").strip()
            elif choice == '3':
                new_rpc = input("Enter your QuickNode URL: ").strip()
            elif choice == '4':
                new_rpc = input("Enter your Alchemy URL: ").strip()
            elif choice == '5' or choice.startswith('http'):
                new_rpc = choice if choice.startswith('http') else input("Enter custom RPC URL: ").strip()
            else:
                new_rpc = current_rpc
            
            if new_rpc and new_rpc != current_rpc:
                self.config["solana_rpc_url"] = new_rpc
                print(f"✅ RPC URL updated to: {new_rpc}")
        
        # Telegram API credentials
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
            new_id = input("Enter Telegram API ID (optional, for Telegram analysis): ").strip()
            new_hash = input("Enter Telegram API Hash (optional, for Telegram analysis): ").strip()
            if new_id and new_hash:
                self.config["telegram_api_id"] = new_id
                self.config["telegram_api_hash"] = new_hash
                print("✅ Telegram API credentials configured")
        
        save_config(self.config)
        print("\n✅ Configuration saved successfully!")
        input("Press Enter to continue...")
    
    def _check_configuration(self):
        """Check current configuration."""
        print("\n" + "="*50)
        print("    CURRENT CONFIGURATION")
        print("="*50)
        
        print(f"\n💰 Cielo Finance API Key: {'✅ Configured' if self.config.get('cielo_api_key') else '❌ Not configured (REQUIRED for wallets)'}")
        print(f"🔑 Birdeye API Key: {'✅ Configured' if self.config.get('birdeye_api_key') else '❌ Not configured'}")
        print(f"📱 Telegram API ID: {'✅ Configured' if self.config.get('telegram_api_id') else '❌ Not configured'}")
        print(f"📱 Telegram API Hash: {'✅ Configured' if self.config.get('telegram_api_hash') else '❌ Not configured'}")
        print(f"🌐 Solana RPC URL: {self.config.get('solana_rpc_url', 'Default')}")
        
        print(f"\n📊 Telegram Channels: {len(self.config.get('sources', {}).get('telegram_groups', []))}")
        for channel in self.config.get('sources', {}).get('telegram_groups', []):
            print(f"   - {channel}")
        
        # Show wallets from file
        wallets_from_file = load_wallets_from_file("wallets.txt")
        print(f"\n💰 Wallets in wallets.txt: {len(wallets_from_file)}")
        for wallet in wallets_from_file[:5]:  # Show first 5
            print(f"   - {wallet}")
        
        if len(wallets_from_file) > 5:
            print(f"   ... and {len(wallets_from_file) - 5} more")
        
        input("\nPress Enter to continue...")
    
    def _view_wallets_file(self):
        """View contents of wallets.txt file."""
        print("\n" + "="*50)
        print("    WALLETS.TXT FILE CONTENTS")
        print("="*50)
        
        if not os.path.exists("wallets.txt"):
            print("\n❌ wallets.txt file not found!")
            print("\n📝 Create a wallets.txt file with the following format:")
            print("# This is a comment")
            print("DJPKomwbTTsjyc3bZZZayE9mHhwAkJHkpwRvePYjV9VR")
            print("9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM")
            print("# Another comment")
            print("AnotherWalletAddressHere...")
        else:
            try:
                with open("wallets.txt", 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                print(f"\n📁 File: wallets.txt")
                print(f"📊 Total lines: {len(lines)}")
                
                valid_wallets = 0
                print("\n📋 Contents:")
                print("-" * 50)
                
                for i, line in enumerate(lines, 1):
                    line_content = line.strip()
                    if not line_content:
                        print(f"{i:3}. (empty line)")
                    elif line_content.startswith('#'):
                        print(f"{i:3}. {line_content}")
                    else:
                        if 32 <= len(line_content) <= 44:
                            print(f"{i:3}. {line_content} ✅")
                            valid_wallets += 1
                        else:
                            print(f"{i:3}. {line_content} ❌ (invalid format)")
                
                print("-" * 50)
                print(f"✅ Valid wallet addresses: {valid_wallets}")
                
            except Exception as e:
                print(f"\n❌ Error reading wallets.txt: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _show_help(self):
        """Show help and examples."""
        print("\n" + "="*60)
        print("    HELP & EXAMPLES")
        print("="*60)
        
        print("\n📖 GETTING STARTED:")
        print("1. Configure API keys (Option 1)")
        print("   - Cielo Finance API: https://cielo.finance (REQUIRED for wallets)")
        print("   - Birdeye API: https://birdeye.so (optional for Telegram)")
        print("   - Telegram API: https://my.telegram.org (optional for Telegram)")
        print("   - Solana RPC: Your P9 provider or other RPC endpoint")
        print()
        print("2. Create wallets.txt file with wallet addresses")
        print("   - One wallet address per line")
        print("   - Use # for comments")
        print("   - Example:")
        print("     # My favorite wallets")
        print("     DJPKomwbTTsjyc3bZZZayE9mHhwAkJHkpwRvePYjV9VR")
        print("     9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM")
        print()
        print("3. Run analysis (Options 5-7)")
        print("   - Option 6: Auto-analyze all wallets in wallets.txt")
        print("   - Option 7: Analyze a single wallet")
        
        print("\n✨ NEW FEATURES:")
        print("• 💰 Cielo Finance Integration - Professional wallet P&L analysis") 
        print("• ⚔️  Contested Wallet Analysis - Detect copy traders using RPC calls")
        print("• 🌐 Custom RPC Support - Use P9 or other providers")
        print("• 📊 Comprehensive Metrics - ROI distribution, platform analysis")
        print("• 🎯 Smart Strategy Generation - Competition-aware recommendations")
        print("• 📈 Excel Export - Multi-sheet detailed reports")
        
        print("\n💡 TIPS:")
        print("• Cielo Finance API key is REQUIRED for wallet analysis")
        print("• Configure your P9 RPC URL for best performance")
        print("• Contested analysis shows how many traders copy each wallet")
        print("• Use fast RPC providers for better contested detection")
        print("• Check outputs folder for all generated files")
        print("• Results include trading strategies and risk assessments")
        
        print("\n🔧 RPC PROVIDERS:")
        print("• P9: Fast, reliable (your current provider)")
        print("• QuickNode: Professional grade")
        print("• Alchemy: Developer-friendly")
        print("• Free Solana RPC: Basic but functional")
        
        input("\nPress Enter to continue...")
    
    def _interactive_add_sources(self):
        """Interactive add sources menu."""
        print("\n" + "="*42)
        print("    ADD DATA SOURCES")
        print("="*42)
        
        print("\nWhat would you like to add?")
        print("1. Telegram Channel/Group")
        print("2. Edit wallets.txt file")
        print("0. Back to main menu")
        
        choice = input("\nEnter your choice (0-2): ").strip()
        
        if choice == '0':
            return
        elif choice == '1':
            channel = input("\nEnter Telegram channel ID or username: ").strip()
            if channel:
                if channel not in self.config["sources"]["telegram_groups"]:
                    self.config["sources"]["telegram_groups"].append(channel)
                    save_config(self.config)
                    print(f"✅ Added Telegram channel: {channel}")
                else:
                    print(f"ℹ️ Channel already exists: {channel}")
        elif choice == '2':
            print("\n💡 To edit wallets, modify the wallets.txt file directly.")
            print("   - Add one wallet address per line")
            print("   - Use # for comments")
            print("   - Save the file and use Option 9 to verify")
        else:
            print("❌ Invalid choice.")
        
        input("Press Enter to continue...")
    
    def _view_current_sources(self):
        """View current configured sources."""
        print("\n" + "="*50)
        print("    CURRENT DATA SOURCES")
        print("="*50)
        
        # Telegram channels
        telegram_channels = self.config.get('sources', {}).get('telegram_groups', [])
        print(f"\n📊 Telegram Channels ({len(telegram_channels)}):")
        if telegram_channels:
            for i, channel in enumerate(telegram_channels, 1):
                print(f"   {i}. {channel}")
        else:
            print("   None configured")
        
        # Wallets from file
        wallets_from_file = load_wallets_from_file("wallets.txt")
        print(f"\n💰 Wallets from wallets.txt ({len(wallets_from_file)}):")
        if wallets_from_file:
            for i, wallet in enumerate(wallets_from_file[:10], 1):  # Show first 10
                print(f"   {i}. {wallet}")
            if len(wallets_from_file) > 10:
                print(f"   ... and {len(wallets_from_file) - 10} more")
        else:
            print("   None found in wallets.txt")
        
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
        
        if args.cielo_api_key:
            self.config["cielo_api_key"] = args.cielo_api_key
            logger.info("Cielo Finance API key configured.")
        
        if args.rpc_url:
            self.config["solana_rpc_url"] = args.rpc_url
            logger.info(f"Solana RPC URL configured: {args.rpc_url}")
        
        if args.telegram_api_id:
            self.config["telegram_api_id"] = args.telegram_api_id
            logger.info("Telegram API ID configured.")
        
        if args.telegram_api_hash:
            self.config["telegram_api_hash"] = args.telegram_api_hash
            logger.info("Telegram API hash configured.")
        
        save_config(self.config)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
    
    def _handle_telegram_analysis(self, args) -> None:
        """Handle the telegram analysis command."""
        import asyncio
        from telegram_module import TelegramScraper
        from birdeye_api import BirdeyeAPI
        
        channels = getattr(args, 'channels', None) or self.config["sources"]["telegram_groups"]
        if not channels:
            logger.error("No Telegram channels specified.")
            return
        
        if not self.config.get("birdeye_api_key"):
            logger.error("Birdeye API key not configured.")
            return
            
        if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
            logger.error("Telegram API credentials not configured.")
            return
        
        output_file = ensure_output_dir(args.output)
        
        logger.info(f"Analyzing SpyDefi channel for the past {args.days} day(s).")
        logger.info(f"Results will be saved to {output_file}")
        
        birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
        telegram_scraper = TelegramScraper(
            self.config["telegram_api_id"],
            self.config["telegram_api_hash"],
            self.config.get("telegram_session", "phoenix")
        )
        
        telegram_analyses = {"ranked_kols": []}
        
        if any(ch.lower() == "spydefi" for ch in channels):
            logger.info("SpyDefi channel detected. Analyzing KOLs from SpyDefi.")
            
            try:
                async def run_spydefi_analysis():
                    try:
                        await telegram_scraper.connect()
                        analysis = await telegram_scraper.scrape_spydefi(
                            "spydefi",
                            args.days,
                            birdeye_api
                        )
                        await telegram_scraper.export_spydefi_analysis(analysis, output_file)
                        return analysis
                    finally:
                        await telegram_scraper.disconnect()
                
                telegram_analyses = asyncio.run(run_spydefi_analysis())
                logger.info(f"SpyDefi analysis completed.")
                
            except Exception as e:
                logger.error(f"Error analyzing SpyDefi: {str(e)}")
                return
        
        logger.info(f"Telegram analysis completed. Results saved to {output_file}")
        
        if hasattr(args, 'excel') and args.excel:
            try:
                from export_utils import export_to_excel
                excel_file = output_file.replace(".csv", ".xlsx")
                export_to_excel(telegram_analyses, {}, excel_file)
                logger.info(f"Excel export completed: {excel_file}")
            except Exception as e:
                logger.error(f"Error exporting to Excel: {str(e)}")
    
    def _handle_wallet_analysis(self, args: argparse.Namespace) -> None:
        """Handle the wallet analysis command."""
        wallets_to_analyze = load_wallets_from_file(args.wallets_file)
        
        if not wallets_to_analyze:
            logger.error(f"No valid wallets found in {args.wallets_file}")
            return
        
        if not self.config.get("cielo_api_key"):
            logger.error("Cielo Finance API key not configured. This is REQUIRED for wallet analysis.")
            return
        
        logger.info(f"Running enhanced wallet analysis on {len(wallets_to_analyze)} wallets")
        logger.info(f"RPC URL: {self.config.get('solana_rpc_url', 'Default')}")
        
        try:
            self._run_enhanced_wallet_analysis(
                wallets_to_analyze,
                args.days,
                args.min_winrate,
                args.output,
                not args.no_contested  # Include contested unless --no-contested flag
            )
            
        except Exception as e:
            logger.error(f"Enhanced wallet analysis failed: {str(e)}")

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