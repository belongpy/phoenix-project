#!/usr/bin/env python3
"""
Phoenix Project - Solana Chain Analysis CLI Tool

A menu-driven interface tool for analyzing Solana blockchain signals and wallet behaviors
using Birdeye API and Helius API for enhanced wallet transaction data.
"""

import os
import sys
import json
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time
import requests

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

# Configuration and file paths
CONFIG_FILE = os.path.expanduser("~/.phoenix_config.json")
DEFAULT_WALLET_FILE = "wallets.txt" 
DEFAULT_CHANNEL_FILE = "channels.txt"

def load_config() -> Dict[str, Any]:
    """Load configuration from the config file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    
    # Create default config if it doesn't exist
    default_config = {
        "birdeye_api_key": "",
        "helius_api_key": "",
        "telegram_api_id": "",
        "telegram_api_hash": "",
        "telegram_session": "phoenix",
        "sources": {
            "telegram_groups": [],
            "wallets": []
        },
        "settings": {
            "telegram_days": 7,
            "wallet_days": 30,
            "min_winrate": 45.0,
            "use_excel": True,
            "wallet_file": DEFAULT_WALLET_FILE,
            "channel_file": DEFAULT_CHANNEL_FILE,
            "output_dir": "outputs",
            "max_retries": 5,
            "retry_delay": 2,
            "rate_limit_pause": 10
        }
    }
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(default_config, f, indent=4)
    logger.info(f"Created default configuration file at {CONFIG_FILE}")
    
    return default_config

def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to the config file."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {CONFIG_FILE}")

def ensure_output_dir(config: Dict[str, Any]) -> str:
    """
    Ensure the output directory exists and return the path.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        str: Path to the output directory
    """
    output_dir = config.get("settings", {}).get("output_dir", "outputs")
    
    # Create outputs directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created outputs directory: {output_dir}")
    
    return output_dir

def load_wallets_from_file(file_path: str) -> List[str]:
    """
    Load wallet addresses from a text file (one address per line).
    
    Args:
        file_path (str): Path to the wallet addresses file
        
    Returns:
        List[str]: List of wallet addresses
    """
    if not os.path.exists(file_path):
        logger.error(f"Wallet file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            # Read lines and strip whitespace, ignore empty lines and comments
            wallets = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        logger.info(f"Loaded {len(wallets)} wallet addresses from {file_path}")
        return wallets
    except Exception as e:
        logger.error(f"Error reading wallet file: {str(e)}")
        return []

def load_channels_from_file(file_path: str) -> List[str]:
    """
    Load Telegram channel IDs from a text file (one ID per line).
    
    Args:
        file_path (str): Path to the channel IDs file
        
    Returns:
        List[str]: List of channel IDs
    """
    if not os.path.exists(file_path):
        logger.error(f"Channel file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r') as f:
            # Read lines and strip whitespace, ignore empty lines and comments
            channels = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        logger.info(f"Loaded {len(channels)} channel IDs from {file_path}")
        return channels
    except Exception as e:
        logger.error(f"Error reading channel file: {str(e)}")
        return []

def validate_wallet_address(address: str) -> bool:
    """
    Validate a Solana wallet address format.
    
    Args:
        address (str): Wallet address to validate
        
    Returns:
        bool: True if the address format is valid
    """
    # Basic validation - Solana addresses are typically base58 encoded
    # and between 32-44 characters
    if not address:
        return False
    
    # Check length
    if not 32 <= len(address) <= 44:
        return False
    
    # Check for valid base58 characters (alphanumeric without 0, O, I, l)
    valid_chars = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
    return all(c in valid_chars for c in address)

def validate_telegram_id(channel_id: str) -> bool:
    """
    Validate a Telegram channel/group ID or username.
    
    Args:
        channel_id (str): Channel ID to validate
        
    Returns:
        bool: True if the ID format is valid
    """
    if not channel_id:
        return False
    
    # Handle usernames
    if channel_id.startswith('@'):
        username = channel_id[1:]
        # Usernames must be 5-32 characters and contain only allowed characters
        if not 5 <= len(username) <= 32:
            return False
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
        return all(c in allowed_chars for c in username)
    
    # Handle channel IDs (usually numeric)
    if channel_id.isdigit():
        return True
    
    # Handle channel links (t.me/...)
    if channel_id.startswith(('https://t.me/', 't.me/')):
        return True
    
    # Special case for SpyDefi
    if channel_id.lower() == "spydefi":
        return True
    
    # For other formats, just ensure it's not empty and has a reasonable length
    return 3 <= len(channel_id) <= 100

def test_helius_api(api_key: str) -> Dict[str, Any]:
    """
    Test Helius API connectivity and return status.
    
    Args:
        api_key (str): Helius API key to test
        
    Returns:
        Dict[str, Any]: Test result with status and message
    """
    if not api_key:
        return {"success": False, "message": "No API key provided"}
    
    try:
        # Try a simple API call - get transactions for a known wallet
        test_wallet = "11111111111111111111111111111112"  # System program address (always exists)
        url = f"https://api.helius.xyz/v0/addresses/{test_wallet}/transactions"
        params = {"api-key": api_key, "limit": 1}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            return {"success": True, "message": "API connection successful"}
        elif response.status_code == 401 or response.status_code == 403:
            return {"success": False, "message": f"API key unauthorized (Status {response.status_code})"}
        else:
            return {"success": False, "message": f"API request failed with status {response.status_code}"}
            
    except requests.RequestException as e:
        return {"success": False, "message": f"Connection error: {str(e)}"}

def test_birdeye_api(api_key: str) -> Dict[str, Any]:
    """
    Test Birdeye API connectivity and return status.
    
    Args:
        api_key (str): Birdeye API key to test
        
    Returns:
        Dict[str, Any]: Test result with status and message
    """
    if not api_key:
        return {"success": False, "message": "No API key provided"}
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    
    try:
        # Try a simple API call - get SOL token info
        url = "https://public-api.birdeye.so/defi/token_overview?address=So11111111111111111111111111111111111111112"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return {"success": True, "message": "API connection successful"}
        elif response.status_code == 401 or response.status_code == 403:
            return {"success": False, "message": f"API key unauthorized (Status {response.status_code})"}
        else:
            return {"success": False, "message": f"API request failed with status {response.status_code}"}
            
    except requests.RequestException as e:
        return {"success": False, "message": f"Connection error: {str(e)}"}

def print_header():
    """Print the Phoenix CLI header."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "=" * 60)
    print("             PHOENIX - Solana Chain Analysis Tool")
    print("                    with Helius Integration")
    print("=" * 60)

def print_menu():
    """Print the main menu options."""
    print("\nMAIN MENU:")
    print("1. Wallet Analysis")
    print("2. Telegram Analysis")
    print("3. Combined Analysis")
    print("4. Configuration")
    print("5. Exit")
    print("\nSelect an option (1-5): ", end='')

def print_config_menu():
    """Print the configuration menu options."""
    print("\nCONFIGURATION MENU:")
    print("1. Set API Keys")
    print("2. Set Analysis Settings")
    print("3. Manage File Paths")
    print("4. API Connection Test")
    print("5. Advanced Settings")
    print("6. Return to Main Menu")
    print("\nSelect an option (1-6): ", end='')

def set_api_keys(config: Dict[str, Any]):
    """Set API keys in the configuration."""
    print_header()
    print("\nAPI KEY CONFIGURATION")
    print("-" * 60)
    print("\nHelius API is recommended for wallet transaction data.")
    print("Birdeye API is used for token pricing and market data.")
    print()
    
    birdeye_key = input(f"Birdeye API Key [{config.get('birdeye_api_key', '')}]: ").strip()
    if birdeye_key:
        config["birdeye_api_key"] = birdeye_key
    
    helius_key = input(f"Helius API Key (recommended) [{config.get('helius_api_key', '')}]: ").strip()
    if helius_key:
        config["helius_api_key"] = helius_key
    
    telegram_api_id = input(f"Telegram API ID [{config.get('telegram_api_id', '')}]: ").strip()
    if telegram_api_id:
        config["telegram_api_id"] = telegram_api_id
    
    telegram_api_hash = input(f"Telegram API Hash [{config.get('telegram_api_hash', '')}]: ").strip()
    if telegram_api_hash:
        config["telegram_api_hash"] = telegram_api_hash
    
    # Save the configuration
    save_config(config)
    print("\nAPI keys updated successfully!")
    input("\nPress Enter to continue...")

def set_analysis_settings(config: Dict[str, Any]):
    """Set analysis settings in the configuration."""
    print_header()
    print("\nANALYSIS SETTINGS")
    print("-" * 60)
    
    settings = config.get("settings", {})
    
    # Get and validate Telegram days
    try:
        telegram_days_str = input(f"Telegram Analysis Days [{settings.get('telegram_days', 7)}]: ").strip()
        if telegram_days_str:
            telegram_days = int(telegram_days_str)
            if telegram_days > 0:
                settings["telegram_days"] = telegram_days
            else:
                print("Days must be a positive number. Using default.")
    except ValueError:
        print("Invalid input. Keeping existing value.")
    
    # Get and validate Wallet days
    try:
        wallet_days_str = input(f"Wallet Analysis Days [{settings.get('wallet_days', 30)}]: ").strip()
        if wallet_days_str:
            wallet_days = int(wallet_days_str)
            if wallet_days > 0:
                settings["wallet_days"] = wallet_days
            else:
                print("Days must be a positive number. Using default.")
    except ValueError:
        print("Invalid input. Keeping existing value.")
    
    # Get and validate Min Win Rate
    try:
        min_winrate_str = input(f"Minimum Win Rate % [{settings.get('min_winrate', 45.0)}]: ").strip()
        if min_winrate_str:
            min_winrate = float(min_winrate_str)
            if 0 <= min_winrate <= 100:
                settings["min_winrate"] = min_winrate
            else:
                print("Win rate must be between 0 and 100. Using default.")
    except ValueError:
        print("Invalid input. Keeping existing value.")
    
    # Set Excel output option
    use_excel_str = input(f"Always use Excel output (yes/no) [{settings.get('use_excel', True)}]: ").strip().lower()
    if use_excel_str in ("yes", "y", "true", "t"):
        settings["use_excel"] = True
    elif use_excel_str in ("no", "n", "false", "f"):
        settings["use_excel"] = False
    
    # Update config and save
    config["settings"] = settings
    save_config(config)
    print("\nAnalysis settings updated successfully!")
    input("\nPress Enter to continue...")

def manage_file_paths(config: Dict[str, Any]):
    """Manage file paths in the configuration."""
    print_header()
    print("\nFILE PATH SETTINGS")
    print("-" * 60)
    
    settings = config.get("settings", {})
    
    wallet_file = input(f"Wallet File Path [{settings.get('wallet_file', DEFAULT_WALLET_FILE)}]: ").strip()
    if wallet_file:
        settings["wallet_file"] = wallet_file
    
    channel_file = input(f"Channel File Path [{settings.get('channel_file', DEFAULT_CHANNEL_FILE)}]: ").strip()
    if channel_file:
        settings["channel_file"] = channel_file
    
    output_dir = input(f"Output Directory [{settings.get('output_dir', 'outputs')}]: ").strip()
    if output_dir:
        settings["output_dir"] = output_dir
    
    # Update config and save
    config["settings"] = settings
    save_config(config)
    print("\nFile paths updated successfully!")
    input("\nPress Enter to continue...")

def set_advanced_settings(config: Dict[str, Any]):
    """Set advanced settings in the configuration."""
    print_header()
    print("\nADVANCED SETTINGS")
    print("-" * 60)
    
    settings = config.get("settings", {})
    
    try:
        max_retries_str = input(f"API Max Retries [{settings.get('max_retries', 5)}]: ").strip()
        if max_retries_str:
            max_retries = int(max_retries_str)
            if max_retries > 0:
                settings["max_retries"] = max_retries
            else:
                print("Max retries must be a positive number. Using default.")
    except ValueError:
        print("Invalid input. Keeping existing value.")
    
    try:
        retry_delay_str = input(f"Initial Retry Delay (seconds) [{settings.get('retry_delay', 2)}]: ").strip()
        if retry_delay_str:
            retry_delay = int(retry_delay_str)
            if retry_delay > 0:
                settings["retry_delay"] = retry_delay
            else:
                print("Retry delay must be a positive number. Using default.")
    except ValueError:
        print("Invalid input. Keeping existing value.")
    
    try:
        rate_limit_pause_str = input(f"Rate Limit Pause (seconds) [{settings.get('rate_limit_pause', 10)}]: ").strip()
        if rate_limit_pause_str:
            rate_limit_pause = int(rate_limit_pause_str)
            if rate_limit_pause > 0:
                settings["rate_limit_pause"] = rate_limit_pause
            else:
                print("Rate limit pause must be a positive number. Using default.")
    except ValueError:
        print("Invalid input. Keeping existing value.")
    
    # Update config and save
    config["settings"] = settings
    save_config(config)
    print("\nAdvanced settings updated successfully!")
    input("\nPress Enter to continue...")

def test_api_connection(config: Dict[str, Any]):
    """Test API connections and display results."""
    print_header()
    print("\nAPI CONNECTION TEST")
    print("-" * 60)
    
    # Test Birdeye API
    print("\nTesting Birdeye API connection...")
    birdeye_key = config.get("birdeye_api_key", "")
    if not birdeye_key:
        print("  ❌ No Birdeye API key configured!")
    else:
        result = test_birdeye_api(birdeye_key)
        if result["success"]:
            print(f"  ✅ {result['message']}")
        else:
            print(f"  ❌ {result['message']}")
    
    # Test Helius API
    print("\nTesting Helius API connection...")
    helius_key = config.get("helius_api_key", "")
    if not helius_key:
        print("  ❌ No Helius API key configured!")
        print("     Wallet analysis will use Birdeye (may have limitations)")
    else:
        result = test_helius_api(helius_key)
        if result["success"]:
            print(f"  ✅ {result['message']}")
            print("     Wallet analysis will use Helius (recommended)")
        else:
            print(f"  ❌ {result['message']}")
    
    # Check Telegram credentials
    print("\nChecking Telegram API credentials...")
    telegram_api_id = config.get("telegram_api_id", "")
    telegram_api_hash = config.get("telegram_api_hash", "")
    
    if not telegram_api_id or not telegram_api_hash:
        print("  ❌ Telegram API credentials are missing!")
    else:
        print("  ✅ Telegram API credentials are configured")
        print("     (Note: Full connection test requires user interaction)")
    
    # Check for wallet file
    wallet_file = config.get("settings", {}).get("wallet_file", DEFAULT_WALLET_FILE)
    print(f"\nChecking wallet file: {wallet_file}")
    if os.path.exists(wallet_file):
        wallets = load_wallets_from_file(wallet_file)
        print(f"  ✅ Found {len(wallets)} wallet addresses")
        
        # Validate a sample of wallets
        if wallets:
            sample_size = min(5, len(wallets))
            invalid_wallets = []
            
            for i in range(sample_size):
                if not validate_wallet_address(wallets[i]):
                    invalid_wallets.append(wallets[i])
            
            if invalid_wallets:
                print(f"  ⚠️  Warning: {len(invalid_wallets)} wallet(s) have invalid format")
                for w in invalid_wallets[:3]:  # Show first 3 invalid wallets
                    print(f"      - {w}")
                if len(invalid_wallets) > 3:
                    print(f"      - ... and {len(invalid_wallets) - 3} more")
    else:
        print(f"  ❌ Wallet file not found")
    
    # Check for channel file
    channel_file = config.get("settings", {}).get("channel_file", DEFAULT_CHANNEL_FILE)
    print(f"\nChecking channel file: {channel_file}")
    if os.path.exists(channel_file):
        channels = load_channels_from_file(channel_file)
        print(f"  ✅ Found {len(channels)} channel IDs")
        
        # Validate a sample of channels
        if channels:
            sample_size = min(5, len(channels))
            invalid_channels = []
            
            for i in range(sample_size):
                if not validate_telegram_id(channels[i]):
                    invalid_channels.append(channels[i])
            
            if invalid_channels:
                print(f"  ⚠️  Warning: {len(invalid_channels)} channel(s) have suspicious format")
                for c in invalid_channels[:3]:  # Show first 3 invalid channels
                    print(f"      - {c}")
                if len(invalid_channels) > 3:
                    print(f"      - ... and {len(invalid_channels) - 3} more")
                    
            # Check for SpyDefi
            has_spydefi = any(ch.lower() == "spydefi" for ch in channels)
            if has_spydefi:
                print("  ✓ SpyDefi channel found (KOL analysis available)")
    else:
        print(f"  ❌ Channel file not found")
    
    # Check for output directory
    output_dir = config.get("settings", {}).get("output_dir", "outputs")
    print(f"\nChecking output directory: {output_dir}")
    if os.path.exists(output_dir):
        print(f"  ✅ Output directory exists")
    else:
        print(f"  ⚠️  Output directory does not exist (will be created when needed)")
    
    print("\nAPI connection test completed!")
    input("\nPress Enter to continue...")

def handle_wallet_analysis(config: Dict[str, Any]):
    """Handle wallet analysis functionality."""
    try:
        from wallet_module import WalletAnalyzer
        from birdeye_api import BirdeyeAPI
        
        print_header()
        print("\nWALLET ANALYSIS")
        print("-" * 60)
        
        # Check for API keys
        if not config.get("birdeye_api_key"):
            print("\nError: Birdeye API key not configured!")
            print("Please set up your API keys in the Configuration menu.")
            input("\nPress Enter to continue...")
            return
        
        # Load settings from config
        settings = config.get("settings", {})
        days = settings.get("wallet_days", 30)
        min_winrate = settings.get("min_winrate", 45.0)
        use_excel = settings.get("use_excel", True)
        wallet_file = settings.get("wallet_file", DEFAULT_WALLET_FILE)
        output_dir = ensure_output_dir(config)
        
        # Load wallets from file
        print(f"\nLoading wallets from file: {wallet_file}")
        wallets = load_wallets_from_file(wallet_file)
        
        if not wallets:
            print(f"\nNo wallet addresses found in {wallet_file}!")
            print(f"Please add wallet addresses to {wallet_file} (one per line).")
            input("\nPress Enter to continue...")
            return
        
        # Validate wallets
        invalid_wallets = [w for w in wallets if not validate_wallet_address(w)]
        if invalid_wallets:
            print(f"\nWarning: {len(invalid_wallets)} wallet addresses have invalid format.")
            print("Invalid addresses may cause errors during analysis.")
            
            if len(invalid_wallets) <= 5:
                print("\nInvalid addresses:")
                for w in invalid_wallets:
                    print(f"- {w}")
            else:
                print(f"\nFirst 5 invalid addresses:")
                for w in invalid_wallets[:5]:
                    print(f"- {w}")
                print(f"... and {len(invalid_wallets) - 5} more")
            
            continue_anyway = input("\nContinue anyway? (yes/no): ").strip().lower()
            if continue_anyway not in ("yes", "y", "true", "t"):
                print("\nAnalysis cancelled.")
                input("\nPress Enter to continue...")
                return
        
        print(f"\nFound {len(wallets)} wallet addresses.")
        print(f"Analysis period: {days} days")
        print(f"Minimum win rate: {min_winrate}%")
        print(f"Output directory: {output_dir}")
        
        # Show API status
        if config.get("helius_api_key"):
            print("✅ Using Helius API for wallet transaction data (recommended)")
        else:
            print("⚠️  Using Birdeye API for wallet data (Helius recommended for better results)")
        
        proceed = input("\nProceed with analysis? (yes/no): ").strip().lower()
        if proceed not in ("yes", "y", "true", "t"):
            print("\nAnalysis cancelled.")
            input("\nPress Enter to continue...")
            return
        
        # Initialize API client and wallet analyzer
        print("\nInitializing APIs...")
        
        # Create BirdeyeAPI with both keys
        birdeye_api = BirdeyeAPI(
            api_key=config["birdeye_api_key"],
            helius_api_key=config.get("helius_api_key"),
            max_retries=settings.get("max_retries", 5),
            retry_delay=settings.get("retry_delay", 2),
            rate_limit_pause=settings.get("rate_limit_pause", 10)
        )
        
        wallet_analyzer = WalletAnalyzer(birdeye_api)
        
        # Prepare output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(output_dir, f"wallet_analysis_{timestamp}")
        output_file = f"{output_base}.csv"
        
        print(f"\nAnalyzing {len(wallets)} wallets...")
        
        try:
            # Batch analyze wallets
            wallet_analyses = wallet_analyzer.batch_analyze_wallets(
                wallets,
                days,
                min_winrate
            )
            
            wallet_analyzer.export_batch_analysis(wallet_analyses, output_file)
            print(f"\nBatch analysis completed!")
            
            # Print summary
            print("\nAnalysis Summary:")
            print(f"- Total wallets: {wallet_analyses['total_wallets']}")
            print(f"- Analyzed wallets: {wallet_analyses['analyzed_wallets']}")
            print(f"- Filtered wallets: {wallet_analyses['filtered_wallets']}")
            print(f"- Gem Finders: {len(wallet_analyses.get('gem_finders', []))}")
            print(f"- Consistent: {len(wallet_analyses.get('consistent', []))}")
            print(f"- Flippers: {len(wallet_analyses.get('flippers', []))}")
            
            print(f"\nResults saved to CSV: {output_file}")
            
            # Export to Excel if requested
            if use_excel:
                try:
                    from export_utils import export_to_excel
                    excel_file = f"{output_base}.xlsx"
                    export_to_excel({}, wallet_analyses, excel_file)
                    print(f"Results saved to Excel: {excel_file}")
                except Exception as e:
                    logger.error(f"Error exporting to Excel: {str(e)}")
                    print(f"Error exporting to Excel: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error during wallet analysis: {str(e)}", exc_info=True)
            print(f"\nError during analysis: {str(e)}")
        
        print("\nWallet analysis completed!")
        input("\nPress Enter to continue...")
    
    except ImportError as e:
        print(f"\nError: Could not import required module: {str(e)}")
        print("Please ensure all dependencies are installed.")
        input("\nPress Enter to continue...")

def handle_telegram_analysis(config: Dict[str, Any]):
    """Handle Telegram analysis functionality."""
    try:
        import asyncio
        from telegram_module import TelegramScraper
        from birdeye_api import BirdeyeAPI
        
        print_header()
        print("\nTELEGRAM ANALYSIS")
        print("-" * 60)
        
        # Check for required API keys
        if not config.get("birdeye_api_key"):
            print("\nError: Birdeye API key not configured!")
            print("Please set up your API keys in the Configuration menu.")
            input("\nPress Enter to continue...")
            return
        
        if not config.get("telegram_api_id") or not config.get("telegram_api_hash"):
            print("\nError: Telegram API credentials not configured!")
            print("Please set up your API keys in the Configuration menu.")
            input("\nPress Enter to continue...")
            return
        
        # Load settings from config
        settings = config.get("settings", {})
        days = settings.get("telegram_days", 7)
        use_excel = settings.get("use_excel", True)
        channel_file = settings.get("channel_file", DEFAULT_CHANNEL_FILE)
        output_dir = ensure_output_dir(config)
        
        # Load channels from file
        print(f"\nLoading channels from file: {channel_file}")
        channels = load_channels_from_file(channel_file)
        
        if not channels:
            print(f"\nNo channel IDs found in {channel_file}!")
            print(f"Please add channel IDs to {channel_file} (one per line).")
            input("\nPress Enter to continue...")
            return
        
        # Validate channels
        invalid_channels = [c for c in channels if not validate_telegram_id(c)]
        if invalid_channels:
            print(f"\nWarning: {len(invalid_channels)} channel IDs have suspicious format.")
            print("Invalid channels may cause errors during analysis.")
            
            if len(invalid_channels) <= 5:
                print("\nSuspicious channels:")
                for c in invalid_channels:
                    print(f"- {c}")
            else:
                print(f"\nFirst 5 suspicious channels:")
                for c in invalid_channels[:5]:
                    print(f"- {c}")
                print(f"... and {len(invalid_channels) - 5} more")
            
            continue_anyway = input("\nContinue anyway? (yes/no): ").strip().lower()
            if continue_anyway not in ("yes", "y", "true", "t"):
                print("\nAnalysis cancelled.")
                input("\nPress Enter to continue...")
                return
        
        print(f"\nFound {len(channels)} channel IDs.")
        print(f"Analysis period: {days} days")
        print(f"Output directory: {output_dir}")
        
        # Check for Spydefi channel
        has_spydefi = any(ch.lower() == "spydefi" for ch in channels)
        if has_spydefi:
            print("\nNote: Spydefi channel detected! KOL analysis will be performed.")
            
            # Inform about the need to provide phone number for first run
            print("\nIMPORTANT: When connecting to Telegram for the first time, you'll need to:")
            print("1. Enter your phone number (e.g., +15623778250)")
            print("2. Enter the verification code sent to your Telegram app")
            print("3. This is only required once as the session will be saved")
        
        proceed = input("\nProceed with analysis? (yes/no): ").strip().lower()
        if proceed not in ("yes", "y", "true", "t"):
            print("\nAnalysis cancelled.")
            input("\nPress Enter to continue...")
            return
        
        # Initialize API clients
        print("\nInitializing APIs...")
        
        # Create BirdeyeAPI with both keys
        birdeye_api = BirdeyeAPI(
            api_key=config["birdeye_api_key"],
            helius_api_key=config.get("helius_api_key"),
            max_retries=settings.get("max_retries", 5),
            retry_delay=settings.get("retry_delay", 2),
            rate_limit_pause=settings.get("rate_limit_pause", 10)
        )
        
        telegram_scraper = TelegramScraper(
            config["telegram_api_id"],
            config["telegram_api_hash"],
            config.get("telegram_session", "phoenix"),
            max_days=days
        )
        
        # Prepare output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(output_dir, f"telegram_analysis_{timestamp}")
        output_file = f"{output_base}.csv"
        
        print(f"\nAnalyzing {len(channels)} Telegram channels...")
        print("\nNote: If this is your first time, you'll need to authenticate with Telegram.")
        print("Please check the console for prompts to enter your phone number and verification code.")
        
        # Store analysis results
        telegram_analyses = {"ranked_kols": []}
        
        # Handle Spydefi specifically
        if has_spydefi:
            try:
                # Find the Spydefi channel ID
                spydefi_channel = next((ch for ch in channels if ch.lower() == "spydefi"), None)
                print(f"\nAnalyzing Spydefi and discovering KOLs...")
                
                # Run the analysis asynchronously
                async def run_spydefi_analysis():
                    try:
                        await telegram_scraper.connect()
                        analysis = await telegram_scraper.scrape_spydefi(
                            spydefi_channel,
                            days,
                            birdeye_api
                        )
                        spydefi_output = f"{output_base}_spydefi.csv"
                        await telegram_scraper.export_spydefi_analysis(analysis, spydefi_output)
                        return analysis
                    finally:
                        await telegram_scraper.disconnect()
                
                # Run the async function
                telegram_analyses = asyncio.run(run_spydefi_analysis())
                print(f"\nSpydefi analysis completed!")
                print(f"Found {len(telegram_analyses.get('ranked_kols', []))} KOL channels.")
                
            except Exception as e:
                logger.error(f"Error analyzing Spydefi: {str(e)}", exc_info=True)
                print(f"\nError analyzing Spydefi: {str(e)}")
                
                if "Cannot find any entity" in str(e):
                    print("\nTroubleshooting tip: Make sure the Spydefi channel name/ID is correct.")
                    print("The channel might have changed its username or been deleted.")
                elif "SessionPasswordNeededError" in str(e):
                    print("\nTroubleshooting tip: Your Telegram account has two-factor authentication.")
                    print("Please delete the 'phoenix.session' file and try again.")
                
        # Handle regular channels
        else:
            channel_analyses = []
            for idx, channel in enumerate(channels):
                try:
                    print(f"\nAnalyzing channel {idx+1}/{len(channels)}: {channel}...")
                    
                    # Run the analysis asynchronously
                    async def run_channel_analysis():
                        try:
                            await telegram_scraper.connect()
                            analysis = await telegram_scraper.analyze_channel(
                                channel,
                                days,
                                birdeye_api
                            )
                            
                            # Customize output file for each channel
                            channel_output = f"{output_base}_{channel.replace('@', '')}.csv"
                            await telegram_scraper.export_channel_analysis(analysis, channel_output)
                            return analysis
                        finally:
                            await telegram_scraper.disconnect()
                    
                    # Run the async function
                    analysis = asyncio.run(run_channel_analysis())
                    channel_analyses.append(analysis)
                    print(f"Analysis for channel {channel} completed")
                    
                except Exception as e:
                    logger.error(f"Error analyzing channel {channel}: {str(e)}", exc_info=True)
                    print(f"Error analyzing channel {channel}: {str(e)}")
                    
                    if "Cannot find any entity" in str(e):
                        print(f"Troubleshooting tip: Unable to find channel '{channel}'.")
                        print("Make sure the channel ID/username is correct and your account has access to it.")
                    elif "FloodWaitError" in str(e):
                        wait_time = int(str(e).split("of ")[1].split(" seconds")[0])
                        print(f"Telegram rate limit hit. Must wait {wait_time} seconds.")
                        print(f"Pausing for {wait_time + 5} seconds before continuing...")
                        time.sleep(wait_time + 5)
            
            telegram_analyses = {
                "ranked_kols": channel_analyses
            }
        
        # Export to Excel if requested
        if use_excel:
            try:
                from export_utils import export_to_excel
                excel_file = f"{output_base}.xlsx"
                export_to_excel(telegram_analyses, {}, excel_file)
                print(f"\nResults saved to Excel: {excel_file}")
            except Exception as e:
                logger.error(f"Error exporting to Excel: {str(e)}")
                print(f"Error exporting to Excel: {str(e)}")
        
        print("\nTelegram analysis completed!")
        input("\nPress Enter to continue...")
    
    except ImportError as e:
        print(f"\nError: Could not import required module: {str(e)}")
        print("Please ensure all dependencies are installed.")
        input("\nPress Enter to continue...")

def handle_combined_analysis(config: Dict[str, Any]):
    """Handle combined Telegram and wallet analysis."""
    try:
        import asyncio
        from telegram_module import TelegramScraper
        from wallet_module import WalletAnalyzer
        from birdeye_api import BirdeyeAPI
        from export_utils import export_to_excel
        
        print_header()
        print("\nCOMBINED ANALYSIS")
        print("-" * 60)
        
        # Check for required API keys
        if not config.get("birdeye_api_key"):
            print("\nError: Birdeye API key not configured!")
            print("Please set up your API keys in the Configuration menu.")
            input("\nPress Enter to continue...")
            return
        
        if not config.get("telegram_api_id") or not config.get("telegram_api_hash"):
            print("\nError: Telegram API credentials not configured!")
            print("Please set up your API keys in the Configuration menu.")
            input("\nPress Enter to continue...")
            return
        
        # Load settings from config
        settings = config.get("settings", {})
        telegram_days = settings.get("telegram_days", 7)
        wallet_days = settings.get("wallet_days", 30)
        min_winrate = settings.get("min_winrate", 45.0)
        wallet_file = settings.get("wallet_file", DEFAULT_WALLET_FILE)
        channel_file = settings.get("channel_file", DEFAULT_CHANNEL_FILE)
        output_dir = ensure_output_dir(config)
        
        # Load channels and wallets from files
        print(f"\nLoading channels from file: {channel_file}")
        channels = load_channels_from_file(channel_file)
        
        print(f"Loading wallets from file: {wallet_file}")
        wallets = load_wallets_from_file(wallet_file)
        
        if not channels and not wallets:
            print("\nNo channels or wallets found in config files!")
            print("Please add data sources before running analysis.")
            input("\nPress Enter to continue...")
            return
        
        # Validate wallets and channels
        invalid_wallets = [w for w in wallets if not validate_wallet_address(w)]
        invalid_channels = [c for c in channels if not validate_telegram_id(c)]
        
        if invalid_wallets or invalid_channels:
            print("\nWarnings:")
            if invalid_wallets:
                print(f"- {len(invalid_wallets)} wallet addresses have invalid format")
            if invalid_channels:
                print(f"- {len(invalid_channels)} channel IDs have suspicious format")
            print("These invalid inputs may cause errors during analysis.")
            
            continue_anyway = input("\nContinue anyway? (yes/no): ").strip().lower()
            if continue_anyway not in ("yes", "y", "true", "t"):
                print("\nAnalysis cancelled.")
                input("\nPress Enter to continue...")
                return
        
        print(f"\nFound {len(channels)} channel IDs and {len(wallets)} wallet addresses.")
        print(f"Telegram analysis period: {telegram_days} days")
        print(f"Wallet analysis period: {wallet_days} days")
        print(f"Minimum win rate: {min_winrate}%")
        print(f"Output directory: {output_dir}")
        
        # Show API status
        if config.get("helius_api_key"):
            print("✅ Using Helius API for wallet transaction data (recommended)")
        else:
            print("⚠️  Using Birdeye API for wallet data (Helius recommended for better results)")
        
        # Prepare output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"combined_analysis_{timestamp}.xlsx")
        
        proceed = input("\nProceed with analysis? (yes/no): ").strip().lower()
        if proceed not in ("yes", "y", "true", "t"):
            print("\nAnalysis cancelled.")
            input("\nPress Enter to continue...")
            return
        
        # Initialize API clients
        print("\nInitializing APIs...")
        
        # Create BirdeyeAPI with both keys
        birdeye_api = BirdeyeAPI(
            api_key=config["birdeye_api_key"],
            helius_api_key=config.get("helius_api_key"),
            max_retries=settings.get("max_retries", 5),
            retry_delay=settings.get("retry_delay", 2),
            rate_limit_pause=settings.get("rate_limit_pause", 10)
        )
        
        # Run Telegram analysis
        telegram_analyses = {"ranked_kols": []}
        if channels:
            print(f"\nAnalyzing {len(channels)} Telegram channels...")
            
            telegram_scraper = TelegramScraper(
                config["telegram_api_id"],
                config["telegram_api_hash"],
                config.get("telegram_session", "phoenix"),
                max_days=telegram_days
            )
            
            # Check for Spydefi channel
            has_spydefi = any(ch.lower() == "spydefi" for ch in channels)
            if has_spydefi:
                try:
                    # Find the Spydefi channel ID
                    spydefi_channel = next((ch for ch in channels if ch.lower() == "spydefi"), None)
                    print(f"\nAnalyzing Spydefi and discovering KOLs...")
                    
                    # Run the analysis asynchronously
                    async def run_spydefi_analysis():
                        try:
                            await telegram_scraper.connect()
                            return await telegram_scraper.scrape_spydefi(
                                spydefi_channel,
                                telegram_days,
                                birdeye_api
                            )
                        finally:
                            await telegram_scraper.disconnect()
                    
                    # Run the async function
                    telegram_analyses = asyncio.run(run_spydefi_analysis())
                    print(f"Spydefi analysis completed!")
                    print(f"Found {len(telegram_analyses.get('ranked_kols', []))} KOL channels.")
                    
                except Exception as e:
                    logger.error(f"Error analyzing Spydefi: {str(e)}", exc_info=True)
                    print(f"Error analyzing Spydefi: {str(e)}")
                    
                    if "Cannot find any entity" in str(e):
                        print("\nTroubleshooting tip: Make sure the Spydefi channel name/ID is correct.")
                        print("The channel might have changed its username or been deleted.")
            else:
                # Regular channel analysis
                channel_analyses = []
                for idx, channel in enumerate(channels):
                    try:
                        print(f"\nAnalyzing channel {idx+1}/{len(channels)}: {channel}...")
                        
                        # Run the analysis asynchronously
                        async def run_channel_analysis():
                            try:
                                await telegram_scraper.connect()
                                return await telegram_scraper.analyze_channel(
                                    channel,
                                    telegram_days,
                                    birdeye_api
                                )
                            finally:
                                await telegram_scraper.disconnect()
                        
                        # Run the async function
                        analysis = asyncio.run(run_channel_analysis())
                        channel_analyses.append(analysis)
                        print(f"Analysis for channel {channel} completed")
                        
                    except Exception as e:
                        logger.error(f"Error analyzing channel {channel}: {str(e)}", exc_info=True)
                        print(f"Error analyzing channel {channel}: {str(e)}")
                        
                        if "Cannot find any entity" in str(e):
                            print(f"Troubleshooting tip: Unable to find channel '{channel}'.")
                            print("Make sure the channel ID/username is correct and your account has access to it.")
                
                telegram_analyses = {
                    "ranked_kols": channel_analyses
                }
            
            print(f"Telegram analysis completed!")
        
        # Run wallet analysis
        wallet_analyses = {}
        if wallets:
            print(f"\nAnalyzing {len(wallets)} wallets...")
            
            wallet_analyzer = WalletAnalyzer(birdeye_api)
            
            try:
                wallet_analyses = wallet_analyzer.batch_analyze_wallets(
                    wallets,
                    wallet_days,
                    min_winrate
                )
                
                print(f"Wallet analysis completed!")
                if wallet_analyses.get('gem_finders'):
                    print(f"- Gem Finders: {len(wallet_analyses['gem_finders'])}")
                if wallet_analyses.get('consistent'):
                    print(f"- Consistent: {len(wallet_analyses['consistent'])}")
                if wallet_analyses.get('flippers'):
                    print(f"- Flippers: {len(wallet_analyses['flippers'])}")
                
            except Exception as e:
                logger.error(f"Error during wallet analysis: {str(e)}", exc_info=True)
                print(f"Error during wallet analysis: {str(e)}")
        
        # Export combined results to Excel
        try:
            export_to_excel(telegram_analyses, wallet_analyses, output_file)
            print(f"\nCombined analysis exported to Excel: {output_file}")
        except Exception as e:
            logger.error(f"Error exporting to Excel: {str(e)}", exc_info=True)
            print(f"Error exporting to Excel: {str(e)}")
        
        print("\nCombined analysis completed!")
        input("\nPress Enter to continue...")
    
    except ImportError as e:
        print(f"\nError: Could not import required module: {str(e)}")
        print("Please ensure all dependencies are installed.")
        input("\nPress Enter to continue...")

def handle_config_menu(config: Dict[str, Any]):
    """Handle the configuration menu."""
    while True:
        print_header()
        print_config_menu()
        
        try:
            choice = int(input().strip())
            
            if choice == 1:
                set_api_keys(config)
            elif choice == 2:
                set_analysis_settings(config)
            elif choice == 3:
                manage_file_paths(config)
            elif choice == 4:
                test_api_connection(config)
            elif choice == 5:
                set_advanced_settings(config)
            elif choice == 6:
                return
            else:
                print("\nInvalid choice. Please try again.")
                time.sleep(1)
        
        except ValueError:
            print("\nInvalid input. Please enter a number.")
            time.sleep(1)

def main():
    """Main entry point for the Phoenix CLI."""
    try:
        config = load_config()
        
        while True:
            print_header()
            print_menu()
            
            try:
                choice = int(input().strip())
                
                if choice == 1:
                    handle_wallet_analysis(config)
                elif choice == 2:
                    handle_telegram_analysis(config)
                elif choice == 3:
                    handle_combined_analysis(config)
                elif choice == 4:
                    handle_config_menu(config)
                elif choice == 5:
                    print("\nExiting Phoenix. Goodbye!")
                    sys.exit(0)
                else:
                    print("\nInvalid choice. Please try again.")
                    time.sleep(1)
            
            except ValueError:
                print("\nInvalid input. Please enter a number.")
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("Check the log file for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()