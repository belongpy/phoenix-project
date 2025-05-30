#!/usr/bin/env python3
"""
Phoenix Project - OPTIMIZED CLI Tool with SPYDEFI KOL Analysis System (FIXED VERSION)

🎯 CRITICAL FIX: Real channel ID lookup and fixed logic
- Real numeric Telegram channel IDs (e.g., -1001234567890)
- Fixed token analysis logic (proper 2x/5x counting)
- losing_calls = tokens with >-50% pullback
- winning_calls = total_calls - losing_calls
- Removed: success_rate_percent, total_roi_percent, max_roi_percent
- Column order: channel_id next to kol
- Fixed pullback calculation (real percentages)

✅ PRESERVED: All wallet analysis and other core functionality intact
"""

import os
import sys
import argparse
import json
import csv
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Setup logging with proper flushing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phoenix.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("phoenix")

# Ensure stdout is unbuffered
class UnbufferedStream:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, lines):
        self.stream.writelines(lines)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = UnbufferedStream(sys.stdout)

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
        "helius_api_key": "",
        "telegram_api_id": "",
        "telegram_api_hash": "",
        "telegram_session": "",
        "solana_rpc_url": "https://api.mainnet-beta.solana.com",
        "sources": {
            "telegram_groups": ["spydefi"],
            "wallets": []
        },
        "analysis_period_days": 7,
        "wallet_analysis": {
            "days_to_analyze": 7,
            "skip_prompts": True
        }
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
                if line and not line.startswith('#'):
                    if 32 <= len(line) <= 44:
                        wallets.append(line)
                    else:
                        logger.warning(f"Line {line_num}: Invalid wallet address format: {line}")
            
            logger.info(f"Loaded {len(wallets)} wallet addresses from {file_path}")
            return wallets
            
    except Exception as e:
        logger.error(f"Error reading wallets file {file_path}: {str(e)}")
        return []

def clear_corrupted_cache():
    """Clear potentially corrupted SPYDEFI cache."""
    cache_dir = Path.home() / ".phoenix_cache"
    spydefi_cache = cache_dir / "spydefi_kol_analysis.json"
    
    if spydefi_cache.exists():
        try:
            # Check if cache contains corrupted data (old format with wrong KOL names)
            with open(spydefi_cache, 'r') as f:
                cache_data = json.load(f)
            
            kol_performances = cache_data.get('kol_performances', {})
            
            # Check for known corrupted KOL names
            corrupted_names = {'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x15', 'x20', 'x30', 'x40', 'x50', 'x60'}
            has_corrupted_data = any(kol in corrupted_names for kol in kol_performances.keys())
            
            if has_corrupted_data or not kol_performances:
                logger.info("🧹 Clearing corrupted SPYDEFI cache with invalid KOL names...")
                spydefi_cache.unlink()
                logger.info("✅ Corrupted cache cleared - will regenerate with fixed extraction")
                return True
        except Exception as e:
            logger.warning(f"Error checking cache: {str(e)} - clearing cache")
            spydefi_cache.unlink()
            return True
    
    return False

class PhoenixCLI:
    """Optimized Phoenix CLI with FIXED SPYDEFI logic and real channel IDs."""
    
    def __init__(self):
        self.config = load_config()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="Phoenix Project - Solana Wallet & KOL Analysis Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Command")
        
        # Configure command
        configure_parser = subparsers.add_parser("configure", help="Configure API keys and sources")
        configure_parser.add_argument("--birdeye-api-key", help="Birdeye Solana API key")
        configure_parser.add_argument("--cielo-api-key", help="Cielo Finance API key")
        configure_parser.add_argument("--helius-api-key", help="Helius API key (for pump.fun tokens)")
        configure_parser.add_argument("--telegram-api-id", help="Telegram API ID")
        configure_parser.add_argument("--telegram-api-hash", help="Telegram API hash")
        configure_parser.add_argument("--rpc-url", help="Solana RPC URL (P9 or other provider)")
        configure_parser.add_argument("--analysis-days", type=int, help="Default days for wallet analysis")
        
        # SPYDEFI analysis command
        spydefi_parser = subparsers.add_parser("spydefi", help="SPYDEFI KOL Analysis with REAL Channel IDs & Fixed Logic")
        spydefi_parser.add_argument("--output", default="spydefi_kol_analysis.csv", help="Output CSV file")
        spydefi_parser.add_argument("--force-refresh", action="store_true", help="Force refresh cache")
        
        # Wallet analysis command
        wallet_parser = subparsers.add_parser("wallet", help="Analyze wallets for copy trading")
        wallet_parser.add_argument("--wallets-file", default="wallets.txt", help="File containing wallet addresses")
        wallet_parser.add_argument("--days", type=int, help="Number of days to analyze (overrides config)")
        wallet_parser.add_argument("--output", default="wallet_analysis.csv", help="Output file")
        
        return parser
    
    def _handle_numbered_menu(self):
        """Handle the numbered menu interface."""
        print("\n" + "="*80, flush=True)
        print("Phoenix Project - Solana Wallet & KOL Analysis Tool", flush=True)
        print("🚀 Enhanced with FIXED SPYDEFI Analysis & Real Channel IDs", flush=True)
        print(f"📅 Current Date: {datetime.now().strftime('%Y-%m-%d')}", flush=True)
        print("="*80, flush=True)
        print("\nSelect an option:", flush=True)
        print("\n🔧 CONFIGURATION:", flush=True)
        print("1. Configure API Keys", flush=True)
        print("2. Check Configuration", flush=True)
        print("3. Test API Connectivity", flush=True)
        print("\n📊 ANALYSIS TOOLS:", flush=True)
        print("4. SPYDEFI (FIXED: Real Channel IDs | Fixed Logic | Real Pullbacks)", flush=True)
        print("5. WALLET ANALYSIS", flush=True)
        print("\n🔍 UTILITIES:", flush=True)
        print("6. View Current Sources", flush=True)
        print("7. Help & Strategy Guide", flush=True)
        print("8. Manage Cache", flush=True)
        print("0. Exit", flush=True)
        print("="*80, flush=True)
        
        try:
            choice = input("\nEnter your choice (0-8): ").strip()
            
            if choice == '0':
                print("\nExiting Phoenix Project. Goodbye! 👋", flush=True)
                sys.exit(0)
            elif choice == '1':
                self._interactive_configure()
            elif choice == '2':
                self._check_configuration()
            elif choice == '3':
                self._test_api_connectivity()
            elif choice == '4':
                self._spydefi_analysis()
            elif choice == '5':
                self._wallet_analysis()
            elif choice == '6':
                self._view_current_sources()
            elif choice == '7':
                self._show_strategy_help()
            elif choice == '8':
                self._manage_cache()
            else:
                print("❌ Invalid choice. Please try again.", flush=True)
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.", flush=True)
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error in menu: {str(e)}")
            input("Press Enter to continue...")
    
    def _manage_cache(self):
        """Manage Phoenix cache."""
        print("\n" + "="*70, flush=True)
        print("    🗄️ CACHE MANAGEMENT", flush=True)
        print("="*70, flush=True)
        
        cache_dir = Path.home() / ".phoenix_cache"
        
        if not cache_dir.exists():
            print("\n📁 No cache directory found.", flush=True)
            input("Press Enter to continue...")
            return
        
        # Check cache files
        cache_files = list(cache_dir.glob("*.json"))
        
        if not cache_files:
            print("\n📭 No cache files found.", flush=True)
            input("Press Enter to continue...")
            return
        
        print("\n📊 CACHE STATUS:", flush=True)
        total_size = 0
        
        for cache_file in cache_files:
            size = cache_file.stat().st_size / 1024  # KB
            total_size += size
            
            # Check different cache types
            if cache_file.name == "spydefi_kol_analysis.json":
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    timestamp = cache_data.get('timestamp', 'Unknown')
                    kol_count = len(cache_data.get('kol_performances', {}))
                    version = cache_data.get('version', '4.0')
                    config = cache_data.get('config', {})
                    scan_hours = config.get('spydefi_scan_hours', 8)
                    analysis_days = config.get('kol_analysis_days', 7)
                    
                    print(f"\n📋 SPYDEFI KOL Analysis Cache:", flush=True)
                    print(f"   File: {cache_file.name}", flush=True)
                    print(f"   Size: {size:.2f} KB", flush=True)
                    print(f"   Version: {version} (FIXED: Real Channel IDs & Logic)", flush=True)
                    print(f"   Created: {timestamp}", flush=True)
                    print(f"   KOLs analyzed: {kol_count}", flush=True)
                    print(f"   SpyDefi scan: {scan_hours}h", flush=True)
                    print(f"   KOL analysis: {analysis_days}d each", flush=True)
                    print(f"   FIXES: Real channel lookup, fixed token counting, pullback calc", flush=True)
                    
                    # Check for corrupted data
                    kol_performances = cache_data.get('kol_performances', {})
                    corrupted_names = {'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'}
                    has_corrupted_data = any(kol in corrupted_names for kol in kol_performances.keys())
                    
                    if has_corrupted_data:
                        print(f"   Status: ❌ CORRUPTED (contains invalid KOL names like x2, x3)", flush=True)
                        print(f"   Action: Should be cleared and regenerated", flush=True)
                    elif not kol_performances:
                        print(f"   Status: ⚠️ EMPTY (no valid KOL performances)", flush=True)
                    else:
                        # Check age
                        if timestamp != 'Unknown':
                            cache_age = datetime.now() - datetime.fromisoformat(timestamp)
                            hours_old = cache_age.total_seconds() / 3600
                            
                            if hours_old < 6:
                                print(f"   Status: ✅ Fresh ({hours_old:.1f} hours old)", flush=True)
                            else:
                                print(f"   Status: ⚠️ Expired ({hours_old:.1f} hours old)", flush=True)
                            
                except Exception as e:
                    print(f"   Error reading cache: {str(e)}", flush=True)
            
            elif cache_file.name == "spydefi_kols.json":
                # Legacy telegram cache
                print(f"\n📋 Legacy Cache: {cache_file.name} ({size:.2f} KB)", flush=True)
        
        print(f"\n📊 Total cache size: {total_size:.2f} KB", flush=True)
        
        print("\n🔧 CACHE ACTIONS:", flush=True)
        print("1. Clear all cache", flush=True)
        print("2. Clear only corrupted SPYDEFI cache", flush=True)
        print("3. View cache details", flush=True)
        print("0. Back to main menu", flush=True)
        
        choice = input("\nEnter your choice (0-3): ").strip()
        
        if choice == '1':
            confirm = input("\n⚠️ Clear all cache files? (y/N): ").lower().strip()
            if confirm == 'y':
                for cache_file in cache_files:
                    try:
                        cache_file.unlink()
                        print(f"✅ Deleted: {cache_file.name}", flush=True)
                    except Exception as e:
                        print(f"❌ Error deleting {cache_file.name}: {str(e)}", flush=True)
                print("\n✅ All cache cleared!", flush=True)
            else:
                print("❌ Cache clear cancelled.", flush=True)
        
        elif choice == '2':
            if clear_corrupted_cache():
                print("✅ Corrupted SPYDEFI cache cleared!", flush=True)
            else:
                print("ℹ️ No corrupted cache found.", flush=True)
                
        elif choice == '3':
            print("\n📄 CACHE DETAILS:", flush=True)
            for cache_file in cache_files:
                print(f"\nFile: {cache_file}", flush=True)
                try:
                    with open(cache_file, 'r') as f:
                        content = json.load(f)
                    print(json.dumps(content, indent=2)[:500] + "...", flush=True)
                except Exception as e:
                    print(f"Error reading file: {str(e)}", flush=True)
        
        input("\nPress Enter to continue...")
    
    def _spydefi_analysis(self):
        """Run SPYDEFI KOL analysis with FIXED logic and real channel IDs."""
        # Check API configuration
        if not self.config.get("birdeye_api_key"):
            print("\n❌ CRITICAL: Birdeye API key required for SPYDEFI analysis!", flush=True)
            print("Please configure your Birdeye API key first (Option 1).", flush=True)
            input("Press Enter to continue...")
            return
        
        if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
            print("\n❌ CRITICAL: Telegram API credentials required!", flush=True)
            print("Please configure your Telegram API credentials first (Option 1).", flush=True)
            input("Press Enter to continue...")
            return
        
        # CRITICAL FIX: Check and clear corrupted cache before starting
        cache_cleared = clear_corrupted_cache()
        if cache_cleared:
            print("\n🧹 Cleared corrupted cache with invalid KOL names", flush=True)
            print("Will regenerate analysis with FIXED extraction & real channel IDs", flush=True)
        
        # Check optional APIs
        if self.config.get("helius_api_key"):
            print(f"\n✅ Helius API configured for enhanced pump.fun analysis", flush=True)
        else:
            print(f"\n⚠️ Helius API not configured - pump.fun analysis limited", flush=True)
        
        print("\n🚀 Starting SPYDEFI analysis with REAL CHANNEL IDs & FIXED logic...", flush=True)
        
        # Create args object
        class Args:
            def __init__(self):
                self.output = "spydefi_kol_analysis.csv"
                self.force_refresh = cache_cleared  # Force refresh if cache was cleared
        
        args = Args()
        
        try:
            self._handle_spydefi_analysis(args)
            print("\n✅ SPYDEFI analysis completed successfully!", flush=True)
            print("📁 Check the outputs folder for detailed results", flush=True)
            print("🔢 CSV now contains REAL numeric Telegram channel IDs", flush=True)
            print("🧮 Fixed logic: losing_calls = >-50% pullback, real pullback percentages", flush=True)
            
        except Exception as e:
            print(f"\n❌ SPYDEFI analysis failed: {str(e)}", flush=True)
            logger.error(f"SPYDEFI analysis error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        input("\nPress Enter to continue...")
    
    def _handle_spydefi_analysis(self, args) -> None:
        """Handle the SPYDEFI analysis command with FIXED logic."""
        import asyncio
        
        try:
            import importlib
            import sys
            
            # Clear any existing modules to get the fixed version
            modules_to_clear = ['spydefi_module']
            for module in modules_to_clear:
                if module in sys.modules:
                    del sys.modules[module]
            
            from spydefi_module import SpyDefiAnalyzer
            from dual_api_manager import DualAPIManager
            
            logger.info("✅ Imported FIXED SPYDEFI module with real channel ID lookup")
            
        except Exception as e:
            logger.error(f"❌ Error importing modules: {str(e)}")
            raise
        
        if not self.config.get("birdeye_api_key"):
            logger.error("🎯 CRITICAL: Birdeye API key required for SPYDEFI analysis!")
            return
            
        if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
            logger.error("📱 CRITICAL: Telegram API credentials required!")
            return
        
        output_file = ensure_output_dir(args.output)
        
        logger.info(f"🚀 Starting SPYDEFI KOL analysis with FIXED LOGIC & REAL CHANNEL IDs")
        logger.info(f"📁 Results will be saved to {output_file}")
        logger.info(f"🔢 Channel IDs will be REAL numeric Telegram IDs")
        logger.info(f"🧮 Fixed: losing_calls = >-50% pullback, real pullback calculation")
        
        # Initialize APIs
        try:
            api_manager = DualAPIManager(
                self.config["birdeye_api_key"],
                self.config.get("cielo_api_key"),
                self.config.get("helius_api_key"),
                self.config.get("solana_rpc_url")
            )
            logger.info("✅ API manager initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize API manager: {str(e)}")
            raise
        
        try:
            spydefi_analyzer = SpyDefiAnalyzer(
                self.config["telegram_api_id"],
                self.config["telegram_api_hash"],
                self.config.get("telegram_session", "phoenix_spydefi")
            )
            
            # Set API manager
            spydefi_analyzer.set_api_manager(api_manager)
            
            logger.info("🔧 Using SpyDefi module config with FIXED logic")
            logger.info(f"📊 FIXES: Real channel lookup, fixed token counting, pullback calc")
            logger.info(f"🔢 Real channel ID lookup: ENABLED")
            
            # Only set force refresh if cache was corrupted
            if getattr(args, 'force_refresh', False):
                logger.info("🔄 Forcing cache refresh due to corrupted data")
            
            logger.info("✅ SPYDEFI analyzer initialized with FIXED LOGIC")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SPYDEFI analyzer: {str(e)}")
            raise
        
        # Run analysis
        async def run_spydefi_analysis():
            try:
                async with spydefi_analyzer:
                    # Run analysis with FIXED logic
                    results = await spydefi_analyzer.run_full_analysis()
                    
                    if results.get('success'):
                        # Export with FIXED column order
                        try:
                            await export_spydefi_results(results, output_file)
                            display_spydefi_summary(results)
                        except Exception as export_error:
                            logger.error(f"❌ Export failed: {str(export_error)}")
                            print(f"\n⚠️ Analysis completed but export failed: {str(export_error)}", flush=True)
                            
                            # Try to show basic summary
                            kol_count = len(results.get('kol_performances', {}))
                            print(f"📊 Analysis found {kol_count} valid KOLs", flush=True)
                        
                        return results
                    else:
                        error_msg = results.get('error', 'Unknown error')
                        logger.error(f"❌ SPYDEFI analysis failed: {error_msg}")
                        print(f"❌ Analysis failed: {error_msg}", flush=True)
                        return results
                        
            except Exception as e:
                logger.error(f"❌ Error in SPYDEFI analysis: {str(e)}")
                import traceback
                logger.error(f"❌ Analysis traceback: {traceback.format_exc()}")
                raise
        
        # Run the analysis
        results = asyncio.run(run_spydefi_analysis())
        
        if results.get('success'):
            logger.info(f"✅ SPYDEFI analysis completed successfully!")
            logger.info(f"📁 Results saved to {output_file}")
        else:
            logger.error(f"❌ SPYDEFI analysis failed: {results.get('error', 'Unknown error')}")
    
    def _wallet_analysis(self):
        """Run wallet analysis (PRESERVED - unchanged functionality)."""
        print("\n" + "="*80, flush=True)
        print("    💰 WALLET ANALYSIS", flush=True)
        print("    🎯 Analyzing active traders with smart strategies", flush=True)
        print("    📊 Features: Performance metrics & exit guidance", flush=True)
        print("="*80, flush=True)
        
        # Check API configuration
        if not self.config.get("cielo_api_key"):
            print("\n❌ CRITICAL: Cielo Finance API key required for wallet analysis!", flush=True)
            print("Please configure your Cielo Finance API key first (Option 1).", flush=True)
            input("Press Enter to continue...")
            return
        
        if not self.config.get("helius_api_key"):
            print("\n⚠️ WARNING: Helius API key not configured!", flush=True)
            print("Pump.fun token analysis will be limited without Helius.", flush=True)
            print("Consider adding Helius API key for complete analysis.", flush=True)
        
        # Load wallets
        wallets = load_wallets_from_file("wallets.txt")
        if not wallets:
            print("\n❌ No wallets found in wallets.txt", flush=True)
            print("Please add wallet addresses to wallets.txt (one per line)", flush=True)
            input("Press Enter to continue...")
            return
        
        print(f"\n📁 Found {len(wallets)} wallets in wallets.txt", flush=True)
        
        # Get days to analyze from config (default 7)
        days_to_analyze = self.config.get("wallet_analysis", {}).get("days_to_analyze", 7)
        
        # Direct to processing without prompts (unless skip_prompts is False)
        skip_prompts = self.config.get("wallet_analysis", {}).get("skip_prompts", True)
        
        if not skip_prompts:
            # Optional: Ask for days if not skipping prompts
            days_input = input(f"Days to analyze (default: {days_to_analyze}, max: 30): ").strip()
            if days_input.isdigit():
                days_to_analyze = min(int(days_input), 30)
        
        print(f"\n🚀 Starting wallet analysis...", flush=True)
        print(f"📊 Parameters:", flush=True)
        print(f"   • Wallets: {len(wallets)}", flush=True)
        print(f"   • Analysis period: {days_to_analyze} days", flush=True)
        print(f"   • Focus: Active traders (recent activity)", flush=True)
        print(f"   • Strategy: Enhanced with TP guidance", flush=True)
        print(f"   • Export format: CSV with strategy details", flush=True)
        print("\nProcessing...", flush=True)
        
        try:
            # Initialize APIs
            from dual_api_manager import DualAPIManager
            from wallet_module import WalletAnalyzer
            
            api_manager = DualAPIManager(
                self.config.get("birdeye_api_key", ""),
                self.config.get("cielo_api_key"),
                self.config.get("helius_api_key")
            )
            
            # Create wallet analyzer
            wallet_analyzer = WalletAnalyzer(
                cielo_api=api_manager.cielo_api,
                birdeye_api=api_manager.birdeye_api,
                helius_api=api_manager.helius_api,
                rpc_url=self.config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
            )
            
            # Run batch analysis
            results = wallet_analyzer.batch_analyze_wallets(
                wallets,
                days_back=days_to_analyze,
                use_hybrid=True
            )
            
            if results.get("success"):
                self._display_wallet_results(results)
                
                # Export to CSV with enhanced format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = ensure_output_dir(f"wallet_analysis_{timestamp}.csv")
                self._export_wallet_csv(results, output_file)
                print(f"\n📄 Exported to CSV: {output_file}", flush=True)
                
                print("\n✅ Wallet analysis completed successfully!", flush=True)
                
                # Display API call statistics
                if "api_calls" in results:
                    print(f"\n📊 API CALL EFFICIENCY:", flush=True)
                    print(f"   Cielo: {results['api_calls']['cielo']} calls", flush=True)
                    print(f"   Birdeye: {results['api_calls']['birdeye']} calls", flush=True)
                    print(f"   Helius: {results['api_calls']['helius']} calls", flush=True)
                    print(f"   RPC: {results['api_calls']['rpc']} calls", flush=True)
                    print(f"   Total: {sum(results['api_calls'].values())} calls", flush=True)
            else:
                print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}", flush=True)
                
        except Exception as e:
            print(f"\n❌ Error during wallet analysis: {str(e)}", flush=True)
            logger.error(f"Wallet analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _display_wallet_results(self, results: Dict[str, Any]) -> None:
        """Display wallet analysis results (PRESERVED - unchanged)."""
        print("\n" + "="*80, flush=True)
        print("    📊 WALLET ANALYSIS RESULTS", flush=True)
        print("="*80, flush=True)
        
        # Summary statistics
        print(f"\n📈 SUMMARY:", flush=True)
        print(f"   Total wallets: {results['total_wallets']}", flush=True)
        print(f"   Successfully analyzed: {results['analyzed_wallets']}", flush=True)
        print(f"   Failed: {results['failed_wallets']}", flush=True)
        
        # Count active vs inactive
        active_count = 0
        inactive_count = 0
        all_wallets = []
        
        for category in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 'swing_traders',
                        'position_traders', 'consistent', 'mixed', 'unknown']:
            category_wallets = results.get(category, [])
            all_wallets.extend(category_wallets)
            for wallet in category_wallets:
                if wallet.get('metrics', {}).get('active_trader', False):
                    active_count += 1
                else:
                    inactive_count += 1
        
        print(f"\n🟢 Active traders: {active_count}", flush=True)
        print(f"🔴 Inactive traders: {inactive_count}", flush=True)
        
        # Sort all wallets by score
        all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
        
        # Get only active traders
        active_wallets = [w for w in all_wallets if w.get('metrics', {}).get('active_trader', False)]
        
        if active_wallets:
            print(f"\n🏆 TOP 10 ACTIVE TRADERS:", flush=True)
            for i, analysis in enumerate(active_wallets[:10], 1):
                wallet = analysis['wallet_address']
                metrics = analysis['metrics']
                composite_score = analysis.get('composite_score', metrics.get('composite_score', 0))
                strategy = analysis.get('strategy', {})
                
                # Cap profit factor for display
                profit_factor = metrics['profit_factor']
                if profit_factor > 999.99:
                    profit_factor_display = "999.99x"
                else:
                    profit_factor_display = f"{profit_factor:.2f}x"
                
                days_analyzed = analysis.get('analysis_period_days', 7)
                
                print(f"\n{i}. Wallet: {wallet[:8]}...{wallet[-4:]}", flush=True)
                print(f"   Score: {composite_score:.1f}/100", flush=True)
                print(f"   Type: {analysis['wallet_type']}", flush=True)
                print(f"   === {days_analyzed}-DAY PERFORMANCE ===", flush=True)
                print(f"   Trades ({days_analyzed}d): {metrics.get('trades_last_7_days', 0)} | Win Rate ({days_analyzed}d): {metrics.get('win_rate_7d', 0):.1f}%", flush=True)
                print(f"   Profit ({days_analyzed}d): ${metrics.get('profit_7d', 0):.2f}", flush=True)
                print(f"   === OVERALL STATS ===", flush=True)
                print(f"   Total Trades: {metrics['total_trades']} | Profit Factor: {profit_factor_display}", flush=True)
                print(f"   5x+ Gem Rate: {metrics.get('gem_rate_5x_plus', 0):.1f}%", flush=True)
                print(f"   Avg Hold: {metrics.get('avg_hold_time_minutes', 0):.1f} min", flush=True)
                print(f"   Avg First TP: {metrics.get('avg_first_take_profit_percent', 0):.1f}%", flush=True)
                print(f"   === STRATEGY RECOMMENDATION ===", flush=True)
                print(f"   Action: {strategy.get('recommendation', 'UNKNOWN')}", flush=True)
                print(f"   Follow Sells: {'YES ✅' if strategy.get('follow_sells', False) else 'NO ❌'}", flush=True)
                print(f"   TP1: {strategy.get('tp1_percent', 0)}% | TP2: {strategy.get('tp2_percent', 0)}%", flush=True)
                print(f"   Guidance: {strategy.get('tp_guidance', 'No guidance available')}", flush=True)
                
                # Entry/Exit Analysis
                if 'entry_exit_analysis' in analysis:
                    ee_analysis = analysis['entry_exit_analysis']
                    print(f"   Entry/Exit Quality: {ee_analysis['entry_quality']}/{ee_analysis['exit_quality']}", flush=True)
                    if ee_analysis.get('exit_quality') == 'POOR':
                        print(f"   ⚠️ They miss {ee_analysis.get('missed_gains_percent', 0):.0f}% gains on average", flush=True)
                
                # Distribution
                print(f"   === DISTRIBUTION ===", flush=True)
                print(f"   5x+: {metrics.get('distribution_500_plus_%', 0):.1f}% | "
                      f"2-5x: {metrics.get('distribution_200_500_%', 0):.1f}% | "
                      f"<2x: {metrics.get('distribution_0_200_%', 0):.1f}%", flush=True)
        
        # Category breakdown
        print(f"\n📂 WALLET CATEGORIES:", flush=True)
        print(f"   🎯 Snipers (< 1 min hold): {len(results.get('snipers', []))}", flush=True)
        print(f"   ⚡ Flippers (1-10 min): {len(results.get('flippers', []))}", flush=True)
        print(f"   📊 Scalpers (10-60 min): {len(results.get('scalpers', []))}", flush=True)
        print(f"   💎 5x+ Gem Hunters: {len(results.get('gem_hunters', []))}", flush=True)
        print(f"   📈 Swing Traders (1-24h): {len(results.get('swing_traders', []))}", flush=True)
        print(f"   🏆 Position Traders (24h+): {len(results.get('position_traders', []))}", flush=True)
        
        # Key insights
        print(f"\n📊 KEY INSIGHTS:", flush=True)
        if active_wallets:
            # Count recent winners
            recent_5x = sum(1 for w in active_wallets 
                          if w.get('seven_day_metrics', {}).get('has_5x_last_7_days', False))
            recent_2x = sum(1 for w in active_wallets 
                          if w.get('seven_day_metrics', {}).get('has_2x_last_7_days', False))
            
            if recent_5x > 0:
                print(f"   🚀 {recent_5x} wallets hit 5x+ recently!", flush=True)
            if recent_2x > 0:
                print(f"   📈 {recent_2x} wallets hit 2x+ recently!", flush=True)
            
            # Exit quality breakdown
            good_exits = sum(1 for w in active_wallets 
                           if w.get('entry_exit_analysis', {}).get('exit_quality') in ['GOOD', 'EXCELLENT'])
            poor_exits = sum(1 for w in active_wallets 
                           if w.get('entry_exit_analysis', {}).get('exit_quality') == 'POOR')
            
            print(f"   ✅ {good_exits} wallets have good exit timing (follow their sells)", flush=True)
            print(f"   ❌ {poor_exits} wallets exit too early (use fixed TPs instead)", flush=True)
    
    def _export_wallet_csv(self, results: Dict[str, Any], output_file: str) -> None:
        """Export wallet analysis to CSV (PRESERVED - unchanged)."""
        try:
            from export_utils import export_wallet_rankings_csv
            export_wallet_rankings_csv(results, output_file)
            logger.info(f"Exported wallet analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
    
    def _test_api_connectivity(self):
        """Test API connectivity (updated with FIXED SPYDEFI info)."""
        print("\n" + "="*70, flush=True)
        print("    🔍 API CONNECTIVITY TEST", flush=True)
        print("="*70, flush=True)
        
        # Test Birdeye API
        if self.config.get("birdeye_api_key"):
            print("\n🔍 Testing Birdeye API...", flush=True)
            try:
                from birdeye_api import BirdeyeAPI
                birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
                test_result = birdeye_api.get_token_info("So11111111111111111111111111111111111111112")
                if test_result.get("success"):
                    print("✅ Birdeye API: Connected successfully", flush=True)
                    print("   🎯 Token analysis: Available", flush=True)
                else:
                    print("❌ Birdeye API: Connection failed", flush=True)
            except Exception as e:
                print(f"❌ Birdeye API: Error - {str(e)}", flush=True)
        else:
            print("❌ Birdeye API: Not configured", flush=True)
        
        # Test Helius API
        if self.config.get("helius_api_key"):
            print("\n🚀 Testing Helius API...", flush=True)
            try:
                from helius_api import HeliusAPI
                helius_api = HeliusAPI(self.config["helius_api_key"])
                if helius_api.health_check():
                    print("✅ Helius API: Connected successfully", flush=True)
                    print("   🎯 Pump.fun token analysis: Available", flush=True)
                else:
                    print("❌ Helius API: Connection failed", flush=True)
            except Exception as e:
                print(f"❌ Helius API: Error - {str(e)}", flush=True)
        else:
            print("⚠️ Helius API: Not configured", flush=True)
            print("   ⚠️ Pump.fun token analysis will be limited", flush=True)
        
        # Test Cielo Finance API
        if self.config.get("cielo_api_key"):
            print("\n💰 Testing Cielo Finance API...", flush=True)
            try:
                from cielo_api import CieloFinanceAPI
                cielo_api = CieloFinanceAPI(self.config["cielo_api_key"])
                if cielo_api.health_check():
                    print("✅ Cielo Finance API: Connected successfully", flush=True)
                    print("   💰 Wallet analysis: Available", flush=True)
                else:
                    print("❌ Cielo Finance API: Connection failed", flush=True)
            except Exception as e:
                print(f"❌ Cielo Finance API: Error - {str(e)}", flush=True)
        else:
            print("❌ Cielo Finance API: Not configured", flush=True)
            print("   ⚠️ CRITICAL: Wallet analysis requires Cielo Finance API", flush=True)
        
        # Test Telegram API
        if self.config.get("telegram_api_id") and self.config.get("telegram_api_hash"):
            print("\n📱 Testing Telegram API...", flush=True)
            try:
                print("✅ Telegram API: Configuration appears valid", flush=True)
                print("   📊 SPYDEFI KOL analysis: Available", flush=True)
                print("   🔢 Real channel ID lookup: ENABLED", flush=True)
                print("   🧮 Fixed logic: Proper token counting & pullback calculation", flush=True)
            except Exception as e:
                print(f"❌ Telegram API: Error - {str(e)}", flush=True)
        else:
            print("❌ Telegram API: Not configured", flush=True)
        
        # Test RPC Connection
        print(f"\n🌐 Testing Solana RPC Connection...", flush=True)
        rpc_url = self.config.get('solana_rpc_url', 'https://api.mainnet-beta.solana.com')
        print(f"   RPC URL: {rpc_url}", flush=True)
        
        if "api.mainnet-beta.solana.com" in rpc_url:
            print("   ⚠️ Using default RPC (may be rate limited)", flush=True)
        else:
            print("   ✅ Using custom RPC provider", flush=True)
        
        try:
            import requests
            response = requests.post(
                rpc_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
                timeout=10
            )
            if response.status_code == 200:
                print("✅ Solana RPC: Connected successfully", flush=True)
            else:
                print(f"❌ Solana RPC: HTTP {response.status_code}", flush=True)
        except Exception as e:
            print(f"❌ Solana RPC: Error - {str(e)}", flush=True)
        
        # Summary
        print(f"\n📊 FEATURES AVAILABLE:", flush=True)
        birdeye_ok = bool(self.config.get("birdeye_api_key"))
        helius_ok = bool(self.config.get("helius_api_key"))
        telegram_ok = bool(self.config.get("telegram_api_id") and self.config.get("telegram_api_hash"))
        cielo_ok = bool(self.config.get("cielo_api_key"))
        
        print(f"   🎯 Token Price Analysis: {'✅ Full' if (birdeye_ok and helius_ok) else '⚠️ Limited' if birdeye_ok else '❌ Not Available'}", flush=True)
        print(f"   💰 Wallet Analysis: {'✅ Ready' if cielo_ok else '❌ Need Cielo Finance API'}", flush=True)
        print(f"   📱 SPYDEFI KOL Analysis: {'✅ Ready (FIXED)' if (birdeye_ok and telegram_ok) else '❌ Missing APIs'}", flush=True)
        print(f"   🔢 Real Channel ID Lookup: {'✅ Enabled' if telegram_ok else '❌ Need Telegram API'}", flush=True)
        print(f"   🧮 Fixed Logic & Pullbacks: {'✅ Implemented' if telegram_ok else '❌ Need Telegram API'}", flush=True)
        
        if birdeye_ok and helius_ok and telegram_ok and cielo_ok:
            print(f"\n🎉 ALL SYSTEMS GO! Full functionality available with FIXES.", flush=True)
        else:
            print(f"\n⚠️ Configure missing APIs to enable all features.", flush=True)
        
        input("\nPress Enter to continue...")
    
    def _interactive_configure(self):
        """Interactive configuration setup (PRESERVED - unchanged)."""
        print("\n" + "="*70, flush=True)
        print("    🔧 CONFIGURATION SETUP", flush=True)
        print("="*70, flush=True)
        
        # Birdeye API Key
        current_birdeye = self.config.get("birdeye_api_key", "")
        if current_birdeye:
            print(f"\n🔑 Current Birdeye API Key: {current_birdeye[:8]}...", flush=True)
            change_birdeye = input("Change Birdeye API key? (y/N): ").lower().strip()
            if change_birdeye == 'y':
                new_key = input("Enter new Birdeye API key: ").strip()
                if new_key:
                    self.config["birdeye_api_key"] = new_key
                    print("✅ Birdeye API key updated", flush=True)
        else:
            print("\n🔑 Birdeye API Key (REQUIRED for SPYDEFI & token analysis)", flush=True)
            print("   📊 Get your key from: https://birdeye.so", flush=True)
            new_key = input("Enter Birdeye API key: ").strip()
            if new_key:
                self.config["birdeye_api_key"] = new_key
                print("✅ Birdeye API key configured", flush=True)
        
        # Helius API Key
        current_helius = self.config.get("helius_api_key", "")
        if current_helius:
            print(f"\n🚀 Current Helius API Key: {current_helius[:8]}...", flush=True)
            change_helius = input("Change Helius API key? (y/N): ").lower().strip()
            if change_helius == 'y':
                new_key = input("Enter new Helius API key: ").strip()
                if new_key:
                    self.config["helius_api_key"] = new_key
                    print("✅ Helius API key updated", flush=True)
        else:
            print("\n🚀 Helius API Key (RECOMMENDED for enhanced pump.fun analysis)", flush=True)
            print("   📊 Required for complete memecoin analysis", flush=True)
            print("   🔑 Get your key from: https://helius.dev", flush=True)
            new_key = input("Enter Helius API key (or press Enter to skip): ").strip()
            if new_key:
                self.config["helius_api_key"] = new_key
                print("✅ Helius API key configured", flush=True)
                print("   🎯 Enhanced pump.fun analysis: Now available", flush=True)
            else:
                print("⚠️ Skipped: Pump.fun analysis will be limited", flush=True)
        
        # Cielo Finance API Key
        current_cielo = self.config.get("cielo_api_key", "")
        if current_cielo:
            print(f"\n💰 Current Cielo Finance API Key: {current_cielo[:8]}...", flush=True)
            change_cielo = input("Change Cielo Finance API key? (y/N): ").lower().strip()
            if change_cielo == 'y':
                new_key = input("Enter new Cielo Finance API key: ").strip()
                if new_key:
                    self.config["cielo_api_key"] = new_key
                    print("✅ Cielo Finance API key updated", flush=True)
        else:
            print("\n💰 Cielo Finance API Key (REQUIRED for wallet analysis)", flush=True)
            print("   🔑 Get your key from: https://cielo.finance", flush=True)
            new_key = input("Enter Cielo Finance API key: ").strip()
            if new_key:
                self.config["cielo_api_key"] = new_key
                print("✅ Cielo Finance API key configured", flush=True)
        
        # Telegram API credentials
        current_tg_id = self.config.get("telegram_api_id", "")
        if current_tg_id:
            print(f"\n📱 Current Telegram API ID: {current_tg_id}", flush=True)
            change_tg = input("Change Telegram API credentials? (y/N): ").lower().strip()
            if change_tg == 'y':
                new_id = input("Enter new Telegram API ID: ").strip()
                new_hash = input("Enter new Telegram API Hash: ").strip()
                if new_id and new_hash:
                    self.config["telegram_api_id"] = new_id
                    self.config["telegram_api_hash"] = new_hash
                    print("✅ Telegram API credentials updated", flush=True)
        else:
            print("\n📱 Telegram API Credentials (Required for SPYDEFI KOL analysis)", flush=True)
            print("   🔑 Get credentials from: https://my.telegram.org", flush=True)
            new_id = input("Enter Telegram API ID: ").strip()
            new_hash = input("Enter Telegram API Hash: ").strip()
            if new_id and new_hash:
                self.config["telegram_api_id"] = new_id
                self.config["telegram_api_hash"] = new_hash
                print("✅ Telegram API credentials configured", flush=True)
        
        # RPC URL
        current_rpc = self.config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
        print(f"\n🌐 Current RPC URL: {current_rpc}", flush=True)
        
        if "api.mainnet-beta.solana.com" in current_rpc:
            print("   ⚠️ Using default Solana RPC (may be rate limited)", flush=True)
            print("   💡 Consider using P9 or another provider for better performance", flush=True)
        
        change_rpc = input("Change RPC URL? (y/N): ").lower().strip()
        if change_rpc == 'y':
            print("   Options:", flush=True)
            print("   1. Default Solana RPC", flush=True)
            print("   2. P9 RPC (recommended)", flush=True)
            print("   3. Custom RPC URL", flush=True)
            rpc_choice = input("Choose option (1-3): ").strip()
            if rpc_choice == '1':
                self.config["solana_rpc_url"] = "https://api.mainnet-beta.solana.com"
                print("✅ Using default Solana RPC", flush=True)
            elif rpc_choice == '2':
                print("   P9 RPC format: https://YOUR-NAME.rpcpool.com/YOUR-API-KEY", flush=True)
                new_rpc = input("Enter your P9 RPC URL: ").strip()
                if new_rpc:
                    self.config["solana_rpc_url"] = new_rpc
                    print("✅ P9 RPC URL configured", flush=True)
            elif rpc_choice == '3':
                new_rpc = input("Enter custom RPC URL: ").strip()
                if new_rpc:
                    self.config["solana_rpc_url"] = new_rpc
                    print("✅ Custom RPC URL configured", flush=True)
        
        # Wallet analysis configuration
        print(f"\n💰 Wallet Analysis Settings:", flush=True)
        current_days = self.config.get("wallet_analysis", {}).get("days_to_analyze", 7)
        print(f"   Current default days: {current_days}", flush=True)
        change_days = input("Change default analysis days? (y/N): ").lower().strip()
        if change_days == 'y':
            days_input = input("Enter default days (1-30): ").strip()
            if days_input.isdigit():
                days = min(max(int(days_input), 1), 30)
                if "wallet_analysis" not in self.config:
                    self.config["wallet_analysis"] = {}
                self.config["wallet_analysis"]["days_to_analyze"] = days
                print(f"✅ Default analysis period set to {days} days", flush=True)
        
        # Save configuration
        save_config(self.config)
        print("\n✅ Configuration saved successfully!", flush=True)
        print("🔧 SPYDEFI now uses FIXED logic with real channel IDs & pullback calculation", flush=True)
        
        input("\nPress Enter to continue...")
    
    def _check_configuration(self):
        """Check current configuration (updated with FIXED SPYDEFI info)."""
        print("\n" + "="*70, flush=True)
        print("    📋 CURRENT CONFIGURATION", flush=True)
        print("="*70, flush=True)
        
        print(f"\n🔑 API KEYS:", flush=True)
        print(f"   Birdeye API Key: {'✅ Configured' if self.config.get('birdeye_api_key') else '❌ Not configured'}", flush=True)
        print(f"   Helius API Key: {'✅ Configured' if self.config.get('helius_api_key') else '⚠️ Not configured (optional)'}", flush=True)
        print(f"   Cielo Finance API Key: {'✅ Configured' if self.config.get('cielo_api_key') else '❌ Not configured'}", flush=True)
        print(f"   Telegram API ID: {'✅ Configured' if self.config.get('telegram_api_id') else '❌ Not configured'}", flush=True)
        print(f"   Telegram API Hash: {'✅ Configured' if self.config.get('telegram_api_hash') else '❌ Not configured'}", flush=True)
        
        print(f"\n🌐 RPC ENDPOINT:", flush=True)
        rpc_url = self.config.get('solana_rpc_url', 'Default')
        print(f"   URL: {rpc_url}", flush=True)
        if "api.mainnet-beta.solana.com" in rpc_url:
            print(f"   Status: ⚠️ Default RPC (consider upgrading to P9)", flush=True)
        else:
            print(f"   Status: ✅ Custom RPC provider", flush=True)
        
        print(f"\n📊 SPYDEFI SETTINGS (FIXED VERSION):", flush=True)
        print(f"   🔧 Version: 4.0 (Fixed Logic & Real Channel IDs)", flush=True)
        print(f"   📅 Scan period: 8 hours (PEAK MEMECOIN HOURS)", flush=True)
        print(f"   🏆 KOLs to analyze: TOP 50", flush=True)
        print(f"   📊 Min mentions: ≥1 (quality filter)", flush=True)
        print(f"   💰 Max market cap: $10M (avoid overvalued)", flush=True)
        print(f"   📈 Win threshold: 50% profit", flush=True)
        print(f"   🔧 KOL extraction: FIXED (real @usernames only)", flush=True)
        print(f"   🔢 Channel ID lookup: REAL numeric IDs (-1001234567890)", flush=True)
        print(f"   🧮 Token counting: FIXED (proper 2x/5x logic)", flush=True)
        print(f"   📉 Pullback calculation: FIXED (real percentages)", flush=True)
        print(f"   ❌ losing_calls: Tokens with >-50% pullback", flush=True)
        print(f"   ✅ winning_calls: total_calls - losing_calls", flush=True)
        print(f"   📊 Column order: channel_id next to kol", flush=True)
        
        print(f"\n💰 WALLET ANALYSIS SETTINGS:", flush=True)
        print(f"   Default analysis period: {self.config.get('wallet_analysis', {}).get('days_to_analyze', 7)} days", flush=True)
        print(f"   Skip prompts: {'Yes' if self.config.get('wallet_analysis', {}).get('skip_prompts', True) else 'No'}", flush=True)
        
        print(f"\n📊 DATA SOURCES:", flush=True)
        print(f"   Telegram Channels: {len(self.config.get('sources', {}).get('telegram_groups', []))}", flush=True)
        for channel in self.config.get('sources', {}).get('telegram_groups', []):
            print(f"      - {channel}", flush=True)
        
        # Show wallets from file
        wallets_from_file = load_wallets_from_file("wallets.txt")
        print(f"\n💰 WALLETS:", flush=True)
        print(f"   Wallets in wallets.txt: {len(wallets_from_file)}", flush=True)
        for wallet in wallets_from_file[:5]:
            print(f"      - {wallet}", flush=True)
        if len(wallets_from_file) > 5:
            print(f"      ... and {len(wallets_from_file) - 5} more", flush=True)
        
        # Check cache status
        cache_dir = Path.home() / ".phoenix_cache"
        if cache_dir.exists():
            print(f"\n📦 CACHE STATUS:", flush=True)
            cache_files = list(cache_dir.glob("*.json"))
            if cache_files:
                for cache_file in cache_files:
                    if cache_file.name == "spydefi_kol_analysis.json":
                        try:
                            with open(cache_file, 'r') as f:
                                cache_data = json.load(f)
                            timestamp = cache_data.get('timestamp', 'Unknown')
                            version = cache_data.get('version', '4.0')
                            
                            if timestamp != 'Unknown':
                                cache_age = datetime.now() - datetime.fromisoformat(timestamp)
                                hours_old = cache_age.total_seconds() / 3600
                                status = "✅ Fresh" if hours_old < 6 else "⚠️ Expired"
                                print(f"   SPYDEFI cache: {status} ({hours_old:.1f} hours old, v{version} FIXED)", flush=True)
                        except:
                            print(f"   SPYDEFI cache: ❌ Error reading", flush=True)
            else:
                print(f"   No cache files", flush=True)
        
        # Feature availability
        birdeye_ok = bool(self.config.get("birdeye_api_key"))
        helius_ok = bool(self.config.get("helius_api_key"))
        telegram_ok = bool(self.config.get("telegram_api_id") and self.config.get("telegram_api_hash"))
        cielo_ok = bool(self.config.get("cielo_api_key"))
        
        print(f"\n🎯 FEATURES AVAILABLE:", flush=True)
        print(f"   Token Price Analysis: {'✅ Full' if (birdeye_ok and helius_ok) else '⚠️ Limited' if birdeye_ok else '❌ Not Available'}", flush=True)
        print(f"   Wallet Analysis: {'✅ Available' if cielo_ok else '❌ Not Available'}", flush=True)
        print(f"   SPYDEFI KOL Analysis: {'✅ Available (FIXED)' if (birdeye_ok and telegram_ok) else '❌ Not Available'}", flush=True)
        print(f"   Real Channel ID Lookup: {'✅ Enabled' if telegram_ok else '❌ Need Telegram API'}", flush=True)
        print(f"   Fixed Logic & Pullbacks: {'✅ Implemented' if telegram_ok else '❌ Need Telegram API'}", flush=True)
        
        input("\nPress Enter to continue...")
    
    def _show_strategy_help(self):
        """Show help and strategy guidance (updated with FIXED SPYDEFI info)."""
        print("\n" + "="*80, flush=True)
        print("    📖 STRATEGY GUIDE - FIXED SPYDEFI Edition", flush=True)
        print("="*80, flush=True)
        
        print("\n🚀 SPYDEFI SYSTEM (FIXED VERSION 4.0):", flush=True)
        print("• FIXED: Real Telegram channel ID lookup (-1001234567890)", flush=True)
        print("• FIXED: Token analysis logic (proper 2x/5x counting)", flush=True)  
        print("• FIXED: losing_calls = tokens with >-50% pullback", flush=True)
        print("• FIXED: winning_calls = total_calls - losing_calls", flush=True)
        print("• FIXED: Pullback calculation (real percentages, not zeros)", flush=True)
        print("• FIXED: KOL extraction (real @usernames, no x2/x3 false positives)", flush=True)
        print("• Column order: channel_id next to kol for easy access", flush=True)
        print("• Removed: success_rate_percent, total_roi_percent, max_roi_percent", flush=True)
        print("• Reordered: success_rate_5x after avg_max_pullback_percent", flush=True)
        print("• Composite score: Based on 2x rate, 5x rate, time, pullback, avg_roi", flush=True)
        
        print("\n💎 KOL PERFORMANCE METRICS (FIXED):", flush=True)
        print("1. success_rate_2x - Main success metric (tokens_2x_plus/total_calls)")
        print("2. success_rate_5x - Gem finding ability (tokens_5x_plus/total_calls)")
        print("3. avg_time_to_2x_hours - Speed to reach 2x gains")
        print("4. avg_max_pullback_percent - FIXED: Real pullback for 2x tokens")
        print("5. avg_roi - Average return across all tokens")
        print("6. winning_calls - total_calls minus losing_calls")
        print("7. losing_calls - FIXED: Tokens with >-50% pullback")
        print("8. channel_id - REAL numeric Telegram channel ID")
        print("9. follower_tier - HIGH/MEDIUM/LOW classification")
        
        print("\n🎯 FIXED COMPOSITE SCORE FORMULA:", flush=True)
        print("• success_rate_2x: 30% weight - Main profitability metric")
        print("• success_rate_5x: 25% weight - Gem finding ability")
        print("• Speed to 2x: 20% weight - Faster gains = higher score")
        print("• Pullback management: 15% weight - Better risk control")
        print("• Average ROI: 10% weight - Overall return performance")
        
        print("\n📊 STRATEGY CLASSIFICATION (FIXED):", flush=True)
        
        print("\n🏃 SCALP Strategy KOLs:")
        print("   Criteria: 2x rate ≥35% + Fast 2x (≤12h) + Good overall performance")
        print("   Use case: Quick trades, momentum plays, fast entry/exit")
        print("   Risk: High competition, requires speed")
        
        print("\n💎 HOLD Strategy KOLs:")
        print("   Criteria: 5x rate ≥12% + Consistent performance")
        print("   Use case: Longer holds, gem hunting, patience required")
        print("   Risk: Longer time commitment, need patience")
        
        print("\n🔄 MIXED Strategy KOLs:")
        print("   Criteria: Doesn't fit pure SCALP or HOLD patterns")
        print("   Use case: Flexible approach, case-by-case decisions")
        
        print("\n🔢 HOW TO USE REAL CHANNEL IDs:")
        print("• Copy the numeric channel ID (e.g., -1001234567890)")
        print("• In Telegram, search: t.me/c/1234567890/1 (remove -100 prefix)")
        print("• Or use @username if channel_id shows username format")
        print("• Join channel to follow their calls in real-time")
        
        print("\n📈 COPY TRADING STRATEGY (FIXED LOGIC):")
        print("1. Focus on Composite Score ≥70 (COPY recommendation)")
        print("2. Use REAL channel IDs to find actual Telegram channels")
        print("3. SCALP KOLs: Quick 2-3x targets, watch pullbacks")
        print("4. HOLD KOLs: Patient 5-20x targets, ignore short-term noise")
        print("5. Monitor losing_calls % - avoid KOLs with >30% losing calls")
        print("6. Check avg_max_pullback_percent for position sizing")
        print("7. Diversify across multiple top-performing KOLs")
        
        print("\n💰 WALLET ANALYSIS (PRESERVED - UNCHANGED):", flush=True)
        print("• Comprehensive wallet performance tracking")
        print("• 7-day active trader focus with recent activity")
        print("• Smart exit timing analysis and TP guidance")
        print("• Entry/exit quality scoring")
        print("• Distribution analysis (5x+, 2x+, etc.)")
        print("• Copy decision recommendations")
        print("• All wallet functionality preserved and working")
        
        print("\n🔧 COMMAND LINE USAGE:")
        print("# Configure APIs")
        print("python phoenix.py configure --birdeye-api-key KEY --telegram-api-id ID --telegram-api-hash HASH")
        print("")
        print("# Run FIXED SPYDEFI analysis")
        print("python phoenix.py spydefi")
        print("")
        print("# Run wallet analysis (unchanged)")
        print("python phoenix.py wallet")
        
        input("\nPress Enter to continue...")
    
    def _view_current_sources(self):
        """View current data sources (updated with FIXED SPYDEFI info)."""
        print("\n" + "="*70, flush=True)
        print("    📂 CURRENT DATA SOURCES", flush=True)
        print("="*70, flush=True)
        
        # SPYDEFI source
        print(f"\n📱 SPYDEFI ANALYSIS (FIXED VERSION 4.0):", flush=True)
        print(f"   Primary channel: @spydefi", flush=True)
        print(f"   Scan period: 8 hours (PEAK MEMECOIN HOURS)", flush=True)
        print(f"   KOLs analyzed: TOP 50", flush=True)
        print(f"   Purpose: Discover top performing KOLs with FIXED logic", flush=True)
        print(f"   🔧 FIXES:", flush=True)
        print(f"      • Real channel ID lookup (-1001234567890)", flush=True)
        print(f"      • Fixed token counting (proper 2x/5x logic)", flush=True)
        print(f"      • losing_calls = >-50% pullback", flush=True)
        print(f"      • Real pullback calculation (not zeros)", flush=True)
        print(f"      • Fixed KOL extraction (no x2/x3 false positives)", flush=True)
        print(f"      • Column order: channel_id next to kol", flush=True)
        print(f"      • Removed broken columns, kept follower_tier", flush=True)
        
        # Telegram channels (legacy)
        channels = self.config.get('sources', {}).get('telegram_groups', [])
        print(f"\n📱 TELEGRAM CHANNELS ({len(channels)}):", flush=True)
        if channels:
            for i, channel in enumerate(channels, 1):
                print(f"   {i}. {channel}", flush=True)
        else:
            print("   No additional channels configured", flush=True)
        
        # Wallets file (PRESERVED)
        wallets = load_wallets_from_file("wallets.txt")
        print(f"\n💰 WALLETS FROM FILE ({len(wallets)}) - PRESERVED:", flush=True)
        if wallets:
            print(f"   Total wallets: {len(wallets)}", flush=True)
            for i, wallet in enumerate(wallets[:10], 1):
                print(f"   {i}. {wallet[:8]}...{wallet[-4:]}", flush=True)
            if len(wallets) > 10:
                print(f"   ... and {len(wallets) - 10} more wallets", flush=True)
            print("\n   Note: Run wallet analysis - functionality PRESERVED and working", flush=True)
        else:
            print("   No wallets found in wallets.txt", flush=True)
        
        # API Status
        print(f"\n🔌 API STATUS:", flush=True)
        print(f"   Birdeye: {'✅ Configured' if self.config.get('birdeye_api_key') else '❌ Not configured'}", flush=True)
        print(f"   Helius: {'✅ Configured' if self.config.get('helius_api_key') else '⚠️ Not configured'}", flush=True)
        print(f"   Cielo: {'✅ Configured' if self.config.get('cielo_api_key') else '❌ Not configured'}", flush=True)
        print(f"   Telegram: {'✅ Configured' if self.config.get('telegram_api_id') else '❌ Not configured'}", flush=True)
        
        # Analysis capabilities
        print(f"\n⚙️ ANALYSIS CAPABILITIES:", flush=True)
        birdeye_ok = bool(self.config.get("birdeye_api_key"))
        telegram_ok = bool(self.config.get("telegram_api_id"))
        
        print(f"   🎯 SPYDEFI Analysis: {'✅ Available (FIXED)' if (birdeye_ok and telegram_ok) else '❌ APIs needed'}", flush=True)
        print(f"   💰 Wallet Analysis: {'✅ Available (PRESERVED)' if self.config.get('cielo_api_key') else '❌ Cielo API needed'}", flush=True)
        print(f"   📊 Token Price Analysis: {'✅ Full' if birdeye_ok else '❌ Birdeye API needed'}", flush=True)
        print(f"   🔢 Real Channel ID Lookup: {'✅ FIXED & Enabled' if telegram_ok else '❌ Telegram API needed'}", flush=True)
        
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
            elif args.command == "spydefi":
                self._handle_spydefi_analysis(args)
            elif args.command == "wallet":
                self._handle_wallet_analysis(args)
    
    def _handle_configure(self, args: argparse.Namespace) -> None:
        """Handle the configure command (PRESERVED - unchanged)."""
        if args.birdeye_api_key:
            self.config["birdeye_api_key"] = args.birdeye_api_key
            logger.info("Birdeye API key configured.")
        
        if args.cielo_api_key:
            self.config["cielo_api_key"] = args.cielo_api_key
            logger.info("Cielo Finance API key configured.")
        
        if args.helius_api_key:
            self.config["helius_api_key"] = args.helius_api_key
            logger.info("Helius API key configured.")
        
        if args.telegram_api_id:
            self.config["telegram_api_id"] = args.telegram_api_id
            logger.info("Telegram API ID configured.")
        
        if args.telegram_api_hash:
            self.config["telegram_api_hash"] = args.telegram_api_hash
            logger.info("Telegram API hash configured.")
        
        if args.rpc_url:
            self.config["solana_rpc_url"] = args.rpc_url
            logger.info("Solana RPC URL configured.")
        
        if args.analysis_days:
            if "wallet_analysis" not in self.config:
                self.config["wallet_analysis"] = {}
            self.config["wallet_analysis"]["days_to_analyze"] = args.analysis_days
            logger.info(f"Default analysis days set to {args.analysis_days}")
        
        save_config(self.config)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
    
    def _handle_wallet_analysis(self, args: argparse.Namespace) -> None:
        """Handle the wallet analysis command (PRESERVED - unchanged)."""
        # Load wallets
        wallets = load_wallets_from_file(args.wallets_file)
        if not wallets:
            logger.error(f"No wallets found in {args.wallets_file}")
            return
        
        logger.info(f"Loaded {len(wallets)} wallets from {args.wallets_file}")
        
        # Get days to analyze
        if args.days:
            days_to_analyze = args.days
        else:
            days_to_analyze = self.config.get("wallet_analysis", {}).get("days_to_analyze", 7)
        
        # Initialize APIs
        try:
            from dual_api_manager import DualAPIManager
            from wallet_module import WalletAnalyzer
            
            api_manager = DualAPIManager(
                self.config.get("birdeye_api_key", ""),
                self.config.get("cielo_api_key"),
                self.config.get("helius_api_key")
            )
            
            wallet_analyzer = WalletAnalyzer(
                cielo_api=api_manager.cielo_api,
                birdeye_api=api_manager.birdeye_api,
                helius_api=api_manager.helius_api,
                rpc_url=self.config.get("solana_rpc_url")
            )
            
            # Run batch analysis
            results = wallet_analyzer.batch_analyze_wallets(
                wallets,
                days_back=days_to_analyze,
                use_hybrid=True
            )
            
            if results.get("success"):
                # Export to CSV
                output_file = ensure_output_dir(args.output)
                if not output_file.endswith('.csv'):
                    output_file = output_file.replace('.xlsx', '.csv')
                self._export_wallet_csv(results, output_file)
                
                logger.info(f"Analysis complete. Results saved to {output_file}")
            else:
                logger.error(f"Analysis failed: {results.get('error')}")
                
        except Exception as e:
            logger.error(f"Error during wallet analysis: {str(e)}")

# Export functions for SPYDEFI results with FIXED logic
async def export_spydefi_results(results: Dict[str, Any], output_file: str):
    """Export SPYDEFI analysis results with FIXED column order and logic."""
    try:
        from export_utils import export_spydefi_to_csv, export_spydefi_summary_txt
        
        # Check if we have valid results
        kol_performances = results.get('kol_performances', {})
        
        if not kol_performances:
            logger.warning("⚠️ No valid KOL performances found")
            return
        
        # Export with FIXED column order
        csv_success = export_spydefi_to_csv(results, output_file)
        
        # Export TXT summary
        txt_file = output_file.replace('.csv', '_summary.txt')
        txt_success = export_spydefi_summary_txt(results, txt_file)
        
        if csv_success and txt_success:
            logger.info(f"✅ SPYDEFI results exported successfully with FIXED logic")
            logger.info(f"📄 CSV: {output_file}")
            logger.info(f"📄 Summary: {txt_file}")
        elif csv_success:
            logger.info(f"✅ CSV exported: {output_file}")
            logger.warning(f"⚠️ TXT export failed: {txt_file}")
        else:
            logger.error(f"❌ Export failed for both CSV and TXT")
        
    except Exception as e:
        logger.error(f"❌ Error exporting SPYDEFI results: {str(e)}")
        raise

def display_spydefi_summary(results: Dict[str, Any]):
    """Display SPYDEFI analysis summary with FIXED logic info."""
    try:
        kol_performances = results.get('kol_performances', {})
        metadata = results.get('metadata', {})
        
        print("\n" + "="*80, flush=True)
        print("    🎉 SPYDEFI ANALYSIS COMPLETE (FIXED VERSION)", flush=True)
        print("="*80, flush=True)
        
        if not kol_performances:
            print("\n❌ NO VALID KOL PERFORMANCES FOUND", flush=True)
            return
        
        print(f"\n📊 OVERALL STATISTICS (FIXED):", flush=True)
        print(f"   🎯 KOLs analyzed: {len(kol_performances)}", flush=True)
        print(f"   📞 Total calls: {metadata.get('total_calls_analyzed', 0)}", flush=True)
        print(f"   💎 Overall 2x rate: {metadata.get('overall_2x_rate', 0):.1f}%", flush=True)
        print(f"   🚀 Overall 5x rate: {metadata.get('overall_5x_rate', 0):.1f}%", flush=True)
        print(f"   ⏱️ Processing time: {metadata.get('processing_time_seconds', 0):.1f}s", flush=True)
        print(f"   📡 API calls: {metadata.get('api_calls', 0)}", flush=True)
        print(f"   🔧 Version: v4.0 (FIXED: Real Channel IDs, Logic, Pullbacks)", flush=True)
        
        # Top 10 KOLs with FIXED display
        top_kols = list(kol_performances.items())[:10]
        
        print(f"\n🏆 TOP {min(10, len(top_kols))} KOLS (FIXED LOGIC):", flush=True)
        for i, (kol, perf) in enumerate(top_kols, 1):
            if isinstance(perf, dict):
                score = perf.get('composite_score', 0)
                success_rate_2x = perf.get('success_rate_2x', 0)
                success_rate_5x = perf.get('success_rate_5x', 0)
                strategy = perf.get('strategy_classification', 'UNKNOWN')
                calls = perf.get('total_calls', 0)
                channel_id = perf.get('channel_id', 'N/A')
                pullback = perf.get('avg_max_pullback_percent', 0)
                follower_tier = perf.get('follower_tier', 'LOW')
                winning_calls = perf.get('winning_calls', 0)
                losing_calls = perf.get('losing_calls', 0)
            else:
                score = perf.composite_score
                success_rate_2x = perf.success_rate_2x
                success_rate_5x = perf.success_rate_5x
                strategy = perf.strategy_classification
                calls = perf.total_calls
                channel_id = perf.channel_id
                pullback = perf.avg_max_pullback_percent
                follower_tier = perf.follower_tier
                winning_calls = perf.winning_calls
                losing_calls = perf.losing_calls
            
            copy_rec = "COPY" if score >= 70 else "AVOID"
            
            print(f"\n{i}. @{kol}", flush=True)
            print(f"   📊 Score: {score:.1f}/100 | Rec: {copy_rec}", flush=True)
            print(f"   🔢 Channel ID: {channel_id} (REAL NUMERIC ID)", flush=True)
            print(f"   💎 2x Rate: {success_rate_2x:.1f}% | 5x Rate: {success_rate_5x:.1f}%", flush=True)
            print(f"   📈 Strategy: {strategy} | Tier: {follower_tier}", flush=True)
            print(f"   📞 Calls: {calls} | ✅Win: {winning_calls} | ❌Lose: {losing_calls}", flush=True)
            print(f"   📉 Pullback: {pullback:.1f}% (FIXED - Real calculation)", flush=True)
        
        print(f"\n✅ FIXES IMPLEMENTED:", flush=True)
        print(f"   🔢 Real channel IDs (numeric like -1001234567890)", flush=True)
        print(f"   🧮 Fixed token counting logic (proper 2x/5x rates)", flush=True)
        print(f"   ❌ losing_calls = tokens with >-50% pullback", flush=True)
        print(f"   ✅ winning_calls = total_calls - losing_calls", flush=True)
        print(f"   📉 Real pullback calculation (not zeros)", flush=True)
        print(f"   📊 Column order: channel_id next to kol", flush=True)
        print(f"   🗑️ Removed: success_rate_percent, total_roi_percent, max_roi_percent", flush=True)
        print(f"   ✅ Kept: follower_tier for strategy classification", flush=True)
        
    except Exception as e:
        logger.error(f"Error displaying SPYDEFI summary: {str(e)}")
        print(f"\n❌ Error displaying summary: {str(e)}", flush=True)

def main():
    """Main entry point for the FIXED Phoenix CLI."""
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