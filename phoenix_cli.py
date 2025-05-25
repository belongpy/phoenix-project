#!/usr/bin/env python3
"""
Phoenix Project - UPDATED CLI Tool for 7-Day Active Trader Analysis

🎯 MAJOR UPDATES:
- 7-day recency focus for all wallet analysis
- Enhanced strategy recommendations with TP guidance
- Removed redundant columns (confidence, tier, rating)
- Smart sell following based on exit quality
- Active trader detection and prioritization
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
        "helius_api_key": "",
        "telegram_api_id": "",
        "telegram_api_hash": "",
        "telegram_session": "",
        "solana_rpc_url": "https://api.mainnet-beta.solana.com",
        "sources": {
            "telegram_groups": ["spydefi"],
            "wallets": []
        },
        "analysis_period_days": 7  # Changed default to 7 days
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

class PhoenixCLI:
    """Phoenix CLI with 7-day active trader focus."""
    
    def __init__(self):
        self.config = load_config()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="Phoenix Project - 7-Day Active Solana Trader Analysis Tool",
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
        
        # Enhanced telegram analysis command
        telegram_parser = subparsers.add_parser("telegram", help="Enhanced SpyDefi analysis")
        telegram_parser.add_argument("--hours", type=int, default=24, help="Hours to analyze (default: 24)")
        telegram_parser.add_argument("--output", default="spydefi_analysis_enhanced.csv", help="Output CSV file")
        telegram_parser.add_argument("--excel", action="store_true", help="Also export to Excel format")
        
        # Wallet analysis command with 7-day default
        wallet_parser = subparsers.add_parser("wallet", help="Analyze wallets for copy trading (7-day focus)")
        wallet_parser.add_argument("--wallets-file", default="wallets.txt", help="File containing wallet addresses")
        wallet_parser.add_argument("--days", type=int, default=7, help="Number of days to analyze (default: 7)")
        wallet_parser.add_argument("--output", default="wallet_analysis_7day.csv", help="Output file")
        
        return parser
    
    def _handle_numbered_menu(self):
        """Handle the numbered menu interface."""
        print("\n" + "="*80)
        print("Phoenix Project - 7-Day Active Trader Analysis Tool")
        print("🚀 Focus on ACTIVE traders with recent wins")
        print(f"📅 Current Date: {datetime.now().strftime('%Y-%m-%d')}")
        print("="*80)
        print("\nSelect an option:")
        print("\n🔧 CONFIGURATION:")
        print("1. Configure API Keys")
        print("2. Check Configuration")
        print("3. Test API Connectivity")
        print("\n📊 TOOLS:")
        print("4. SPYDEFI ANALYSIS")
        print("5. 7-DAY ACTIVE WALLET ANALYSIS (Enhanced Strategies)")
        print("\n🔍 UTILITIES:")
        print("6. View Current Sources")
        print("7. Help & Strategy Guide")
        print("0. Exit")
        print("="*80)
        
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
            
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
                self._enhanced_telegram_analysis()
            elif choice == '5':
                self._active_trader_wallet_analysis()
            elif choice == '6':
                self._view_current_sources()
            elif choice == '7':
                self._show_strategy_help()
            else:
                print("❌ Invalid choice. Please try again.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error in menu: {str(e)}")
            input("Press Enter to continue...")
    
    def _active_trader_wallet_analysis(self):
        """Run 7-day active trader wallet analysis with enhanced strategies."""
        print("\n" + "="*80)
        print("    💰 7-DAY ACTIVE TRADER ANALYSIS")
        print("    🎯 Focus: Recent winners with smart exit strategies")
        print("    📊 Features: Custom TPs based on trader behavior")
        print("="*80)
        
        # Check API configuration
        if not self.config.get("cielo_api_key"):
            print("\n❌ CRITICAL: Cielo Finance API key required for wallet analysis!")
            print("Please configure your Cielo Finance API key first (Option 1).")
            input("Press Enter to continue...")
            return
        
        if not self.config.get("helius_api_key"):
            print("\n⚠️ WARNING: Helius API key not configured!")
            print("Pump.fun token analysis will be limited without Helius.")
            print("Consider adding Helius API key for complete analysis.")
        
        # Load wallets
        wallets = load_wallets_from_file("wallets.txt")
        if not wallets:
            print("\n❌ No wallets found in wallets.txt")
            print("Please add wallet addresses to wallets.txt (one per line)")
            input("Press Enter to continue...")
            return
        
        print(f"\n📁 Found {len(wallets)} wallets in wallets.txt")
        
        # Get analysis parameters
        print("\n🔧 ANALYSIS PARAMETERS:")
        
        # Days to analyze (default 7)
        days_input = input("Days to analyze (default: 7, max: 30): ").strip()
        if days_input.isdigit():
            days_to_analyze = min(int(days_input), 30)
        else:
            days_to_analyze = 7
        
        print(f"\n🚀 Starting 7-day active trader analysis...")
        print(f"📊 Parameters:")
        print(f"   • Wallets: {len(wallets)}")
        print(f"   • Analysis period: {days_to_analyze} days")
        print(f"   • Focus: Active traders only (traded in last 7 days)")
        print(f"   • Strategy: Enhanced with TP guidance")
        print(f"   • Export format: CSV with strategy details")
        print("\nProcessing...")
        
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
                self._display_active_trader_results(results)
                
                # Export to CSV with enhanced format
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = ensure_output_dir(f"active_traders_7day_{timestamp}.csv")
                self._export_active_trader_csv(results, output_file)
                print(f"\n📄 Exported to CSV: {output_file}")
                
                print("\n✅ 7-day active trader analysis completed successfully!")
                
                # Display API call statistics
                if "api_calls" in results:
                    print(f"\n📊 API CALL EFFICIENCY:")
                    print(f"   Cielo: {results['api_calls']['cielo']} calls")
                    print(f"   Birdeye: {results['api_calls']['birdeye']} calls")
                    print(f"   Helius: {results['api_calls']['helius']} calls")
                    print(f"   RPC: {results['api_calls']['rpc']} calls")
                    print(f"   Total: {sum(results['api_calls'].values())} calls")
            else:
                print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"\n❌ Error during wallet analysis: {str(e)}")
            logger.error(f"Wallet analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _display_active_trader_results(self, results: Dict[str, Any]) -> None:
        """Display 7-day active trader analysis results."""
        print("\n" + "="*80)
        print("    📊 7-DAY ACTIVE TRADER ANALYSIS RESULTS")
        print("="*80)
        
        # Summary statistics
        print(f"\n📈 SUMMARY:")
        print(f"   Total wallets: {results['total_wallets']}")
        print(f"   Successfully analyzed: {results['analyzed_wallets']}")
        print(f"   Failed: {results['failed_wallets']}")
        
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
        
        print(f"\n🟢 Active traders (7-day): {active_count}")
        print(f"🔴 Inactive traders: {inactive_count}")
        
        # Sort all wallets by score
        all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
        
        # Get only active traders
        active_wallets = [w for w in all_wallets if w.get('metrics', {}).get('active_trader', False)]
        
        if active_wallets:
            print(f"\n🏆 TOP 10 ACTIVE TRADERS (Last 7 Days):")
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
                
                print(f"\n{i}. Wallet: {wallet[:8]}...{wallet[-4:]}")
                print(f"   Score: {composite_score:.1f}/100")
                print(f"   Type: {analysis['wallet_type']}")
                print(f"   === 7-DAY PERFORMANCE ===")
                print(f"   Trades (7d): {metrics.get('trades_last_7_days', 0)} | Win Rate (7d): {metrics.get('win_rate_7d', 0):.1f}%")
                print(f"   Profit (7d): ${metrics.get('profit_7d', 0):.2f}")
                print(f"   Days since trade: {metrics.get('days_since_last_trade', 999)}")
                print(f"   === OVERALL STATS ===")
                print(f"   Total Trades: {metrics['total_trades']} | Win Rate: {metrics['win_rate']:.1f}%")
                print(f"   Profit Factor: {profit_factor_display} | Net Profit: ${metrics['net_profit_usd']:.2f}")
                print(f"   5x+ Gem Rate: {metrics.get('gem_rate_5x_plus', 0):.1f}%")
                print(f"   Avg Hold: {metrics.get('avg_hold_time_minutes', 0):.1f} min")
                print(f"   === STRATEGY RECOMMENDATION ===")
                print(f"   Action: {strategy.get('recommendation', 'UNKNOWN')}")
                print(f"   Follow Sells: {'YES ✅' if strategy.get('follow_sells', False) else 'NO ❌'}")
                print(f"   TP1: {strategy.get('tp1_percent', 0)}% | TP2: {strategy.get('tp2_percent', 0)}%")
                print(f"   Guidance: {strategy.get('tp_guidance', 'No guidance available')}")
                
                # Entry/Exit Analysis
                if 'entry_exit_analysis' in analysis:
                    ee_analysis = analysis['entry_exit_analysis']
                    print(f"   Entry/Exit Quality: {ee_analysis['entry_quality']}/{ee_analysis['exit_quality']}")
                    if ee_analysis.get('exit_quality') == 'POOR':
                        print(f"   ⚠️ They miss {ee_analysis.get('missed_gains_percent', 0):.0f}% gains on average")
                
                # Bundle detection
                if 'bundle_analysis' in analysis and analysis['bundle_analysis'].get('is_likely_bundler'):
                    print(f"   ⚠️ WARNING: Possible bundler detected!")
                
                # 7-day distribution
                print(f"   === 7-DAY DISTRIBUTION ===")
                print(f"   5x+: {metrics.get('distribution_500_plus_%', 0):.1f}% | "
                      f"2-5x: {metrics.get('distribution_200_500_%', 0):.1f}% | "
                      f"<2x: {metrics.get('distribution_0_200_%', 0):.1f}%")
        
        # Category breakdown
        print(f"\n📂 WALLET CATEGORIES:")
        print(f"   🎯 Snipers (< 1 min hold): {len(results.get('snipers', []))}")
        print(f"   ⚡ Flippers (1-10 min): {len(results.get('flippers', []))}")
        print(f"   📊 Scalpers (10-60 min): {len(results.get('scalpers', []))}")
        print(f"   💎 5x+ Gem Hunters: {len(results.get('gem_hunters', []))}")
        print(f"   📈 Swing Traders (1-24h): {len(results.get('swing_traders', []))}")
        print(f"   🏆 Position Traders (24h+): {len(results.get('position_traders', []))}")
        
        # Key insights
        print(f"\n📊 KEY INSIGHTS:")
        if active_wallets:
            # Count recent winners
            recent_5x = sum(1 for w in active_wallets 
                          if w.get('seven_day_metrics', {}).get('has_5x_last_7_days', False))
            recent_2x = sum(1 for w in active_wallets 
                          if w.get('seven_day_metrics', {}).get('has_2x_last_7_days', False))
            
            if recent_5x > 0:
                print(f"   🚀 {recent_5x} wallets hit 5x+ in last 7 days!")
            if recent_2x > 0:
                print(f"   📈 {recent_2x} wallets hit 2x+ in last 7 days!")
            
            # Exit quality breakdown
            good_exits = sum(1 for w in active_wallets 
                           if w.get('entry_exit_analysis', {}).get('exit_quality') in ['GOOD', 'EXCELLENT'])
            poor_exits = sum(1 for w in active_wallets 
                           if w.get('entry_exit_analysis', {}).get('exit_quality') == 'POOR')
            
            print(f"   ✅ {good_exits} wallets have good exit timing (follow their sells)")
            print(f"   ❌ {poor_exits} wallets exit too early (use fixed TPs instead)")
    
    def _export_active_trader_csv(self, results: Dict[str, Any], output_file: str) -> None:
        """Export 7-day active trader analysis to CSV with enhanced strategy columns."""
        try:
            from export_utils import export_wallet_rankings_csv
            export_wallet_rankings_csv(results, output_file)
            logger.info(f"Exported active trader analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
    
    def _enhanced_telegram_analysis(self):
        """Run enhanced Telegram analysis with 5x+ focus."""
        print("\n" + "="*80)
        print("    🎯 ENHANCED SPYDEFI TELEGRAM ANALYSIS")
        print("    📉 Max Pullback % + ⏱️ Time to 5x Analysis")
        print("="*80)
        
        # Check API configuration
        if not self.config.get("birdeye_api_key"):
            print("\n❌ CRITICAL: Birdeye API key required for enhanced analysis!")
            print("Please configure your Birdeye API key first (Option 1).")
            input("Press Enter to continue...")
            return
        
        if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
            print("\n❌ CRITICAL: Telegram API credentials required!")
            print("Please configure your Telegram API credentials first (Option 1).")
            input("Press Enter to continue...")
            return
        
        print("\n🚀 Starting enhanced SpyDefi analysis...")
        print("📅 Analysis period: 24 hours")
        print("📁 Output: spydefi_analysis_enhanced.csv")
        print("📊 Excel export: Enabled")
        print("🎯 Enhanced features:")
        print("   • ✅ Max pullback % for stop loss calculation")
        print("   • ✅ Average time to reach 5x for gem hunting")
        print("   • ✅ Enhanced contract address detection")
        print("   • ✅ Detailed price analysis using Birdeye API")
        if self.config.get("helius_api_key"):
            print("   • ✅ Helius API for pump.fun token analysis")
        else:
            print("   • ⚠️ Helius API not configured - pump.fun analysis limited")
        print("\nProcessing...")
        
        # Create args object with defaults
        class Args:
            def __init__(self):
                self.channels = ["spydefi"]
                self.days = 1  # 24 hours
                self.hours = 24
                self.output = "spydefi_analysis_enhanced.csv"
                self.excel = True
        
        args = Args()
        
        try:
            self._handle_telegram_analysis(args)
            print("\n✅ Enhanced analysis completed successfully!")
            print("📁 Check the outputs folder for results")
            
        except Exception as e:
            print(f"\n❌ Enhanced analysis failed: {str(e)}")
            logger.error(f"Enhanced telegram analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _handle_telegram_analysis(self, args) -> None:
        """Handle the enhanced telegram analysis command."""
        import asyncio
        
        try:
            import importlib
            import sys
            
            if 'telegram_module' in sys.modules:
                del sys.modules['telegram_module']
            
            from telegram_module import TelegramScraper
            from birdeye_api import BirdeyeAPI
            
            logger.info("✅ Imported telegram module")
            
        except Exception as e:
            logger.error(f"❌ Error importing modules: {str(e)}")
            raise
        
        channels = getattr(args, 'channels', None) or self.config["sources"]["telegram_groups"]
        if not channels:
            logger.error("No Telegram channels specified.")
            return
        
        if not self.config.get("birdeye_api_key"):
            logger.error("🎯 CRITICAL: Birdeye API key required for enhanced analysis!")
            return
            
        if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
            logger.error("📱 CRITICAL: Telegram API credentials required!")
            return
        
        output_file = ensure_output_dir(args.output)
        hours = getattr(args, 'hours', 24)
        days = getattr(args, 'days', 1)
        
        logger.info(f"🚀 Starting enhanced SpyDefi analysis for the past {hours} hours.")
        logger.info(f"📁 Results will be saved to {output_file}")
        
        try:
            birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
            logger.info("✅ Birdeye API initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Birdeye API: {str(e)}")
            raise
        
        # Initialize Helius API if configured
        helius_api = None
        if self.config.get("helius_api_key"):
            try:
                from helius_api import HeliusAPI
                helius_api = HeliusAPI(self.config["helius_api_key"])
                logger.info("✅ Helius API initialized successfully for pump.fun tokens")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Helius API: {str(e)}")
                logger.warning("Pump.fun token analysis will be limited")
        
        try:
            telegram_scraper = TelegramScraper(
                self.config["telegram_api_id"],
                self.config["telegram_api_hash"],
                self.config.get("telegram_session", "phoenix")
            )
            logger.info("✅ Telegram scraper initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Telegram scraper: {str(e)}")
            raise
        
        telegram_analyses = {"ranked_kols": []}
        
        if any(ch.lower() == "spydefi" for ch in channels):
            logger.info("🎯 SpyDefi channel detected. Running enhanced analysis...")
            
            try:
                async def run_enhanced_spydefi_analysis():
                    try:
                        await telegram_scraper.connect()
                        logger.info("📞 Connected to Telegram")
                        
                        telegram_scraper.birdeye_api = birdeye_api
                        telegram_scraper.helius_api = helius_api
                        
                        analysis = await telegram_scraper.redesigned_spydefi_analysis(hours)
                        
                        logger.info("📊 Analysis completed, exporting results...")
                        
                        await telegram_scraper.export_spydefi_analysis(analysis, output_file)
                        
                        return analysis
                        
                    except Exception as e:
                        logger.error(f"❌ Error in analysis: {str(e)}")
                        import traceback
                        logger.error(f"❌ Analysis traceback: {traceback.format_exc()}")
                        raise
                    finally:
                        await telegram_scraper.disconnect()
                        logger.info("📞 Disconnected from Telegram")
                
                telegram_analyses = asyncio.run(run_enhanced_spydefi_analysis())
                
                if telegram_analyses.get('success'):
                    enhanced_count = sum(kol.get('detailed_analysis_count', 0) for kol in telegram_analyses.get('ranked_kols', {}).values())
                    total_count = telegram_analyses.get('total_calls', 0)
                    pump_count = telegram_analyses.get('total_pump_tokens', 0)
                    
                    if enhanced_count > 0:
                        logger.info(f"✅ Enhanced SpyDefi analysis completed successfully!")
                        logger.info(f"🎯 Enhanced analysis coverage: {enhanced_count}/{total_count} tokens ({(enhanced_count/total_count*100):.1f}%)")
                        if pump_count > 0:
                            logger.info(f"🚀 Pump.fun tokens analyzed: {pump_count}")
                else:
                    logger.error(f"❌ Analysis failed: {telegram_analyses.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"❌ Error in enhanced SpyDefi analysis: {str(e)}")
                return
        
        logger.info(f"📁 Enhanced telegram analysis completed. Results saved to {output_file}")
        
        # Enhanced Excel export
        if hasattr(args, 'excel') and args.excel:
            try:
                from export_utils import export_to_excel
                excel_file = output_file.replace(".csv", "_enhanced.xlsx")
                
                if isinstance(telegram_analyses, dict) and telegram_analyses.get('ranked_kols'):
                    enhanced_telegram_data = {"ranked_kols": []}
                    
                    for kol, performance in telegram_analyses['ranked_kols'].items():
                        enhanced_kol_data = {
                            "channel_id": performance.get('channel_id', ''),
                            "total_calls": performance.get('tokens_mentioned', 0),
                            "success_rate": performance.get('success_rate_5x', 0),
                            "avg_roi": performance.get('avg_ath_roi', 0),
                            "avg_max_roi": performance.get('avg_ath_roi', 0),
                            "composite_score": performance.get('composite_score', 0),
                            "avg_max_pullback_percent": performance.get('avg_max_pullback_percent', 0),
                            "avg_time_to_5x_formatted": performance.get('avg_time_to_5x_formatted', 'N/A'),
                            "detailed_analysis_count": performance.get('detailed_analysis_count', 0),
                            "pump_tokens_analyzed": performance.get('pump_tokens_analyzed', 0),
                            "pump_success_rate_5x": performance.get('pump_success_rate_5x', 0),
                            "strategy": {
                                "recommendation": "ENHANCED_ANALYSIS",
                                "entry_type": "IMMEDIATE",
                                "take_profit_1": 100,
                                "take_profit_2": 300,
                                "take_profit_3": 500,
                                "stop_loss": -(performance.get('avg_max_pullback_percent', 25) + 10)
                            }
                        }
                        enhanced_telegram_data["ranked_kols"].append(enhanced_kol_data)
                    
                    export_to_excel(enhanced_telegram_data, {}, excel_file)
                    logger.info(f"📊 Enhanced Excel export completed: {excel_file}")
                    
            except Exception as e:
                logger.error(f"❌ Error in Excel export: {str(e)}")
    
    def _test_api_connectivity(self):
        """Test API connectivity."""
        print("\n" + "="*70)
        print("    🔍 API CONNECTIVITY TEST")
        print("="*70)
        
        # Test Birdeye API
        if self.config.get("birdeye_api_key"):
            print("\n🔍 Testing Birdeye API...")
            try:
                from birdeye_api import BirdeyeAPI
                birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
                test_result = birdeye_api.get_token_info("So11111111111111111111111111111111111111112")
                if test_result.get("success"):
                    print("✅ Birdeye API: Connected successfully")
                    print("   🎯 Mainstream token analysis: Available")
                else:
                    print("❌ Birdeye API: Connection failed")
            except Exception as e:
                print(f"❌ Birdeye API: Error - {str(e)}")
        else:
            print("❌ Birdeye API: Not configured")
        
        # Test Helius API
        if self.config.get("helius_api_key"):
            print("\n🚀 Testing Helius API...")
            try:
                from helius_api import HeliusAPI
                helius_api = HeliusAPI(self.config["helius_api_key"])
                if helius_api.health_check():
                    print("✅ Helius API: Connected successfully")
                    print("   🎯 Pump.fun token analysis: Available")
                else:
                    print("❌ Helius API: Connection failed")
            except Exception as e:
                print(f"❌ Helius API: Error - {str(e)}")
        else:
            print("⚠️ Helius API: Not configured")
            print("   ⚠️ Pump.fun token analysis will be limited")
        
        # Test Cielo Finance API
        if self.config.get("cielo_api_key"):
            print("\n💰 Testing Cielo Finance API...")
            try:
                from cielo_api import CieloFinanceAPI
                cielo_api = CieloFinanceAPI(self.config["cielo_api_key"])
                if cielo_api.health_check():
                    print("✅ Cielo Finance API: Connected successfully")
                    print("   💰 Wallet analysis: Available")
                else:
                    print("❌ Cielo Finance API: Connection failed")
            except Exception as e:
                print(f"❌ Cielo Finance API: Error - {str(e)}")
        else:
            print("❌ Cielo Finance API: Not configured")
            print("   ⚠️ CRITICAL: Wallet analysis requires Cielo Finance API")
        
        # Test Telegram API
        if self.config.get("telegram_api_id") and self.config.get("telegram_api_hash"):
            print("\n📱 Testing Telegram API...")
            try:
                from telegram_module import TelegramScraper
                print("✅ Telegram API: Configuration appears valid")
                print("   📊 SpyDefi analysis: Available")
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
        
        # Summary
        print(f"\n📊 7-DAY ACTIVE TRADER FEATURES:")
        birdeye_ok = bool(self.config.get("birdeye_api_key"))
        helius_ok = bool(self.config.get("helius_api_key"))
        telegram_ok = bool(self.config.get("telegram_api_id") and self.config.get("telegram_api_hash"))
        cielo_ok = bool(self.config.get("cielo_api_key"))
        
        print(f"   🎯 Token Price Analysis: {'✅ Full' if (birdeye_ok and helius_ok) else '⚠️ Limited' if birdeye_ok else '❌ Not Available'}")
        print(f"   💰 Wallet Analysis: {'✅ Ready' if cielo_ok else '❌ Need Cielo Finance API'}")
        print(f"   📱 Telegram/SpyDefi: {'✅ Ready' if (birdeye_ok and telegram_ok) else '❌ Missing APIs'}")
        print(f"   🎯 Market Cap Tracking: {'✅ Active' if birdeye_ok else '❌ Need Birdeye'}")
        print(f"   ⚡ Bundle Detection: {'✅ Active' if cielo_ok else '❌ Need Cielo'}")
        print(f"   📊 Entry/Exit Quality: {'✅ Full Analysis' if (birdeye_ok and helius_ok) else '⚠️ Basic Only'}")
        print(f"   🚀 5x+ Gem Detection: {'✅ Active' if cielo_ok else '❌ Need Cielo'}")
        print(f"   📈 7-Day Focus: {'✅ Active' if cielo_ok else '❌ Need Cielo'}")
        print(f"   🎯 Enhanced Strategy: {'✅ Active' if cielo_ok else '❌ Need Cielo'}")
        
        if birdeye_ok and helius_ok and telegram_ok and cielo_ok:
            print(f"\n🎉 ALL SYSTEMS GO! Full 7-day active trader capabilities available.")
        else:
            print(f"\n⚠️ Configure missing APIs to enable all active trader features.")
        
        input("\nPress Enter to continue...")
    
    def _interactive_configure(self):
        """Interactive configuration setup."""
        print("\n" + "="*70)
        print("    🔧 CONFIGURATION SETUP")
        print("="*70)
        
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
            print("\n🔑 Birdeye API Key (REQUIRED for token analysis)")
            print("   📊 Get your key from: https://birdeye.so")
            new_key = input("Enter Birdeye API key: ").strip()
            if new_key:
                self.config["birdeye_api_key"] = new_key
                print("✅ Birdeye API key configured")
        
        # Helius API Key
        current_helius = self.config.get("helius_api_key", "")
        if current_helius:
            print(f"\n🚀 Current Helius API Key: {current_helius[:8]}...")
            change_helius = input("Change Helius API key? (y/N): ").lower().strip()
            if change_helius == 'y':
                new_key = input("Enter new Helius API key: ").strip()
                if new_key:
                    self.config["helius_api_key"] = new_key
                    print("✅ Helius API key updated")
        else:
            print("\n🚀 Helius API Key (RECOMMENDED for pump.fun tokens)")
            print("   📊 Required for complete memecoin analysis")
            print("   🔑 Get your key from: https://helius.dev")
            new_key = input("Enter Helius API key (or press Enter to skip): ").strip()
            if new_key:
                self.config["helius_api_key"] = new_key
                print("✅ Helius API key configured")
                print("   🎯 Pump.fun token analysis: Now available")
            else:
                print("⚠️ Skipped: Pump.fun token analysis will be limited")
        
        # Cielo Finance API Key
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
            print("\n💰 Cielo Finance API Key (REQUIRED for wallet analysis)")
            print("   🔑 Get your key from: https://cielo.finance")
            new_key = input("Enter Cielo Finance API key: ").strip()
            if new_key:
                self.config["cielo_api_key"] = new_key
                print("✅ Cielo Finance API key configured")
        
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
            print("\n📱 Telegram API Credentials (Required for SpyDefi analysis)")
            print("   🔑 Get credentials from: https://my.telegram.org")
            new_id = input("Enter Telegram API ID: ").strip()
            new_hash = input("Enter Telegram API Hash: ").strip()
            if new_id and new_hash:
                self.config["telegram_api_id"] = new_id
                self.config["telegram_api_hash"] = new_hash
                print("✅ Telegram API credentials configured")
        
        # RPC URL
        current_rpc = self.config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
        print(f"\n🌐 Current RPC URL: {current_rpc}")
        change_rpc = input("Change RPC URL? (y/N): ").lower().strip()
        if change_rpc == 'y':
            print("   Options:")
            print("   1. Default Solana RPC")
            print("   2. Custom RPC URL (P9, QuickNode, etc.)")
            rpc_choice = input("Choose option (1-2): ").strip()
            if rpc_choice == '1':
                self.config["solana_rpc_url"] = "https://api.mainnet-beta.solana.com"
                print("✅ Using default Solana RPC")
            elif rpc_choice == '2':
                new_rpc = input("Enter custom RPC URL: ").strip()
                if new_rpc:
                    self.config["solana_rpc_url"] = new_rpc
                    print("✅ Custom RPC URL configured")
        
        # Save configuration
        save_config(self.config)
        print("\n✅ Configuration saved successfully!")
        
        input("\nPress Enter to continue...")
    
    def _check_configuration(self):
        """Check current configuration."""
        print("\n" + "="*70)
        print("    📋 CURRENT CONFIGURATION")
        print("="*70)
        
        print(f"\n🔑 API KEYS:")
        print(f"   Birdeye API Key: {'✅ Configured' if self.config.get('birdeye_api_key') else '❌ Not configured'}")
        print(f"   Helius API Key: {'✅ Configured' if self.config.get('helius_api_key') else '⚠️ Not configured (optional)'}")
        print(f"   Cielo Finance API Key: {'✅ Configured' if self.config.get('cielo_api_key') else '❌ Not configured'}")
        print(f"   Telegram API ID: {'✅ Configured' if self.config.get('telegram_api_id') else '❌ Not configured'}")
        print(f"   Telegram API Hash: {'✅ Configured' if self.config.get('telegram_api_hash') else '❌ Not configured'}")
        
        print(f"\n🌐 RPC ENDPOINT:")
        print(f"   URL: {self.config.get('solana_rpc_url', 'Default')}")
        
        print(f"\n📊 DATA SOURCES:")
        print(f"   Telegram Channels: {len(self.config.get('sources', {}).get('telegram_groups', []))}")
        for channel in self.config.get('sources', {}).get('telegram_groups', []):
            print(f"      - {channel}")
        
        # Show wallets from file
        wallets_from_file = load_wallets_from_file("wallets.txt")
        print(f"\n💰 WALLETS:")
        print(f"   Wallets in wallets.txt: {len(wallets_from_file)}")
        for wallet in wallets_from_file[:5]:
            print(f"      - {wallet}")
        if len(wallets_from_file) > 5:
            print(f"      ... and {len(wallets_from_file) - 5} more")
        
        # Feature availability
        birdeye_ok = bool(self.config.get("birdeye_api_key"))
        helius_ok = bool(self.config.get("helius_api_key"))
        telegram_ok = bool(self.config.get("telegram_api_id") and self.config.get("telegram_api_hash"))
        cielo_ok = bool(self.config.get("cielo_api_key"))
        
        print(f"\n🎯 7-DAY ACTIVE TRADER FEATURES:")
        print(f"   Token Price Analysis: {'✅ Full' if (birdeye_ok and helius_ok) else '⚠️ Limited' if birdeye_ok else '❌ Not Available'}")
        print(f"   Wallet Analysis: {'✅ Available' if cielo_ok else '❌ Not Available'}")
        print(f"   Entry/Exit Quality: {'✅ Full' if (birdeye_ok and helius_ok) else '⚠️ Basic' if birdeye_ok else '❌ Not Available'}")
        print(f"   Market Cap Tracking: {'✅ Active' if birdeye_ok else '❌ Not Available'}")
        print(f"   Bundle Detection: {'✅ Active' if cielo_ok else '❌ Not Available'}")
        print(f"   Enhanced Strategy: {'✅ Active' if cielo_ok else '❌ Not Available'}")
        print(f"   7-Day Focus: {'✅ Active' if cielo_ok else '❌ Not Available'}")
        
        input("\nPress Enter to continue...")
    
    def _show_strategy_help(self):
        """Show help and strategy guidance."""
        print("\n" + "="*80)
        print("    📖 STRATEGY GUIDE - 7-Day Active Trader Edition")
        print("="*80)
        
        print("\n🎯 WALLET SELECTION CRITERIA:")
        print("• Active in last 7 days (recent trades)")
        print("• Win rate 40%+ in last 7 days")
        print("• At least 3 trades in last week")
        print("• Hit 2x+ or 5x+ recently")
        
        print("\n📊 ENHANCED STRATEGY RECOMMENDATIONS:")
        
        print("\n1️⃣ FOLLOW SELLS = YES ✅")
        print("   When: Exit Quality = GOOD or EXCELLENT")
        print("   Why: They capture 60-80%+ of gains consistently")
        print("   Action: Copy their exit timing directly")
        
        print("\n2️⃣ FOLLOW SELLS = NO ❌")
        print("   When: Exit Quality = POOR")
        print("   Why: They exit too early, missing gains")
        print("   Action: Use fixed TPs instead:")
        print("   • If they avg 30% TP but miss 100%+ → Set TP1=60%, TP2=150%")
        print("   • If they avg 50% TP but miss 200%+ → Set TP1=100%, TP2=300%")
        
        print("\n📈 SELL STRATEGIES EXPLAINED:")
        print("• COPY_EXITS: Follow their sells exactly")
        print("• USE_FIXED_TP: Ignore their sells, use your TPs")
        print("• HYBRID: Consider their exits but hold longer")
        print("• FOLLOW_GEMS: For 5x+ hunters, let winners run")
        
        print("\n💎 WALLET TYPES & TYPICAL TPs:")
        print("• Sniper: TP1=50-100% (quick profits)")
        print("• Flipper: TP1=30-50% (fast turnover)")
        print("• Scalper: TP1=20-50% (consistent gains)")
        print("• Gem Hunter: TP1=400%+ (hold for 5x+)")
        print("• Swing Trader: TP1=100-200% (patience pays)")
        
        print("\n⚠️ RED FLAGS TO WATCH:")
        print("• Days since trade > 3 = Getting inactive")
        print("• Exit quality = POOR = Don't follow sells")
        print("• Bundle warning = Verify on-chain first")
        print("• Missed gains > 200% = They panic sell")
        
        print("\n📊 7-DAY DISTRIBUTION FOCUS:")
        print("Look for wallets with high % in:")
        print("• 500%+ bucket (5x+ trades)")
        print("• 200-500% bucket (2x-5x trades)")
        print("And low % in:")
        print("• Below -50% bucket (catastrophic losses)")
        
        print("\n🔧 COMMAND LINE USAGE:")
        print("# Configure all APIs")
        print("python phoenix.py configure --birdeye-api-key KEY --helius-api-key KEY --cielo-api-key KEY")
        print()
        print("# Analyze active traders (7-day default)")
        print("python phoenix.py wallet --days 7")
        
        input("\nPress Enter to continue...")
    
    def _view_current_sources(self):
        """View current data sources."""
        print("\n" + "="*70)
        print("    📂 CURRENT DATA SOURCES")
        print("="*70)
        
        # Telegram channels
        channels = self.config.get('sources', {}).get('telegram_groups', [])
        print(f"\n📱 TELEGRAM CHANNELS ({len(channels)}):")
        if channels:
            for i, channel in enumerate(channels, 1):
                print(f"   {i}. {channel}")
        else:
            print("   No channels configured")
        
        # Wallets file
        wallets = load_wallets_from_file("wallets.txt")
        print(f"\n💰 WALLETS FROM FILE ({len(wallets)}):")
        if wallets:
            # Count active wallets (would need actual analysis to be accurate)
            print(f"   Total wallets: {len(wallets)}")
            for i, wallet in enumerate(wallets[:10], 1):
                print(f"   {i}. {wallet[:8]}...{wallet[-4:]}")
            if len(wallets) > 10:
                print(f"   ... and {len(wallets) - 10} more wallets")
            print("\n   Note: Run analysis to see active/inactive breakdown")
        else:
            print("   No wallets found in wallets.txt")
        
        # API Status
        print(f"\n🔌 API STATUS:")
        print(f"   Birdeye: {'✅ Configured' if self.config.get('birdeye_api_key') else '❌ Not configured'}")
        print(f"   Helius: {'✅ Configured' if self.config.get('helius_api_key') else '⚠️ Not configured'}")
        print(f"   Cielo: {'✅ Configured' if self.config.get('cielo_api_key') else '❌ Not configured'}")
        print(f"   Telegram: {'✅ Configured' if self.config.get('telegram_api_id') else '❌ Not configured'}")
        
        # Analysis settings
        print(f"\n⚙️ ANALYSIS SETTINGS:")
        print(f"   Default period: 7 days (active traders)")
        print(f"   Focus: Recent performance & activity")
        print(f"   Strategy: Enhanced with TP guidance")
        
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
        
        save_config(self.config)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
    
    def _handle_wallet_analysis(self, args: argparse.Namespace) -> None:
        """Handle the wallet analysis command."""
        # Load wallets
        wallets = load_wallets_from_file(args.wallets_file)
        if not wallets:
            logger.error(f"No wallets found in {args.wallets_file}")
            return
        
        logger.info(f"Loaded {len(wallets)} wallets from {args.wallets_file}")
        
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
                days_back=args.days,
                use_hybrid=True
            )
            
            if results.get("success"):
                # Export to CSV
                output_file = ensure_output_dir(args.output)
                if not output_file.endswith('.csv'):
                    output_file = output_file.replace('.xlsx', '.csv')
                self._export_active_trader_csv(results, output_file)
                
                logger.info(f"Analysis complete. Results saved to {output_file}")
            else:
                logger.error(f"Analysis failed: {results.get('error')}")
                
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