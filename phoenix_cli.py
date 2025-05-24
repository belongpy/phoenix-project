#!/usr/bin/env python3
"""
Phoenix Project - OPTIMIZED CLI Tool with Tiered Analysis & Advanced Metrics

🎯 MAJOR UPDATES:
- Tiered wallet analysis (Quick/Standard/Deep)
- New gem hunter criteria (5x+ focus)
- Advanced metrics display (Sharpe ratio, Diamond Hands, etc.)
- Smart caching to reduce API calls by ~40%
- API budget tracking and management
- Smart transaction sampling for deep analysis
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
        "analysis_period_days": 1,
        "cache_size_mb": 200,
        "api_budgets": {
            "birdeye_daily": 1000,
            "cielo_daily": 5000,
            "helius_daily": 2000
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

class PhoenixCLI:
    """Phoenix CLI with tiered analysis and advanced metrics."""
    
    def __init__(self):
        self.config = load_config()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="Phoenix Project - Optimized Solana Chain Analysis with Advanced Metrics",
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
        configure_parser.add_argument("--rpc-url", help="Solana RPC URL")
        configure_parser.add_argument("--cache-size", type=int, help="Cache size in MB (default: 200)")
        
        # Enhanced telegram analysis command
        telegram_parser = subparsers.add_parser("telegram", help="Enhanced SpyDefi analysis (5x+ gem focus)")
        telegram_parser.add_argument("--hours", type=int, default=24, help="Hours to analyze (default: 24)")
        telegram_parser.add_argument("--output", default="spydefi_analysis_enhanced.csv", help="Output CSV file")
        telegram_parser.add_argument("--excel", action="store_true", help="Also export to Excel format")
        
        # Wallet analysis command with tiers
        wallet_parser = subparsers.add_parser("wallet", help="Analyze wallets with tiered approach")
        wallet_parser.add_argument("--wallets-file", default="wallets.txt", help="File containing wallet addresses")
        wallet_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
        wallet_parser.add_argument("--output", default="wallet_analysis.xlsx", help="Output file")
        wallet_parser.add_argument("--tier", choices=["quick", "standard", "deep", "auto"], 
                                 default="auto", help="Analysis tier (auto = smart selection)")
        
        # Cache management
        cache_parser = subparsers.add_parser("cache", help="Manage cache")
        cache_parser.add_argument("--stats", action="store_true", help="Show cache statistics")
        cache_parser.add_argument("--clear", action="store_true", help="Clear cache")
        
        return parser
    
    def _handle_numbered_menu(self):
        """Handle the numbered menu interface."""
        print("\n" + "="*80)
        print("Phoenix Project - Optimized Analysis with Advanced Metrics")
        print("🚀 5x+ Gem Hunter Focus | 🧠 Smart Analysis | 💾 Intelligent Caching")
        print(f"📅 Current Date: {datetime.now().strftime('%B %d, %Y')}")
        print("="*80)
        print("\n🔧 CONFIGURATION:")
        print("1. Configure API Keys")
        print("2. Check Configuration")
        print("3. Test API Connectivity")
        print("\n📊 ANALYSIS TOOLS:")
        print("4. SPYDEFI TELEGRAM ANALYSIS (5x+ Gems)")
        print("5. WALLET ANALYSIS (Tiered)")
        print("6. QUICK WALLET SCAN")
        print("7. DEEP WALLET ANALYSIS")
        print("\n🔍 UTILITIES:")
        print("8. View Cache Statistics")
        print("9. Clear Cache")
        print("10. View API Budget Status")
        print("11. Help & Examples")
        print("0. Exit")
        print("="*80)
        
        try:
            choice = input("\nEnter your choice (0-11): ").strip()
            
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
                self._tiered_wallet_analysis()
            elif choice == '6':
                self._quick_wallet_scan()
            elif choice == '7':
                self._deep_wallet_analysis()
            elif choice == '8':
                self._view_cache_stats()
            elif choice == '9':
                self._clear_cache()
            elif choice == '10':
                self._view_api_budget()
            elif choice == '11':
                self._show_help_advanced()
            else:
                print("❌ Invalid choice. Please try again.")
                input("Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error in menu: {str(e)}")
            input("Press Enter to continue...")
    
    def _tiered_wallet_analysis(self):
        """Run tiered wallet analysis with smart tier selection."""
        print("\n" + "="*80)
        print("    💰 TIERED WALLET ANALYSIS")
        print("    🎯 Smart API Usage | 📊 Advanced Metrics")
        print("="*80)
        
        # Check API configuration
        if not self.config.get("cielo_api_key"):
            print("\n❌ CRITICAL: Cielo Finance API key required for wallet analysis!")
            print("Please configure your Cielo Finance API key first (Option 1).")
            input("Press Enter to continue...")
            return
        
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
        
        # Analysis tier selection
        print("\nAnalysis Tiers:")
        print("  • AUTO: Smart selection based on wallet performance")
        print("  • QUICK: Basic stats only (2 API calls)")
        print("  • STANDARD: Last 5 trades (5 API calls)")
        print("  • DEEP: Smart sampling with advanced metrics (20 API calls)")
        
        tier_input = input("\nSelect tier (auto/quick/standard/deep) [default: auto]: ").strip().lower()
        tier = tier_input if tier_input in ["quick", "standard", "deep"] else "auto"
        
        # Days to analyze
        days_input = input("Days to analyze (default: 30): ").strip()
        days_to_analyze = int(days_input) if days_input.isdigit() else 30
        
        print(f"\n🚀 Starting tiered wallet analysis...")
        print(f"📊 Parameters:")
        print(f"   • Wallets: {len(wallets)}")
        print(f"   • Analysis tier: {tier.upper()}")
        print(f"   • Analysis period: {days_to_analyze} days")
        print(f"   • New gem criteria: 5x+ (not 2x+)")
        print(f"   • Min trade volume: $100")
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
            
            # Run tiered batch analysis
            if tier == "auto":
                results = wallet_analyzer.batch_analyze_wallets_tiered(
                    wallets,
                    days_back=days_to_analyze,
                    default_tier="standard",
                    top_performers_deep=True
                )
            else:
                # Manual tier selection
                results = {}
                wallet_analyses = []
                failed_analyses = []
                
                for i, wallet in enumerate(wallets, 1):
                    print(f"\nAnalyzing wallet {i}/{len(wallets)}: {wallet[:8]}...{wallet[-4:]}")
                    
                    try:
                        analysis = wallet_analyzer.analyze_wallet_tiered(wallet, tier, days_to_analyze)
                        if analysis.get("success"):
                            wallet_analyses.append(analysis)
                        else:
                            failed_analyses.append({
                                "wallet_address": wallet,
                                "error": analysis.get("error", "Analysis failed")
                            })
                    except Exception as e:
                        logger.error(f"Error analyzing wallet {wallet}: {str(e)}")
                        failed_analyses.append({
                            "wallet_address": wallet,
                            "error": str(e)
                        })
                
                # Package results
                results = self._package_analysis_results(wallet_analyses, failed_analyses)
            
            if results.get("success"):
                self._display_tiered_wallet_results(results)
                
                # Export results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_base = f"wallet_analysis_tiered_{timestamp}"
                
                # Excel export
                excel_file = ensure_output_dir(f"{output_base}.xlsx")
                self._export_tiered_wallet_excel(results, excel_file)
                print(f"\n📊 Exported to Excel: {excel_file}")
                
                # CSV export
                csv_file = ensure_output_dir(f"{output_base}.csv")
                self._export_tiered_wallet_csv(results, csv_file)
                print(f"📄 Exported to CSV: {csv_file}")
                
                # Show cache stats
                if "cache_stats" in results:
                    cache_stats = results["cache_stats"]
                    print(f"\n💾 CACHE PERFORMANCE:")
                    print(f"   • Hit rate: {cache_stats.get('hit_rate_percent', 0):.1f}%")
                    print(f"   • API calls saved: {cache_stats.get('api_calls_saved', 0)}")
                    print(f"   • Estimated cost saved: ${cache_stats.get('estimated_api_cost_saved', 0):.2f}")
                
                print("\n✅ Tiered analysis completed successfully!")
            else:
                print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"\n❌ Error during wallet analysis: {str(e)}")
            logger.error(f"Wallet analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _quick_wallet_scan(self):
        """Run quick scan on all wallets."""
        print("\n" + "="*80)
        print("    ⚡ QUICK WALLET SCAN")
        print("    📊 Basic metrics only (2 API calls per wallet)")
        print("="*80)
        
        wallets = load_wallets_from_file("wallets.txt")
        if not wallets:
            print("\n❌ No wallets found in wallets.txt")
            input("Press Enter to continue...")
            return
        
        self._run_tier_analysis(wallets, "quick", 30)
    
    def _deep_wallet_analysis(self):
        """Run deep analysis on top wallets."""
        print("\n" + "="*80)
        print("    🔬 DEEP WALLET ANALYSIS")
        print("    📊 Full metrics with smart sampling (up to 20 API calls)")
        print("="*80)
        
        wallets = load_wallets_from_file("wallets.txt")
        if not wallets:
            print("\n❌ No wallets found in wallets.txt")
            input("Press Enter to continue...")
            return
        
        # Limit deep analysis to top 10 wallets
        max_deep = min(10, len(wallets))
        print(f"\n⚠️ Deep analysis will be limited to {max_deep} wallets to conserve API budget")
        
        self._run_tier_analysis(wallets[:max_deep], "deep", 30)
    
    def _run_tier_analysis(self, wallets: List[str], tier: str, days: int):
        """Helper to run specific tier analysis."""
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
            
            wallet_analyses = []
            failed_analyses = []
            
            print(f"\n🚀 Running {tier.upper()} analysis on {len(wallets)} wallets...")
            
            for i, wallet in enumerate(wallets, 1):
                print(f"Analyzing {i}/{len(wallets)}: {wallet[:8]}...{wallet[-4:]}")
                
                try:
                    analysis = wallet_analyzer.analyze_wallet_tiered(wallet, tier, days)
                    if analysis.get("success"):
                        wallet_analyses.append(analysis)
                        score = analysis.get("composite_score", 0)
                        wallet_type = analysis.get("wallet_type", "unknown")
                        print(f"  └─ Score: {score}/100, Type: {wallet_type}")
                    else:
                        failed_analyses.append({
                            "wallet_address": wallet,
                            "error": analysis.get("error", "Analysis failed")
                        })
                except Exception as e:
                    logger.error(f"Error analyzing wallet {wallet}: {str(e)}")
                    failed_analyses.append({
                        "wallet_address": wallet,
                        "error": str(e)
                    })
            
            # Package and display results
            results = self._package_analysis_results(wallet_analyses, failed_analyses)
            if results.get("success"):
                self._display_tiered_wallet_results(results)
                
                # Export
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = ensure_output_dir(f"wallet_{tier}_{timestamp}.xlsx")
                self._export_tiered_wallet_excel(results, output_file)
                print(f"\n📊 Exported to: {output_file}")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            logger.error(f"Tier analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _package_analysis_results(self, wallet_analyses: List[Dict], 
                                 failed_analyses: List[Dict]) -> Dict[str, Any]:
        """Package wallet analyses into result format."""
        # Categorize wallets
        gem_hunters = [a for a in wallet_analyses if a.get("wallet_type") == "gem_hunter"]
        smart_traders = [a for a in wallet_analyses if a.get("wallet_type") == "smart_trader"]
        diamond_hands = [a for a in wallet_analyses if a.get("wallet_type") == "diamond_hands"]
        consistent = [a for a in wallet_analyses if a.get("wallet_type") == "consistent"]
        flippers = [a for a in wallet_analyses if a.get("wallet_type") == "flipper"]
        mixed = [a for a in wallet_analyses if a.get("wallet_type") == "mixed"]
        underperformers = [a for a in wallet_analyses if a.get("wallet_type") == "underperformer"]
        unknown = [a for a in wallet_analyses if a.get("wallet_type") == "unknown"]
        
        # Sort by score
        for category in [gem_hunters, smart_traders, diamond_hands, consistent, flippers, mixed, underperformers, unknown]:
            category.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        
        return {
            "success": True,
            "total_wallets": len(wallet_analyses) + len(failed_analyses),
            "analyzed_wallets": len(wallet_analyses),
            "failed_wallets": len(failed_analyses),
            "gem_hunters": gem_hunters,
            "smart_traders": smart_traders,
            "diamond_hands": diamond_hands,
            "consistent": consistent,
            "flippers": flippers,
            "mixed": mixed,
            "underperformers": underperformers,
            "unknown": unknown,
            "failed_analyses": failed_analyses
        }
    
    def _display_tiered_wallet_results(self, results: Dict[str, Any]) -> None:
        """Display tiered wallet analysis results with advanced metrics."""
        print("\n" + "="*80)
        print("    📊 WALLET ANALYSIS RESULTS")
        print("="*80)
        
        # Summary statistics
        print(f"\n📈 SUMMARY:")
        print(f"   Total wallets: {results['total_wallets']}")
        print(f"   Successfully analyzed: {results['analyzed_wallets']}")
        print(f"   Failed: {results['failed_wallets']}")
        
        # Helper function to format score with emoji
        def format_score(score: float) -> str:
            if score >= 81:
                return f"{score:.1f}/100 🟣 EXCELLENT"
            elif score >= 61:
                return f"{score:.1f}/100 🟢 GOOD"
            elif score >= 41:
                return f"{score:.1f}/100 🟡 AVERAGE"
            elif score >= 21:
                return f"{score:.1f}/100 🟠 POOR"
            else:
                return f"{score:.1f}/100 🔴 VERY POOR"
        
        # Display top performers
        all_wallets = []
        for category in ['gem_hunters', 'smart_traders', 'diamond_hands', 'consistent', 'flippers', 'mixed', 'underperformers', 'unknown']:
            all_wallets.extend(results.get(category, []))
        
        # Sort by composite score
        all_wallets.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        if all_wallets:
            print(f"\n🏆 TOP PERFORMERS:")
            for i, analysis in enumerate(all_wallets[:10], 1):
                wallet = analysis['wallet_address']
                metrics = analysis['metrics']
                advanced = analysis.get('advanced_metrics', {})
                composite_score = analysis.get('composite_score', 0)
                
                print(f"\n{i}. Wallet: {wallet[:8]}...{wallet[-4:]}")
                print(f"   Score: {format_score(composite_score)}")
                print(f"   Type: {analysis['wallet_type']} | Tier: {analysis.get('analysis_tier', 'standard')}")
                print(f"   Win Rate: {metrics['win_rate']:.1f}% | Trades: {metrics['total_trades']}")
                print(f"   Net Profit: ${metrics['net_profit_usd']:.2f} | Max ROI: {metrics['max_roi']:.1f}%")
                
                # Show advanced metrics if available
                if advanced:
                    print(f"   📊 Advanced Metrics:")
                    print(f"      • Sharpe Ratio: {advanced.get('sharpe_ratio', 0):.2f}")
                    print(f"      • Entry Precision: {advanced.get('entry_precision_score', 0):.1f}%")
                    print(f"      • Diamond Hands: {advanced.get('diamond_hands_score', 0):.1f}%")
                    print(f"      • Risk-Adjusted Return: {advanced.get('risk_adjusted_return', 0):.1f}%")
        
        # Category breakdown with new types
        print(f"\n📂 WALLET CATEGORIES:")
        print(f"   💎 Gem Hunters (5x+): {len(results.get('gem_hunters', []))}")
        print(f"   🧠 Smart Traders: {len(results.get('smart_traders', []))}")
        print(f"   💎👐 Diamond Hands: {len(results.get('diamond_hands', []))}")
        print(f"   📊 Consistent: {len(results.get('consistent', []))}")
        print(f"   ⚡ Flippers: {len(results.get('flippers', []))}")
        print(f"   🔀 Mixed: {len(results.get('mixed', []))}")
        print(f"   📉 Underperformers: {len(results.get('underperformers', []))}")
        print(f"   ❓ Unknown: {len(results.get('unknown', []))}")
        
        # Show gem hunters specifically
        gem_hunters = results.get('gem_hunters', [])
        if gem_hunters:
            print(f"\n💎 GEM HUNTERS (NEW 5x+ CRITERIA):")
            for wallet in gem_hunters[:5]:
                score = wallet.get('composite_score', 0)
                metrics = wallet['metrics']
                roi_dist = metrics.get('roi_distribution', {})
                five_x_count = roi_dist.get('5x_to_10x', 0) + roi_dist.get('10x_plus', 0)
                
                print(f"   {wallet['wallet_address'][:8]}...")
                print(f"   └─ Score: {score:.1f} | 5x+ trades: {five_x_count} | Max ROI: {metrics['max_roi']:.0f}%")
    
    def _view_cache_stats(self):
        """View cache statistics."""
        print("\n" + "="*80)
        print("    💾 CACHE STATISTICS")
        print("="*80)
        
        try:
            from cache_manager import get_cache_manager
            cache = get_cache_manager(self.config.get("cache_size_mb", 200))
            stats = cache.get_stats()
            
            print(f"\n📊 CACHE PERFORMANCE:")
            print(f"   Total entries: {stats['total_entries']}")
            print(f"   Memory usage: {stats['total_size_mb']:.2f} MB / {stats['max_size_mb']} MB ({stats['usage_percent']:.1f}%)")
            print(f"   Hit rate: {stats['hit_rate_percent']:.1f}%")
            print(f"   Total hits: {stats['hits']}")
            print(f"   Total misses: {stats['misses']}")
            print(f"   Evictions: {stats['evictions']}")
            print(f"   API calls saved: {stats['api_calls_saved']}")
            print(f"   Estimated cost saved: ${stats['estimated_api_cost_saved']:.2f}")
            
            print(f"\n📁 CACHED DATA BY CATEGORY:")
            for category, count in stats['category_breakdown'].items():
                print(f"   {category}: {count} entries")
            
            print(f"\n🔥 Popular items (frequently accessed): {stats['popular_items_count']}")
            
        except Exception as e:
            print(f"\n❌ Error viewing cache stats: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _clear_cache(self):
        """Clear the cache."""
        print("\n" + "="*80)
        print("    🗑️ CLEAR CACHE")
        print("="*80)
        
        confirm = input("\n⚠️ Are you sure you want to clear all cached data? (y/N): ").strip().lower()
        if confirm == 'y':
            try:
                from cache_manager import get_cache_manager
                cache = get_cache_manager()
                count = cache.invalidate()  # Clear all
                print(f"\n✅ Cleared {count} cache entries")
            except Exception as e:
                print(f"\n❌ Error clearing cache: {str(e)}")
        else:
            print("\n❌ Cache clear cancelled")
        
        input("\nPress Enter to continue...")
    
    def _view_api_budget(self):
        """View API budget status."""
        print("\n" + "="*80)
        print("    💰 API BUDGET STATUS")
        print("="*80)
        
        try:
            from dual_api_manager import DualAPIManager
            from wallet_module import WalletAnalyzer
            
            # Initialize to get current usage
            api_manager = DualAPIManager(
                self.config.get("birdeye_api_key", ""),
                self.config.get("cielo_api_key"),
                self.config.get("helius_api_key")
            )
            
            if self.config.get("cielo_api_key"):
                wallet_analyzer = WalletAnalyzer(
                    cielo_api=api_manager.cielo_api,
                    birdeye_api=api_manager.birdeye_api,
                    helius_api=api_manager.helius_api
                )
                
                print(f"\n📊 API USAGE TODAY:")
                for api_name, budget in wallet_analyzer.api_budget.items():
                    used = budget["used"]
                    limit = budget["daily_limit"]
                    percent = (used / limit * 100) if limit > 0 else 0
                    
                    # Status emoji
                    if percent >= 90:
                        status = "🔴"
                    elif percent >= 70:
                        status = "🟡"
                    else:
                        status = "🟢"
                    
                    print(f"   {status} {api_name.upper()}: {used}/{limit} calls ({percent:.1f}%)")
                
                print(f"\n💡 RECOMMENDATIONS:")
                if any(b["used"] / b["daily_limit"] > 0.7 for b in wallet_analyzer.api_budget.values()):
                    print("   • Consider using Quick tier analysis to conserve API calls")
                    print("   • Enable caching to reduce redundant API calls")
                    print("   • Focus on high-value wallets for Deep analysis")
            else:
                print("\n❌ Cielo Finance API not configured")
                
        except Exception as e:
            print(f"\n❌ Error viewing API budget: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _enhanced_telegram_analysis(self):
        """Run enhanced Telegram analysis with new gem criteria."""
        print("\n" + "="*80)
        print("    🎯 ENHANCED SPYDEFI ANALYSIS")
        print("    💎 New Gem Hunter Focus: 5x+ (not 2x+)")
        print("="*80)
        
        # Check API configuration
        if not self.config.get("birdeye_api_key"):
            print("\n❌ CRITICAL: Birdeye API key required!")
            print("Please configure your Birdeye API key first (Option 1).")
            input("Press Enter to continue...")
            return
        
        if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
            print("\n❌ CRITICAL: Telegram API credentials required!")
            print("Please configure your Telegram API credentials first (Option 1).")
            input("Press Enter to continue...")
            return
        
        print("\n🚀 Starting enhanced SpyDefi analysis...")
        print("📊 New Features:")
        print("   • 5x+ gem criteria (15% threshold)")
        print("   • Max pullback % tracking")
        print("   • Time to 2x/5x analysis")
        print("   • Enhanced composite scoring")
        print("\nProcessing...")
        
        # Create args object
        class Args:
            def __init__(self):
                self.channels = ["spydefi"]
                self.days = 1
                self.hours = 24
                self.output = "spydefi_analysis_enhanced.csv"
                self.excel = True
        
        args = Args()
        
        try:
            self._handle_enhanced_telegram_analysis(args)
            print("\n✅ Enhanced analysis completed successfully!")
            print("📁 Check the outputs folder for results")
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            logger.error(f"Enhanced telegram analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _handle_enhanced_telegram_analysis(self, args) -> None:
        """Handle the enhanced telegram analysis command."""
        import asyncio
        
        try:
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
        
        output_file = ensure_output_dir(args.output)
        hours = getattr(args, 'hours', 24)
        
        logger.info(f"🚀 Starting enhanced SpyDefi analysis for the past {hours} hours.")
        logger.info(f"📁 Results will be saved to {output_file}")
        
        try:
            birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
            logger.info("✅ Birdeye API initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Birdeye API: {str(e)}")
            raise
        
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
        
        if any(ch.lower() == "spydefi" for ch in channels):
            logger.info("🎯 SpyDefi channel detected. Running enhanced analysis...")
            
            try:
                async def run_enhanced_spydefi_analysis():
                    try:
                        await telegram_scraper.connect()
                        logger.info("📞 Connected to Telegram")
                        
                        telegram_scraper.birdeye_api = birdeye_api
                        
                        analysis = await telegram_scraper.redesigned_spydefi_analysis(hours)
                        
                        logger.info("📊 Analysis completed, exporting results...")
                        
                        await telegram_scraper.export_spydefi_analysis(analysis, output_file)
                        
                        return analysis
                        
                    except Exception as e:
                        logger.error(f"❌ Error in analysis: {str(e)}")
                        raise
                    finally:
                        await telegram_scraper.disconnect()
                        logger.info("📞 Disconnected from Telegram")
                
                telegram_analyses = asyncio.run(run_enhanced_spydefi_analysis())
                
                if telegram_analyses.get('success'):
                    logger.info(f"✅ Enhanced SpyDefi analysis completed successfully!")
                else:
                    logger.error(f"❌ Analysis failed: {telegram_analyses.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"❌ Error in enhanced SpyDefi analysis: {str(e)}")
                return
        
        logger.info(f"📁 Enhanced telegram analysis completed. Results saved to {output_file}")
    
    def _show_help_advanced(self):
        """Show help with advanced features."""
        print("\n" + "="*80)
        print("    📖 HELP - Phoenix Project Optimized")
        print("="*80)
        
        print("\n🚀 NEW FEATURES:")
        print("• Tiered Analysis: Quick (2 calls), Standard (5), Deep (20)")
        print("• Smart Caching: ~40% API call reduction")
        print("• New Gem Criteria: 5x+ focus (was 2x+)")
        print("• Advanced Metrics: Sharpe ratio, Diamond Hands, Entry Precision")
        print("• Smart Sampling: Top wins/losses/recent/random")
        print("• Dust Filter: Skip trades <$100")
        
        print("\n💎 NEW GEM HUNTER CRITERIA:")
        print("• 15%+ of trades are 5x or higher (was 10% for 2x+)")
        print("• Max ROI ≥ 500% (was 200%)")
        print("• At least 2 trades with 5x+ returns")
        print("• Composite score ≥ 70")
        
        print("\n📊 ADVANCED METRICS EXPLAINED:")
        print("• Sharpe Ratio: Risk-adjusted returns (>1.5 is excellent)")
        print("• Entry Precision: % of entries near local bottom")
        print("• Diamond Hands: % held through 30%+ pullback")
        print("• Portfolio Concentration: % of trades that are 5x+")
        
        print("\n🎯 ANALYSIS TIERS:")
        print("• QUICK: Basic P&L stats only (fastest)")
        print("• STANDARD: Last 5 trades analyzed")
        print("• DEEP: Smart sampling with full metrics")
        print("• AUTO: Quick scan all, deep dive top performers")
        
        print("\n💾 CACHE BENEFITS:")
        print("• Token metadata: 24hr cache")
        print("• Wallet analysis: 6hr cache")
        print("• Price history: 1hr cache")
        print("• Popular tokens get 50% longer TTL")
        
        print("\n💡 BEST PRACTICES:")
        print("1. Start with AUTO tier for new wallets")
        print("2. Use QUICK for large batches (100+ wallets)")
        print("3. Use DEEP for your top 10 performers")
        print("4. Check cache stats regularly")
        print("5. Monitor API budget to avoid limits")
        
        print("\n📈 STRATEGY RECOMMENDATIONS:")
        print("• Gem Hunters: Follow aggressively, scale in on dips")
        print("• Smart Traders: Wait for their exact entries")
        print("• Diamond Hands: Follow and hold, wide stops")
        print("• Consistent: Safe for larger positions")
        
        input("\nPress Enter to continue...")
    
    def _export_tiered_wallet_csv(self, results: Dict[str, Any], output_file: str) -> None:
        """Export tiered wallet analysis to CSV with advanced metrics."""
        try:
            all_wallets = []
            for category in ['gem_hunters', 'smart_traders', 'diamond_hands', 'consistent', 'flippers', 'mixed', 'underperformers', 'unknown']:
                all_wallets.extend(results.get(category, []))
            
            # Sort by composite score
            all_wallets.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'rank', 'wallet_address', 'composite_score', 'score_rating',
                    'wallet_type', 'analysis_tier', 'total_trades', 'win_rate', 
                    'profit_factor', 'net_profit_usd', 'avg_roi', 'max_roi',
                    'avg_hold_time_minutes',
                    # Advanced metrics
                    'sharpe_ratio', 'entry_precision_score', 'diamond_hands_score',
                    'portfolio_concentration', 'risk_adjusted_return',
                    'max_consecutive_losses', 'conviction_score',
                    # Strategy
                    'strategy_recommendation', 'confidence', 'position_size',
                    'take_profit_1', 'take_profit_2', 'take_profit_3', 'stop_loss',
                    # API usage
                    'api_calls_used', 'data_quality_factor'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for rank, analysis in enumerate(all_wallets, 1):
                    metrics = analysis['metrics']
                    advanced = analysis.get('advanced_metrics', {})
                    strategy = analysis.get('strategy', {})
                    score = analysis.get('composite_score', 0)
                    
                    # Determine score rating
                    if score >= 81:
                        rating = "EXCELLENT"
                    elif score >= 61:
                        rating = "GOOD"
                    elif score >= 41:
                        rating = "AVERAGE"
                    elif score >= 21:
                        rating = "POOR"
                    else:
                        rating = "VERY POOR"
                    
                    row = {
                        'rank': rank,
                        'wallet_address': analysis['wallet_address'],
                        'composite_score': round(score, 1),
                        'score_rating': rating,
                        'wallet_type': analysis['wallet_type'],
                        'analysis_tier': analysis.get('analysis_tier', 'standard'),
                        'total_trades': metrics['total_trades'],
                        'win_rate': round(metrics['win_rate'], 2),
                        'profit_factor': min(999.99, metrics.get('profit_factor', 0)),
                        'net_profit_usd': round(metrics['net_profit_usd'], 2),
                        'avg_roi': round(metrics['avg_roi'], 2),
                        'max_roi': round(metrics['max_roi'], 2),
                        'avg_hold_time_minutes': round(metrics.get('avg_hold_time_hours', 0) * 60, 2),
                        # Advanced metrics
                        'sharpe_ratio': advanced.get('sharpe_ratio', 0),
                        'entry_precision_score': advanced.get('entry_precision_score', 0),
                        'diamond_hands_score': advanced.get('diamond_hands_score', 0),
                        'portfolio_concentration': advanced.get('portfolio_concentration', 0),
                        'risk_adjusted_return': advanced.get('risk_adjusted_return', 0),
                        'max_consecutive_losses': advanced.get('max_consecutive_losses', 0),
                        'conviction_score': advanced.get('conviction_score', 0),
                        # Strategy
                        'strategy_recommendation': strategy.get('recommendation', ''),
                        'confidence': strategy.get('confidence', ''),
                        'position_size': strategy.get('position_size', ''),
                        'take_profit_1': strategy.get('take_profit_1', 0),
                        'take_profit_2': strategy.get('take_profit_2', 0),
                        'take_profit_3': strategy.get('take_profit_3', 0),
                        'stop_loss': strategy.get('stop_loss', 0),
                        # API usage
                        'api_calls_used': analysis.get('api_calls_used', 0),
                        'data_quality_factor': metrics.get('data_quality_factor', 1.0)
                    }
                    
                    writer.writerow(row)
            
            logger.info(f"Exported tiered analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
    
    def _export_tiered_wallet_excel(self, results: Dict[str, Any], output_file: str) -> None:
        """Export tiered wallet analysis to Excel with advanced formatting."""
        try:
            import pandas as pd
            import xlsxwriter
            
            # Prepare data
            all_wallets = []
            for category in ['gem_hunters', 'smart_traders', 'diamond_hands', 'consistent', 'flippers', 'mixed', 'underperformers', 'unknown']:
                all_wallets.extend(results.get(category, []))
            
            # Sort by composite score
            all_wallets.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
            
            # Create Excel writer
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_data = {
                    'Metric': ['Total Wallets', 'Analyzed', 'Failed', 
                              'Gem Hunters (5x+)', 'Smart Traders', 'Diamond Hands',
                              'Consistent', 'Flippers', 'Mixed', 'Underperformers', 'Unknown'],
                    'Value': [
                        results.get('total_wallets', 0),
                        results.get('analyzed_wallets', 0),
                        results.get('failed_wallets', 0),
                        len(results.get('gem_hunters', [])),
                        len(results.get('smart_traders', [])),
                        len(results.get('diamond_hands', [])),
                        len(results.get('consistent', [])),
                        len(results.get('flippers', [])),
                        len(results.get('mixed', [])),
                        len(results.get('underperformers', [])),
                        len(results.get('unknown', []))
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Main results sheet
                data = []
                for rank, wallet in enumerate(all_wallets, 1):
                    metrics = wallet['metrics']
                    advanced = wallet.get('advanced_metrics', {})
                    composite_score = wallet.get('composite_score', 0)
                    
                    # Determine rating
                    if composite_score >= 81:
                        rating = "EXCELLENT"
                    elif composite_score >= 61:
                        rating = "GOOD"
                    elif composite_score >= 41:
                        rating = "AVERAGE"
                    elif composite_score >= 21:
                        rating = "POOR"
                    else:
                        rating = "VERY POOR"
                    
                    row = {
                        'Rank': rank,
                        'Wallet': wallet['wallet_address'],
                        'Score': composite_score,
                        'Rating': rating,
                        'Type': wallet['wallet_type'],
                        'Tier': wallet.get('analysis_tier', 'standard'),
                        'Trades': metrics['total_trades'],
                        'Win Rate %': metrics['win_rate'],
                        'Profit Factor': min(999.99, metrics.get('profit_factor', 0)),
                        'Net Profit': metrics['net_profit_usd'],
                        'Avg ROI %': metrics['avg_roi'],
                        'Max ROI %': metrics['max_roi'],
                        'Hold Time (min)': round(metrics.get('avg_hold_time_hours', 0) * 60, 2),
                        # Advanced metrics columns
                        'Sharpe Ratio': advanced.get('sharpe_ratio', 0),
                        'Entry Precision %': advanced.get('entry_precision_score', 0),
                        'Diamond Hands %': advanced.get('diamond_hands_score', 0),
                        'Portfolio Conc %': advanced.get('portfolio_concentration', 0),
                        'Risk-Adj Return %': advanced.get('risk_adjusted_return', 0),
                        'Strategy': wallet.get('strategy', {}).get('recommendation', '')
                    }
                    
                    data.append(row)
                
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name='Wallet Rankings', index=False)
                
                # Format worksheets
                workbook = writer.book
                
                # Define formats
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#1a1a2e',
                    'font_color': 'white',
                    'border': 1
                })
                
                excellent_format = workbook.add_format({'bg_color': '#e6e6fa', 'border': 1})
                good_format = workbook.add_format({'bg_color': '#90ee90', 'border': 1})
                average_format = workbook.add_format({'bg_color': '#ffffe0', 'border': 1})
                poor_format = workbook.add_format({'bg_color': '#ffdab9', 'border': 1})
                very_poor_format = workbook.add_format({'bg_color': '#ffcccb', 'border': 1})
                
                # Format main sheet
                worksheet = writer.sheets['Wallet Rankings']
                
                # Apply conditional formatting based on rating
                for row_num, row_data in enumerate(data, 1):
                    rating = row_data['Rating']
                    if rating == 'EXCELLENT':
                        worksheet.set_row(row_num, None, excellent_format)
                    elif rating == 'GOOD':
                        worksheet.set_row(row_num, None, good_format)
                    elif rating == 'AVERAGE':
                        worksheet.set_row(row_num, None, average_format)
                    elif rating == 'POOR':
                        worksheet.set_row(row_num, None, poor_format)
                    elif rating == 'VERY POOR':
                        worksheet.set_row(row_num, None, very_poor_format)
                
                # Set column widths
                worksheet.set_column('A:A', 8)   # Rank
                worksheet.set_column('B:B', 50)  # Wallet
                worksheet.set_column('C:C', 10)  # Score
                worksheet.set_column('D:D', 12)  # Rating
                worksheet.set_column('E:F', 15)  # Type, Tier
                worksheet.set_column('G:R', 15)  # Metrics
            
            logger.info(f"Exported tiered analysis to Excel: {output_file}")
            
        except ImportError:
            logger.error("pandas and xlsxwriter required for Excel export")
            print("\n⚠️ Excel export requires pandas and xlsxwriter. Using CSV fallback.")
        except Exception as e:
            logger.error(f"Error exporting Excel: {str(e)}")
    
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
            print("   📊 Required for complete entry/exit analysis")
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
        
        # Cache settings
        print(f"\n💾 Current cache size: {self.config.get('cache_size_mb', 200)} MB")
        change_cache = input("Change cache size? (y/N): ").lower().strip()
        if change_cache == 'y':
            new_size = input("Enter new cache size in MB (50-500): ").strip()
            if new_size.isdigit():
                size = int(new_size)
                if 50 <= size <= 500:
                    self.config["cache_size_mb"] = size
                    print(f"✅ Cache size set to {size} MB")
                else:
                    print("❌ Cache size must be between 50 and 500 MB")
        
        # API budgets
        print(f"\n💰 API Daily Budgets:")
        print(f"   Birdeye: {self.config.get('api_budgets', {}).get('birdeye_daily', 1000)}")
        print(f"   Cielo: {self.config.get('api_budgets', {}).get('cielo_daily', 5000)}")
        print(f"   Helius: {self.config.get('api_budgets', {}).get('helius_daily', 2000)}")
        
        change_budgets = input("Change API budgets? (y/N): ").lower().strip()
        if change_budgets == 'y':
            if "api_budgets" not in self.config:
                self.config["api_budgets"] = {}
            
            birdeye_budget = input("Birdeye daily limit (default: 1000): ").strip()
            if birdeye_budget.isdigit():
                self.config["api_budgets"]["birdeye_daily"] = int(birdeye_budget)
            
            cielo_budget = input("Cielo daily limit (default: 5000): ").strip()
            if cielo_budget.isdigit():
                self.config["api_budgets"]["cielo_daily"] = int(cielo_budget)
            
            helius_budget = input("Helius daily limit (default: 2000): ").strip()
            if helius_budget.isdigit():
                self.config["api_budgets"]["helius_daily"] = int(helius_budget)
            
            print("✅ API budgets updated")
        
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
        
        print(f"\n💾 CACHE SETTINGS:")
        print(f"   Cache size: {self.config.get('cache_size_mb', 200)} MB")
        
        print(f"\n💰 API BUDGETS (daily):")
        budgets = self.config.get('api_budgets', {})
        print(f"   Birdeye: {budgets.get('birdeye_daily', 1000)} calls")
        print(f"   Cielo: {budgets.get('cielo_daily', 5000)} calls")
        print(f"   Helius: {budgets.get('helius_daily', 2000)} calls")
        
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
        
        input("\nPress Enter to continue...")
    
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
                    print("   📊 Enhanced transaction parsing: Available")
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
        
        # Test Cache
        print("\n💾 Testing Cache System...")
        try:
            from cache_manager import get_cache_manager
            cache = get_cache_manager(self.config.get("cache_size_mb", 200))
            cache.set("test", "connectivity", {"test": True})
            if cache.get("test", "connectivity"):
                print("✅ Cache: Working properly")
                stats = cache.get_stats()
                print(f"   💾 Memory: {stats['total_size_mb']:.2f} MB / {stats['max_size_mb']} MB")
            else:
                print("❌ Cache: Not working")
        except Exception as e:
            print(f"❌ Cache: Error - {str(e)}")
        
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
                self._handle_enhanced_telegram_analysis(args)
            elif args.command == "wallet":
                self._handle_wallet_analysis(args)
            elif args.command == "cache":
                self._handle_cache_command(args)
    
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
        
        if args.cache_size:
            self.config["cache_size_mb"] = args.cache_size
            logger.info(f"Cache size set to {args.cache_size} MB.")
        
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
            
            # Run analysis based on tier
            if args.tier == "auto":
                results = wallet_analyzer.batch_analyze_wallets_tiered(
                    wallets,
                    days_back=args.days,
                    default_tier="standard",
                    top_performers_deep=True
                )
            else:
                # Manual tier
                results = {"success": True, "analyzed_wallets": [], "failed_wallets": []}
                for wallet in wallets:
                    analysis = wallet_analyzer.analyze_wallet_tiered(wallet, args.tier, args.days)
                    if analysis.get("success"):
                        results["analyzed_wallets"].append(analysis)
                    else:
                        results["failed_wallets"].append({
                            "wallet_address": wallet,
                            "error": analysis.get("error", "Analysis failed")
                        })
                
                results = self._package_analysis_results(results["analyzed_wallets"], results["failed_wallets"])
            
            if results.get("success"):
                # Export results
                output_file = ensure_output_dir(args.output)
                
                # Excel export
                if args.output.endswith('.xlsx'):
                    self._export_tiered_wallet_excel(results, output_file)
                else:
                    # If output doesn't end with .xlsx, add it
                    excel_file = output_file.replace('.csv', '.xlsx')
                    if not excel_file.endswith('.xlsx'):
                        excel_file += '.xlsx'
                    self._export_tiered_wallet_excel(results, excel_file)
                    output_file = excel_file
                
                # Also export CSV
                csv_file = output_file.replace('.xlsx', '.csv')
                self._export_tiered_wallet_csv(results, csv_file)
                
                logger.info(f"Analysis complete. Results saved to {output_file} and {csv_file}")
            else:
                logger.error(f"Analysis failed: {results.get('error')}")
                
        except Exception as e:
            logger.error(f"Error during wallet analysis: {str(e)}")
    
    def _handle_cache_command(self, args: argparse.Namespace) -> None:
        """Handle cache management commands."""
        try:
            from cache_manager import get_cache_manager
            cache = get_cache_manager(self.config.get("cache_size_mb", 200))
            
            if args.stats:
                stats = cache.get_stats()
                print(f"\n💾 CACHE STATISTICS:")
                print(f"   Total entries: {stats['total_entries']}")
                print(f"   Memory usage: {stats['total_size_mb']:.2f} MB / {stats['max_size_mb']} MB")
                print(f"   Hit rate: {stats['hit_rate_percent']:.1f}%")
                print(f"   API calls saved: {stats['api_calls_saved']}")
                print(f"   Estimated cost saved: ${stats['estimated_api_cost_saved']:.2f}")
            
            elif args.clear:
                count = cache.invalidate()
                print(f"\n✅ Cleared {count} cache entries")
                
        except Exception as e:
            logger.error(f"Error with cache command: {str(e)}")

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