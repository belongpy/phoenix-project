#!/usr/bin/env python3
"""
Phoenix Project - UPDATED Enhanced CLI Tool

🎯 UPDATED CHANGES:
- Removed contested wallet analysis options
- Removed min win rate input (hardcoded to 30%)
- Always export to Excel (no need to ask)
- Simplified wallet analysis flow
- Updated help text to reflect new thresholds
- Fixed profit factor display (capped at 999.99)
- Changed hold time display to minutes instead of hours
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
        "solana_rpc_url": "https://api.mainnet-beta.solana.com",
        "sources": {
            "telegram_groups": ["spydefi"],
            "wallets": []
        },
        "analysis_period_days": 1
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
    """UPDATED Phoenix CLI with simplified wallet analysis."""
    
    def __init__(self):
        self.config = load_config()
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser."""
        parser = argparse.ArgumentParser(
            description="Phoenix Project - UPDATED Enhanced Solana Chain Analysis CLI Tool",
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
        
        # Enhanced telegram analysis command
        telegram_parser = subparsers.add_parser("telegram", help="FIXED Enhanced SpyDefi analysis")
        telegram_parser.add_argument("--hours", type=int, default=24, help="Hours to analyze (default: 24)")
        telegram_parser.add_argument("--output", default="spydefi_analysis_enhanced.csv", help="Output CSV file")
        telegram_parser.add_argument("--excel", action="store_true", help="Also export to Excel format")
        
        # Wallet analysis command
        wallet_parser = subparsers.add_parser("wallet", help="Analyze wallets for copy trading")
        wallet_parser.add_argument("--wallets-file", default="wallets.txt", help="File containing wallet addresses")
        wallet_parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
        wallet_parser.add_argument("--output", default="wallet_analysis.xlsx", help="Output file")
        
        return parser
    
    def _handle_numbered_menu(self):
        """Handle the numbered menu interface."""
        print("\n" + "="*80)
        print("Phoenix Project - UPDATED Enhanced Solana Chain Analysis Tool")
        print("🎯 FIXED: Pullback % & Time-to-2x Analysis + Enhanced Contract Detection")
        print("="*80)
        print("\nSelect an option:")
        print("\n🔧 CONFIGURATION:")
        print("1. Configure API Keys")
        print("2. Check Configuration")
        print("3. Test API Connectivity")
        print("\n📊 TOOLS:")
        print("4. SPYDEFI")
        print("5. ANALYZE WALLETS")
        print("\n🔍 UTILITIES:")
        print("6. View Current Sources")
        print("7. Help & Examples")
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
                self._fixed_enhanced_telegram_analysis()
            elif choice == '5':
                self._auto_wallet_analysis()
            elif choice == '6':
                self._view_current_sources()
            elif choice == '7':
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
    
    def _auto_wallet_analysis(self):
        """Run comprehensive wallet analysis with composite scoring."""
        print("\n" + "="*80)
        print("    💰 COMPREHENSIVE WALLET ANALYSIS")
        print("    📊 Composite Scoring System (0-100)")
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
        
        # Days to analyze
        days_input = input("Days to analyze (default: 30): ").strip()
        days_to_analyze = int(days_input) if days_input.isdigit() else 30
        
        print(f"\n🚀 Starting wallet analysis...")
        print(f"📊 Parameters:")
        print(f"   • Wallets: {len(wallets)}")
        print(f"   • Analysis period: {days_to_analyze} days")
        print(f"   • Min win rate: 30% (optimized for memecoins)")
        print(f"   • Entry/Exit Analysis: Last 5 trades")
        print(f"   • Export format: Excel + CSV")
        print("\nProcessing...")
        
        try:
            # Initialize APIs
            from dual_api_manager import DualAPIManager
            from wallet_module import WalletAnalyzer
            
            api_manager = DualAPIManager(
                self.config.get("birdeye_api_key", ""),
                self.config.get("cielo_api_key")
            )
            
            # Create wallet analyzer
            wallet_analyzer = WalletAnalyzer(
                cielo_api=api_manager.cielo_api,
                birdeye_api=api_manager.birdeye_api,
                rpc_url=self.config.get("solana_rpc_url", "https://api.mainnet-beta.solana.com")
            )
            
            # Run batch analysis (removed min_winrate and include_contested parameters)
            results = wallet_analyzer.batch_analyze_wallets(
                wallets,
                days_back=days_to_analyze,
                use_hybrid=True  # Always use hybrid approach
            )
            
            if results.get("success"):
                self._display_wallet_analysis_results(results)
                
                # Always export to both Excel and CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_base = f"wallet_analysis_{timestamp}"
                
                # Excel export
                excel_file = ensure_output_dir(f"{output_base}.xlsx")
                self._export_wallet_analysis_excel(results, excel_file)
                print(f"\n📊 Exported to Excel: {excel_file}")
                
                # CSV export
                csv_file = ensure_output_dir(f"{output_base}.csv")
                self._export_wallet_analysis_csv(results, csv_file)
                print(f"📄 Exported to CSV: {csv_file}")
                
                print("\n✅ Wallet analysis completed successfully!")
            else:
                print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"\n❌ Error during wallet analysis: {str(e)}")
            logger.error(f"Wallet analysis error: {str(e)}")
        
        input("\nPress Enter to continue...")
    
    def _display_wallet_analysis_results(self, results: Dict[str, Any]) -> None:
        """Display wallet analysis results with composite scores."""
        print("\n" + "="*80)
        print("    📊 WALLET ANALYSIS RESULTS")
        print("="*80)
        
        # Summary statistics
        print(f"\n📈 SUMMARY:")
        print(f"   Total wallets: {results['total_wallets']}")
        print(f"   Successfully analyzed: {results['analyzed_wallets']}")
        print(f"   Failed: {results['failed_wallets']}")
        print(f"   Total shown: {results['filtered_wallets']} (all wallets displayed)")
        
        # Helper function to format composite score with emoji
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
        
        # Display top performers by composite score
        all_wallets = []
        for category in ['gem_finders', 'consistent', 'flippers', 'mixed', 'underperformers', 'unknown']:
            all_wallets.extend(results.get(category, []))
        
        # Sort by composite score
        all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
        
        if all_wallets:
            print(f"\n🏆 TOP PERFORMERS BY COMPOSITE SCORE:")
            for i, analysis in enumerate(all_wallets[:10], 1):
                wallet = analysis['wallet_address']
                metrics = analysis['metrics']
                composite_score = analysis.get('composite_score', metrics.get('composite_score', 0))
                
                # Cap profit factor for display
                profit_factor = metrics['profit_factor']
                if profit_factor > 999.99:
                    profit_factor_display = "999.99x"
                else:
                    profit_factor_display = f"{profit_factor:.2f}x"
                
                print(f"\n{i}. Wallet: {wallet[:8]}...{wallet[-4:]}")
                print(f"   Score: {format_score(composite_score)}")
                print(f"   Type: {analysis['wallet_type']} | Win Rate: {metrics['win_rate']:.1f}%")
                print(f"   Profit Factor: {profit_factor_display} | Total Trades: {metrics['total_trades']}")
                print(f"   Net Profit: ${metrics['net_profit_usd']:.2f} | Avg ROI: {metrics['avg_roi']:.1f}%")
                
                # Entry/Exit Analysis
                if 'entry_exit_analysis' in analysis:
                    ee_analysis = analysis['entry_exit_analysis']
                    print(f"   Entry/Exit: {ee_analysis['pattern']} | Missed Gains: {ee_analysis['missed_gains_percent']:.1f}%")
        
        # Category breakdown
        print(f"\n📂 WALLET CATEGORIES:")
        print(f"   🎯 Gem Finders: {len(results.get('gem_finders', []))}")
        print(f"   📊 Consistent Traders: {len(results.get('consistent', []))}")
        print(f"   ⚡ Quick Flippers: {len(results.get('flippers', []))}")
        print(f"   🔀 Mixed Results: {len(results.get('mixed', []))}")
        print(f"   📉 Underperformers: {len(results.get('underperformers', []))}")
        print(f"   ❓ Unknown/Low Activity: {len(results.get('unknown', []))}")
        
        # Top in each category
        for category_name, category_key in [
            ("GEM FINDERS", "gem_finders"),
            ("CONSISTENT TRADERS", "consistent"),
            ("QUICK FLIPPERS", "flippers"),
            ("MIXED RESULTS", "mixed")
        ]:
            category_wallets = results.get(category_key, [])
            if category_wallets:
                print(f"\n🏅 TOP {category_name}:")
                for wallet in category_wallets[:3]:
                    score = wallet.get('composite_score', wallet['metrics'].get('composite_score', 0))
                    print(f"   {wallet['wallet_address'][:8]}... | {format_score(score)}")
    
    def _export_wallet_analysis_csv(self, results: Dict[str, Any], output_file: str) -> None:
        """Export wallet analysis results to CSV with hold time in minutes."""
        try:
            all_wallets = []
            for category in ['gem_finders', 'consistent', 'flippers', 'mixed', 'underperformers', 'unknown']:
                all_wallets.extend(results.get(category, []))
            
            # Sort by composite score
            all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'rank', 'wallet_address', 'composite_score', 'score_rating',
                    'wallet_type', 'total_trades', 'win_rate', 'profit_factor',
                    'net_profit_usd', 'avg_roi', 'median_roi', 'max_roi',
                    'avg_hold_time_minutes',  # Changed from hours to minutes
                    'total_tokens_traded',
                    'entry_exit_pattern', 'entry_quality', 'exit_quality',
                    'missed_gains_percent', 'early_exit_rate',
                    'strategy_recommendation', 'confidence'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for rank, analysis in enumerate(all_wallets, 1):
                    metrics = analysis['metrics']
                    score = analysis.get('composite_score', metrics.get('composite_score', 0))
                    
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
                    
                    # Convert hold time from hours to minutes
                    hold_time_minutes = round(metrics.get('avg_hold_time_hours', 0) * 60, 2)
                    
                    # Cap profit factor at 999.99 for display
                    profit_factor = metrics.get('profit_factor', 0)
                    if profit_factor > 999.99:
                        profit_factor = 999.99
                    
                    row = {
                        'rank': rank,
                        'wallet_address': analysis['wallet_address'],
                        'composite_score': round(score, 1),
                        'score_rating': rating,
                        'wallet_type': analysis['wallet_type'],
                        'total_trades': metrics['total_trades'],
                        'win_rate': round(metrics['win_rate'], 2),
                        'profit_factor': profit_factor,
                        'net_profit_usd': round(metrics['net_profit_usd'], 2),
                        'avg_roi': round(metrics['avg_roi'], 2),
                        'median_roi': round(metrics['median_roi'], 2),
                        'max_roi': round(metrics['max_roi'], 2),
                        'avg_hold_time_minutes': hold_time_minutes,
                        'total_tokens_traded': metrics['total_tokens_traded'],
                        'strategy_recommendation': analysis['strategy']['recommendation'],
                        'confidence': analysis['strategy'].get('confidence', 'LOW')
                    }
                    
                    # Add entry/exit analysis if available
                    if 'entry_exit_analysis' in analysis:
                        ee_analysis = analysis['entry_exit_analysis']
                        row['entry_exit_pattern'] = ee_analysis.get('pattern', '')
                        row['entry_quality'] = ee_analysis.get('entry_quality', '')
                        row['exit_quality'] = ee_analysis.get('exit_quality', '')
                        row['missed_gains_percent'] = ee_analysis.get('missed_gains_percent', 0)
                        row['early_exit_rate'] = ee_analysis.get('early_exit_rate', 0)
                    else:
                        row['entry_exit_pattern'] = ''
                        row['entry_quality'] = ''
                        row['exit_quality'] = ''
                        row['missed_gains_percent'] = 0
                        row['early_exit_rate'] = 0
                    
                    writer.writerow(row)
            
            logger.info(f"Exported wallet analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {str(e)}")
    
    def _export_wallet_analysis_excel(self, results: Dict[str, Any], output_file: str) -> None:
        """Export wallet analysis results to Excel with hold time in minutes."""
        try:
            import pandas as pd
            import xlsxwriter
            
            # Prepare data
            all_wallets = []
            for category in ['gem_finders', 'consistent', 'flippers', 'mixed', 'underperformers', 'unknown']:
                all_wallets.extend(results.get(category, []))
            
            # Sort by composite score
            all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
            
            # Create DataFrame
            data = []
            for rank, analysis in enumerate(all_wallets, 1):
                metrics = analysis['metrics']
                score = analysis.get('composite_score', metrics.get('composite_score', 0))
                
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
                
                # Convert hold time from hours to minutes
                hold_time_minutes = round(metrics.get('avg_hold_time_hours', 0) * 60, 2)
                
                # Cap profit factor at 999.99 for display
                profit_factor = metrics.get('profit_factor', 0)
                if profit_factor > 999.99:
                    profit_factor = 999.99
                
                row = {
                    'Rank': rank,
                    'Wallet Address': analysis['wallet_address'],
                    'Composite Score': score,
                    'Rating': rating,
                    'Type': analysis['wallet_type'],
                    'Total Trades': metrics['total_trades'],
                    'Win Rate %': metrics['win_rate'],
                    'Profit Factor': profit_factor,
                    'Net Profit USD': metrics['net_profit_usd'],
                    'Avg ROI %': metrics['avg_roi'],
                    'Max ROI %': metrics['max_roi'],
                    'Avg Hold Time (Minutes)': hold_time_minutes,  # Changed to minutes
                    'Strategy': analysis['strategy']['recommendation'],
                    'Confidence': analysis['strategy'].get('confidence', 'LOW')
                }
                
                # Add entry/exit analysis if available
                if 'entry_exit_analysis' in analysis:
                    ee_analysis = analysis['entry_exit_analysis']
                    row['Entry/Exit Pattern'] = ee_analysis.get('pattern', '')
                    row['Entry Quality'] = ee_analysis.get('entry_quality', '')
                    row['Exit Quality'] = ee_analysis.get('exit_quality', '')
                    row['Missed Gains %'] = ee_analysis.get('missed_gains_percent', 0)
                else:
                    row['Entry/Exit Pattern'] = ''
                    row['Entry Quality'] = ''
                    row['Exit Quality'] = ''
                    row['Missed Gains %'] = 0
                
                data.append(row)
            
            # Create Excel writer
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                # Summary sheet
                summary_data = {
                    'Metric': ['Total Wallets', 'Analyzed', 'Failed', 'Gem Finders', 
                              'Consistent', 'Flippers', 'Mixed', 'Underperformers', 'Unknown'],
                    'Value': [
                        results['total_wallets'],
                        results['analyzed_wallets'],
                        results['failed_wallets'],
                        len(results.get('gem_finders', [])),
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
                
                excellent_format = workbook.add_format({
                    'bg_color': '#e6e6fa',  # Light purple
                    'border': 1
                })
                
                good_format = workbook.add_format({
                    'bg_color': '#90ee90',  # Light green
                    'border': 1
                })
                
                average_format = workbook.add_format({
                    'bg_color': '#ffffe0',  # Light yellow
                    'border': 1
                })
                
                poor_format = workbook.add_format({
                    'bg_color': '#ffdab9',  # Light orange
                    'border': 1
                })
                
                very_poor_format = workbook.add_format({
                    'bg_color': '#ffcccb',  # Light red
                    'border': 1
                })
                
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
                worksheet.set_column('B:B', 50)  # Wallet Address
                worksheet.set_column('C:C', 15)  # Composite Score
                worksheet.set_column('D:D', 12)  # Rating
                worksheet.set_column('E:E', 15)  # Type
                worksheet.set_column('F:K', 15)  # Metrics
                worksheet.set_column('L:L', 20)  # Avg Hold Time (Minutes)
                worksheet.set_column('M:M', 20)  # Strategy
                worksheet.set_column('N:N', 12)  # Confidence
                worksheet.set_column('O:R', 15)  # Entry/Exit analysis
            
            logger.info(f"Exported wallet analysis to Excel: {output_file}")
            
        except ImportError:
            logger.error("pandas and xlsxwriter required for Excel export. Install with: pip install pandas xlsxwriter")
            print("\n⚠️ Excel export requires pandas and xlsxwriter. Using CSV fallback.")
        except Exception as e:
            logger.error(f"Error exporting Excel: {str(e)}")
    
    def _fixed_enhanced_telegram_analysis(self):
        """Run FIXED enhanced Telegram analysis with proper pullback % and time-to-2x metrics."""
        print("\n" + "="*80)
        print("    🎯 FIXED ENHANCED SPYDEFI TELEGRAM ANALYSIS")
        print("    📉 Max Pullback % + ⏱️ Time to 2x Analysis (WORKING)")
        print("="*80)
        
        # Check API configuration first
        if not self.config.get("birdeye_api_key"):
            print("\n❌ CRITICAL: Birdeye API key required for enhanced analysis!")
            print("The enhanced features WILL NOT WORK without this API key.")
            print("Please configure your Birdeye API key first (Option 1).")
            input("Press Enter to continue...")
            return
        
        if not self.config.get("telegram_api_id") or not self.config.get("telegram_api_hash"):
            print("\n❌ CRITICAL: Telegram API credentials required!")
            print("Please configure your Telegram API credentials first (Option 1).")
            input("Press Enter to continue...")
            return
        
        print("\n🚀 Starting FIXED enhanced SpyDefi analysis...")
        print("📅 Analysis period: 24 hours")
        print("📁 Output: spydefi_analysis_enhanced.csv")
        print("📊 Excel export: Enabled")
        print("🎯 FIXED enhanced features:")
        print("   • ✅ Max pullback % for stop loss calculation (WORKING)")
        print("   • ✅ Average time to reach 2x for holding strategy (WORKING)")
        print("   • ✅ Enhanced contract address detection (IMPROVED)")
        print("   • ✅ Expanded token name to contract lookup (200+ tokens)")
        print("   • ✅ Detailed price analysis using Birdeye API (FIXED)")
        print("   • ✅ Proper CSV export with all enhanced metrics (FIXED)")
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
            self._handle_fixed_enhanced_telegram_analysis(args)
            print("\n✅ FIXED enhanced analysis completed successfully!")
            print("📁 Check the outputs folder for results:")
            print("   • spydefi_analysis_enhanced.csv - Token call details with enhanced metrics")
            print("   • spydefi_analysis_enhanced_kol_performance_enhanced.csv - KOL metrics with pullback data")
            print("   • spydefi_analysis_enhanced_enhanced_summary.txt - Analysis summary")
            print("   • spydefi_analysis_enhanced_enhanced.xlsx - Excel workbook")
            print("\n🎯 FIXED metrics now available:")
            print("   • avg_max_pullback_percent - Use this + buffer for stop loss")
            print("   • avg_time_to_2x_formatted - Average time to reach 2x (e.g., '2h 15m 30s')")
            print("   • detailed_analysis_count - Tokens with full price data")
            print("   • pullback_data_available - TRUE/FALSE indicator")
            print("   • time_to_2x_data_available - TRUE/FALSE indicator")
            print("\n📊 Expected coverage: 20-50% of tokens (depending on contract address availability)")
            
        except Exception as e:
            print(f"\n❌ Enhanced analysis failed: {str(e)}")
            logger.error(f"Fixed enhanced telegram analysis error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
        
        input("\nPress Enter to continue...")
    
    def _handle_fixed_enhanced_telegram_analysis(self, args) -> None:
        """Handle the FIXED enhanced telegram analysis command."""
        import asyncio
        
        # Import the FIXED telegram module
        try:
            # Make sure we're importing the fixed version
            import importlib
            import sys
            
            # Remove any cached telegram_module
            if 'telegram_module' in sys.modules:
                del sys.modules['telegram_module']
            
            from telegram_module import TelegramScraper
            from birdeye_api import BirdeyeAPI
            
            logger.info("✅ Imported FIXED telegram module")
            
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
        
        logger.info(f"🚀 Starting FIXED enhanced SpyDefi analysis for the past {hours} hours.")
        logger.info(f"📁 Results will be saved to {output_file}")
        logger.info(f"🎯 Enhanced features: FIXED Pullback % + Time-to-2x analysis")
        
        # Initialize Birdeye API
        try:
            birdeye_api = BirdeyeAPI(self.config["birdeye_api_key"])
            logger.info("✅ Birdeye API initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Birdeye API: {str(e)}")
            raise
        
        # Initialize Telegram scraper
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
            logger.info("🎯 SpyDefi channel detected. Running FIXED enhanced analysis...")
            
            try:
                async def run_fixed_enhanced_spydefi_analysis():
                    try:
                        await telegram_scraper.connect()
                        logger.info("📞 Connected to Telegram")
                        
                        # Set the birdeye API instance to the scraper
                        telegram_scraper.birdeye_api = birdeye_api
                        
                        # Use the redesigned analysis method
                        analysis = await telegram_scraper.redesigned_spydefi_analysis(hours)
                        
                        logger.info("📊 Analysis completed, exporting results...")
                        
                        # FIXED: Use the correct export method name
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
                
                telegram_analyses = asyncio.run(run_fixed_enhanced_spydefi_analysis())
                
                # Check results
                if telegram_analyses.get('success'):
                    enhanced_count = sum(kol.get('detailed_analysis_count', 0) for kol in telegram_analyses.get('ranked_kols', {}).values())
                    total_count = telegram_analyses.get('total_calls', 0)
                    
                    if enhanced_count > 0:
                        logger.info(f"✅ FIXED enhanced SpyDefi analysis completed successfully!")
                        logger.info(f"🎯 Enhanced analysis coverage: {enhanced_count}/{total_count} tokens ({(enhanced_count/total_count*100):.1f}%)")
                        logger.info(f"📉 Pullback data calculated for stop loss optimization")
                        logger.info(f"⏱️ Time-to-2x data calculated for holding strategy")
                        
                        # Log some sample enhanced metrics
                        if telegram_analyses.get('ranked_kols'):
                            logger.info("📊 Sample enhanced KOL metrics:")
                            count = 0
                            for kol, perf in telegram_analyses['ranked_kols'].items():
                                if count >= 3:
                                    break
                                enhanced_count_kol = perf.get('detailed_analysis_count', 0)
                                pullback = perf.get('avg_max_pullback_percent', 0)
                                time_to_2x = perf.get('avg_time_to_2x_formatted', 'N/A')
                                logger.info(f"   @{kol}: {enhanced_count_kol} enhanced, avg pullback: {pullback}%, time to 2x: {time_to_2x}")
                                count += 1
                    else:
                        logger.warning(f"⚠️ Enhanced analysis coverage: 0/{total_count} tokens")
                        logger.warning(f"   Possible reasons:")
                        logger.warning(f"   • SpyDefi messages contain token names instead of contract addresses")
                        logger.warning(f"   • Token names not in our expanded lookup table (200+ tokens)")
                        logger.warning(f"   • Birdeye API rate limiting or connectivity issues")
                        logger.warning(f"   • Limited price history availability for recent tokens")
                else:
                    logger.error(f"❌ Analysis failed: {telegram_analyses.get('error', 'Unknown error')}")
                
            except Exception as e:
                logger.error(f"❌ Error in FIXED enhanced SpyDefi analysis: {str(e)}")
                import traceback
                logger.error(f"❌ Full error traceback: {traceback.format_exc()}")
                return
        
        logger.info(f"📁 FIXED enhanced telegram analysis completed. Results saved to {output_file}")
        
        # Enhanced Excel export
        if hasattr(args, 'excel') and args.excel:
            try:
                from export_utils import export_to_excel
                excel_file = output_file.replace(".csv", "_enhanced.xlsx")
                
                # Convert analysis format for Excel export
                if isinstance(telegram_analyses, dict) and telegram_analyses.get('ranked_kols'):
                    enhanced_telegram_data = {"ranked_kols": []}
                    
                    for kol, performance in telegram_analyses['ranked_kols'].items():
                        enhanced_kol_data = {
                            "channel_id": performance.get('channel_id', ''),
                            "total_calls": performance.get('tokens_mentioned', 0),
                            "success_rate": performance.get('success_rate_2x', 0),
                            "avg_roi": performance.get('avg_ath_roi', 0),
                            "avg_max_roi": performance.get('avg_ath_roi', 0),
                            "confidence_level": performance.get('composite_score', 0),
                            # Enhanced metrics for Excel
                            "avg_max_pullback_percent": performance.get('avg_max_pullback_percent', 0),
                            "avg_time_to_2x_formatted": performance.get('avg_time_to_2x_formatted', 'N/A'),
                            "detailed_analysis_count": performance.get('detailed_analysis_count', 0),
                            "strategy": {
                                "recommendation": "ENHANCED_ANALYSIS",
                                "entry_type": "IMMEDIATE",
                                "take_profit_1": 50,
                                "take_profit_2": 100,
                                "take_profit_3": 200,
                                "stop_loss": -(performance.get('avg_max_pullback_percent', 25) + 10)
                            }
                        }
                        enhanced_telegram_data["ranked_kols"].append(enhanced_kol_data)
                    
                    export_to_excel(enhanced_telegram_data, {}, excel_file)
                    logger.info(f"📊 FIXED enhanced Excel export completed: {excel_file}")
                else:
                    logger.warning("⚠️ Excel export skipped - analysis format issue")
                    
            except Exception as e:
                logger.error(f"❌ Error in Excel export: {str(e)}")
    
    def _test_api_connectivity(self):
        """Test API connectivity."""
        print("\n" + "="*70)
        print("    🔍 API CONNECTIVITY TEST (FIXED)")
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
                    print("   🎯 FIXED enhanced telegram analysis: Available")
                    print("   📉 Pullback % calculation: Ready")
                    print("   ⏱️ Time-to-2x analysis: Ready")
                else:
                    print("❌ Birdeye API: Connection failed")
                    print("   ⚠️ Enhanced features will not work")
            except Exception as e:
                print(f"❌ Birdeye API: Error - {str(e)}")
                print("   ⚠️ Enhanced features will not work")
        else:
            print("❌ Birdeye API: Not configured")
            print("   ⚠️ CRITICAL: Enhanced analysis requires Birdeye API key")
        
        # Test Cielo Finance API
        if self.config.get("cielo_api_key"):
            print("\n💰 Testing Cielo Finance API...")
            try:
                from cielo_api import CieloFinanceAPI
                cielo_api = CieloFinanceAPI(self.config["cielo_api_key"])
                if cielo_api.health_check():
                    print("✅ Cielo Finance API: Connected successfully")
                    print("   💰 Wallet analysis: Available")
                    print("   📊 Trading statistics: Ready")
                else:
                    print("❌ Cielo Finance API: Connection failed")
                    print("   ⚠️ Wallet analysis will not work")
            except Exception as e:
                print(f"❌ Cielo Finance API: Error - {str(e)}")
                print("   ⚠️ Wallet analysis will not work")
        else:
            print("❌ Cielo Finance API: Not configured")
            print("   ⚠️ CRITICAL: Wallet analysis requires Cielo Finance API key")
        
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
            print("   ⚠️ SpyDefi analysis: Not available")
        
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
        print(f"\n📊 FEATURE AVAILABILITY SUMMARY:")
        birdeye_ok = bool(self.config.get("birdeye_api_key"))
        telegram_ok = bool(self.config.get("telegram_api_id") and self.config.get("telegram_api_hash"))
        cielo_ok = bool(self.config.get("cielo_api_key"))
        
        print(f"   🎯 Enhanced Telegram Analysis: {'✅ Ready' if (birdeye_ok and telegram_ok) else '❌ Missing APIs'}")
        print(f"   💰 Wallet Analysis: {'✅ Ready' if cielo_ok else '❌ Need Cielo Finance API'}")
        print(f"   📉 Pullback % Calculation: {'✅ Ready' if birdeye_ok else '❌ Need Birdeye API'}")
        print(f"   ⏱️ Time-to-2x Analysis: {'✅ Ready' if birdeye_ok else '❌ Need Birdeye API'}")
        print(f"   🔍 Enhanced Contract Detection: {'✅ Ready' if telegram_ok else '❌ Need Telegram API'}")
        print(f"   📊 Composite Scoring: {'✅ Ready' if cielo_ok else '❌ Need Cielo Finance API'}")
        
        if birdeye_ok and telegram_ok and cielo_ok:
            print(f"\n🎉 ALL SYSTEMS GO! Full functionality is available.")
        else:
            print(f"\n⚠️ Configure missing APIs to enable all features.")
        
        input("\nPress Enter to continue...")
    
    def _interactive_configure(self):
        """Interactive configuration setup."""
        print("\n" + "="*70)
        print("    🔧 CONFIGURATION SETUP (FIXED)")
        print("="*70)
        
        # Birdeye API Key (CRITICAL for enhanced features)
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
            print("\n🔑 Birdeye API Key (CRITICAL for enhanced analysis)")
            print("   🎯 Required for pullback % and time-to-2x calculations")
            print("   📊 Get your key from: https://birdeye.so")
            new_key = input("Enter Birdeye API key: ").strip()
            if new_key:
                self.config["birdeye_api_key"] = new_key
                print("✅ Birdeye API key configured")
                print("   🎯 Enhanced telegram analysis: Now available")
            else:
                print("⚠️ CRITICAL: Enhanced analysis will NOT work without Birdeye API key")
        
        # Cielo Finance API Key (CRITICAL for wallet analysis)
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
            print("\n💰 Cielo Finance API Key (CRITICAL for wallet analysis)")
            print("   📊 Required for wallet trading statistics")
            print("   🔑 Get your key from: https://cielo.finance")
            new_key = input("Enter Cielo Finance API key: ").strip()
            if new_key:
                self.config["cielo_api_key"] = new_key
                print("✅ Cielo Finance API key configured")
                print("   💰 Wallet analysis: Now available")
            else:
                print("⚠️ CRITICAL: Wallet analysis will NOT work without Cielo Finance API key")
        
        # Telegram API credentials (CRITICAL for telegram analysis)
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
            print("\n📱 Telegram API Credentials (CRITICAL for SpyDefi analysis)")
            print("   📊 Required for accessing SpyDefi channel")
            print("   🔑 Get credentials from: https://my.telegram.org")
            new_id = input("Enter Telegram API ID: ").strip()
            new_hash = input("Enter Telegram API Hash: ").strip()
            if new_id and new_hash:
                self.config["telegram_api_id"] = new_id
                self.config["telegram_api_hash"] = new_hash
                print("✅ Telegram API credentials configured")
                print("   📊 SpyDefi analysis: Now available")
            else:
                print("⚠️ CRITICAL: SpyDefi analysis will NOT work without Telegram API credentials")
        
        # Save configuration
        save_config(self.config)
        print("\n✅ Configuration saved successfully!")
        
        # Show feature availability
        birdeye_ok = bool(self.config.get("birdeye_api_key"))
        telegram_ok = bool(self.config.get("telegram_api_id") and self.config.get("telegram_api_hash"))
        cielo_ok = bool(self.config.get("cielo_api_key"))
        
        print(f"\n🎯 ENHANCED FEATURES STATUS:")
        print(f"   Enhanced Telegram Analysis: {'✅ Ready' if (birdeye_ok and telegram_ok) else '❌ Missing APIs'}")
        print(f"   Wallet Analysis: {'✅ Ready' if cielo_ok else '❌ Need Cielo Finance API'}")
        print(f"   Pullback % Calculation: {'✅ Ready' if birdeye_ok else '❌ Need Birdeye API'}")
        print(f"   Time-to-2x Analysis: {'✅ Ready' if birdeye_ok else '❌ Need Birdeye API'}")
        print(f"   Enhanced Contract Detection: {'✅ Ready' if telegram_ok else '❌ Need Telegram API'}")
        print(f"   Composite Scoring: {'✅ Ready' if cielo_ok else '❌ Need Cielo Finance API'}")
        
        if birdeye_ok and telegram_ok and cielo_ok:
            print(f"\n🎉 ALL SYSTEMS READY! You can now use all features.")
        else:
            print(f"\n⚠️ Configure the missing APIs above to enable all features.")
        
        input("Press Enter to continue...")
    
    def _check_configuration(self):
        """Check current configuration."""
        print("\n" + "="*70)
        print("    📋 CURRENT CONFIGURATION (FIXED)")
        print("="*70)
        
        print(f"\n🔑 API KEYS:")
        print(f"   Birdeye API Key: {'✅ Configured' if self.config.get('birdeye_api_key') else '❌ Not configured'}")
        print(f"   Cielo Finance API Key: {'✅ Configured' if self.config.get('cielo_api_key') else '❌ Not configured'}")
        print(f"   Telegram API ID: {'✅ Configured' if self.config.get('telegram_api_id') else '❌ Not configured'}")
        print(f"   Telegram API Hash: {'✅ Configured' if self.config.get('telegram_api_hash') else '❌ Not configured'}")
        
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
        
        # Enhanced features availability
        birdeye_ok = bool(self.config.get("birdeye_api_key"))
        telegram_ok = bool(self.config.get("telegram_api_id") and self.config.get("telegram_api_hash"))
        cielo_ok = bool(self.config.get("cielo_api_key"))
        
        print(f"\n🎯 ENHANCED FEATURES AVAILABILITY:")
        print(f"   Enhanced Telegram Analysis: {'✅ Available' if (birdeye_ok and telegram_ok) else '❌ Not Available'}")
        print(f"   Wallet Analysis: {'✅ Available' if cielo_ok else '❌ Not Available'}")
        print(f"   Pullback % Calculation: {'✅ Available' if birdeye_ok else '❌ Need Birdeye API'}")
        print(f"   Time-to-2x Analysis: {'✅ Available' if birdeye_ok else '❌ Need Birdeye API'}")
        print(f"   Enhanced Contract Detection: {'✅ Available' if telegram_ok else '❌ Need Telegram API'}")
        print(f"   Composite Scoring: {'✅ Available' if cielo_ok else '❌ Need Cielo Finance API'}")
        
        input("\nPress Enter to continue...")
    
    def _show_help(self):
        """Show help and examples."""
        print("\n" + "="*80)
        print("    📖 HELP & EXAMPLES - FIXED Enhanced Phoenix Project")
        print("="*80)
        
        print("\n🚀 GETTING STARTED:")
        print("1. Configure API keys (Option 1)")
        print("   - Birdeye API: https://birdeye.so (for enhanced features)")
        print("   - Cielo Finance API: https://cielo.finance (for wallet analysis)")
        print("   - Telegram API: https://my.telegram.org (for SpyDefi)")
        print()
        print("2. Run enhanced analysis")
        print("   - SpyDefi analysis with pullback & time-to-2x (Option 4)")
        print("   - Wallet analysis with composite scoring (Option 5)")
        
        print("\n🎯 ENHANCED FEATURES:")
        print("• 📉 Max Pullback % Analysis - Calculate average maximum drawdown")
        print("• ⏱️ Time-to-2x Analysis - Average time to reach 100% ROI")
        print("• 🔍 Enhanced Contract Detection - Better Solana address extraction")
        print("• 💰 Wallet Composite Score (0-100) - Bad to Great scale")
        print("• 🏆 Strategy Optimization - Use pullback data for stop loss")
        print("• 📊 Entry/Exit Behavior Analysis - Detect if selling too early")
        print("• ⚡ Parallel Processing - Faster wallet analysis")
        print("• 💾 RPC Caching - Avoid duplicate calls")
        
        print("\n💯 COMPOSITE SCORE BREAKDOWN:")
        print("• 0-20: 🔴 VERY POOR - Avoid copying")
        print("• 21-40: 🟠 POOR - High risk, proceed with caution")
        print("• 41-60: 🟡 AVERAGE - Mixed results, selective copying")
        print("• 61-80: 🟢 GOOD - Solid performer, recommended")
        print("• 81-100: 🟣 EXCELLENT - Top tier trader, priority follow")
        
        print("\n📂 WALLET CATEGORIES (Optimized for Solana Memecoins):")
        print("• 🎯 GEM FINDER: 10%+ big wins AND 200%+ max ROI")
        print("• 📊 CONSISTENT: 35%+ win rate AND -10%+ median ROI")
        print("• ⚡ FLIPPER: <24h hold time AND 40%+ win rate")
        print("• 🔀 MIXED: Some positive traits but inconsistent")
        print("• 📉 UNDERPERFORMER: Below thresholds but active")
        print("• ❓ UNKNOWN: Low activity (<3 trades)")
        
        print("\n📊 OUTPUT FILES:")
        print("SpyDefi Analysis:")
        print("• spydefi_analysis_enhanced.csv - Individual token call data")
        print("• spydefi_analysis_enhanced_kol_performance_enhanced.csv - KOL metrics")
        print("• spydefi_analysis_enhanced_enhanced_summary.txt - Analysis summary")
        print("• spydefi_analysis_enhanced_enhanced.xlsx - Excel workbook")
        print()
        print("Wallet Analysis:")
        print("• wallet_analysis_[timestamp].csv - Ranked wallets with scores")
        print("• wallet_analysis_[timestamp].xlsx - Excel with formatting")
        
        print("\n💡 TRADING STRATEGY USAGE:")
        print("📉 Stop Loss Calculation:")
        print("   Recommended SL = avg_max_pullback_percent + 5-10% buffer")
        print("   Example: If KOL has 35% avg pullback, set SL at -40% to -45%")
        print()
        print("⏱️ Holding Time Strategy:")
        print("   Minimum Hold Time = avg_time_to_2x_formatted")
        print("   Example: If avg time to 2x is '3h 20m', hold for at least 3.5 hours")
        print()
        print("💯 Composite Score Usage:")
        print("   80+ Score: Copy all trades immediately (EXCELLENT)")
        print("   60-80 Score: Copy selectively, use tight stops (GOOD)")
        print("   40-60 Score: Only copy high-confidence setups (AVERAGE)")
        print("   20-40 Score: Paper trade or very small positions (POOR)")
        print("   Below 20: Avoid copying (VERY POOR)")
        print()
        print("📊 Entry/Exit Analysis:")
        print("   EARLY_SELLER: Consider holding positions longer")
        print("   MISSING_RUNNERS: Use trailing stops to capture gains")
        print("   BALANCED: Good risk/reward balance")
        
        print("\n🔧 COMMAND LINE USAGE:")
        print("# Enhanced telegram analysis")
        print("python phoenix.py telegram --hours 24 --output enhanced_analysis.csv --excel")
        print()
        print("# Wallet analysis (simplified)")
        print("python phoenix.py wallet --days 30")
        print()
        print("# Configure APIs")
        print("python phoenix.py configure --birdeye-api-key YOUR_KEY --cielo-api-key YOUR_KEY")
        
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
            # Show first 10 wallets
            for i, wallet in enumerate(wallets[:10], 1):
                print(f"   {i}. {wallet[:8]}...{wallet[-4:]}")
            if len(wallets) > 10:
                print(f"   ... and {len(wallets) - 10} more wallets")
        else:
            print("   No wallets found in wallets.txt")
        
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
                self._handle_fixed_enhanced_telegram_analysis(args)
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
                self.config.get("cielo_api_key")
            )
            
            wallet_analyzer = WalletAnalyzer(
                cielo_api=api_manager.cielo_api,
                birdeye_api=api_manager.birdeye_api,
                rpc_url=self.config.get("solana_rpc_url")
            )
            
            # Run batch analysis (removed min_winrate and include_contested)
            results = wallet_analyzer.batch_analyze_wallets(
                wallets,
                days_back=args.days,
                use_hybrid=True  # Always use hybrid approach
            )
            
            if results.get("success"):
                # Always export to both Excel and CSV
                output_file = ensure_output_dir(args.output)
                
                # Excel export
                if args.output.endswith('.xlsx'):
                    self._export_wallet_analysis_excel(results, output_file)
                else:
                    # If output doesn't end with .xlsx, add it
                    excel_file = output_file.replace('.csv', '.xlsx')
                    if not excel_file.endswith('.xlsx'):
                        excel_file += '.xlsx'
                    self._export_wallet_analysis_excel(results, excel_file)
                    output_file = excel_file
                
                # Also export CSV
                csv_file = output_file.replace('.xlsx', '.csv')
                self._export_wallet_analysis_csv(results, csv_file)
                
                logger.info(f"Analysis complete. Results saved to {output_file} and {csv_file}")
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