"""
Export Utilities Module - Phoenix Project (UPDATED WITH DATA QUALITY)

Handles Excel and CSV exports for analysis results with enhanced formatting.
UPDATES:
- Added data quality factor support
- Added base composite score tracking
- Added data source breakdown
- Hold time now in minutes instead of hours
"""

import os
import csv
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("phoenix.export")

try:
    import pandas as pd
    import xlsxwriter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logger.warning("pandas and xlsxwriter not installed. Excel export will be limited.")

def export_to_excel(telegram_data: Dict[str, Any], wallet_data: Dict[str, Any], 
                   output_file: str) -> bool:
    """
    Export combined analysis results to Excel with formatting.
    
    Args:
        telegram_data: Telegram analysis results
        wallet_data: Wallet analysis results
        output_file: Output Excel file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not EXCEL_AVAILABLE:
        logger.error("Excel export requires pandas and xlsxwriter. Install with: pip install pandas xlsxwriter")
        return False
    
    try:
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#1a1a2e',
                'font_color': 'white',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            score_excellent_format = workbook.add_format({
                'bg_color': '#e6e6fa',  # Light purple
                'border': 1,
                'num_format': '0.0'
            })
            
            score_good_format = workbook.add_format({
                'bg_color': '#90ee90',  # Light green
                'border': 1,
                'num_format': '0.0'
            })
            
            score_average_format = workbook.add_format({
                'bg_color': '#ffffe0',  # Light yellow
                'border': 1,
                'num_format': '0.0'
            })
            
            score_poor_format = workbook.add_format({
                'bg_color': '#ffdab9',  # Light orange
                'border': 1,
                'num_format': '0.0'
            })
            
            score_very_poor_format = workbook.add_format({
                'bg_color': '#ffcccb',  # Light red
                'border': 1,
                'num_format': '0.0'
            })
            
            percent_format = workbook.add_format({
                'num_format': '0.00%',
                'border': 1
            })
            
            money_format = workbook.add_format({
                'num_format': '$#,##0.00',
                'border': 1
            })
            
            number_format = workbook.add_format({
                'num_format': '#,##0.00',
                'border': 1
            })
            
            # Export Telegram data if available
            if telegram_data and "ranked_kols" in telegram_data:
                telegram_df = pd.DataFrame(telegram_data["ranked_kols"])
                telegram_df.to_excel(writer, sheet_name='Telegram KOLs', index=False)
                
                # Format Telegram sheet
                telegram_sheet = writer.sheets['Telegram KOLs']
                
                # Apply header format
                for col_num, value in enumerate(telegram_df.columns.values):
                    telegram_sheet.write(0, col_num, value, header_format)
                
                # Set column widths
                telegram_sheet.set_column('A:A', 20)  # Channel ID
                telegram_sheet.set_column('B:B', 15)  # Total calls
                telegram_sheet.set_column('C:E', 15)  # Success rates
                telegram_sheet.set_column('F:G', 20)  # Metrics
            
            # Export Wallet data if available
            if wallet_data:
                # Create summary sheet
                if "total_wallets" in wallet_data:
                    summary_data = {
                        'Metric': [
                            'Total Wallets',
                            'Successfully Analyzed',
                            'Failed Analysis',
                            'Passed Filters',
                            'Gem Finders',
                            'Consistent Traders',
                            'Quick Flippers',
                            'Mixed Results',
                            'Underperformers',
                            'Unknown/Low Activity'
                        ],
                        'Value': [
                            wallet_data.get('total_wallets', 0),
                            wallet_data.get('analyzed_wallets', 0),
                            wallet_data.get('failed_wallets', 0),
                            wallet_data.get('filtered_wallets', 0),
                            len(wallet_data.get('gem_finders', [])),
                            len(wallet_data.get('consistent', [])),
                            len(wallet_data.get('flippers', [])),
                            len(wallet_data.get('mixed', [])),
                            len(wallet_data.get('underperformers', [])),
                            len(wallet_data.get('unknown', []))
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Create detailed wallet analysis sheet
                all_wallets = []
                for category in ['gem_finders', 'consistent', 'flippers', 'mixed', 'underperformers', 'unknown']:
                    all_wallets.extend(wallet_data.get(category, []))
                
                if all_wallets:
                    # Sort by composite score
                    all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
                    
                    wallet_rows = []
                    for rank, wallet in enumerate(all_wallets, 1):
                        metrics = wallet['metrics']
                        composite_score = wallet.get('composite_score', metrics.get('composite_score', 0))
                        
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
                        
                        # Convert hold time from hours to minutes
                        hold_time_minutes = round(metrics.get('avg_hold_time_hours', 0) * 60, 2)
                        
                        # Cap profit factor at 999.99
                        profit_factor = metrics.get('profit_factor', 0)
                        if profit_factor > 999.99:
                            profit_factor = 999.99
                        
                        row = {
                            'Rank': rank,
                            'Wallet': wallet['wallet_address'],
                            'Score': composite_score,
                            'Data Quality': metrics.get('data_quality_factor', 1.0),
                            'Base Score': metrics.get('base_composite_score', composite_score),
                            'Rating': rating,
                            'Type': wallet['wallet_type'],
                            'Trades': metrics['total_trades'],
                            'Win Rate': metrics['win_rate'] / 100,  # For percentage formatting
                            'Profit Factor': profit_factor,
                            'Net Profit': metrics['net_profit_usd'],
                            'Avg ROI': metrics['avg_roi'] / 100,  # For percentage formatting
                            'Max ROI': metrics['max_roi'] / 100,  # For percentage formatting
                            'Hold Time (min)': hold_time_minutes,
                            'Strategy': wallet['strategy']['recommendation']
                        }
                        
                        # Add entry/exit analysis if available
                        if 'entry_exit_analysis' in wallet and wallet['entry_exit_analysis']:
                            ee_analysis = wallet['entry_exit_analysis']
                            row['Entry Pattern'] = ee_analysis.get('pattern', '')
                            row['Entry Quality'] = ee_analysis.get('entry_quality', '')
                            row['Exit Quality'] = ee_analysis.get('exit_quality', '')
                            row['Missed Gains %'] = ee_analysis.get('missed_gains_percent', 0) / 100
                        
                        # Add data source breakdown
                        if 'data_quality_breakdown' in wallet:
                            breakdown = wallet['data_quality_breakdown']
                            row['Full Analysis'] = breakdown.get('full_analysis', 0)
                            row['Helius Analysis'] = breakdown.get('helius_analysis', 0)
                            row['Basic Analysis'] = breakdown.get('basic_analysis', 0)
                        
                        wallet_rows.append(row)
                    
                    wallet_df = pd.DataFrame(wallet_rows)
                    wallet_df.to_excel(writer, sheet_name='Wallet Analysis', index=False)
                    
                    # Format wallet sheet
                    wallet_sheet = writer.sheets['Wallet Analysis']
                    
                    # Apply header format
                    for col_num, value in enumerate(wallet_df.columns.values):
                        wallet_sheet.write(0, col_num, value, header_format)
                    
                    # Apply conditional formatting for scores
                    wallet_sheet.conditional_format('C2:C{}'.format(len(wallet_df) + 1), {
                        'type': 'cell',
                        'criteria': '>=',
                        'value': 81,
                        'format': score_excellent_format
                    })
                    
                    wallet_sheet.conditional_format('C2:C{}'.format(len(wallet_df) + 1), {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 61,
                        'maximum': 80.99,
                        'format': score_good_format
                    })
                    
                    wallet_sheet.conditional_format('C2:C{}'.format(len(wallet_df) + 1), {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 41,
                        'maximum': 60.99,
                        'format': score_average_format
                    })
                    
                    wallet_sheet.conditional_format('C2:C{}'.format(len(wallet_df) + 1), {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 21,
                        'maximum': 40.99,
                        'format': score_poor_format
                    })
                    
                    wallet_sheet.conditional_format('C2:C{}'.format(len(wallet_df) + 1), {
                        'type': 'cell',
                        'criteria': '<=',
                        'value': 20.99,
                        'format': score_very_poor_format
                    })
                    
                    # Apply percentage format to percentage columns
                    for row_num in range(1, len(wallet_df) + 1):
                        wallet_sheet.write(row_num, 8, wallet_df.iloc[row_num-1]['Win Rate'], percent_format)
                        wallet_sheet.write(row_num, 11, wallet_df.iloc[row_num-1]['Avg ROI'], percent_format)
                        wallet_sheet.write(row_num, 12, wallet_df.iloc[row_num-1]['Max ROI'], percent_format)
                        if 'Missed Gains %' in wallet_df.columns:
                            col_idx = wallet_df.columns.get_loc('Missed Gains %')
                            wallet_sheet.write(row_num, col_idx, wallet_df.iloc[row_num-1]['Missed Gains %'], percent_format)
                    
                    # Apply money format to profit column
                    for row_num in range(1, len(wallet_df) + 1):
                        wallet_sheet.write(row_num, 10, wallet_df.iloc[row_num-1]['Net Profit'], money_format)
                    
                    # Set column widths
                    wallet_sheet.set_column('A:A', 8)   # Rank
                    wallet_sheet.set_column('B:B', 50)  # Wallet
                    wallet_sheet.set_column('C:C', 10)  # Score
                    wallet_sheet.set_column('D:D', 12)  # Data Quality
                    wallet_sheet.set_column('E:E', 12)  # Base Score
                    wallet_sheet.set_column('F:F', 12)  # Rating
                    wallet_sheet.set_column('G:G', 15)  # Type
                    wallet_sheet.set_column('H:H', 10)  # Trades
                    wallet_sheet.set_column('I:I', 12)  # Win Rate
                    wallet_sheet.set_column('J:J', 15)  # Profit Factor
                    wallet_sheet.set_column('K:K', 15)  # Net Profit
                    wallet_sheet.set_column('L:M', 12)  # ROI columns
                    wallet_sheet.set_column('N:N', 15)  # Hold Time
                    wallet_sheet.set_column('O:O', 20)  # Strategy
                    wallet_sheet.set_column('P:S', 15)  # Entry/Exit columns
                    wallet_sheet.set_column('T:V', 12)  # Data source columns
            
            logger.info(f"Successfully exported analysis to Excel: {output_file}")
            return True
            
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        return False

def export_wallet_rankings_csv(wallet_data: Dict[str, Any], output_file: str) -> bool:
    """
    Export wallet rankings to CSV with composite scores and data quality.
    
    Args:
        wallet_data: Wallet analysis results
        output_file: Output CSV file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare all wallets
        all_wallets = []
        for category in ['gem_finders', 'consistent', 'flippers', 'mixed', 'underperformers', 'unknown']:
            all_wallets.extend(wallet_data.get(category, []))
        
        # Sort by composite score
        all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'rank', 'wallet_address', 'composite_score', 'score_rating',
                'data_quality_factor', 'base_composite_score',
                'wallet_type', 'total_trades', 'win_rate', 'profit_factor',
                'net_profit_usd', 'avg_roi', 'median_roi', 'max_roi',
                'avg_hold_time_minutes',  # Changed from hours to minutes
                'total_tokens_traded',
                'entry_pattern', 'entry_quality', 'exit_quality',
                'missed_gains_percent', 'early_exit_rate',
                'full_analysis_count', 'helius_analysis_count', 'basic_analysis_count',
                'strategy_recommendation', 'confidence',
                'competition_level'
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
                
                # Cap profit factor at 999.99
                profit_factor = metrics.get('profit_factor', 0)
                if profit_factor > 999.99:
                    profit_factor = 999.99
                
                row = {
                    'rank': rank,
                    'wallet_address': analysis['wallet_address'],
                    'composite_score': round(score, 1),
                    'score_rating': rating,
                    'data_quality_factor': round(metrics.get('data_quality_factor', 1.0), 2),
                    'base_composite_score': round(metrics.get('base_composite_score', score), 1),
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
                    'confidence': analysis['strategy'].get('confidence', ''),
                    'competition_level': analysis['strategy'].get('competition_level', '')
                }
                
                # Add entry/exit analysis if available
                if 'entry_exit_analysis' in analysis and analysis['entry_exit_analysis']:
                    ee_analysis = analysis['entry_exit_analysis']
                    row['entry_pattern'] = ee_analysis.get('pattern', '')
                    row['entry_quality'] = ee_analysis.get('entry_quality', '')
                    row['exit_quality'] = ee_analysis.get('exit_quality', '')
                    row['missed_gains_percent'] = ee_analysis.get('missed_gains_percent', 0)
                    row['early_exit_rate'] = ee_analysis.get('early_exit_rate', 0)
                else:
                    row['entry_pattern'] = ''
                    row['entry_quality'] = ''
                    row['exit_quality'] = ''
                    row['missed_gains_percent'] = 0
                    row['early_exit_rate'] = 0
                
                # Add data quality breakdown
                if 'data_quality_breakdown' in analysis:
                    breakdown = analysis['data_quality_breakdown']
                    row['full_analysis_count'] = breakdown.get('full_analysis', 0)
                    row['helius_analysis_count'] = breakdown.get('helius_analysis', 0)
                    row['basic_analysis_count'] = breakdown.get('basic_analysis', 0)
                else:
                    row['full_analysis_count'] = 0
                    row['helius_analysis_count'] = 0
                    row['basic_analysis_count'] = 0
                
                writer.writerow(row)
        
        logger.info(f"Successfully exported wallet rankings to CSV: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting wallet rankings to CSV: {str(e)}")
        return False

def generate_analysis_report(telegram_data: Dict[str, Any], wallet_data: Dict[str, Any], 
                           output_file: str) -> bool:
    """
    Generate a comprehensive text report of the analysis.
    
    Args:
        telegram_data: Telegram analysis results
        wallet_data: Wallet analysis results
        output_file: Output text file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PHOENIX PROJECT - COMPREHENSIVE ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Telegram Analysis Section
            if telegram_data and "ranked_kols" in telegram_data:
                f.write("📱 TELEGRAM ANALYSIS (SPYDEFI)\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total KOLs analyzed: {telegram_data.get('total_kols_analyzed', 0)}\n")
                f.write(f"Total calls analyzed: {telegram_data.get('total_calls', 0)}\n")
                f.write(f"2x success rate: {telegram_data.get('success_rate_2x', 0):.2f}%\n")
                f.write(f"5x success rate: {telegram_data.get('success_rate_5x', 0):.2f}%\n\n")
                
                f.write("Top 5 KOLs:\n")
                ranked_kols = telegram_data.get('ranked_kols', {})
                for i, (kol, data) in enumerate(list(ranked_kols.items())[:5], 1):
                    f.write(f"{i}. @{kol}\n")
                    f.write(f"   Composite Score: {data.get('composite_score', 0):.1f}\n")
                    f.write(f"   Success Rate (2x): {data.get('success_rate_2x', 0):.1f}%\n")
                    if data.get('avg_max_pullback_percent', 0) > 0:
                        f.write(f"   Avg Pullback: {data.get('avg_max_pullback_percent', 0):.1f}%\n")
                    if data.get('avg_time_to_2x_formatted', 'N/A') != 'N/A':
                        f.write(f"   Avg Time to 2x: {data.get('avg_time_to_2x_formatted', 'N/A')}\n")
                    f.write("\n")
            
            # Wallet Analysis Section
            if wallet_data and wallet_data.get('success'):
                f.write("\n💰 WALLET ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total wallets: {wallet_data.get('total_wallets', 0)}\n")
                f.write(f"Successfully analyzed: {wallet_data.get('analyzed_wallets', 0)}\n")
                f.write(f"Failed analysis: {wallet_data.get('failed_wallets', 0)}\n")
                f.write(f"Passed filters: {wallet_data.get('filtered_wallets', 0)}\n\n")
                
                f.write("Category Breakdown:\n")
                f.write(f"🎯 Gem Finders: {len(wallet_data.get('gem_finders', []))}\n")
                f.write(f"📊 Consistent Traders: {len(wallet_data.get('consistent', []))}\n")
                f.write(f"⚡ Quick Flippers: {len(wallet_data.get('flippers', []))}\n")
                f.write(f"🔀 Mixed Results: {len(wallet_data.get('mixed', []))}\n")
                f.write(f"📉 Underperformers: {len(wallet_data.get('underperformers', []))}\n")
                f.write(f"❓ Unknown: {len(wallet_data.get('unknown', []))}\n\n")
                
                # Top wallets by composite score
                all_wallets = []
                for category in ['gem_finders', 'consistent', 'flippers', 'mixed', 'underperformers', 'unknown']:
                    all_wallets.extend(wallet_data.get(category, []))
                
                all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
                
                f.write("🏆 TOP 10 WALLETS BY COMPOSITE SCORE:\n")
                for i, wallet in enumerate(all_wallets[:10], 1):
                    metrics = wallet['metrics']
                    score = wallet.get('composite_score', metrics.get('composite_score', 0))
                    data_quality = metrics.get('data_quality_factor', 1.0)
                    
                    # Rating
                    if score >= 81:
                        rating = "🟣 EXCELLENT"
                    elif score >= 61:
                        rating = "🟢 GOOD"
                    elif score >= 41:
                        rating = "🟡 AVERAGE"
                    elif score >= 21:
                        rating = "🟠 POOR"
                    else:
                        rating = "🔴 VERY POOR"
                    
                    # Cap profit factor
                    profit_factor = metrics['profit_factor']
                    if profit_factor > 999.99:
                        profit_factor_display = "999.99x"
                    else:
                        profit_factor_display = f"{profit_factor:.2f}x"
                    
                    f.write(f"\n{i}. {wallet['wallet_address'][:8]}...{wallet['wallet_address'][-4:]}\n")
                    f.write(f"   Score: {score:.1f}/100 {rating} (Data Quality: {data_quality:.2f})\n")
                    f.write(f"   Type: {wallet['wallet_type']}\n")
                    f.write(f"   Win Rate: {metrics['win_rate']:.1f}%\n")
                    f.write(f"   Profit Factor: {profit_factor_display}\n")
                    f.write(f"   Net Profit: ${metrics['net_profit_usd']:.2f}\n")
                    f.write(f"   Total Trades: {metrics['total_trades']}\n")
                    f.write(f"   Avg Hold Time: {round(metrics.get('avg_hold_time_hours', 0) * 60, 2)} minutes\n")
                    
                    # Entry/exit analysis
                    if 'entry_exit_analysis' in wallet and wallet['entry_exit_analysis']:
                        ee_analysis = wallet['entry_exit_analysis']
                        if ee_analysis.get('pattern') != 'INSUFFICIENT_DATA':
                            f.write(f"   Entry/Exit: {ee_analysis.get('pattern', 'UNKNOWN')} ")
                            f.write(f"(Missed: {ee_analysis.get('missed_gains_percent', 0):.1f}%)\n")
                    
                    # Data source breakdown
                    if 'data_quality_breakdown' in wallet:
                        breakdown = wallet['data_quality_breakdown']
                        f.write(f"   Data Sources: Full={breakdown.get('full_analysis', 0)}, ")
                        f.write(f"Helius={breakdown.get('helius_analysis', 0)}, ")
                        f.write(f"Basic={breakdown.get('basic_analysis', 0)}\n")
                    
                    f.write(f"   Strategy: {wallet['strategy']['recommendation']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            
        logger.info(f"Successfully generated analysis report: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating analysis report: {str(e)}")
        return False