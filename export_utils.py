"""
Export Utilities Module - Phoenix Project (FIXED EXCEL EXPORT VERSION)

Handles Excel and CSV exports for memecoin analysis results.
FIXES:
- Fixed 'list' object has no attribute 'items' error
- Ensures telegram_data['ranked_kols'] is always treated as dictionary
- Maintains compatibility with wallet_module
- All existing functionality preserved
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
    Export combined analysis results to Excel with 7-day focus formatting.
    
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
            
            # Export Telegram data if available (2x-focused)
            if telegram_data and "ranked_kols" in telegram_data:
                # Prepare telegram data for export
                telegram_rows = []
                
                # Handle both dictionary and list formats for ranked_kols
                ranked_kols = telegram_data["ranked_kols"]
                
                # FIX: Check if ranked_kols is a list (the error case)
                if isinstance(ranked_kols, list):
                    logger.warning("ranked_kols is a list, converting to dictionary format")
                    # Convert list to dictionary if needed
                    # Assuming each item in list has 'kol' key
                    temp_dict = {}
                    for item in ranked_kols:
                        if isinstance(item, dict) and 'kol' in item:
                            kol_name = item['kol']
                            temp_dict[kol_name] = item
                    ranked_kols = temp_dict
                
                # Now process as dictionary
                if isinstance(ranked_kols, dict):
                    for kol, performance in ranked_kols.items():
                        # Handle case where performance might be nested
                        if isinstance(performance, dict):
                            row = {
                                'KOL': f"@{kol}",
                                'Channel ID': performance.get('channel_id', ''),
                                'Calls Analyzed': performance.get('tokens_mentioned', 0),
                                '2x Success Rate %': performance.get('success_rate_2x', 0),
                                'Avg ATH ROI %': performance.get('avg_ath_roi', 0),
                                'Composite Score': performance.get('composite_score', 0),
                                'Avg Max Pullback %': performance.get('avg_max_pullback_percent', 0),
                                'Avg Time to 2x (min)': performance.get('avg_time_to_2x_minutes', 0),
                                'Analysis Type': performance.get('analysis_type', 'initial')
                            }
                            telegram_rows.append(row)
                
                # Sort by composite score
                telegram_rows.sort(key=lambda x: x['Composite Score'], reverse=True)
                
                if telegram_rows:
                    telegram_df = pd.DataFrame(telegram_rows)
                    telegram_df.to_excel(writer, sheet_name='Telegram KOLs', index=False)
                    
                    # Format Telegram sheet
                    telegram_sheet = writer.sheets['Telegram KOLs']
                    
                    # Apply header format
                    for col_num, value in enumerate(telegram_df.columns.values):
                        telegram_sheet.write(0, col_num, value, header_format)
                    
                    # Apply conditional formatting for composite scores
                    telegram_sheet.conditional_format('F2:F{}'.format(len(telegram_df) + 1), {
                        'type': 'cell',
                        'criteria': '>=',
                        'value': 81,
                        'format': score_excellent_format
                    })
                    
                    telegram_sheet.conditional_format('F2:F{}'.format(len(telegram_df) + 1), {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 61,
                        'maximum': 80.99,
                        'format': score_good_format
                    })
                    
                    telegram_sheet.conditional_format('F2:F{}'.format(len(telegram_df) + 1), {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 41,
                        'maximum': 60.99,
                        'format': score_average_format
                    })
                    
                    telegram_sheet.conditional_format('F2:F{}'.format(len(telegram_df) + 1), {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 21,
                        'maximum': 40.99,
                        'format': score_poor_format
                    })
                    
                    telegram_sheet.conditional_format('F2:F{}'.format(len(telegram_df) + 1), {
                        'type': 'cell',
                        'criteria': '<=',
                        'value': 20.99,
                        'format': score_very_poor_format
                    })
                    
                    # Set column widths
                    telegram_sheet.set_column('A:A', 20)  # KOL
                    telegram_sheet.set_column('B:B', 15)  # Channel ID
                    telegram_sheet.set_column('C:C', 15)  # Calls Analyzed
                    telegram_sheet.set_column('D:D', 18)  # 2x Success Rate
                    telegram_sheet.set_column('E:E', 15)  # Avg ATH ROI
                    telegram_sheet.set_column('F:F', 15)  # Composite Score
                    telegram_sheet.set_column('G:G', 18)  # Avg Max Pullback
                    telegram_sheet.set_column('H:H', 20)  # Avg Time to 2x
                    telegram_sheet.set_column('I:I', 15)  # Analysis Type
            
            # Export Wallet data if available (unchanged - preserved for wallet_module compatibility)
            if wallet_data:
                # Create summary sheet for 7-day active traders
                if "total_wallets" in wallet_data:
                    summary_data = {
                        'Metric': [
                            'Total Wallets',
                            'Successfully Analyzed',
                            'Failed Analysis',
                            '',
                            '7-DAY ACTIVE TRADERS',
                            'Active (traded in 7 days)',
                            'Inactive (no trades in 7 days)',
                            '',
                            'COPY DECISIONS',
                            'Copy YES',
                            'Copy NO',
                            '',
                            'WALLET TYPES',
                            'Snipers (<1 min)',
                            'Flippers (1-10 min)',
                            'Scalpers (10-60 min)',
                            'Gem Hunters (5x+ focus)',
                            'Swing Traders (1-24h)',
                            'Position Traders (24h+)',
                            'Consistent',
                            'Mixed',
                            'Unknown',
                            '',
                            'MARKET CAP INSIGHTS',
                            'Ultra Low Cap ($5K-$50K)',
                            'Low Cap ($50K-$500K)',
                            'Mid Cap ($500K-$5M)',
                            'High Cap ($5M+)'
                        ],
                        'Value': [
                            wallet_data.get('total_wallets', 0),
                            wallet_data.get('analyzed_wallets', 0),
                            wallet_data.get('failed_wallets', 0),
                            '',
                            '',
                            sum(1 for cat in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 
                                            'swing_traders', 'position_traders', 'consistent', 'mixed']
                                for w in wallet_data.get(cat, []) 
                                if w.get('metrics', {}).get('active_trader', False)),
                            sum(1 for cat in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 
                                            'swing_traders', 'position_traders', 'consistent', 'mixed', 'unknown']
                                for w in wallet_data.get(cat, []) 
                                if not w.get('metrics', {}).get('active_trader', False)),
                            '',
                            '',
                            sum(1 for cat in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 
                                            'swing_traders', 'position_traders', 'consistent', 'mixed']
                                for w in wallet_data.get(cat, []) 
                                if w.get('composite_score', w['metrics'].get('composite_score', 0)) >= 60 
                                and w.get('metrics', {}).get('active_trader', False)),
                            sum(1 for cat in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 
                                            'swing_traders', 'position_traders', 'consistent', 'mixed', 'unknown']
                                for w in wallet_data.get(cat, []) 
                                if w.get('composite_score', w['metrics'].get('composite_score', 0)) < 60 
                                or not w.get('metrics', {}).get('active_trader', False)),
                            '',
                            '',
                            len(wallet_data.get('snipers', [])),
                            len(wallet_data.get('flippers', [])),
                            len(wallet_data.get('scalpers', [])),
                            len(wallet_data.get('gem_hunters', [])),
                            len(wallet_data.get('swing_traders', [])),
                            len(wallet_data.get('position_traders', [])),
                            len(wallet_data.get('consistent', [])),
                            len(wallet_data.get('mixed', [])),
                            len(wallet_data.get('unknown', [])),
                            '',
                            '',
                            'Most common for new launches',
                            'Sweet spot for most traders',
                            'Safer but lower multiples',
                            'Established tokens'
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Create detailed wallet analysis sheet
                all_wallets = []
                for category in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 
                               'swing_traders', 'position_traders', 'consistent', 'mixed', 'unknown']:
                    all_wallets.extend(wallet_data.get(category, []))
                
                if all_wallets:
                    # Sort by composite score
                    all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
                    
                    wallet_rows = []
                    for rank, wallet in enumerate(all_wallets, 1):
                        metrics = wallet['metrics']
                        composite_score = wallet.get('composite_score', metrics.get('composite_score', 0))
                        strategy = wallet.get('strategy', {})
                        
                        # Determine binary copy decision
                        copy_decision = "YES" if (
                            composite_score >= 60 and 
                            metrics.get('active_trader', False) and
                            metrics.get('trades_last_7_days', 0) > 0
                        ) else "NO"
                        
                        # Cap profit factor at 999.99
                        profit_factor = metrics.get('profit_factor', 0)
                        if profit_factor > 999.99:
                            profit_factor = 999.99
                        
                        # Build row data
                        row = {
                            'Rank': rank,
                            'Wallet': wallet['wallet_address'],
                            'Copy Decision': copy_decision,
                            'Score': composite_score,
                            'Type': wallet['wallet_type'],
                            'Trades': metrics['total_trades'],
                            'Trades 7d': metrics.get('trades_last_7_days', 0),
                            'Win Rate 7d': metrics.get('win_rate_7d', 0) / 100,
                            'Profit Factor': profit_factor,
                            'Profit 7d': metrics.get('profit_7d', 0),
                            'Avg ROI': metrics['avg_roi'] / 100,
                            'Max ROI': metrics['max_roi'] / 100,
                            'Gem Rate (5x+)': metrics.get('gem_rate_5x_plus', 0) / 100,
                            'Hold Time (min)': metrics.get('avg_hold_time_minutes', 0),
                            'Avg First TP %': metrics.get('avg_first_take_profit_percent', 0) / 100,
                            'Active': 'YES' if metrics.get('active_trader', False) else 'NO',
                            'Follow Sells': 'YES' if strategy.get('follow_sells', False) else 'NO',
                            'TP1 %': strategy.get('tp1_percent', 0) / 100,
                            'TP2 %': strategy.get('tp2_percent', 0) / 100,
                            'Sell Strategy': strategy.get('sell_strategy', ''),
                            'TP Guidance': strategy.get('tp_guidance', ''),
                            'Avg MCap Buy': metrics.get('avg_buy_market_cap_usd', 0)
                        }
                        
                        # Add entry/exit analysis if available
                        if 'entry_exit_analysis' in wallet and wallet['entry_exit_analysis']:
                            ee_analysis = wallet['entry_exit_analysis']
                            row['Entry Quality'] = ee_analysis.get('entry_quality', '')
                            row['Exit Quality'] = ee_analysis.get('exit_quality', '')
                            row['Pattern'] = ee_analysis.get('pattern', '')
                            row['Missed Gains %'] = ee_analysis.get('missed_gains_percent', 0) / 100
                            row['Avg Exit ROI'] = ee_analysis.get('avg_exit_roi', 0) / 100
                        
                        # Add distribution data (7-day)
                        row['Dist 500%+'] = metrics.get('distribution_500_plus_%', 0) / 100
                        row['Dist 200-500%'] = metrics.get('distribution_200_500_%', 0) / 100
                        row['Dist 0-200%'] = metrics.get('distribution_0_200_%', 0) / 100
                        row['Dist -50-0%'] = metrics.get('distribution_neg50_0_%', 0) / 100
                        row['Dist <-50%'] = metrics.get('distribution_below_neg50_%', 0) / 100
                        
                        wallet_rows.append(row)
                    
                    wallet_df = pd.DataFrame(wallet_rows)
                    wallet_df.to_excel(writer, sheet_name='Active Traders', index=False)
                    
                    # Format wallet sheet
                    wallet_sheet = writer.sheets['Active Traders']
                    
                    # Apply header format
                    for col_num, value in enumerate(wallet_df.columns.values):
                        wallet_sheet.write(0, col_num, value, header_format)
                    
                    # Apply conditional formatting for scores
                    wallet_sheet.conditional_format('D2:D{}'.format(len(wallet_df) + 1), {
                        'type': 'cell',
                        'criteria': '>=',
                        'value': 81,
                        'format': score_excellent_format
                    })
                    
                    wallet_sheet.conditional_format('D2:D{}'.format(len(wallet_df) + 1), {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 61,
                        'maximum': 80.99,
                        'format': score_good_format
                    })
                    
                    wallet_sheet.conditional_format('D2:D{}'.format(len(wallet_df) + 1), {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 41,
                        'maximum': 60.99,
                        'format': score_average_format
                    })
                    
                    wallet_sheet.conditional_format('D2:D{}'.format(len(wallet_df) + 1), {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 21,
                        'maximum': 40.99,
                        'format': score_poor_format
                    })
                    
                    wallet_sheet.conditional_format('D2:D{}'.format(len(wallet_df) + 1), {
                        'type': 'cell',
                        'criteria': '<=',
                        'value': 20.99,
                        'format': score_very_poor_format
                    })
                    
                    # Apply percentage format to percentage columns
                    percent_cols = ['Win Rate 7d', 'Avg ROI', 'Max ROI', 'Gem Rate (5x+)', 
                                   'Avg First TP %', 'Missed Gains %', 'Avg Exit ROI',
                                   'TP1 %', 'TP2 %',
                                   'Dist 500%+', 'Dist 200-500%', 'Dist 0-200%', 
                                   'Dist -50-0%', 'Dist <-50%']
                    
                    for row_num in range(1, len(wallet_df) + 1):
                        for col_name in percent_cols:
                            if col_name in wallet_df.columns:
                                col_idx = wallet_df.columns.get_loc(col_name)
                                value = wallet_df.iloc[row_num-1][col_name]
                                wallet_sheet.write(row_num, col_idx, value, percent_format)
                    
                    # Apply money format to profit and market cap columns
                    money_cols = ['Profit 7d', 'Avg MCap Buy']
                    for row_num in range(1, len(wallet_df) + 1):
                        for col_name in money_cols:
                            if col_name in wallet_df.columns:
                                col_idx = wallet_df.columns.get_loc(col_name)
                                value = wallet_df.iloc[row_num-1][col_name]
                                wallet_sheet.write(row_num, col_idx, value, money_format)
                    
                    # Set column widths
                    wallet_sheet.set_column('A:A', 8)   # Rank
                    wallet_sheet.set_column('B:B', 50)  # Wallet
                    wallet_sheet.set_column('C:C', 12)  # Copy Decision
                    wallet_sheet.set_column('D:D', 10)  # Score
                    wallet_sheet.set_column('E:E', 15)  # Type
                    wallet_sheet.set_column('F:G', 10)  # Trades
                    wallet_sheet.set_column('H:H', 12)  # Win Rate 7d
                    wallet_sheet.set_column('I:I', 15)  # Profit Factor
                    wallet_sheet.set_column('J:J', 15)  # Profit 7d
                    wallet_sheet.set_column('K:L', 12)  # ROI columns
                    wallet_sheet.set_column('M:M', 15)  # Gem Rate
                    wallet_sheet.set_column('N:N', 15)  # Hold Time
                    wallet_sheet.set_column('O:O', 15)  # First TP
                    wallet_sheet.set_column('P:P', 10)  # Active
                    wallet_sheet.set_column('Q:Q', 12)  # Follow Sells
                    wallet_sheet.set_column('R:S', 10)  # TPs
                    wallet_sheet.set_column('T:T', 15)  # Sell Strategy
                    wallet_sheet.set_column('U:U', 30)  # TP Guidance
                    wallet_sheet.set_column('V:V', 15)  # Avg MCap Buy
                    wallet_sheet.set_column('W:AA', 15) # Entry/Exit
                    wallet_sheet.set_column('AB:AF', 12) # Distribution
            
            logger.info(f"Successfully exported analysis to Excel: {output_file}")
            return True
            
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        return False

def export_wallet_rankings_csv(wallet_data: Dict[str, Any], output_file: str) -> bool:
    """
    Export 7-day focused wallet rankings to CSV with binary copy decision.
    
    Args:
        wallet_data: Wallet analysis results
        output_file: Output CSV file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare all wallets
        all_wallets = []
        for category in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 
                        'swing_traders', 'position_traders', 'consistent', 'mixed', 'unknown']:
            all_wallets.extend(wallet_data.get(category, []))
        
        # Sort by composite score
        all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            # Updated fieldnames - added copy_decision after wallet_address, removed strategy_recommendation and market cap filters
            fieldnames = [
                'rank', 'wallet_address', 'copy_decision', 'composite_score',
                'wallet_type', 'total_trades', 'trades_last_7_days',
                'win_rate_7d', 'profit_factor',
                'profit_7d', 'avg_roi', 'median_roi', 'max_roi',
                'avg_hold_time_minutes', 'total_tokens_traded',
                'active_trader',
                'distribution_500_plus_%', 'distribution_200_500_%',
                'distribution_0_200_%', 'distribution_neg50_0_%',
                'distribution_below_neg50_%',
                'gem_rate_5x_plus_%', 'gem_rate_2x_plus_%',
                'avg_buy_market_cap_usd', 'avg_buy_amount_usd',
                'avg_first_take_profit_percent',
                'entry_exit_pattern', 'entry_quality', 'exit_quality',
                'missed_gains_percent', 'early_exit_rate', 'avg_exit_roi',
                'hold_pattern',
                'follow_sells', 'tp1_percent', 'tp2_percent',
                'sell_strategy', 'tp_guidance'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for rank, analysis in enumerate(all_wallets, 1):
                metrics = analysis['metrics']
                score = analysis.get('composite_score', metrics.get('composite_score', 0))
                strategy = analysis.get('strategy', {})
                
                # Binary copy decision based on score >= 60, active trader, and recent trades
                copy_decision = "YES" if (
                    score >= 60 and 
                    metrics.get('active_trader', False) and
                    metrics.get('trades_last_7_days', 0) > 0
                ) else "NO"
                
                # Cap profit factor at 999.99
                profit_factor = metrics.get('profit_factor', 0)
                if profit_factor > 999.99:
                    profit_factor = 999.99
                
                row = {
                    'rank': rank,
                    'wallet_address': analysis['wallet_address'],
                    'copy_decision': copy_decision,
                    'composite_score': round(score, 1),
                    'wallet_type': analysis['wallet_type'],
                    'total_trades': metrics['total_trades'],
                    'trades_last_7_days': metrics.get('trades_last_7_days', 0),
                    'win_rate_7d': round(metrics.get('win_rate_7d', 0), 2),
                    'profit_factor': profit_factor,
                    'profit_7d': round(metrics.get('profit_7d', 0), 2),
                    'avg_roi': round(metrics['avg_roi'], 2),
                    'median_roi': round(metrics.get('median_roi', 0), 2),
                    'max_roi': round(metrics['max_roi'], 2),
                    'avg_hold_time_minutes': round(metrics.get('avg_hold_time_minutes', 0), 2),
                    'total_tokens_traded': metrics['total_tokens_traded'],
                    'active_trader': 'YES' if metrics.get('active_trader', False) else 'NO',
                    'distribution_500_plus_%': metrics.get('distribution_500_plus_%', 0),
                    'distribution_200_500_%': metrics.get('distribution_200_500_%', 0),
                    'distribution_0_200_%': metrics.get('distribution_0_200_%', 0),
                    'distribution_neg50_0_%': metrics.get('distribution_neg50_0_%', 0),
                    'distribution_below_neg50_%': metrics.get('distribution_below_neg50_%', 0),
                    'gem_rate_5x_plus_%': metrics.get('gem_rate_5x_plus', 0),
                    'gem_rate_2x_plus_%': metrics.get('gem_rate_2x_plus', 0),
                    'avg_buy_market_cap_usd': metrics.get('avg_buy_market_cap_usd', 0),
                    'avg_buy_amount_usd': metrics.get('avg_buy_amount_usd', 0),
                    'avg_first_take_profit_percent': metrics.get('avg_first_take_profit_percent', 0),
                    'follow_sells': 'YES' if strategy.get('follow_sells', False) else 'NO',
                    'tp1_percent': strategy.get('tp1_percent', 0),
                    'tp2_percent': strategy.get('tp2_percent', 0),
                    'sell_strategy': strategy.get('sell_strategy', ''),
                    'tp_guidance': strategy.get('tp_guidance', '')
                }
                
                # Add entry/exit analysis if available
                if 'entry_exit_analysis' in analysis and analysis['entry_exit_analysis']:
                    ee_analysis = analysis['entry_exit_analysis']
                    row['entry_exit_pattern'] = ee_analysis.get('pattern', '')
                    row['entry_quality'] = ee_analysis.get('entry_quality', '')
                    row['exit_quality'] = ee_analysis.get('exit_quality', '')
                    row['missed_gains_percent'] = ee_analysis.get('missed_gains_percent', 0)
                    row['early_exit_rate'] = ee_analysis.get('early_exit_rate', 0)
                    row['avg_exit_roi'] = ee_analysis.get('avg_exit_roi', 0)
                    row['hold_pattern'] = ee_analysis.get('hold_pattern', '')
                else:
                    row['entry_exit_pattern'] = ''
                    row['entry_quality'] = ''
                    row['exit_quality'] = ''
                    row['missed_gains_percent'] = 0
                    row['early_exit_rate'] = 0
                    row['avg_exit_roi'] = 0
                    row['hold_pattern'] = ''
                
                writer.writerow(row)
        
        logger.info(f"Successfully exported 7-day active trader rankings to CSV: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting wallet rankings to CSV: {str(e)}")
        return False

def generate_memecoin_analysis_report(telegram_data: Dict[str, Any], 
                                    wallet_data: Dict[str, Any], 
                                    output_file: str) -> bool:
    """
    Generate a comprehensive 7-day active trader analysis report.
    
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
            f.write("PHOENIX PROJECT - 7-DAY ACTIVE TRADER ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Telegram Analysis Section (2x-focused)
            if telegram_data and "ranked_kols" in telegram_data:
                f.write("📱 TELEGRAM ANALYSIS (SPYDEFI - 2X HOT STREAK FOCUS)\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total KOLs analyzed: {telegram_data.get('total_kols_analyzed', 0)}\n")
                f.write(f"Deep analyses performed: {telegram_data.get('deep_analyses_performed', 0)}\n")
                f.write(f"Total calls analyzed: {telegram_data.get('total_calls', 0)}\n")
                f.write(f"2x success rate: {telegram_data.get('success_rate_2x', 0):.2f}%\n\n")
                
                f.write("Top 5 KOLs (2x Hot Streaks):\n")
                ranked_kols = telegram_data.get('ranked_kols', {})
                
                # Handle both dictionary and list formats
                if isinstance(ranked_kols, list):
                    # Convert list to dictionary format
                    temp_dict = {}
                    for item in ranked_kols:
                        if isinstance(item, dict) and 'kol' in item:
                            kol_name = item['kol']
                            temp_dict[kol_name] = item
                    ranked_kols = temp_dict
                
                if isinstance(ranked_kols, dict):
                    for i, (kol, data) in enumerate(list(ranked_kols.items())[:5], 1):
                        f.write(f"{i}. @{kol}\n")
                        f.write(f"   Composite Score: {data.get('composite_score', 0):.1f}\n")
                        f.write(f"   2x Success Rate: {data.get('success_rate_2x', 0):.1f}%\n")
                        f.write(f"   Avg Time to 2x: {data.get('avg_time_to_2x_minutes', 0):.1f} minutes\n")
                        f.write(f"   Avg ATH ROI: {data.get('avg_ath_roi', 0):.1f}%\n")
                        f.write(f"   Avg Max Pullback: {data.get('avg_max_pullback_percent', 0):.1f}%\n")
                        f.write(f"   Analysis Type: {data.get('analysis_type', 'initial')}\n")
                        f.write("\n")
            
            # Wallet Analysis Section (unchanged)
            if wallet_data and wallet_data.get('success'):
                f.write("\n💰 7-DAY ACTIVE TRADER WALLET ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total wallets: {wallet_data.get('total_wallets', 0)}\n")
                f.write(f"Successfully analyzed: {wallet_data.get('analyzed_wallets', 0)}\n")
                f.write(f"Failed analysis: {wallet_data.get('failed_wallets', 0)}\n\n")
                
                # Count active traders and copy decisions
                active_count = 0
                inactive_count = 0
                copy_yes_count = 0
                copy_no_count = 0
                
                for category in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 
                               'swing_traders', 'position_traders', 'consistent', 'mixed', 'unknown']:
                    for wallet in wallet_data.get(category, []):
                        if wallet.get('metrics', {}).get('active_trader', False):
                            active_count += 1
                        else:
                            inactive_count += 1
                        
                        # Calculate copy decision
                        score = wallet.get('composite_score', wallet.get('metrics', {}).get('composite_score', 0))
                        if (score >= 60 and 
                            wallet.get('metrics', {}).get('active_trader', False) and
                            wallet.get('metrics', {}).get('trades_last_7_days', 0) > 0):
                            copy_yes_count += 1
                        else:
                            copy_no_count += 1
                
                f.write(f"🟢 Active traders (7-day): {active_count}\n")
                f.write(f"🔴 Inactive traders: {inactive_count}\n\n")
                
                f.write("COPY DECISIONS:\n")
                f.write(f"✅ Copy YES: {copy_yes_count}\n")
                f.write(f"❌ Copy NO: {copy_no_count}\n\n")
                
                f.write("WALLET TYPE BREAKDOWN:\n")
                f.write(f"🎯 Snipers (<1 min): {len(wallet_data.get('snipers', []))}\n")
                f.write(f"⚡ Flippers (1-10 min): {len(wallet_data.get('flippers', []))}\n")
                f.write(f"📊 Scalpers (10-60 min): {len(wallet_data.get('scalpers', []))}\n")
                f.write(f"💎 Gem Hunters (5x+ focus): {len(wallet_data.get('gem_hunters', []))}\n")
                f.write(f"📈 Swing Traders (1-24h): {len(wallet_data.get('swing_traders', []))}\n")
                f.write(f"🏆 Position Traders (24h+): {len(wallet_data.get('position_traders', []))}\n")
                f.write(f"✅ Consistent: {len(wallet_data.get('consistent', []))}\n")
                f.write(f"🔀 Mixed: {len(wallet_data.get('mixed', []))}\n")
                f.write(f"❓ Unknown: {len(wallet_data.get('unknown', []))}\n\n")
                
                # Combine all wallets and sort by score
                all_wallets = []
                for category in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 
                               'swing_traders', 'position_traders', 'consistent', 'mixed', 'unknown']:
                    all_wallets.extend(wallet_data.get(category, []))
                
                all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
                
                # Filter for active traders only in top 10
                active_wallets = [w for w in all_wallets if w.get('metrics', {}).get('active_trader', False)]
                
                f.write("🏆 TOP 10 ACTIVE TRADERS (7-DAY):\n")
                for i, wallet in enumerate(active_wallets[:10], 1):
                    metrics = wallet['metrics']
                    score = wallet.get('composite_score', metrics.get('composite_score', 0))
                    strategy = wallet.get('strategy', {})
                    
                    # Calculate copy decision
                    copy_decision = "YES" if (
                        score >= 60 and 
                        metrics.get('active_trader', False) and
                        metrics.get('trades_last_7_days', 0) > 0
                    ) else "NO"
                    
                    # Cap profit factor
                    profit_factor = metrics['profit_factor']
                    if profit_factor > 999.99:
                        profit_factor_display = "999.99x"
                    else:
                        profit_factor_display = f"{profit_factor:.2f}x"
                    
                    f.write(f"\n{i}. {wallet['wallet_address'][:8]}...{wallet['wallet_address'][-4:]}\n")
                    f.write(f"   Copy Decision: {copy_decision}\n")
                    f.write(f"   Score: {score:.1f}/100\n")
                    f.write(f"   Type: {wallet['wallet_type']}\n")
                    f.write(f"   7-day trades: {metrics.get('trades_last_7_days', 0)}\n")
                    f.write(f"   7-day win rate: {metrics.get('win_rate_7d', 0):.1f}%\n")
                    f.write(f"   7-day profit: ${metrics.get('profit_7d', 0):.2f}\n")
                    f.write(f"   Profit Factor: {profit_factor_display}\n")
                    f.write(f"   Total Trades: {metrics['total_trades']}\n")
                    f.write(f"   Gem Rate (5x+): {metrics.get('gem_rate_5x_plus', 0):.1f}%\n")
                    f.write(f"   Avg First TP: {metrics.get('avg_first_take_profit_percent', 0):.1f}%\n")
                    f.write(f"   Avg Hold Time: {metrics.get('avg_hold_time_minutes', 0):.1f} minutes\n")
                    f.write(f"   Avg Buy Market Cap: ${metrics.get('avg_buy_market_cap_usd', 0):,.0f}\n")
                    
                    # Strategy info
                    f.write(f"   Follow Sells: {'YES' if strategy.get('follow_sells', False) else 'NO'}\n")
                    f.write(f"   TP1: {strategy.get('tp1_percent', 0)}% | TP2: {strategy.get('tp2_percent', 0)}%\n")
                    f.write(f"   Guidance: {strategy.get('tp_guidance', '')}\n")
                    
                    # Entry/exit analysis
                    if 'entry_exit_analysis' in wallet and wallet['entry_exit_analysis']:
                        ee_analysis = wallet['entry_exit_analysis']
                        f.write(f"   Entry/Exit: {ee_analysis.get('entry_quality', 'UNKNOWN')}/{ee_analysis.get('exit_quality', 'UNKNOWN')}\n")
                        if ee_analysis.get('missed_gains_percent', 0) > 0:
                            f.write(f"   Missed Gains: {ee_analysis.get('missed_gains_percent', 0):.1f}%\n")
                
                # Key insights
                f.write("\n\n📊 KEY INSIGHTS:\n")
                f.write("-" * 40 + "\n")
                
                # Calculate averages for active traders only
                if active_wallets:
                    avg_7d_trades = sum(w['metrics'].get('trades_last_7_days', 0) for w in active_wallets) / len(active_wallets)
                    avg_7d_win_rate = sum(w['metrics'].get('win_rate_7d', 0) for w in active_wallets) / len(active_wallets)
                    avg_gem_rate = sum(w['metrics'].get('gem_rate_5x_plus', 0) for w in active_wallets) / len(active_wallets)
                    
                    f.write(f"Average 7-day trades: {avg_7d_trades:.1f}\n")
                    f.write(f"Average 7-day win rate: {avg_7d_win_rate:.1f}%\n")
                    f.write(f"Average Gem Rate (5x+): {avg_gem_rate:.1f}%\n")
                    
                    # Who hit 5x in last 7 days
                    recent_5x_count = sum(1 for w in active_wallets 
                                        if w.get('seven_day_metrics', {}).get('has_5x_last_7_days', False))
                    if recent_5x_count > 0:
                        f.write(f"\n🚀 {recent_5x_count} wallets hit 5x+ in the last 7 days!\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF 7-DAY ACTIVE TRADER ANALYSIS REPORT\n")
            
        logger.info(f"Successfully generated 7-day active trader report: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating analysis report: {str(e)}")
        return False

def export_distribution_analysis(wallet_data: Dict[str, Any], output_file: str) -> bool:
    """
    Export detailed 7-day distribution analysis for active traders.
    
    Args:
        wallet_data: Wallet analysis results
        output_file: Output CSV file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Combine all wallets
        all_wallets = []
        for category in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 
                        'swing_traders', 'position_traders', 'consistent', 'mixed', 'unknown']:
            all_wallets.extend(wallet_data.get(category, []))
        
        # Sort by composite score
        all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'wallet_address', 'wallet_type', 'composite_score',
                'copy_decision',
                'total_trades', 'trades_last_7_days', 'active_trader',
                'win_rate_7d', 'profit_7d',
                'distribution_sum_%',
                'distribution_500_plus_%', 'distribution_200_500_%',
                'distribution_0_200_%', 'distribution_neg50_0_%',
                'distribution_below_neg50_%',
                'gem_rate_5x_plus_%', 'distribution_quality',
                'has_5x_last_7_days', 'has_2x_last_7_days',
                'avg_buy_market_cap_usd'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for wallet in all_wallets:
                metrics = wallet['metrics']
                seven_day = wallet.get('seven_day_metrics', {})
                score = wallet.get('composite_score', metrics.get('composite_score', 0))
                
                # Calculate copy decision
                copy_decision = "YES" if (
                    score >= 60 and 
                    metrics.get('active_trader', False) and
                    metrics.get('trades_last_7_days', 0) > 0
                ) else "NO"
                
                # Calculate distribution sum to verify it equals 100%
                dist_sum = (
                    metrics.get('distribution_500_plus_%', 0) +
                    metrics.get('distribution_200_500_%', 0) +
                    metrics.get('distribution_0_200_%', 0) +
                    metrics.get('distribution_neg50_0_%', 0) +
                    metrics.get('distribution_below_neg50_%', 0)
                )
                
                # Determine distribution quality
                if abs(dist_sum - 100) < 1:  # Allow 1% tolerance
                    dist_quality = "VERIFIED"
                elif dist_sum == 0:
                    dist_quality = "NO_DATA"
                else:
                    dist_quality = f"ERROR ({dist_sum:.1f}%)"
                
                row = {
                    'wallet_address': wallet['wallet_address'],
                    'wallet_type': wallet['wallet_type'],
                    'composite_score': score,
                    'copy_decision': copy_decision,
                    'total_trades': metrics['total_trades'],
                    'trades_last_7_days': metrics.get('trades_last_7_days', 0),
                    'active_trader': 'YES' if metrics.get('active_trader', False) else 'NO',
                    'win_rate_7d': round(metrics.get('win_rate_7d', 0), 2),
                    'profit_7d': round(metrics.get('profit_7d', 0), 2),
                    'distribution_sum_%': round(dist_sum, 1),
                    'distribution_500_plus_%': metrics.get('distribution_500_plus_%', 0),
                    'distribution_200_500_%': metrics.get('distribution_200_500_%', 0),
                    'distribution_0_200_%': metrics.get('distribution_0_200_%', 0),
                    'distribution_neg50_0_%': metrics.get('distribution_neg50_0_%', 0),
                    'distribution_below_neg50_%': metrics.get('distribution_below_neg50_%', 0),
                    'gem_rate_5x_plus_%': metrics.get('gem_rate_5x_plus', 0),
                    'distribution_quality': dist_quality,
                    'has_5x_last_7_days': 'YES' if seven_day.get('has_5x_last_7_days', False) else 'NO',
                    'has_2x_last_7_days': 'YES' if seven_day.get('has_2x_last_7_days', False) else 'NO',
                    'avg_buy_market_cap_usd': metrics.get('avg_buy_market_cap_usd', 0)
                }
                
                writer.writerow(row)
        
        logger.info(f"Successfully exported 7-day distribution analysis to CSV: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting distribution analysis: {str(e)}")
        return False