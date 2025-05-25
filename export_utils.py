"""
Export Utilities Module - Phoenix Project (ENHANCED 2x+ AND 5x+ EDITION)

Handles Excel and CSV exports for enhanced SpyDefi analysis results.
UPDATES:
- Separate columns for strategy components (recommendation, entry_type, TP levels, SL)
- Tracks both 2x+ and 5x+ metrics
- Separate pullback percentages for 2x and 5x milestones
- Removed detailed_analysis_count and pump_tokens_analyzed from exports
- Enhanced formatting for better readability
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
    Export combined analysis results to Excel with enhanced formatting.
    
    Args:
        telegram_data: Telegram analysis results with 2x+ and 5x+ metrics
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
            
            percent_int_format = workbook.add_format({
                'num_format': '0%',
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
            
            time_format = workbook.add_format({
                'border': 1,
                'align': 'center'
            })
            
            warning_format = workbook.add_format({
                'bg_color': '#ff0000',
                'font_color': 'white',
                'bold': True,
                'border': 1
            })
            
            # Export Enhanced Telegram data if available
            if telegram_data and "ranked_kols" in telegram_data:
                # Prepare data for DataFrame
                telegram_rows = []
                for kol_data in telegram_data["ranked_kols"]:
                    row = {
                        'Channel ID': kol_data.get('channel_id', ''),
                        'Total Calls': kol_data.get('total_calls', 0),
                        '2x Success Rate': kol_data.get('success_rate', 0) / 100,  # Convert to decimal for %
                        '5x Success Rate': kol_data.get('success_rate_5x', 0) / 100,
                        'Avg ROI': kol_data.get('avg_roi', 0),
                        'Avg Max ROI': kol_data.get('avg_max_roi', 0),
                        'Confidence Level': kol_data.get('confidence_level', 0),
                        'Avg Max Pullback %': kol_data.get('avg_max_pullback_percent', 0),
                        'Avg Pullback After 2x %': kol_data.get('avg_max_pullback_percent_2x', 0),
                        'Avg Pullback After 5x %': kol_data.get('avg_max_pullback_percent_5x', 0),
                        'Avg Time to 2x': kol_data.get('avg_time_to_2x_formatted', 'N/A'),
                        'Avg Time to 5x': kol_data.get('avg_time_to_5x_formatted', 'N/A'),
                        'Pump 2x Success Rate': kol_data.get('pump_success_rate_2x', 0) / 100,
                        'Pump 5x Success Rate': kol_data.get('pump_success_rate_5x', 0) / 100,
                        'Recommendation': kol_data.get('recommendation', 'ENHANCED_ANALYSIS'),
                        'Entry Type': kol_data.get('entry_type', 'IMMEDIATE'),
                        'Take Profit 1 %': kol_data.get('take_profit_1', 100),
                        'Take Profit 2 %': kol_data.get('take_profit_2', 300),
                        'Take Profit 3 %': kol_data.get('take_profit_3', 500),
                        'Stop Loss %': kol_data.get('stop_loss', -35)
                    }
                    telegram_rows.append(row)
                
                telegram_df = pd.DataFrame(telegram_rows)
                telegram_df.to_excel(writer, sheet_name='SpyDefi KOLs', index=False)
                
                # Format Telegram sheet
                telegram_sheet = writer.sheets['SpyDefi KOLs']
                
                # Apply header format
                for col_num, value in enumerate(telegram_df.columns.values):
                    telegram_sheet.write(0, col_num, value, header_format)
                
                # Apply conditional formatting for confidence scores
                confidence_col = telegram_df.columns.get_loc('Confidence Level')
                telegram_sheet.conditional_format(1, confidence_col, len(telegram_df), confidence_col, {
                    'type': 'cell',
                    'criteria': '>=',
                    'value': 300,
                    'format': score_excellent_format
                })
                
                telegram_sheet.conditional_format(1, confidence_col, len(telegram_df), confidence_col, {
                    'type': 'cell',
                    'criteria': 'between',
                    'minimum': 200,
                    'maximum': 299.99,
                    'format': score_good_format
                })
                
                telegram_sheet.conditional_format(1, confidence_col, len(telegram_df), confidence_col, {
                    'type': 'cell',
                    'criteria': 'between',
                    'minimum': 100,
                    'maximum': 199.99,
                    'format': score_average_format
                })
                
                telegram_sheet.conditional_format(1, confidence_col, len(telegram_df), confidence_col, {
                    'type': 'cell',
                    'criteria': 'between',
                    'minimum': 50,
                    'maximum': 99.99,
                    'format': score_poor_format
                })
                
                telegram_sheet.conditional_format(1, confidence_col, len(telegram_df), confidence_col, {
                    'type': 'cell',
                    'criteria': '<',
                    'value': 50,
                    'format': score_very_poor_format
                })
                
                # Apply percentage format to percentage columns
                percent_cols = ['2x Success Rate', '5x Success Rate', 'Pump 2x Success Rate', 'Pump 5x Success Rate']
                for col_name in percent_cols:
                    if col_name in telegram_df.columns:
                        col_idx = telegram_df.columns.get_loc(col_name)
                        for row_num in range(1, len(telegram_df) + 1):
                            value = telegram_df.iloc[row_num-1][col_name]
                            telegram_sheet.write(row_num, col_idx, value, percent_format)
                
                # Apply number format to ROI and numeric columns
                number_cols = ['Avg ROI', 'Avg Max ROI', 'Avg Max Pullback %', 
                              'Avg Pullback After 2x %', 'Avg Pullback After 5x %']
                for col_name in number_cols:
                    if col_name in telegram_df.columns:
                        col_idx = telegram_df.columns.get_loc(col_name)
                        for row_num in range(1, len(telegram_df) + 1):
                            value = telegram_df.iloc[row_num-1][col_name]
                            telegram_sheet.write(row_num, col_idx, value, number_format)
                
                # Apply percent integer format to TP and SL columns
                tp_sl_cols = ['Take Profit 1 %', 'Take Profit 2 %', 'Take Profit 3 %', 'Stop Loss %']
                for col_name in tp_sl_cols:
                    if col_name in telegram_df.columns:
                        col_idx = telegram_df.columns.get_loc(col_name)
                        for row_num in range(1, len(telegram_df) + 1):
                            value = telegram_df.iloc[row_num-1][col_name]
                            telegram_sheet.write(row_num, col_idx, value, number_format)
                
                # Set column widths
                telegram_sheet.set_column('A:A', 15)   # Channel ID
                telegram_sheet.set_column('B:B', 12)   # Total calls
                telegram_sheet.set_column('C:D', 15)   # Success rates
                telegram_sheet.set_column('E:F', 12)   # ROI columns
                telegram_sheet.set_column('G:G', 15)   # Confidence
                telegram_sheet.set_column('H:J', 18)   # Pullback columns
                telegram_sheet.set_column('K:L', 15)   # Time columns
                telegram_sheet.set_column('M:N', 18)   # Pump success rates
                telegram_sheet.set_column('O:O', 20)   # Recommendation
                telegram_sheet.set_column('P:P', 12)   # Entry Type
                telegram_sheet.set_column('Q:S', 15)   # TP columns
                telegram_sheet.set_column('T:T', 12)   # Stop Loss
                
                # Add summary sheet
                summary_data = {
                    'Metric': [
                        'Total KOLs Analyzed',
                        'Total Calls Analyzed',
                        'Overall 2x Success Rate',
                        'Overall 5x Success Rate',
                        '',
                        'TOP PERFORMERS',
                        'KOLs with 50%+ 2x Rate',
                        'KOLs with 30%+ 5x Rate',
                        'KOLs with <30% Pullback',
                        '',
                        'TIME METRICS',
                        'Fastest Avg Time to 2x',
                        'Fastest Avg Time to 5x',
                        '',
                        'PUMP.FUN METRICS',
                        'KOLs with Pump Data',
                        'Best Pump 2x Rate',
                        'Best Pump 5x Rate'
                    ],
                    'Value': [
                        telegram_data.get('total_kols_analyzed', len(telegram_rows)),
                        telegram_data.get('total_calls', sum(row['Total Calls'] for row in telegram_rows)),
                        f"{telegram_data.get('success_rate_2x', 0):.2f}%",
                        f"{telegram_data.get('success_rate_5x', 0):.2f}%",
                        '',
                        '',
                        sum(1 for row in telegram_rows if row['2x Success Rate'] >= 0.5),
                        sum(1 for row in telegram_rows if row['5x Success Rate'] >= 0.3),
                        sum(1 for row in telegram_rows if row['Avg Max Pullback %'] < 30),
                        '',
                        '',
                        min((row['Avg Time to 2x'] for row in telegram_rows if row['Avg Time to 2x'] != 'N/A'), default='N/A'),
                        min((row['Avg Time to 5x'] for row in telegram_rows if row['Avg Time to 5x'] != 'N/A'), default='N/A'),
                        '',
                        '',
                        sum(1 for row in telegram_rows if row['Pump 2x Success Rate'] > 0),
                        f"{max((row['Pump 2x Success Rate'] * 100 for row in telegram_rows), default=0):.1f}%",
                        f"{max((row['Pump 5x Success Rate'] * 100 for row in telegram_rows), default=0):.1f}%"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format summary sheet
                summary_sheet = writer.sheets['Summary']
                for col_num, value in enumerate(summary_df.columns.values):
                    summary_sheet.write(0, col_num, value, header_format)
                
                summary_sheet.set_column('A:A', 30)
                summary_sheet.set_column('B:B', 20)
            
            # Export Wallet data if available
            if wallet_data:
                # Create wallet analysis sheet similar to memecoin edition
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
                        
                        # Cap profit factor at 999.99
                        profit_factor = metrics.get('profit_factor', 0)
                        if profit_factor > 999.99:
                            profit_factor = 999.99
                        
                        # Build row data
                        row = {
                            'Rank': rank,
                            'Wallet': wallet['wallet_address'],
                            'Score': composite_score,
                            'Rating': rating,
                            'Type': wallet['wallet_type'],
                            'Trades': metrics['total_trades'],
                            'Win Rate': metrics['win_rate'] / 100,  # For percentage formatting
                            'Profit Factor': profit_factor,
                            'Net Profit': metrics['net_profit_usd'],
                            'Avg ROI': metrics['avg_roi'] / 100,
                            'Max ROI': metrics['max_roi'] / 100,
                            'Gem Rate (2x+)': metrics.get('gem_rate_2x_plus', 0) / 100,
                            'Gem Rate (5x+)': metrics.get('gem_rate_5x_plus', 0) / 100,
                            'Hold Time (min)': metrics.get('avg_hold_time_minutes', 0),
                            'Avg First TP %': metrics.get('avg_first_take_profit_percent', 0) / 100,
                            'Strategy': strategy.get('recommendation', ''),
                            'MC Min': strategy.get('filter_market_cap_min', 0),
                            'MC Max': strategy.get('filter_market_cap_max', 0)
                        }
                        
                        # Add entry/exit analysis if available
                        if 'entry_exit_analysis' in wallet and wallet['entry_exit_analysis']:
                            ee_analysis = wallet['entry_exit_analysis']
                            row['Entry Quality'] = ee_analysis.get('entry_quality', '')
                            row['Exit Quality'] = ee_analysis.get('exit_quality', '')
                            row['Pattern'] = ee_analysis.get('pattern', '')
                            row['Missed Gains %'] = ee_analysis.get('missed_gains_percent', 0) / 100
                            row['Avg Exit ROI'] = ee_analysis.get('avg_exit_roi', 0) / 100
                        
                        # Add bundle warning
                        if 'bundle_analysis' in wallet and wallet['bundle_analysis'].get('is_likely_bundler'):
                            row['Warning'] = '⚠️ BUNDLER'
                        else:
                            row['Warning'] = ''
                        
                        # Add distribution data
                        row['Dist 500%+'] = metrics.get('distribution_500_plus_%', 0) / 100
                        row['Dist 200-500%'] = metrics.get('distribution_200_500_%', 0) / 100
                        row['Dist 0-200%'] = metrics.get('distribution_0_200_%', 0) / 100
                        row['Dist -50-0%'] = metrics.get('distribution_neg50_0_%', 0) / 100
                        row['Dist <-50%'] = metrics.get('distribution_below_neg50_%', 0) / 100
                        
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
                    percent_cols = ['Win Rate', 'Avg ROI', 'Max ROI', 'Gem Rate (2x+)', 'Gem Rate (5x+)',
                                   'Avg First TP %', 'Missed Gains %', 'Avg Exit ROI',
                                   'Dist 500%+', 'Dist 200-500%', 'Dist 0-200%', 
                                   'Dist -50-0%', 'Dist <-50%']
                    
                    for row_num in range(1, len(wallet_df) + 1):
                        for col_name in percent_cols:
                            if col_name in wallet_df.columns:
                                col_idx = wallet_df.columns.get_loc(col_name)
                                value = wallet_df.iloc[row_num-1][col_name]
                                wallet_sheet.write(row_num, col_idx, value, percent_format)
                    
                    # Apply money format to profit and market cap columns
                    money_cols = ['Net Profit', 'MC Min', 'MC Max']
                    for row_num in range(1, len(wallet_df) + 1):
                        for col_name in money_cols:
                            if col_name in wallet_df.columns:
                                col_idx = wallet_df.columns.get_loc(col_name)
                                value = wallet_df.iloc[row_num-1][col_name]
                                wallet_sheet.write(row_num, col_idx, value, money_format)
                    
                    # Apply warning format to bundler warnings
                    if 'Warning' in wallet_df.columns:
                        warning_col = wallet_df.columns.get_loc('Warning')
                        for row_num in range(1, len(wallet_df) + 1):
                            if wallet_df.iloc[row_num-1]['Warning']:
                                wallet_sheet.write(row_num, warning_col, 
                                                 wallet_df.iloc[row_num-1]['Warning'], 
                                                 warning_format)
                    
                    # Set column widths
                    wallet_sheet.set_column('A:A', 8)   # Rank
                    wallet_sheet.set_column('B:B', 50)  # Wallet
                    wallet_sheet.set_column('C:C', 10)  # Score
                    wallet_sheet.set_column('D:D', 12)  # Rating
                    wallet_sheet.set_column('E:E', 15)  # Type
                    wallet_sheet.set_column('F:F', 10)  # Trades
                    wallet_sheet.set_column('G:G', 12)  # Win Rate
                    wallet_sheet.set_column('H:H', 15)  # Profit Factor
                    wallet_sheet.set_column('I:I', 15)  # Net Profit
                    wallet_sheet.set_column('J:K', 12)  # ROI columns
                    wallet_sheet.set_column('L:M', 15)  # Gem Rate columns
                    wallet_sheet.set_column('N:N', 15)  # Hold Time
                    wallet_sheet.set_column('O:O', 15)  # First TP
                    wallet_sheet.set_column('P:P', 20)  # Strategy
                    wallet_sheet.set_column('Q:R', 15)  # Market Cap filters
            
            logger.info(f"Successfully exported enhanced analysis to Excel: {output_file}")
            return True
            
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        return False

def export_wallet_rankings_csv(wallet_data: Dict[str, Any], output_file: str) -> bool:
    """
    Export memecoin wallet rankings to CSV with all metrics.
    
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
            fieldnames = [
                'rank', 'wallet_address', 'composite_score', 'score_rating',
                'wallet_type', 'total_trades', 'win_rate', 'profit_factor',
                'net_profit_usd', 'avg_roi', 'median_roi', 'max_roi',
                'avg_hold_time_minutes', 'avg_hold_time_seconds',
                'total_tokens_traded',
                'distribution_500_plus_%', 'distribution_200_500_%',
                'distribution_0_200_%', 'distribution_neg50_0_%',
                'distribution_below_neg50_%',
                'gem_rate_2x_plus_%', 'gem_rate_5x_plus_%',
                'avg_buy_market_cap_usd',
                'avg_buy_amount_usd', 'avg_first_take_profit_percent',
                'entry_exit_pattern', 'entry_quality', 'exit_quality',
                'missed_gains_percent', 'early_exit_rate', 'avg_exit_roi',
                'hold_pattern', 'strategy_recommendation', 'confidence',
                'filter_market_cap_min', 'filter_market_cap_max',
                'is_likely_bundler', 'bundle_indicators',
                'estimated_copytraders', 'suggested_slippage', 'suggested_gas'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for rank, analysis in enumerate(all_wallets, 1):
                metrics = analysis['metrics']
                score = analysis.get('composite_score', metrics.get('composite_score', 0))
                strategy = analysis.get('strategy', {})
                
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
                
                # Cap profit factor at 999.99
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
                    'median_roi': round(metrics.get('median_roi', 0), 2),
                    'max_roi': round(metrics['max_roi'], 2),
                    'avg_hold_time_minutes': round(metrics.get('avg_hold_time_minutes', 0), 2),
                    'avg_hold_time_seconds': round(metrics.get('avg_hold_time_seconds', 0), 2),
                    'total_tokens_traded': metrics['total_tokens_traded'],
                    'distribution_500_plus_%': metrics.get('distribution_500_plus_%', 0),
                    'distribution_200_500_%': metrics.get('distribution_200_500_%', 0),
                    'distribution_0_200_%': metrics.get('distribution_0_200_%', 0),
                    'distribution_neg50_0_%': metrics.get('distribution_neg50_0_%', 0),
                    'distribution_below_neg50_%': metrics.get('distribution_below_neg50_%', 0),
                    'gem_rate_2x_plus_%': metrics.get('gem_rate_2x_plus', 0),
                    'gem_rate_5x_plus_%': metrics.get('gem_rate_5x_plus', 0),
                    'avg_buy_market_cap_usd': metrics.get('avg_buy_market_cap_usd', 0),
                    'avg_buy_amount_usd': metrics.get('avg_buy_amount_usd', 0),
                    'avg_first_take_profit_percent': metrics.get('avg_first_take_profit_percent', 0),
                    'strategy_recommendation': strategy.get('recommendation', ''),
                    'confidence': strategy.get('confidence', ''),
                    'filter_market_cap_min': strategy.get('filter_market_cap_min', 0),
                    'filter_market_cap_max': strategy.get('filter_market_cap_max', 0),
                    'suggested_slippage': strategy.get('suggested_slippage', 15),
                    'suggested_gas': strategy.get('suggested_gas', 'medium')
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
                
                # Add bundle detection if available
                if 'bundle_analysis' in analysis and analysis['bundle_analysis']:
                    bundle = analysis['bundle_analysis']
                    row['is_likely_bundler'] = 'YES' if bundle.get('is_likely_bundler') else 'NO'
                    row['bundle_indicators'] = bundle.get('bundle_indicators', 0)
                    row['estimated_copytraders'] = bundle.get('estimated_copytraders', '0-5')
                else:
                    row['is_likely_bundler'] = 'NO'
                    row['bundle_indicators'] = 0
                    row['estimated_copytraders'] = 'UNKNOWN'
                
                writer.writerow(row)
        
        logger.info(f"Successfully exported memecoin wallet rankings to CSV: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting wallet rankings to CSV: {str(e)}")
        return False

def generate_memecoin_analysis_report(telegram_data: Dict[str, Any], 
                                    wallet_data: Dict[str, Any], 
                                    output_file: str) -> bool:
    """
    Generate a comprehensive memecoin analysis report.
    
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
            f.write("PHOENIX PROJECT - ENHANCED MEMECOIN ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Enhanced Telegram Analysis Section
            if telegram_data and "ranked_kols" in telegram_data:
                f.write("📱 ENHANCED TELEGRAM ANALYSIS (SPYDEFI) - 2x+ AND 5x+ TRACKING\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total KOLs analyzed: {telegram_data.get('total_kols_analyzed', 0)}\n")
                f.write(f"Total calls analyzed: {telegram_data.get('total_calls', 0)}\n")
                f.write(f"2x success rate: {telegram_data.get('success_rate_2x', 0):.2f}%\n")
                f.write(f"5x success rate: {telegram_data.get('success_rate_5x', 0):.2f}%\n\n")
                
                f.write("Top 5 KOLs (by composite score):\n")
                ranked_kols = telegram_data.get('ranked_kols', [])
                for i, kol_data in enumerate(ranked_kols[:5], 1):
                    channel_id = kol_data.get('channel_id', 'Unknown')
                    f.write(f"{i}. Channel ID: {channel_id}\n")
                    f.write(f"   Composite Score: {kol_data.get('confidence_level', 0):.1f}\n")
                    f.write(f"   Success Rate (2x): {kol_data.get('success_rate', 0):.1f}%\n")
                    f.write(f"   Success Rate (5x): {kol_data.get('success_rate_5x', 0):.1f}%\n")
                    f.write(f"   Avg Max Pullback: {kol_data.get('avg_max_pullback_percent', 0):.1f}%\n")
                    f.write(f"   Avg Pullback After 2x: {kol_data.get('avg_max_pullback_percent_2x', 0):.1f}%\n")
                    f.write(f"   Avg Pullback After 5x: {kol_data.get('avg_max_pullback_percent_5x', 0):.1f}%\n")
                    f.write(f"   Avg Time to 2x: {kol_data.get('avg_time_to_2x_formatted', 'N/A')}\n")
                    f.write(f"   Avg Time to 5x: {kol_data.get('avg_time_to_5x_formatted', 'N/A')}\n")
                    if kol_data.get('pump_success_rate_2x', 0) > 0:
                        f.write(f"   Pump.fun 2x Rate: {kol_data.get('pump_success_rate_2x', 0):.1f}%\n")
                        f.write(f"   Pump.fun 5x Rate: {kol_data.get('pump_success_rate_5x', 0):.1f}%\n")
                    f.write("\n")
            
            # Wallet Analysis Section
            if wallet_data and wallet_data.get('success'):
                f.write("\n💰 MEMECOIN WALLET ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total wallets: {wallet_data.get('total_wallets', 0)}\n")
                f.write(f"Successfully analyzed: {wallet_data.get('analyzed_wallets', 0)}\n")
                f.write(f"Failed analysis: {wallet_data.get('failed_wallets', 0)}\n\n")
                
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
                
                f.write("🏆 TOP 10 MEMECOIN TRADERS:\n")
                for i, wallet in enumerate(all_wallets[:10], 1):
                    metrics = wallet['metrics']
                    score = wallet.get('composite_score', metrics.get('composite_score', 0))
                    strategy = wallet.get('strategy', {})
                    
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
                    f.write(f"   Score: {score:.1f}/100 {rating}\n")
                    f.write(f"   Type: {wallet['wallet_type']}\n")
                    f.write(f"   Win Rate: {metrics['win_rate']:.1f}%\n")
                    f.write(f"   Profit Factor: {profit_factor_display}\n")
                    f.write(f"   Net Profit: ${metrics['net_profit_usd']:.2f}\n")
                    f.write(f"   Total Trades: {metrics['total_trades']}\n")
                    f.write(f"   Gem Rate (2x+): {metrics.get('gem_rate_2x_plus', 0):.1f}%\n")
                    f.write(f"   Gem Rate (5x+): {metrics.get('gem_rate_5x_plus', 0):.1f}%\n")
                    f.write(f"   Avg First TP: {metrics.get('avg_first_take_profit_percent', 0):.1f}%\n")
                    f.write(f"   Avg Hold Time: {metrics.get('avg_hold_time_minutes', 0):.1f} minutes\n")
                    
                    # Market cap range
                    mc_min = strategy.get('filter_market_cap_min', 0)
                    mc_max = strategy.get('filter_market_cap_max', 0)
                    if mc_min >= 1000000:
                        mc_min_str = f"${mc_min/1000000:.1f}M"
                    elif mc_min >= 1000:
                        mc_min_str = f"${mc_min/1000:.0f}K"
                    else:
                        mc_min_str = f"${mc_min:.0f}"
                    
                    if mc_max >= 1000000:
                        mc_max_str = f"${mc_max/1000000:.1f}M"
                    elif mc_max >= 1000:
                        mc_max_str = f"${mc_max/1000:.0f}K"
                    else:
                        mc_max_str = f"${mc_max:.0f}"
                    
                    f.write(f"   Market Cap Range: {mc_min_str} - {mc_max_str}\n")
                    
                    # Entry/exit analysis
                    if 'entry_exit_analysis' in wallet and wallet['entry_exit_analysis']:
                        ee_analysis = wallet['entry_exit_analysis']
                        f.write(f"   Entry/Exit: {ee_analysis.get('entry_quality', 'UNKNOWN')}/{ee_analysis.get('exit_quality', 'UNKNOWN')}\n")
                        f.write(f"   Pattern: {ee_analysis.get('pattern', 'UNKNOWN')}\n")
                        if ee_analysis.get('missed_gains_percent', 0) > 0:
                            f.write(f"   Missed Gains: {ee_analysis.get('missed_gains_percent', 0):.1f}%\n")
                    
                    # Bundle warning
                    if 'bundle_analysis' in wallet and wallet['bundle_analysis'].get('is_likely_bundler'):
                        f.write(f"   ⚠️ WARNING: Possible bundler detected!\n")
                    
                    f.write(f"   Strategy: {strategy.get('recommendation', '')} ({strategy.get('confidence', '')})\n")
                
                # Key insights
                f.write("\n\n📊 KEY INSIGHTS:\n")
                f.write("-" * 40 + "\n")
                
                # Calculate averages
                if all_wallets:
                    avg_win_rate = sum(w['metrics']['win_rate'] for w in all_wallets) / len(all_wallets)
                    avg_gem_rate_2x = sum(w['metrics'].get('gem_rate_2x_plus', 0) for w in all_wallets) / len(all_wallets)
                    avg_gem_rate_5x = sum(w['metrics'].get('gem_rate_5x_plus', 0) for w in all_wallets) / len(all_wallets)
                    avg_hold_time = sum(w['metrics'].get('avg_hold_time_minutes', 0) for w in all_wallets) / len(all_wallets)
                    
                    f.write(f"Average Win Rate: {avg_win_rate:.1f}%\n")
                    f.write(f"Average Gem Rate (2x+): {avg_gem_rate_2x:.1f}%\n")
                    f.write(f"Average Gem Rate (5x+): {avg_gem_rate_5x:.1f}%\n")
                    f.write(f"Average Hold Time: {avg_hold_time:.1f} minutes\n")
                    
                    # Distribution of wallet types
                    total_analyzed = wallet_data['analyzed_wallets']
                    if total_analyzed > 0:
                        gem_hunters = len(wallet_data.get('gem_hunters', []))
                        flippers = len(wallet_data.get('flippers', []))
                        f.write(f"\n{(gem_hunters/total_analyzed*100):.1f}% are gem hunters (5x+ focused)\n")
                        f.write(f"{(flippers/total_analyzed*100):.1f}% are quick flippers (exit within 10 min)\n")
                    
                    # Bundle detection
                    bundlers = sum(1 for w in all_wallets if 'bundle_analysis' in w and w['bundle_analysis'].get('is_likely_bundler'))
                    if bundlers > 0:
                        f.write(f"\n⚠️ {bundlers} potential bundlers detected - verify on-chain before copying\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF ENHANCED MEMECOIN ANALYSIS REPORT\n")
            
        logger.info(f"Successfully generated enhanced memecoin analysis report: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating memecoin analysis report: {str(e)}")
        return False

def export_distribution_analysis(wallet_data: Dict[str, Any], output_file: str) -> bool:
    """
    Export detailed distribution analysis for memecoin wallets.
    
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
                'total_trades', 'distribution_sum_%',
                'distribution_500_plus_%', 'distribution_200_500_%',
                'distribution_0_200_%', 'distribution_neg50_0_%',
                'distribution_below_neg50_%',
                'gem_rate_2x_plus_%', 'gem_rate_5x_plus_%', 
                'distribution_quality'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for wallet in all_wallets:
                metrics = wallet['metrics']
                
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
                    'composite_score': wallet.get('composite_score', metrics.get('composite_score', 0)),
                    'total_trades': metrics['total_trades'],
                    'distribution_sum_%': round(dist_sum, 1),
                    'distribution_500_plus_%': metrics.get('distribution_500_plus_%', 0),
                    'distribution_200_500_%': metrics.get('distribution_200_500_%', 0),
                    'distribution_0_200_%': metrics.get('distribution_0_200_%', 0),
                    'distribution_neg50_0_%': metrics.get('distribution_neg50_0_%', 0),
                    'distribution_below_neg50_%': metrics.get('distribution_below_neg50_%', 0),
                    'gem_rate_2x_plus_%': metrics.get('gem_rate_2x_plus', 0),
                    'gem_rate_5x_plus_%': metrics.get('gem_rate_5x_plus', 0),
                    'distribution_quality': dist_quality
                }
                
                writer.writerow(row)
        
        logger.info(f"Successfully exported distribution analysis to CSV: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting distribution analysis: {str(e)}")
        return False