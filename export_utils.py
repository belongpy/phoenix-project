"""
Export Utilities Module - Phoenix Project (FIXED VERSION)

MAJOR FIXES:
- Column order: channel_id next to kol  
- Removed: success_rate_percent, total_roi_percent, max_roi_percent
- Kept: follower_tier for strategy classification
- Reordered: success_rate_5x after avg_max_pullback_percent
- Updated all export functions with fixed column order
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

def export_spydefi_to_csv(results: Dict[str, Any], output_file: str) -> bool:
    """
    Export SPYDEFI KOL analysis results to CSV with FIXED column order.
    
    Column Order: rank, kol, channel_id, composite_score, copy_recommendation, 
                  strategy_classification, follower_tier, total_calls, winning_calls, 
                  losing_calls, success_rate_2x_percent, tokens_2x_plus, tokens_5x_plus, 
                  avg_time_to_2x_hours, avg_max_pullback_percent, success_rate_5x_percent, 
                  consistency_score, avg_roi
    
    Args:
        results: SPYDEFI analysis results
        output_file: Output CSV file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        kol_performances = results.get('kol_performances', {})
        
        if not kol_performances:
            logger.warning("No KOL performance data to export")
            return False
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Prepare CSV data
        csv_data = []
        
        for kol, performance in kol_performances.items():
            # Handle both dict and object formats
            if isinstance(performance, dict):
                data = performance
            else:
                # Convert object to dict
                data = {
                    'kol': performance.kol,
                    'channel_id': performance.channel_id,
                    'follower_tier': performance.follower_tier,
                    'total_calls': performance.total_calls,
                    'winning_calls': performance.winning_calls,
                    'losing_calls': performance.losing_calls,
                    'success_rate_2x': performance.success_rate_2x,
                    'success_rate_5x': performance.success_rate_5x,
                    'tokens_2x_plus': performance.tokens_2x_plus,
                    'tokens_5x_plus': performance.tokens_5x_plus,
                    'avg_time_to_2x_hours': performance.avg_time_to_2x_hours,
                    'avg_max_pullback_percent': performance.avg_max_pullback_percent,
                    'consistency_score': performance.consistency_score,
                    'composite_score': performance.composite_score,
                    'strategy_classification': performance.strategy_classification,
                    'avg_roi': performance.avg_roi
                }
            
            # Prepare CSV row with FIXED column order
            row = {
                'rank': len(csv_data) + 1,
                'kol': f"@{data.get('kol', kol)}",
                'channel_id': data.get('channel_id', 'N/A'),  # Real numeric channel ID
                'composite_score': round(data.get('composite_score', 0), 1),
                'copy_recommendation': 'COPY' if data.get('composite_score', 0) >= 70 else 'AVOID',
                'strategy_classification': data.get('strategy_classification', 'UNKNOWN'),
                'follower_tier': data.get('follower_tier', 'LOW'),
                'total_calls': data.get('total_calls', 0),
                'winning_calls': data.get('winning_calls', 0),
                'losing_calls': data.get('losing_calls', 0),
                'success_rate_2x_percent': round(data.get('success_rate_2x', 0), 2),
                'tokens_2x_plus': data.get('tokens_2x_plus', 0),
                'tokens_5x_plus': data.get('tokens_5x_plus', 0),
                'avg_time_to_2x_hours': round(data.get('avg_time_to_2x_hours', 0), 2),
                'avg_max_pullback_percent': round(data.get('avg_max_pullback_percent', 0), 2),
                'success_rate_5x_percent': round(data.get('success_rate_5x', 0), 2),
                'consistency_score': round(data.get('consistency_score', 0), 1),
                'avg_roi': round(data.get('avg_roi', 0), 2)
            }
            
            csv_data.append(row)
        
        # Sort by composite score
        csv_data.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Update ranks
        for i, row in enumerate(csv_data, 1):
            row['rank'] = i
        
        # Write CSV with FIXED column order
        if csv_data:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'rank', 'kol', 'channel_id', 'composite_score', 'copy_recommendation',
                    'strategy_classification', 'follower_tier', 'total_calls', 'winning_calls', 
                    'losing_calls', 'success_rate_2x_percent', 'tokens_2x_plus', 'tokens_5x_plus',
                    'avg_time_to_2x_hours', 'avg_max_pullback_percent', 'success_rate_5x_percent', 
                    'consistency_score', 'avg_roi'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        logger.info(f"✅ Exported {len(csv_data)} KOLs to SPYDEFI CSV: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting SPYDEFI CSV: {str(e)}")
        return False

def export_spydefi_summary_txt(results: Dict[str, Any], output_file: str) -> bool:
    """
    Export SPYDEFI analysis summary to TXT file with FIXED logic.
    
    Args:
        results: SPYDEFI analysis results
        output_file: Output TXT file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        kol_performances = results.get('kol_performances', {})
        metadata = results.get('metadata', {})
        kol_mentions = results.get('kol_mentions', {})
        
        if not kol_performances:
            logger.warning("No KOL performance data for TXT export")
            return False
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert to list and sort by composite score
        kol_list = []
        for kol, performance in kol_performances.items():
            if isinstance(performance, dict):
                data = performance.copy()
                data['kol'] = kol
            else:
                data = {
                    'kol': kol,
                    'composite_score': performance.composite_score,
                    'strategy_classification': performance.strategy_classification,
                    'follower_tier': performance.follower_tier,
                    'channel_id': performance.channel_id,
                    'total_calls': performance.total_calls,
                    'winning_calls': performance.winning_calls,
                    'success_rate_2x': performance.success_rate_2x,
                    'success_rate_5x': performance.success_rate_5x,
                    'tokens_2x_plus': performance.tokens_2x_plus,
                    'tokens_5x_plus': performance.tokens_5x_plus,
                    'avg_time_to_2x_hours': performance.avg_time_to_2x_hours,
                    'avg_max_pullback_percent': performance.avg_max_pullback_percent,
                    'avg_roi': performance.avg_roi
                }
            kol_list.append(data)
        
        kol_list.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        
        # Write comprehensive TXT summary
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PHOENIX PROJECT - SPYDEFI KOL ANALYSIS SUMMARY (FIXED VERSION)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Overall Statistics
            f.write("📊 OVERALL ANALYSIS STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Analysis Date: {metadata.get('timestamp', 'Unknown')}\n")
            f.write(f"KOLs Analyzed: {len(kol_performances)}\n")
            f.write(f"Total Calls Tracked: {metadata.get('total_calls_analyzed', 0)}\n")
            f.write(f"Overall 2x Rate: {metadata.get('overall_2x_rate', 0):.2f}%\n")
            f.write(f"Overall 5x Rate: {metadata.get('overall_5x_rate', 0):.2f}%\n")
            f.write(f"Processing Time: {metadata.get('processing_time_seconds', 0):.1f} seconds\n")
            f.write(f"API Calls Made: {metadata.get('api_calls', 0)}\n")
            f.write(f"Version: v4.0 (Fixed Logic, Real Channel IDs, Fixed Pullbacks)\n\n")
            
            # Configuration Used
            config = metadata.get('config', {})
            f.write("⚙️ ANALYSIS CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"SpyDefi Scan Hours: {config.get('spydefi_scan_hours', 8)}\n")
            f.write(f"KOL Analysis Days: {config.get('kol_analysis_days', 7)}\n")
            f.write(f"Top KOLs Count: {config.get('top_kols_count', 50)}\n")
            f.write(f"Min Mentions Required: {config.get('min_mentions', 1)}\n")
            f.write(f"Max Market Cap: ${config.get('max_market_cap_usd', 10000000):,}\n")
            f.write(f"Win Threshold: {config.get('win_threshold_percent', 50)}%\n")
            f.write(f"Losing Call Definition: >-50% pullback\n\n")
            
            # Top 10 KOLs Detailed Analysis
            f.write("🏆 TOP 10 KOLS - DETAILED ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            top_10 = kol_list[:10]
            for i, kol_data in enumerate(top_10, 1):
                kol = kol_data['kol']
                spydefi_mentions = kol_mentions.get(kol, 0)
                
                f.write(f"\n{i}. @{kol}\n")
                f.write(f"   🎭 SpyDefi Mentions: {spydefi_mentions}\n")
                f.write(f"   📊 Composite Score: {kol_data.get('composite_score', 0):.1f}/100\n")
                f.write(f"   🎯 Strategy: {kol_data.get('strategy_classification', 'UNKNOWN')}\n")
                f.write(f"   👥 Follower Tier: {kol_data.get('follower_tier', 'LOW')}\n")
                f.write(f"   📱 Channel ID: {kol_data.get('channel_id', 'N/A')}\n")
                f.write(f"   📞 Total Calls: {kol_data.get('total_calls', 0)}\n")
                f.write(f"   ✅ Winning Calls: {kol_data.get('winning_calls', 0)}\n")
                f.write(f"   ❌ Losing Calls: {kol_data.get('losing_calls', 0)} (>-50% pullback)\n")
                f.write(f"   💎 2x Rate: {kol_data.get('success_rate_2x', 0):.1f}%\n")
                f.write(f"   🚀 5x Rate: {kol_data.get('success_rate_5x', 0):.1f}%\n")
                f.write(f"   ⏱️ Avg Time to 2x: {kol_data.get('avg_time_to_2x_hours', 0):.1f} hours\n")
                f.write(f"   📉 Avg Max Pullback: {kol_data.get('avg_max_pullback_percent', 0):.1f}%\n")
                f.write(f"   📊 Avg ROI: {kol_data.get('avg_roi', 0):.1f}%\n")
                
                # Copy recommendation
                score = kol_data.get('composite_score', 0)
                if score >= 80:
                    f.write(f"   🟢 RECOMMENDATION: STRONG COPY (Elite Performer)\n")
                elif score >= 70:
                    f.write(f"   🟡 RECOMMENDATION: COPY (Good Performer)\n")
                elif score >= 60:
                    f.write(f"   🟠 RECOMMENDATION: MONITOR (Average Performer)\n")
                else:
                    f.write(f"   🔴 RECOMMENDATION: AVOID (Poor Performer)\n")
            
            # Strategy Breakdown
            f.write(f"\n\n📈 STRATEGY CLASSIFICATION BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            
            scalp_kols = [k for k in kol_list if k.get('strategy_classification') == 'SCALP']
            hold_kols = [k for k in kol_list if k.get('strategy_classification') == 'HOLD']
            mixed_kols = [k for k in kol_list if k.get('strategy_classification') == 'MIXED']
            
            f.write(f"🏃 SCALP Strategy KOLs: {len(scalp_kols)}\n")
            if scalp_kols:
                for kol_data in scalp_kols[:5]:  # Top 5 scalp KOLs
                    f.write(f"   • @{kol_data['kol']} (Score: {kol_data.get('composite_score', 0):.1f}, "
                           f"Channel: {kol_data.get('channel_id', 'N/A')})\n")
            
            f.write(f"\n💎 HOLD Strategy KOLs: {len(hold_kols)}\n")
            if hold_kols:
                for kol_data in hold_kols[:5]:  # Top 5 hold KOLs
                    f.write(f"   • @{kol_data['kol']} (Score: {kol_data.get('composite_score', 0):.1f}, "
                           f"5x Rate: {kol_data.get('success_rate_5x', 0):.1f}%)\n")
            
            if mixed_kols:
                f.write(f"\n🔄 Mixed Strategy KOLs: {len(mixed_kols)}\n")
            
            # Follower Tier Analysis
            f.write(f"\n\n👥 FOLLOWER TIER ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            high_tier = [k for k in kol_list if k.get('follower_tier') == 'HIGH']
            medium_tier = [k for k in kol_list if k.get('follower_tier') == 'MEDIUM']
            low_tier = [k for k in kol_list if k.get('follower_tier') == 'LOW']
            
            f.write(f"🔥 HIGH Tier: {len(high_tier)} KOLs\n")
            f.write(f"   • Best for scalping opportunities\n")
            f.write(f"   • High volume potential\n")
            if high_tier:
                avg_score = sum(k.get('composite_score', 0) for k in high_tier) / len(high_tier)
                f.write(f"   • Average Score: {avg_score:.1f}\n")
            
            f.write(f"\n📊 MEDIUM Tier: {len(medium_tier)} KOLs\n")
            f.write(f"   • Balanced opportunity\n")
            f.write(f"   • Good for both scalp and hold\n")
            if medium_tier:
                avg_score = sum(k.get('composite_score', 0) for k in medium_tier) / len(medium_tier)
                f.write(f"   • Average Score: {avg_score:.1f}\n")
            
            f.write(f"\n🔍 LOW Tier: {len(low_tier)} KOLs\n")
            f.write(f"   • Early alpha potential\n")
            f.write(f"   • Higher risk, higher reward\n")
            if low_tier:
                avg_score = sum(k.get('composite_score', 0) for k in low_tier) / len(low_tier)
                f.write(f"   • Average Score: {avg_score:.1f}\n")
            
            # Key Insights and Recommendations
            f.write(f"\n\n💡 KEY INSIGHTS & RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            # Best performers
            elite_kols = [k for k in kol_list if k.get('composite_score', 0) >= 80]
            good_kols = [k for k in kol_list if 70 <= k.get('composite_score', 0) < 80]
            
            f.write(f"🌟 Elite Performers (80+ score): {len(elite_kols)} KOLs\n")
            if elite_kols:
                f.write(f"   • Top recommendation for copy trading\n")
                f.write(f"   • Consistent high performance\n")
                for kol_data in elite_kols:
                    f.write(f"   • @{kol_data['kol']} ({kol_data.get('composite_score', 0):.1f} score, Channel: {kol_data.get('channel_id', 'N/A')})\n")
            
            f.write(f"\n⭐ Good Performers (70-79 score): {len(good_kols)} KOLs\n")
            if good_kols:
                f.write(f"   • Solid copy trading candidates\n")
                f.write(f"   • Monitor for consistency\n")
            
            # Gem finders
            gem_finders = [k for k in kol_list if k.get('success_rate_5x', 0) >= 20]  # 20%+ 5x rate
            if gem_finders:
                f.write(f"\n💎 Top Gem Finders (20%+ 5x rate): {len(gem_finders)} KOLs\n")
                for kol_data in gem_finders[:3]:
                    f.write(f"   • @{kol_data['kol']} ({kol_data.get('success_rate_5x', 0):.1f}% 5x rate, Channel: {kol_data.get('channel_id', 'N/A')})\n")
            
            # Fast movers
            fast_movers = [k for k in kol_list if k.get('avg_time_to_2x_hours', 0) > 0 and k.get('avg_time_to_2x_hours', 0) <= 6]
            if fast_movers:
                f.write(f"\n⚡ Fastest to 2x (≤6h average): {len(fast_movers)} KOLs\n")
                for kol_data in fast_movers[:3]:
                    f.write(f"   • @{kol_data['kol']} ({kol_data.get('avg_time_to_2x_hours', 0):.1f}h average, Channel: {kol_data.get('channel_id', 'N/A')})\n")
            
            # Final recommendations
            f.write(f"\n\n🎯 FINAL COPY TRADING STRATEGY\n")
            f.write("-" * 40 + "\n")
            f.write(f"1. PRIMARY TARGETS: Elite performers (80+ score)\n")
            f.write(f"2. SECONDARY TARGETS: Good performers (70-79 score)\n")
            f.write(f"3. SCALP STRATEGY: Focus on SCALP classified KOLs for quick trades\n")
            f.write(f"4. HOLD STRATEGY: Focus on gem finders for longer positions\n")
            f.write(f"5. TIER STRATEGY: Mix HIGH/MEDIUM/LOW tiers for diversification\n")
            f.write(f"6. DIVERSIFICATION: Follow 3-5 KOLs across different strategies\n")
            f.write(f"7. RISK MANAGEMENT: Set stop losses and take profits\n")
            f.write(f"8. MONITORING: Track performance and adjust portfolio\n")
            f.write(f"9. CHANNEL ACCESS: Use numeric Channel IDs to find real channels\n")
            f.write(f"10. FIXED LOGIC: losing_calls = >-50% pullback, winning_calls = remainder\n")
            
            f.write(f"\n" + "=" * 80 + "\n")
            f.write("END OF SPYDEFI ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"✅ Exported SPYDEFI summary to TXT: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error exporting SPYDEFI TXT summary: {str(e)}")
        return False

def export_to_excel(telegram_data: Dict[str, Any], wallet_data: Dict[str, Any], 
                   output_file: str) -> bool:
    """
    Export comprehensive analysis results to Excel with FIXED formatting.
    
    Args:
        telegram_data: SPYDEFI analysis results
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
            
            percent_format = workbook.add_format({
                'num_format': '0.00%',
                'border': 1
            })
            
            number_format = workbook.add_format({
                'num_format': '#,##0.00',
                'border': 1
            })
            
            # Export SPYDEFI data if available
            if telegram_data and telegram_data.get('kol_performances'):
                kol_performances = telegram_data['kol_performances']
                
                # Prepare SPYDEFI data for export with FIXED column order
                spydefi_rows = []
                
                for kol, performance in kol_performances.items():
                    # Handle both dictionary and object formats
                    if isinstance(performance, dict):
                        data = performance
                    else:
                        data = {
                            'composite_score': performance.composite_score,
                            'strategy_classification': performance.strategy_classification,
                            'follower_tier': performance.follower_tier,
                            'channel_id': performance.channel_id,
                            'total_calls': performance.total_calls,
                            'winning_calls': performance.winning_calls,
                            'losing_calls': performance.losing_calls,
                            'success_rate_2x': performance.success_rate_2x,
                            'success_rate_5x': performance.success_rate_5x,
                            'tokens_2x_plus': performance.tokens_2x_plus,
                            'tokens_5x_plus': performance.tokens_5x_plus,
                            'avg_time_to_2x_hours': performance.avg_time_to_2x_hours,
                            'avg_max_pullback_percent': performance.avg_max_pullback_percent,
                            'consistency_score': performance.consistency_score,
                            'avg_roi': performance.avg_roi
                        }
                    
                    # Excel row with FIXED column order
                    row = {
                        'KOL': f"@{kol}",
                        'Channel ID': data.get('channel_id', 'N/A'),  # Real numeric channel ID
                        'Composite Score': data.get('composite_score', 0),
                        'Copy Rec': 'COPY' if data.get('composite_score', 0) >= 70 else 'AVOID',
                        'Strategy': data.get('strategy_classification', 'UNKNOWN'),
                        'Follower Tier': data.get('follower_tier', 'LOW'),
                        'Total Calls': data.get('total_calls', 0),
                        'Winning Calls': data.get('winning_calls', 0),
                        'Losing Calls': data.get('losing_calls', 0),
                        '2x Success Rate %': data.get('success_rate_2x', 0),
                        '2x Tokens': data.get('tokens_2x_plus', 0),
                        '5x Tokens': data.get('tokens_5x_plus', 0),
                        'Avg Time to 2x (h)': data.get('avg_time_to_2x_hours', 0),
                        'Max Pullback %': data.get('avg_max_pullback_percent', 0),
                        '5x Success Rate %': data.get('success_rate_5x', 0),
                        'Consistency Score': data.get('consistency_score', 0),
                        'Avg ROI %': data.get('avg_roi', 0)
                    }
                    spydefi_rows.append(row)
                
                # Sort by composite score
                spydefi_rows.sort(key=lambda x: x['Composite Score'], reverse=True)
                
                if spydefi_rows:
                    spydefi_df = pd.DataFrame(spydefi_rows)
                    spydefi_df.to_excel(writer, sheet_name='SPYDEFI KOLs', index=False)
                    
                    # Format SPYDEFI sheet
                    spydefi_sheet = writer.sheets['SPYDEFI KOLs']
                    
                    # Apply header format
                    for col_num, value in enumerate(spydefi_df.columns.values):
                        spydefi_sheet.write(0, col_num, value, header_format)
                    
                    # Apply conditional formatting for composite scores
                    score_col = spydefi_df.columns.get_loc('Composite Score')
                    spydefi_sheet.conditional_format(f'{chr(65 + score_col)}2:{chr(65 + score_col)}{len(spydefi_df) + 1}', {
                        'type': 'cell',
                        'criteria': '>=',
                        'value': 80,
                        'format': score_excellent_format
                    })
                    
                    spydefi_sheet.conditional_format(f'{chr(65 + score_col)}2:{chr(65 + score_col)}{len(spydefi_df) + 1}', {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 70,
                        'maximum': 79.99,
                        'format': score_good_format
                    })
                    
                    spydefi_sheet.conditional_format(f'{chr(65 + score_col)}2:{chr(65 + score_col)}{len(spydefi_df) + 1}', {
                        'type': 'cell',
                        'criteria': 'between',
                        'minimum': 60,
                        'maximum': 69.99,
                        'format': score_average_format
                    })
                    
                    spydefi_sheet.conditional_format(f'{chr(65 + score_col)}2:{chr(65 + score_col)}{len(spydefi_df) + 1}', {
                        'type': 'cell',
                        'criteria': '<',
                        'value': 60,
                        'format': score_poor_format
                    })
                    
                    # Set column widths
                    spydefi_sheet.set_column('A:A', 20)  # KOL
                    spydefi_sheet.set_column('B:B', 20)  # Channel ID
                    spydefi_sheet.set_column('C:C', 15)  # Composite Score
                    spydefi_sheet.set_column('D:D', 10)  # Copy Rec
                    spydefi_sheet.set_column('E:E', 12)  # Strategy
                    spydefi_sheet.set_column('F:F', 12)  # Follower Tier
                    spydefi_sheet.set_column('G:M', 10)  # Calls and metrics
                    spydefi_sheet.set_column('N:Q', 12)  # Performance metrics
            
            # Export Wallet data if available (unchanged for compatibility)
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
                            'Unknown'
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
                            len(wallet_data.get('unknown', []))
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Create detailed wallet analysis sheet (abbreviated for space)
                all_wallets = []
                for category in ['snipers', 'flippers', 'scalpers', 'gem_hunters', 
                               'swing_traders', 'position_traders', 'consistent', 'mixed', 'unknown']:
                    all_wallets.extend(wallet_data.get(category, []))
                
                if all_wallets:
                    # Sort by composite score
                    all_wallets.sort(key=lambda x: x.get('composite_score', x['metrics'].get('composite_score', 0)), reverse=True)
                    
                    wallet_rows = []
                    for rank, wallet in enumerate(all_wallets[:100], 1):  # Top 100 only for Excel
                        metrics = wallet['metrics']
                        composite_score = wallet.get('composite_score', metrics.get('composite_score', 0))
                        
                        # Determine binary copy decision
                        copy_decision = "YES" if (
                            composite_score >= 60 and 
                            metrics.get('active_trader', False) and
                            metrics.get('trades_last_7_days', 0) > 0
                        ) else "NO"
                        
                        # Build row data (abbreviated)
                        row = {
                            'Rank': rank,
                            'Wallet': wallet['wallet_address'],
                            'Copy Decision': copy_decision,
                            'Score': composite_score,
                            'Type': wallet['wallet_type'],
                            'Trades 7d': metrics.get('trades_last_7_days', 0),
                            'Win Rate 7d': metrics.get('win_rate_7d', 0) / 100,
                            'Profit 7d': metrics.get('profit_7d', 0),
                            'Avg ROI': metrics['avg_roi'] / 100,
                            'Max ROI': metrics['max_roi'] / 100,
                            'Active': 'YES' if metrics.get('active_trader', False) else 'NO'
                        }
                        
                        wallet_rows.append(row)
                    
                    if wallet_rows:
                        wallet_df = pd.DataFrame(wallet_rows)
                        wallet_df.to_excel(writer, sheet_name='Top Wallets', index=False)
            
            logger.info(f"Successfully exported comprehensive analysis to Excel: {output_file}")
            return True
            
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        return False

# Wallet analysis export functions remain unchanged for compatibility
def export_wallet_rankings_csv(wallet_data: Dict[str, Any], output_file: str) -> bool:
    """
    Export 7-day focused wallet rankings to CSV with binary copy decision.
    (PRESERVED - unchanged for wallet_module compatibility)
    
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
    Generate a comprehensive analysis report with enhanced SPYDEFI analysis.
    (PRESERVED - unchanged for wallet_module compatibility)
    
    Args:
        telegram_data: SPYDEFI analysis results
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
            
            # SPYDEFI Analysis Section
            if telegram_data and telegram_data.get('kol_performances'):
                f.write("📱 SPYDEFI KOL ANALYSIS\n")
                f.write("-" * 40 + "\n")
                
                kol_performances = telegram_data['kol_performances']
                metadata = telegram_data.get('metadata', {})
                
                f.write(f"Analysis Type: Professional KOL Performance Tracking\n")
                f.write(f"KOLs Analyzed: {len(kol_performances)}\n")
                f.write(f"Total Calls Tracked: {metadata.get('total_calls_analyzed', 0)}\n")
                f.write(f"Overall 2x Rate: {metadata.get('overall_2x_rate', 0):.2f}%\n")
                f.write(f"Overall 5x Rate: {metadata.get('overall_5x_rate', 0):.2f}%\n\n")
                
                f.write("Top 10 SPYDEFI KOLs:\n")
                
                # Convert to list and sort
                kol_list = []
                for kol, performance in kol_performances.items():
                    if isinstance(performance, dict):
                        data = performance.copy()
                        data['kol'] = kol
                    else:
                        data = {
                            'kol': kol,
                            'composite_score': performance.composite_score,
                            'strategy_classification': performance.strategy_classification,
                            'follower_tier': performance.follower_tier,
                            'success_rate_2x': performance.success_rate_2x,
                            'success_rate_5x': performance.success_rate_5x,
                            'total_calls': performance.total_calls,
                            'channel_id': performance.channel_id
                        }
                    kol_list.append(data)
                
                kol_list.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
                
                for i, data in enumerate(kol_list[:10], 1):
                    f.write(f"{i}. @{data['kol']}\n")
                    f.write(f"   Composite Score: {data.get('composite_score', 0):.1f}/100\n")
                    f.write(f"   Strategy: {data.get('strategy_classification', 'UNKNOWN')}\n")
                    f.write(f"   Follower Tier: {data.get('follower_tier', 'LOW')}\n")
                    f.write(f"   Channel ID: {data.get('channel_id', 'N/A')}\n")
                    f.write(f"   Total Calls: {data.get('total_calls', 0)}\n")
                    f.write(f"   2x Rate: {data.get('success_rate_2x', 0):.1f}%\n")
                    f.write(f"   5x Rate: {data.get('success_rate_5x', 0):.1f}%\n")
                    f.write("\n")
            
            # Wallet Analysis Section (unchanged for compatibility)
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
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF COMPREHENSIVE ANALYSIS REPORT\n")
            
        logger.info(f"Successfully generated comprehensive analysis report: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating comprehensive analysis report: {str(e)}")
        return False

def export_distribution_analysis(wallet_data: Dict[str, Any], output_file: str) -> bool:
    """
    Export detailed 7-day distribution analysis for active traders.
    (PRESERVED - unchanged for wallet_module compatibility)
    
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