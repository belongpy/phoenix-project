"""
Export Utilities Module - Phoenix Project

This module handles consolidated exports to Excel workbooks for easier analysis
and integration with automated trading bots.
"""

import os
import logging
import pandas as pd
from typing import Dict, List, Any, Optional

logger = logging.getLogger("phoenix.export")

def export_to_excel(telegram_analyses: Dict[str, Any], wallet_analyses: Dict[str, Any], 
                  output_file: str) -> None:
    """
    Export all analyses to a single Excel workbook with multiple sheets.
    
    Args:
        telegram_analyses: Results from Telegram analysis
        wallet_analyses: Results from wallet analysis
        output_file: Output Excel file path
    """
    try:
        logger.info(f"Creating consolidated Excel export: {output_file}")
        
        # Create outputs directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Create formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Create Summary sheet
            export_summary_sheet(writer, telegram_analyses, wallet_analyses)
            
            # Create KOL Channels sheet
            if telegram_analyses and "ranked_kols" in telegram_analyses:
                export_kol_sheet(writer, telegram_analyses, header_format)
            
            # Create Wallet category sheets
            if wallet_analyses:
                export_wallet_sheets(writer, wallet_analyses, header_format)
            
            # Create Export sheet for easy copying
            export_quick_extract_sheet(writer, telegram_analyses, wallet_analyses, header_format)
            
            # Create Correlated Wallets sheet
            if wallet_analyses and "wallet_clusters" in wallet_analyses:
                export_correlated_wallets_sheet(writer, wallet_analyses, header_format)
        
        logger.info(f"Successfully created Excel workbook: {output_file}")
    
    except Exception as e:
        logger.error(f"Error creating Excel export: {str(e)}")
        raise

def export_summary_sheet(writer: pd.ExcelWriter, 
                       telegram_analyses: Dict[str, Any], 
                       wallet_analyses: Dict[str, Any]) -> None:
    """
    Export summary statistics to the Excel workbook.
    
    Args:
        writer: Excel writer object
        telegram_analyses: Results from Telegram analysis
        wallet_analyses: Results from wallet analysis
    """
    kol_count = len(telegram_analyses.get("ranked_kols", [])) if telegram_analyses else 0
    
    gem_count = len(wallet_analyses.get("gem_finders", [])) if wallet_analyses else 0
    consistent_count = len(wallet_analyses.get("consistent", [])) if wallet_analyses else 0
    flipper_count = len(wallet_analyses.get("flippers", [])) if wallet_analyses else 0
    other_count = len(wallet_analyses.get("others", [])) if wallet_analyses else 0
    wallet_count = gem_count + consistent_count + flipper_count + other_count
    
    # Prepare summary data
    summary_data = {
        "Metric": [
            "Analysis Date",
            "Total KOLs Analyzed",
            "Total Wallets Analyzed",
            "Gem Finder Wallets",
            "Consistent Wallets",
            "Flipper Wallets",
            "Other Wallets"
        ],
        "Value": [
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            kol_count,
            wallet_count,
            gem_count,
            consistent_count,
            flipper_count,
            other_count
        ]
    }
    
    df = pd.DataFrame(summary_data)
    df.to_excel(writer, sheet_name="Summary", index=False)
    
    # Format the sheet
    worksheet = writer.sheets["Summary"]
    worksheet.set_column('A:A', 25)
    worksheet.set_column('B:B', 15)

def export_kol_sheet(writer: pd.ExcelWriter, 
                   telegram_analyses: Dict[str, Any],
                   header_format: Any) -> None:
    """
    Export KOL channel analysis to the Excel workbook.
    
    Args:
        writer: Excel writer object
        telegram_analyses: Results from Telegram analysis
        header_format: Excel cell format for headers
    """
    # Prepare KOL data
    kol_data = []
    
    for kol in telegram_analyses["ranked_kols"]:
        strategy = kol.get("strategy", {})
        kol_data.append({
            "channel_id": kol["channel_id"],
            "total_calls": kol.get("total_calls", 0),
            "success_rate": round(kol.get("success_rate", 0), 2),
            "avg_roi": round(kol.get("avg_roi", 0), 2),
            "avg_max_roi": round(kol.get("avg_max_roi", 0), 2),
            "confidence_level": round(kol.get("confidence_level", 0), 2),
            "strategy": strategy.get("recommendation", "CAUTIOUS"),
            "entry_type": strategy.get("entry_type", "WAIT_FOR_CONFIRMATION"),
            "take_profit_1": strategy.get("take_profit_1", 0),
            "take_profit_2": strategy.get("take_profit_2", 0),
            "take_profit_3": strategy.get("take_profit_3", 0),
            "stop_loss": strategy.get("stop_loss", 0)
        })
    
    # Create DataFrame and export
    if kol_data:
        df = pd.DataFrame(kol_data)
        df.to_excel(writer, sheet_name="KOL_Channels", index=False)
        
        # Format the sheet
        worksheet = writer.sheets["KOL_Channels"]
        for idx, col in enumerate(df.columns):
            worksheet.write(0, idx, col, header_format)
            worksheet.set_column(idx, idx, max(len(col) + 2, 12))

def export_wallet_sheets(writer: pd.ExcelWriter, 
                       wallet_analyses: Dict[str, Any],
                       header_format: Any) -> None:
    """
    Export wallet analysis to separate sheets by category.
    
    Args:
        writer: Excel writer object
        wallet_analyses: Results from wallet analysis
        header_format: Excel cell format for headers
    """
    categories = {
        "Gem_Finders": wallet_analyses.get("gem_finders", []),
        "Consistent": wallet_analyses.get("consistent", []),
        "Flippers": wallet_analyses.get("flippers", []),
        "Others": wallet_analyses.get("others", [])
    }
    
    for sheet_name, wallets in categories.items():
        if not wallets:
            continue
        
        wallet_data = []
        for wallet in wallets:
            metrics = wallet["metrics"]
            strategy = wallet["strategy"]
            
            # Calculate average SOL values and most common platform
            sol_values = [t.get("buy_value_sol", 0) for t in wallet.get("trades", []) if "buy_value_sol" in t]
            market_caps = [t.get("market_cap_at_buy", 0) for t in wallet.get("trades", []) if "market_cap_at_buy" in t]
            platforms = [t.get("platform", "") for t in wallet.get("trades", []) if "platform" in t and t["platform"]]
            
            avg_sol = sum(sol_values) / len(sol_values) if sol_values else 0
            avg_mcap = sum(market_caps) / len(market_caps) if market_caps else 0
            
            # Get most common platform
            most_common_platform = ""
            if platforms:
                from collections import Counter
                platform_counts = Counter(platforms)
                most_common_platform = platform_counts.most_common(1)[0][0] if platform_counts else ""
            
            wallet_data.append({
                "wallet_address": wallet["wallet_address"],
                "entry_type": strategy.get("entry_type", "UNKNOWN"),
                "strategy": strategy.get("recommendation", "CAUTIOUS"),
                "win_rate": round(metrics.get("win_rate", 0), 2),
                "profit_factor": round(metrics.get("profit_factor", 0), 2),
                "median_roi": round(metrics.get("median_roi", 0), 2),
                "avg_roi": round(metrics.get("avg_roi", 0), 2),
                "max_roi": round(metrics.get("max_roi", 0), 2),
                "total_trades": metrics.get("total_trades", 0),
                "avg_hold_time": round(metrics.get("avg_hold_time_hours", 0), 2),
                "avg_bet_size_usd": round(metrics.get("avg_bet_size_usd", 0), 2),
                "avg_bet_size_sol": round(avg_sol, 6),
                "avg_market_cap": int(avg_mcap),
                "common_platform": most_common_platform,
                "take_profit_1": strategy.get("take_profit_1", 0),
                "take_profit_2": strategy.get("take_profit_2", 0),
                "take_profit_3": strategy.get("take_profit_3", 0),
                "stop_loss": strategy.get("stop_loss", 0)
            })
        
        # Create DataFrame and export
        if wallet_data:
            df = pd.DataFrame(wallet_data)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Format the sheet
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                worksheet.write(0, idx, col, header_format)
                worksheet.set_column(idx, idx, max(len(col) + 2, 12))

def export_quick_extract_sheet(writer: pd.ExcelWriter, 
                             telegram_analyses: Dict[str, Any], 
                             wallet_analyses: Dict[str, Any],
                             header_format: Any) -> None:
    """
    Export a quick extract sheet with just wallet addresses and channel IDs for easy copying.
    
    Args:
        writer: Excel writer object
        telegram_analyses: Results from Telegram analysis
        wallet_analyses: Results from wallet analysis
        header_format: Excel cell format for headers
    """
    # Prepare extract data
    extract_data = {
        "Type": [],
        "ID": [],
        "Strategy": [],
        "Entry_Type": [],
        "TP1": [],
        "TP2": [],
        "TP3": [],
        "SL": []
    }
    
    # Add KOL channel IDs
    if telegram_analyses and "ranked_kols" in telegram_analyses:
        for kol in telegram_analyses["ranked_kols"]:
            strategy = kol.get("strategy", {})
            if kol.get("confidence_level", 0) >= 50:  # Only include higher confidence KOLs
                extract_data["Type"].append("Telegram_Channel")
                extract_data["ID"].append(kol["channel_id"])
                extract_data["Strategy"].append(strategy.get("recommendation", "CAUTIOUS"))
                extract_data["Entry_Type"].append(strategy.get("entry_type", "WAIT_FOR_CONFIRMATION"))
                extract_data["TP1"].append(strategy.get("take_profit_1", 0))
                extract_data["TP2"].append(strategy.get("take_profit_2", 0))
                extract_data["TP3"].append(strategy.get("take_profit_3", 0))
                extract_data["SL"].append(strategy.get("stop_loss", 0))
    
    # Add wallet addresses for each category
    if wallet_analyses:
        categories = [
            ("Wallet_GemFinder", wallet_analyses.get("gem_finders", [])),
            ("Wallet_Consistent", wallet_analyses.get("consistent", [])),
            ("Wallet_Flipper", wallet_analyses.get("flippers", []))
        ]
        
        for category_name, wallets in categories:
            for wallet in wallets:
                strategy = wallet.get("strategy", {})
                extract_data["Type"].append(category_name)
                extract_data["ID"].append(wallet["wallet_address"])
                extract_data["Strategy"].append(strategy.get("recommendation", "CAUTIOUS"))
                extract_data["Entry_Type"].append(strategy.get("entry_type", "UNKNOWN"))
                extract_data["TP1"].append(strategy.get("take_profit_1", 0))
                extract_data["TP2"].append(strategy.get("take_profit_2", 0))
                extract_data["TP3"].append(strategy.get("take_profit_3", 0))
                extract_data["SL"].append(strategy.get("stop_loss", 0))
    
    # Create DataFrame and export
    df = pd.DataFrame(extract_data)
    
    # Sort by Strategy (prioritizing HOLD_MOON and SCALP_AND_HOLD) then by Entry_Type
    strategy_order = {
        "HOLD_MOON": 1,
        "SCALP_AND_HOLD": 2,
        "SCALP": 3,
        "CAUTIOUS": 4
    }
    
    entry_order = {
        "IMMEDIATE": 1,
        "WAIT_FOR_CONFIRMATION": 2,
        "UNKNOWN": 3
    }
    
    # Create sorting keys
    df['strategy_sort'] = df['Strategy'].map(lambda x: strategy_order.get(x, 5))
    df['entry_sort'] = df['Entry_Type'].map(lambda x: entry_order.get(x, 3))
    
    # Sort the dataframe
    df = df.sort_values(['strategy_sort', 'entry_sort'], ascending=[True, True])
    
    # Remove sorting columns
    df = df.drop(['strategy_sort', 'entry_sort'], axis=1)
    
    # Export to Excel
    df.to_excel(writer, sheet_name="Quick_Extract", index=False)
    
    # Format the sheet
    worksheet = writer.sheets["Quick_Extract"]
    for idx, col in enumerate(df.columns):
        worksheet.write(0, idx, col, header_format)
        if col == "ID":
            worksheet.set_column(idx, idx, 45)  # Wallet addresses are long
        else:
            worksheet.set_column(idx, idx, max(len(col) + 2, 12))

def export_correlated_wallets_sheet(writer: pd.ExcelWriter, 
                                  wallet_analyses: Dict[str, Any],
                                  header_format: Any) -> None:
    """
    Export correlated wallets and wallet clusters.
    
    Args:
        writer: Excel writer object
        wallet_analyses: Results from wallet analysis
        header_format: Excel cell format for headers
    """
    # Export wallet clusters
    if "wallet_clusters" in wallet_analyses and wallet_analyses["wallet_clusters"]:
        cluster_data = []
        
        for i, cluster in enumerate(wallet_analyses["wallet_clusters"]):
            cluster_data.append({
                "Cluster_ID": i + 1,
                "Size": cluster["size"],
                "Correlation_Strength": cluster["correlation_strength"],
                "Wallets": ", ".join(cluster["wallets"][:3]) + ("..." if len(cluster["wallets"]) > 3 else "")
            })
        
        df = pd.DataFrame(cluster_data)
        df.to_excel(writer, sheet_name="Wallet_Clusters", index=False)
        
        # Format the sheet
        worksheet = writer.sheets["Wallet_Clusters"]
        for idx, col in enumerate(df.columns):
            worksheet.write(0, idx, col, header_format)
            if col == "Wallets":
                worksheet.set_column(idx, idx, 60)
            else:
                worksheet.set_column(idx, idx, max(len(col) + 2, 15))
        
        # Add full wallet listings in separate sheets for each large cluster
        for i, cluster in enumerate(wallet_analyses["wallet_clusters"]):
            if cluster["size"] > 5:  # Only create detailed sheets for larger clusters
                sheet_name = f"Cluster_{i+1}_Wallets"
                try:
                    wallet_list = {"Wallet_Address": cluster["wallets"]}
                    pd.DataFrame(wallet_list).to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Format the sheet
                    worksheet = writer.sheets[sheet_name]
                    worksheet.write(0, 0, "Wallet_Address", header_format)
                    worksheet.set_column(0, 0, 45)
                except Exception as e:
                    logger.warning(f"Could not create detailed cluster sheet: {str(e)}")