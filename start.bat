@echo off
cls
echo ========================================
echo    PHOENIX PROJECT - SOLANA ANALYZER
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 🔧 Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🚀 Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install/update requirements
echo 📦 Installing requirements...
pip install -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ Some packages may have failed to install, continuing...
)

REM Create outputs directory if it doesn't exist
if not exist "outputs" (
    mkdir outputs
    echo 📁 Created outputs directory
)

REM Check if config exists
if not exist "%USERPROFILE%\.phoenix_config.json" (
    echo.
    echo ⚙️ FIRST TIME SETUP REQUIRED
    echo ========================================
    echo You need to configure your API keys first.
    echo.
    echo Required API Keys:
    echo 1. Birdeye API Key (https://birdeye.so)
    echo 2. Telegram API ID and Hash (https://my.telegram.org)
    echo.
    echo Starting configuration...
    echo.
    
    python phoenix_cli.py configure
    if %errorlevel% neq 0 (
        echo ❌ Configuration failed
        pause
        exit /b 1
    )
    
    echo.
    echo ✅ Configuration complete!
    echo.
)

:MENU
cls
echo ========================================
echo    PHOENIX PROJECT - SOLANA ANALYZER
echo ========================================
echo.
echo SELECT ANALYSIS MODE:
echo [1] FAST SpyDefi Analysis (24 hours)
echo [2] CUSTOM SpyDefi Analysis (specify hours)
echo [3] Configure API Keys
echo [4] View Last Results
echo [5] Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto FAST_SPYDEFI
if "%choice%"=="2" goto CUSTOM_SPYDEFI
if "%choice%"=="3" goto CONFIGURE
if "%choice%"=="4" goto VIEW_RESULTS
if "%choice%"=="5" goto EXIT
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto MENU

:FAST_SPYDEFI
cls
echo ==========================================
echo    FAST SPYDEFI ANALYSIS (24 HOURS)
echo ==========================================
echo.
echo Starting SpyDefi analysis for past 24 hours...
echo This will scan for KOL token calls and calculate ROI performance.
echo.

python -c "
import asyncio
import sys
import os
sys.path.append('.')
from telegram_module import TelegramScraper
import json

async def run_analysis():
    try:
        # Load config
        config_file = os.path.expanduser('~/.phoenix_config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Initialize scraper
        scraper = TelegramScraper(
            config['telegram_api_id'],
            config['telegram_api_hash'], 
            config['birdeye_api_key']
        )
        
        # Run analysis
        await scraper.connect()
        results = await scraper.scan_spydefi_channel(hours_back=24)
        
        # Export results
        await scraper.export_analysis_results(results, 'outputs/spydefi_24h_analysis.csv')
        await scraper.disconnect()
        
        print('✅ Analysis complete! Check outputs folder.')
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        return 1
    return 0

exit_code = asyncio.run(run_analysis())
sys.exit(exit_code)
"

if %errorlevel% neq 0 (
    echo.
    echo ❌ Analysis failed. Check your configuration and API keys.
    pause
    goto MENU
)

echo.
echo Analysis complete! Check outputs folder.
pause
goto MENU

:CUSTOM_SPYDEFI
cls
echo ==========================================
echo    CUSTOM SPYDEFI ANALYSIS
echo ==========================================
echo.
set /p hours="Enter hours to analyze (default 24): "
if "%hours%"=="" set hours=24
set /p output="Enter output filename (default spydefi_analysis.csv): "
if "%output%"=="" set output=spydefi_analysis.csv

echo.
echo Starting SpyDefi analysis for past %hours% hours...
echo Results will be saved to: outputs\%output%
echo.

python -c "
import asyncio
import sys
import os
sys.path.append('.')
from telegram_module import TelegramScraper
import json

async def run_analysis():
    try:
        # Load config
        config_file = os.path.expanduser('~/.phoenix_config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Initialize scraper
        scraper = TelegramScraper(
            config['telegram_api_id'],
            config['telegram_api_hash'], 
            config['birdeye_api_key']
        )
        
        # Run analysis
        await scraper.connect()
        results = await scraper.scan_spydefi_channel(hours_back=%hours%)
        
        # Export results
        await scraper.export_analysis_results(results, 'outputs/%output%')
        await scraper.disconnect()
        
        print('✅ Analysis complete! Check outputs folder.')
        
    except Exception as e:
        print(f'❌ Error: {str(e)}')
        return 1
    return 0

exit_code = asyncio.run(run_analysis())
sys.exit(exit_code)
"

if %errorlevel% neq 0 (
    echo.
    echo ❌ Analysis failed. Check your configuration and API keys.
    pause
    goto MENU
)

echo.
echo Analysis complete! Check outputs folder.
pause
goto MENU

:CONFIGURE
cls
echo ==========================================
echo    CONFIGURE API KEYS
echo ==========================================
echo.
python phoenix_cli.py configure
pause
goto MENU

:VIEW_RESULTS
cls
echo ==========================================
echo    VIEW LAST RESULTS
echo ==========================================
echo.
if exist "outputs\spydefi_24h_analysis_summary.txt" (
    type "outputs\spydefi_24h_analysis_summary.txt"
) else if exist "outputs\spydefi_analysis_summary.txt" (
    type "outputs\spydefi_analysis_summary.txt"
) else (
    echo No analysis results found.
    echo Run an analysis first to see results here.
)
echo.
echo Available files in outputs folder:
dir /b outputs\*.csv 2>nul
if %errorlevel% neq 0 (
    echo No CSV files found.
)
echo.
pause
goto MENU

:EXIT
echo.
echo Thanks for using Phoenix Project! 🚀
echo.
timeout /t 2 >nul
exit /b 0