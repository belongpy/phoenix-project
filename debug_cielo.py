#!/usr/bin/env python3
"""
Debug Cielo Finance API Response Format
This script will show the actual response structure from Cielo Finance
"""

import json
import requests
from pprint import pprint

def debug_cielo_response(api_key: str, test_wallet: str = None):
    """Debug the Cielo Finance API response format."""
    
    # Default test wallets - use ones that likely have trading data
    test_wallets = [
        "2PnqznAKcwK6xu7mBjc2XAiwugMYzMx97vZSZsgLgVVd",  # First wallet from your list
        "5f5jNxQuHnAqmRv4fGsvUT91Ac24eJN6N1rWmB6hX5Ex",  # Second wallet
        "So11111111111111111111111111111111111111112"     # Wrapped SOL (system)
    ]
    
    if test_wallet:
        test_wallets.insert(0, test_wallet)
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    base_url = "https://feed-api.cielo.finance"
    
    print("🔍 Debugging Cielo Finance API Response Format")
    print("=" * 60)
    
    for wallet in test_wallets[:2]:  # Test first 2 wallets
        print(f"\n📊 Testing wallet: {wallet}")
        print("-" * 60)
        
        url = f"{base_url}/api/v1/{wallet}/trading-stats"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Success! Status: {response.status_code}")
                print(f"\n📋 Response structure:")
                
                # Save full response for analysis
                filename = f"cielo_response_{wallet[:8]}.json"
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"💾 Full response saved to: {filename}")
                
                # Show response structure
                print("\n🔑 Top-level keys:")
                if isinstance(data, dict):
                    for key in data.keys():
                        print(f"  - {key}: {type(data[key]).__name__}")
                
                # Show data structure if exists
                if isinstance(data, dict) and "data" in data:
                    print(f"\n📦 'data' structure:")
                    data_content = data["data"]
                    
                    if isinstance(data_content, dict):
                        print("  Keys in 'data':")
                        for key, value in data_content.items():
                            if isinstance(value, (list, dict)):
                                print(f"    - {key}: {type(value).__name__} (length: {len(value)})")
                            else:
                                print(f"    - {key}: {value}")
                        
                        # Check for tokens/trades data
                        if "tokens" in data_content:
                            print(f"\n  🪙 Found 'tokens' array with {len(data_content['tokens'])} items")
                            if data_content['tokens']:
                                print("  Sample token structure:")
                                sample = data_content['tokens'][0]
                                for k, v in sample.items():
                                    print(f"      - {k}: {v if not isinstance(v, (dict, list)) else type(v).__name__}")
                        
                        if "swaps" in data_content:
                            print(f"\n  🔄 Found 'swaps' data with {len(data_content.get('swaps', []))} items")
                    
                    # Show a preview of the data
                    print(f"\n📝 Data preview (first 500 chars):")
                    preview = json.dumps(data, indent=2)[:500]
                    print(preview + "..." if len(json.dumps(data)) > 500 else preview)
                
            else:
                print(f"❌ Error: Status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"💥 Exception: {str(e)}")
        
        print()
    
    print("\n" + "=" * 60)
    print("📌 Check the saved JSON files to see the complete response structure")
    print("📌 This will help us understand how to properly parse the data")

if __name__ == "__main__":
    api_key = input("Enter your Cielo Finance API key: ").strip()
    
    if not api_key:
        print("❌ API key is required!")
        exit(1)
    
    custom_wallet = input("Enter a specific wallet to test (or press Enter to use defaults): ").strip()
    
    debug_cielo_response(api_key, custom_wallet if custom_wallet else None)