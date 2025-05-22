#!/usr/bin/env python3
"""
Debug script for Cielo Finance API authentication issues
"""

import requests
import json

def debug_cielo_auth():
    """Debug Cielo Finance API authentication with different methods"""
    
    print("🔍 CIELO FINANCE API AUTHENTICATION DEBUG")
    print("="*50)
    
    # Get API key
    api_key = input("Enter your Cielo Finance API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return
    
    print(f"API Key (first 10 chars): {api_key[:10]}...")
    print(f"API Key length: {len(api_key)} characters")
    
    base_url = "https://feed-api.cielo.finance"
    test_wallet = "7Hy2bJvqPRXGVybQ18zLW2sc8E2DnorXqPrCxARAxWQC"
    
    # Test different authentication methods
    auth_methods = [
        {
            "name": "Bearer Token (Standard)",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Phoenix-Project/1.0"
            }
        },
        {
            "name": "X-API-KEY Header",
            "headers": {
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
                "User-Agent": "Phoenix-Project/1.0"
            }
        },
        {
            "name": "Authorization Direct",
            "headers": {
                "Authorization": api_key,
                "Content-Type": "application/json",
                "User-Agent": "Phoenix-Project/1.0"
            }
        }
    ]
    
    # Test different endpoints
    endpoints = [
        ("Token P&L", f"/api/v1/{test_wallet}/pnl/tokens"),
        ("Trading Stats", f"/api/v1/{test_wallet}/trading-stats"),
        ("Total Stats", f"/api/v1/{test_wallet}/pnl/total-stats"),
        ("Tracked Wallets", "/api/v1/tracked-wallets")
    ]
    
    print(f"\n🎯 Testing with wallet: {test_wallet}")
    print("-" * 50)
    
    for endpoint_name, endpoint_path in endpoints:
        print(f"\n📋 Testing: {endpoint_name}")
        print(f"    Endpoint: {endpoint_path}")
        
        for auth_method in auth_methods:
            try:
                url = f"{base_url}{endpoint_path}"
                response = requests.get(url, headers=auth_method['headers'], timeout=10)
                
                print(f"    {auth_method['name']:20} | Status: {response.status_code:3d} | {response.reason}")
                
                if response.status_code == 200:
                    print(f"    ✅ SUCCESS! Found working authentication method")
                    try:
                        data = response.json()
                        print(f"    📊 Sample response: {str(data)[:100]}...")
                    except:
                        print(f"    📊 Response text: {response.text[:100]}...")
                    break
                elif response.status_code == 401:
                    print(f"    ❌ 401 Unauthorized - Invalid API key")
                elif response.status_code == 403:
                    print(f"    ❌ 403 Forbidden - API key valid but insufficient permissions")
                elif response.status_code == 429:
                    print(f"    ⚠️ 429 Rate Limited - Too many requests")
                else:
                    error_text = response.text[:200] if response.text else "No response body"
                    print(f"    ⚠️ {response.status_code} - {error_text}")
                    
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
        
        print("-" * 50)
    
    # Test API key format variations
    print(f"\n🔧 API KEY FORMAT ANALYSIS")
    print(f"Raw key: {api_key}")
    print(f"Starts with 'cf_': {'Yes' if api_key.startswith('cf_') else 'No'}")
    print(f"Starts with 'ck_': {'Yes' if api_key.startswith('ck_') else 'No'}")
    print(f"Contains dashes: {'Yes' if '-' in api_key else 'No'}")
    print(f"Contains underscores: {'Yes' if '_' in api_key else 'No'}")
    print(f"All alphanumeric: {'Yes' if api_key.replace('-', '').replace('_', '').isalnum() else 'No'}")
    
    # Test basic connectivity
    print(f"\n🌐 BASIC CONNECTIVITY TEST")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"Base URL connectivity: {response.status_code} {response.reason}")
    except Exception as e:
        print(f"Base URL connectivity: Failed - {str(e)}")

if __name__ == "__main__":
    debug_cielo_auth()