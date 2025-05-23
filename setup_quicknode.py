#!/usr/bin/env python3
"""
QuickNode RPC Setup for Phoenix Project
This script helps you configure your QuickNode RPC endpoint
"""

import json
import os
import requests
import sys

def test_rpc_connection(rpc_url: str) -> bool:
    """Test if the RPC URL is working."""
    try:
        response = requests.post(
            rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result:
                return True
            elif "error" in result:
                print(f"❌ RPC Error: {result['error']}")
                return False
        else:
            print(f"❌ HTTP Error {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

def update_config_with_rpc(rpc_url: str):
    """Update the Phoenix config file with the new RPC URL."""
    config_path = os.path.expanduser("~/.phoenix_config.json")
    
    # Load existing config or create new one
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "birdeye_api_key": "",
            "cielo_api_key": "",
            "telegram_api_id": "",
            "telegram_api_hash": "",
            "telegram_session": "phoenix",
            "sources": {
                "telegram_groups": ["spydefi"],
                "wallets": []
            },
            "analysis_period_days": 1
        }
    
    # Update RPC URL
    config["solana_rpc_url"] = rpc_url
    
    # Save config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Config updated: {config_path}")

def main():
    print("🚀 QuickNode RPC Setup for Phoenix Project")
    print("=" * 60)
    
    print("\n📋 QuickNode RPC URL Format:")
    print("https://YOUR-ENDPOINT-NAME.solana-mainnet.quiknode.pro/YOUR-API-KEY/")
    
    print("\n💡 To find your QuickNode RPC URL:")
    print("1. Log into QuickNode Dashboard")
    print("2. Find your Solana Mainnet endpoint")
    print("3. Copy the HTTPS endpoint URL")
    
    print("\n" + "=" * 60)
    
    # Get RPC URL
    rpc_url = input("\nPaste your QuickNode RPC URL: ").strip()
    
    if not rpc_url:
        print("❌ No URL provided!")
        sys.exit(1)
    
    # Validate format
    if "quiknode.pro" not in rpc_url:
        print("⚠️ This doesn't look like a QuickNode URL.")
        confirm = input("Continue anyway? (y/N): ").lower().strip()
        if confirm != 'y':
            sys.exit(1)
    
    # Test connection
    print("\n🔍 Testing RPC connection...")
    if test_rpc_connection(rpc_url):
        print("✅ RPC connection successful!")
        
        # Test performance
        print("\n⚡ Testing RPC performance...")
        import time
        
        start = time.time()
        test_rpc_connection(rpc_url)
        latency = (time.time() - start) * 1000
        
        print(f"📊 Latency: {latency:.0f}ms")
        
        if latency < 100:
            print("🚀 Excellent performance!")
        elif latency < 300:
            print("✅ Good performance")
        else:
            print("⚠️ High latency detected")
        
        # Update config
        print("\n📝 Updating Phoenix configuration...")
        update_config_with_rpc(rpc_url)
        
        print("\n✅ Setup complete! Your QuickNode RPC is configured.")
        print("\n🎯 Next steps:")
        print("1. Run: python phoenix.py")
        print("2. Test wallet analysis (option 6)")
        print("3. The last 3 tokens feature will now work!")
        
    else:
        print("\n❌ RPC connection failed!")
        print("\n🔧 Troubleshooting:")
        print("1. Check if your QuickNode endpoint is active")
        print("2. Verify the URL is copied correctly")
        print("3. Ensure your QuickNode plan hasn't expired")
        print("4. Check if you need to whitelist your IP in QuickNode dashboard")

if __name__ == "__main__":
    main()