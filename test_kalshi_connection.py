#!/usr/bin/env python3
"""
Test script to verify Kalshi API connection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from api.kalshi_client import load_client_from_config


def main():
    print("=" * 50)
    print("Kalshi API Connection Test")
    print("=" * 50)

    try:
        print("\n1. Loading client from config...")
        client = load_client_from_config("config/config.yaml")
        print("   ✓ Client loaded successfully")

        print("\n2. Testing exchange status endpoint...")
        status = client.get_exchange_status()
        print(f"   ✓ Exchange status: {status}")

        print("\n3. Testing portfolio balance endpoint...")
        balance = client.get_balance()
        print(f"   ✓ Account balance: ${balance.get('balance', 0) / 100:.2f}")

        print("\n4. Fetching sample markets...")
        markets_response = client.get_markets(limit=5)
        markets = markets_response.get("markets", [])
        print(f"   ✓ Found {len(markets)} markets")

        if markets:
            print("\n   Sample markets:")
            for market in markets[:3]:
                ticker = market.get("ticker", "N/A")
                title = market.get("title", "N/A")[:50]
                print(f"   - {ticker}: {title}...")

        print("\n" + "=" * 50)
        print("✓ All tests passed! API connection is working.")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"\n✗ Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
