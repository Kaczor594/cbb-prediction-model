"""
Barttorvik Data Fetcher

Automatically downloads team efficiency data from Barttorvik's public CSV endpoints.
Based on: https://barttorvik.com/XXXX_team_results.csv

Usage:
    python src/data/fetch_barttorvik.py --seasons 2024 2025 2026
"""

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.import_barttorvik import import_barttorvik_csv


BARTTORVIK_URL = "https://barttorvik.com/{year}_team_results.csv"


def fetch_barttorvik_csv(season: int, output_dir: str = "data") -> str:
    """
    Fetch Barttorvik CSV for a given season.

    Args:
        season: Season year (e.g., 2024 for 2023-24)
        output_dir: Directory to save the CSV

    Returns:
        Path to the downloaded CSV file
    """
    url = BARTTORVIK_URL.format(year=season)
    output_path = Path(output_dir) / f"barttorvik_{season}.csv"

    print(f"Fetching {url}...")

    response = requests.get(url)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    print(f"  Saved to {output_path}")

    return str(output_path)


def fetch_and_import(seasons: list[int], db_path: str = None) -> dict:
    """
    Fetch and import Barttorvik data for multiple seasons.

    Args:
        seasons: List of season years
        db_path: Database path

    Returns:
        Import statistics by season
    """
    results = {}

    for season in seasons:
        try:
            csv_path = fetch_barttorvik_csv(season)
            result = import_barttorvik_csv(csv_path, season, db_path)
            results[season] = result
            print(f"  Imported: {result['imported']} teams, Matched: {result['matched_to_espn']}")
        except requests.HTTPError as e:
            print(f"  Error fetching {season}: {e}")
            results[season] = {'error': str(e)}
        except Exception as e:
            print(f"  Error processing {season}: {e}")
            results[season] = {'error': str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch and import Barttorvik data")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2024, 2025, 2026],
        help="Season years to collect",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Database path",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Barttorvik Data Fetch & Import")
    print("=" * 60)

    results = fetch_and_import(args.seasons, args.db)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total = 0
    for season, result in results.items():
        if 'error' in result:
            print(f"  {season-1}-{season}: ERROR - {result['error']}")
        else:
            print(f"  {season-1}-{season}: {result['imported']} teams ({result['matched_to_espn']} matched)")
            total += result['imported']
    print(f"\n  Total: {total} team records")


if __name__ == "__main__":
    main()
