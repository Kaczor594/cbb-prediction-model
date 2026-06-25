"""
Geocode team home locations and game venue locations.

Determines each team's home city from their most common home game venue,
then geocodes all unique (city, state) pairs using geopy/Nominatim.
Computes timezone and UTC offset for each location.

Usage:
    python src/data/geocode_locations.py
    python src/data/geocode_locations.py --force   # Re-geocode all
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from timezonefinder import TimezoneFinder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.database import get_connection


# Manual overrides for locations that Nominatim struggles with
# Format: (city, state) -> (latitude, longitude)
LOCATION_OVERRIDES = {
    # US territories and unusual locations
    ('Saint Thomas', 'Virgin Islands'): (18.3358, -64.9307),
    ('Bayamon', 'Puerto Rico'): (18.3985, -66.1536),
    ('San Juan', 'Puerto Rico'): (18.4655, -66.1057),
    # Military academies
    ('West Point', 'NY'): (41.3915, -73.9566),
    ('Annapolis', 'MD'): (38.9784, -76.4922),
    # Common problem cities
    ('University Park', 'TX'): (32.8504, -96.8003),
}


def geocode_location(geolocator, city: str, state: str,
                     max_retries: int = 3) -> tuple[float, float] | None:
    """Geocode a city, state pair. Returns (lat, lng) or None."""
    # Check overrides first
    if (city, state) in LOCATION_OVERRIDES:
        return LOCATION_OVERRIDES[(city, state)]

    query = f"{city}, {state}, USA"
    # For territories, don't append USA
    if state in ('Virgin Islands', 'Puerto Rico', 'Guam'):
        query = f"{city}, {state}"

    for attempt in range(max_retries):
        try:
            result = geolocator.geocode(query, timeout=10)
            if result:
                return (result.latitude, result.longitude)
            # Try without USA suffix
            result = geolocator.geocode(f"{city}, {state}", timeout=10)
            if result:
                return (result.latitude, result.longitude)
            return None
        except (GeocoderTimedOut, GeocoderUnavailable):
            time.sleep(2 ** attempt)
    return None


def get_timezone_info(tf: TimezoneFinder, lat: float, lng: float) -> tuple[str, float]:
    """Get timezone name and UTC offset for a coordinate."""
    from datetime import datetime, timezone as tz
    import zoneinfo

    tz_name = tf.timezone_at(lat=lat, lng=lng)
    if not tz_name:
        # Fallback: estimate from longitude
        tz_name = 'US/Eastern'  # default

    zone = zoneinfo.ZoneInfo(tz_name)
    # Use January 1 to avoid DST complications (standard time offset)
    dt = datetime(2025, 1, 1, tzinfo=zone)
    utc_offset = dt.utcoffset().total_seconds() / 3600

    return tz_name, utc_offset


def main():
    parser = argparse.ArgumentParser(description='Geocode team and venue locations')
    parser.add_argument('--force', action='store_true',
                        help='Re-geocode all locations (ignore existing)')
    args = parser.parse_args()

    conn = get_connection()

    # Ensure tables exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS team_locations (
            espn_id INTEGER PRIMARY KEY REFERENCES teams(espn_id),
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timezone TEXT NOT NULL,
            utc_offset REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS venue_locations (
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timezone TEXT NOT NULL,
            utc_offset REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(city, state)
        )
    """)

    if args.force:
        conn.execute("DELETE FROM team_locations")
        conn.execute("DELETE FROM venue_locations")
        conn.commit()

    geolocator = Nominatim(user_agent="cbb-prediction-model")
    tf = TimezoneFinder()

    # --- Step 1: Determine each team's home city ---
    print("Determining team home cities...")
    home_games = conn.execute("""
        SELECT home_team_id, venue_city, venue_state, COUNT(*) as cnt
        FROM games
        WHERE neutral_site = 0 AND venue_city IS NOT NULL
        GROUP BY home_team_id, venue_city, venue_state
        ORDER BY home_team_id, cnt DESC
    """).fetchall()

    team_home_city = {}
    for row in home_games:
        tid = row['home_team_id']
        if tid not in team_home_city:
            team_home_city[tid] = (row['venue_city'], row['venue_state'])

    print(f"  Found home cities for {len(team_home_city)} teams")

    # --- Step 2: Collect all unique (city, state) pairs to geocode ---
    venue_pairs = conn.execute("""
        SELECT DISTINCT venue_city, venue_state
        FROM games
        WHERE venue_city IS NOT NULL
    """).fetchall()
    all_locations = set()
    for row in venue_pairs:
        all_locations.add((row['venue_city'], row['venue_state']))
    for city, state in team_home_city.values():
        all_locations.add((city, state))

    # Filter out already-geocoded venues
    existing = set()
    for row in conn.execute("SELECT city, state FROM venue_locations").fetchall():
        existing.add((row['city'], row['state']))
    to_geocode = all_locations - existing

    print(f"  {len(all_locations)} unique locations, {len(to_geocode)} need geocoding")

    # --- Step 3: Geocode all locations ---
    if to_geocode:
        print("Geocoding locations...")
        geocoded = {}
        failed = []

        for i, (city, state) in enumerate(sorted(to_geocode)):
            coords = geocode_location(geolocator, city, state)
            if coords:
                lat, lng = coords
                tz_name, utc_offset = get_timezone_info(tf, lat, lng)
                geocoded[(city, state)] = (lat, lng, tz_name, utc_offset)
                conn.execute("""
                    INSERT OR REPLACE INTO venue_locations
                    (city, state, latitude, longitude, timezone, utc_offset)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (city, state, lat, lng, tz_name, utc_offset))
            else:
                failed.append((city, state))

            if (i + 1) % 20 == 0:
                conn.commit()
                print(f"  [{i+1}/{len(to_geocode)}] geocoded")

            # Rate limit: Nominatim requires 1 req/sec
            time.sleep(1.1)

        conn.commit()
        print(f"  Geocoded {len(geocoded)} locations, {len(failed)} failed")
        if failed:
            print(f"  Failed locations: {failed}")

    # --- Step 4: Populate team_locations ---
    existing_teams = set()
    for row in conn.execute("SELECT espn_id FROM team_locations").fetchall():
        existing_teams.add(row['espn_id'])

    teams_to_add = set(team_home_city.keys()) - existing_teams
    if teams_to_add:
        print(f"Populating team_locations for {len(teams_to_add)} teams...")
        added = 0
        for tid in teams_to_add:
            city, state = team_home_city[tid]
            venue = conn.execute(
                "SELECT latitude, longitude, timezone, utc_offset FROM venue_locations WHERE city=? AND state=?",
                (city, state)
            ).fetchone()
            if venue:
                conn.execute("""
                    INSERT OR REPLACE INTO team_locations
                    (espn_id, city, state, latitude, longitude, timezone, utc_offset)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (tid, city, state, venue['latitude'], venue['longitude'],
                      venue['timezone'], venue['utc_offset']))
                added += 1
            else:
                name = conn.execute("SELECT name FROM teams WHERE espn_id=?", (tid,)).fetchone()
                print(f"  WARNING: No geocode for {name['name']} ({city}, {state})")
        conn.commit()
        print(f"  Added {added} team locations")

    # --- Summary ---
    team_count = conn.execute("SELECT COUNT(*) FROM team_locations").fetchone()[0]
    venue_count = conn.execute("SELECT COUNT(*) FROM venue_locations").fetchone()[0]
    print(f"\nDone: {team_count} team locations, {venue_count} venue locations in database")

    # Show a few examples
    print("\nSample team locations:")
    for row in conn.execute("""
        SELECT tl.espn_id, t.name, tl.city, tl.state, tl.latitude, tl.longitude, tl.timezone
        FROM team_locations tl
        JOIN teams t ON t.espn_id = tl.espn_id
        ORDER BY t.name
        LIMIT 5
    """).fetchall():
        print(f"  {row['name']:35s} {row['city']:20s} {row['state']:5s} "
              f"({row['latitude']:.4f}, {row['longitude']:.4f}) {row['timezone']}")

    conn.close()


if __name__ == '__main__':
    main()
