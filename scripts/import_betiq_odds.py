"""
Import clean odds data from BetIQ/TeamRankings into game_odds table.

Matches BetIQ games to our ESPN games by (date, home_team, away_team).
Replaces any existing scrambled SportsData.io odds with clean BetIQ data.

Usage:
    python scripts/import_betiq_odds.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection


def moneyline_to_prob(ml: int) -> float:
    """Convert American moneyline odds to implied probability."""
    if ml is None:
        return None
    if ml > 0:
        return 100 / (ml + 100)
    else:
        return abs(ml) / (abs(ml) + 100)


# BetIQ short name -> ESPN full name
BETIQ_TO_ESPN_NAME = {
    'Abl Christian': 'Abilene Christian Wildcats',
    'Air Force': 'Air Force Falcons',
    'Akron': 'Akron Zips',
    'Alabama': 'Alabama Crimson Tide',
    'Alabama A&M': 'Alabama A&M Bulldogs',
    'Alabama St': 'Alabama State Hornets',
    'Albany': 'UAlbany Great Danes',
    'Alcorn St': 'Alcorn State Braves',
    'American': 'American University Eagles',
    'App State': 'App State Mountaineers',
    'Arizona': 'Arizona Wildcats',
    'Arizona St': 'Arizona State Sun Devils',
    'Arkansas': 'Arkansas Razorbacks',
    'Arkansas St': 'Arkansas State Red Wolves',
    'Army': 'Army Black Knights',
    'Auburn': 'Auburn Tigers',
    'Austin Peay': 'Austin Peay Governors',
    'BYU': 'BYU Cougars',
    'Ball St': 'Ball State Cardinals',
    'Baylor': 'Baylor Bears',
    'Bellarmine': 'Bellarmine Knights',
    'Belmont': 'Belmont Bruins',
    'Bethune': 'Bethune-Cookman Wildcats',
    'Binghamton': 'Binghamton Bearcats',
    'Boise St': 'Boise State Broncos',
    'Boston College': 'Boston College Eagles',
    'Boston U': 'Boston University Terriers',
    'Bowling Green': 'Bowling Green Falcons',
    'Bradley': 'Bradley Braves',
    'Brown': 'Brown Bears',
    'Bryant': 'Bryant Bulldogs',
    'Bucknell': 'Bucknell Bison',
    'Buffalo': 'Buffalo Bulls',
    'Butler': 'Butler Bulldogs',
    'Cal Baptist': 'California Baptist Lancers',
    'Cal Poly': 'Cal Poly Mustangs',
    'Cal St Bakersfld': 'Cal State Bakersfield Roadrunners',
    'Cal St Fullerton': 'Cal State Fullerton Titans',
    'Cal St Northridge': 'Cal State Northridge Matadors',
    'Campbell': 'Campbell Fighting Camels',
    'Canisius': 'Canisius Golden Griffins',
    'Cent Arkansas': 'Central Arkansas Bears',
    'Cent Connecticut': 'Central Connecticut Blue Devils',
    'Cent Michigan': 'Central Michigan Chippewas',
    'Charleston So': 'Charleston Southern Buccaneers',
    'Charlotte': 'Charlotte 49ers',
    'Chattanooga': 'Chattanooga Mocs',
    'Chicago St': 'Chicago State Cougars',
    'Cincinnati': 'Cincinnati Bearcats',
    'Citadel': 'The Citadel Bulldogs',
    'Clemson': 'Clemson Tigers',
    'Cleveland St': 'Cleveland State Vikings',
    'Coastal Carolina': 'Coastal Carolina Chanticleers',
    'Col of Charleston': 'Charleston Cougars',
    'Colgate': 'Colgate Raiders',
    'Colorado': 'Colorado Buffaloes',
    'Colorado St': 'Colorado State Rams',
    'Columbia': 'Columbia Lions',
    'Connecticut': 'UConn Huskies',
    'Coppin St': 'Coppin State Eagles',
    'Cornell': 'Cornell Big Red',
    'Corpus Christi': 'Texas A&M-Corpus Christi Islanders',
    'Creighton': 'Creighton Bluejays',
    'Dartmouth': 'Dartmouth Big Green',
    'Davidson': 'Davidson Wildcats',
    'Dayton': 'Dayton Flyers',
    'DePaul': 'DePaul Blue Demons',
    'Delaware': 'Delaware Blue Hens',
    'Delaware St': 'Delaware State Hornets',
    'Denver': 'Denver Pioneers',
    'Detroit': 'Detroit Mercy Titans',
    'Drake': 'Drake Bulldogs',
    'Drexel': 'Drexel Dragons',
    'Duke': 'Duke Blue Devils',
    'Duquesne': 'Duquesne Dukes',
    'E Illinois': 'Eastern Illinois Panthers',
    'E Kentucky': 'Eastern Kentucky Colonels',
    'E Michigan': 'Eastern Michigan Eagles',
    'E Tennessee St': 'East Tennessee State Buccaneers',
    'E Texas A&M': 'East Texas A&M Lions',
    'E Washington': 'Eastern Washington Eagles',
    'Elon': 'Elon Phoenix',
    'Evansville': 'Evansville Purple Aces',
    'F Dickinson': 'Fairleigh Dickinson Knights',
    'FGCU': 'Florida Gulf Coast Eagles',
    'FIU': 'Florida International Panthers',
    'Fairfield': 'Fairfield Stags',
    'Fl Atlantic': 'Florida Atlantic Owls',
    'Florida': 'Florida Gators',
    'Florida St': 'Florida State Seminoles',
    'Fordham': 'Fordham Rams',
    'Fresno St': 'Fresno State Bulldogs',
    'Furman': 'Furman Paladins',
    'Ga Southern': 'Georgia Southern Eagles',
    'Gardner-Webb': "Gardner-Webb Runnin' Bulldogs",
    'Geo Mason': 'George Mason Patriots',
    'Geo Washington': 'George Washington Revolutionaries',
    'Georgetown': 'Georgetown Hoyas',
    'Georgia': 'Georgia Bulldogs',
    'Georgia St': 'Georgia State Panthers',
    'Georgia Tech': 'Georgia Tech Yellow Jackets',
    'Gonzaga': 'Gonzaga Bulldogs',
    'Grambling': 'Grambling Tigers',
    'Grand Canyon': 'Grand Canyon Lopes',
    'Green Bay': 'Green Bay Phoenix',
    'Hampton': 'Hampton Pirates',
    "Hawai'i": "Hawai'i Rainbow Warriors",
    'High Point': 'High Point Panthers',
    'Hofstra': 'Hofstra Pride',
    'Holy Cross': 'Holy Cross Crusaders',
    'Houston': 'Houston Cougars',
    'Houston Chr': 'Houston Christian Huskies',
    'Howard': 'Howard Bison',
    'IU Indy': 'IU Indianapolis Jaguars',
    'IUPUI': 'IU Indianapolis Jaguars',
    'Idaho': 'Idaho Vandals',
    'Idaho St': 'Idaho State Bengals',
    'Illinois': 'Illinois Fighting Illini',
    'Illinois St': 'Illinois State Redbirds',
    'Incarnate Word': 'Incarnate Word Cardinals',
    'Indiana': 'Indiana Hoosiers',
    'Indiana St': 'Indiana State Sycamores',
    'Iona': 'Iona Gaels',
    'Iowa': 'Iowa Hawkeyes',
    'Iowa St': 'Iowa State Cyclones',
    'Jackson St': 'Jackson State Tigers',
    'Jacksonville': 'Jacksonville Dolphins',
    'Jacksonville St': 'Jacksonville State Gamecocks',
    'Jax State': 'Jacksonville State Gamecocks',
    'James Madison': 'James Madison Dukes',
    'Kansas': 'Kansas Jayhawks',
    'Kansas St': 'Kansas State Wildcats',
    'Kennesaw St': 'Kennesaw State Owls',
    'Kent St': 'Kent State Golden Flashes',
    'Kentucky': 'Kentucky Wildcats',
    'LIU': 'Long Island University Sharks',
    'La Salle': 'La Salle Explorers',
    'La Tech': 'Louisiana Tech Bulldogs',
    'Lafayette': 'Lafayette Leopards',
    'Lamar': 'Lamar Cardinals',
    'Le Moyne': 'Le Moyne Dolphins',
    'Lehigh': 'Lehigh Mountain Hawks',
    'Liberty': 'Liberty Flames',
    'Lipscomb': 'Lipscomb Bisons',
    'Little Rock': 'Little Rock Trojans',
    'Lg Beach St': 'Long Beach State Beach',
    'LMU': 'Loyola Marymount Lions',
    'Longwood': 'Longwood Lancers',
    'Louisville': 'Louisville Cardinals',
    'Loyola MD': 'Loyola Maryland Greyhounds',
    'Loyola-Chicago': 'Loyola Chicago Ramblers',
    'Maine': 'Maine Black Bears',
    'Manhattan': 'Manhattan Jaspers',
    'Marist': 'Marist Red Foxes',
    'Marquette': 'Marquette Golden Eagles',
    'Marshall': 'Marshall Thundering Herd',
    'Maryland': 'Maryland Terrapins',
    'McNeese St': 'McNeese Cowboys',
    'Memphis': 'Memphis Tigers',
    'Mercer': 'Mercer Bears',
    'Mercyhurst': 'Mercyhurst Lakers',
    'Miami FL': 'Miami Hurricanes',
    'Miami OH': 'Miami (OH) RedHawks',
    'Michigan': 'Michigan Wolverines',
    'Michigan St': 'Michigan State Spartans',
    'Mid Tennessee': 'Middle Tennessee Blue Raiders',
    'Minnesota': 'Minnesota Golden Gophers',
    'Miss St': 'Mississippi State Bulldogs',
    'Miss Valley St': 'Mississippi Valley State Delta Devils',
    'Mississippi': 'Ole Miss Rebels',
    'Missouri': 'Missouri Tigers',
    'Missouri St': 'Missouri State Bears',
    'Monmouth': 'Monmouth Hawks',
    'Montana': 'Montana Grizzlies',
    'Montana St': 'Montana State Bobcats',
    'Morehead St': 'Morehead State Eagles',
    'Morgan St': 'Morgan State Bears',
    "Mt St Mary's": "Mount St. Mary's Mountaineers",
    'Murray St': 'Murray State Racers',
    'N Arizona': 'Northern Arizona Lumberjacks',
    'N Colorado': 'Northern Colorado Bears',
    'N Dakota St': 'North Dakota State Bison',
    'N Illinois': 'Northern Illinois Huskies',
    'N Iowa': 'Northern Iowa Panthers',
    'N Kentucky': 'Northern Kentucky Norse',
    'NC A&T': 'North Carolina A&T Aggies',
    'NC Asheville': 'UNC Asheville Bulldogs',
    'NC Central': 'North Carolina Central Eagles',
    'NC Greensboro': 'UNC Greensboro Spartans',
    'NC State': 'NC State Wolfpack',
    'NC Wilmington': 'UNC Wilmington Seahawks',
    'NJIT': 'NJIT Highlanders',
    'NW State': 'Northwestern State Demons',
    'Navy': 'Navy Midshipmen',
    'Nebraska': 'Nebraska Cornhuskers',
    'Nevada': 'Nevada Wolf Pack',
    'New Hampshire': 'New Hampshire Wildcats',
    'New Mexico': 'New Mexico Lobos',
    'New Mexico St': 'New Mexico State Aggies',
    'New Orleans': 'New Orleans Privateers',
    'Niagara': 'Niagara Purple Eagles',
    'Nicholls St': 'Nicholls Colonels',
    'Norfolk St': 'Norfolk State Spartans',
    'North Carolina': 'North Carolina Tar Heels',
    'North Dakota': 'North Dakota Fighting Hawks',
    'North Florida': 'North Florida Ospreys',
    'North Texas': 'North Texas Mean Green',
    'Northeastern': 'Northeastern Huskies',
    'Northwestern': 'Northwestern Wildcats',
    'Notre Dame': 'Notre Dame Fighting Irish',
    'Oakland': 'Oakland Golden Grizzlies',
    'Ohio': 'Ohio Bobcats',
    'Ohio St': 'Ohio State Buckeyes',
    'Oklahoma': 'Oklahoma Sooners',
    'Oklahoma St': 'Oklahoma State Cowboys',
    'Old Dominion': 'Old Dominion Monarchs',
    'Ole Miss': 'Ole Miss Rebels',
    'Omaha': 'Omaha Mavericks',
    'Oral Roberts': 'Oral Roberts Golden Eagles',
    'Oregon': 'Oregon Ducks',
    'Oregon St': 'Oregon State Beavers',
    'Pacific': 'Pacific Tigers',
    'Penn': 'Pennsylvania Quakers',
    'Penn St': 'Penn State Nittany Lions',
    'Pepperdine': 'Pepperdine Waves',
    'Pitt': 'Pittsburgh Panthers',
    'Portland': 'Portland Pilots',
    'Portland St': 'Portland State Vikings',
    'Prairie View': 'Prairie View A&M Panthers',
    'Presbyterian': 'Presbyterian Blue Hose',
    'Princeton': 'Princeton Tigers',
    'Providence': 'Providence Friars',
    'Purdue': 'Purdue Boilermakers',
    'Purdue FW': 'Purdue Fort Wayne Mastodons',
    'Quinnipiac': 'Quinnipiac Bobcats',
    'Radford': 'Radford Highlanders',
    'Rhode Island': 'Rhode Island Rams',
    'Rice': 'Rice Owls',
    'Richmond': 'Richmond Spiders',
    'Rider': 'Rider Broncs',
    'Robert Morris': 'Robert Morris Colonials',
    'Rutgers': 'Rutgers Scarlet Knights',
    'S Alabama': 'South Alabama Jaguars',
    'S Carolina': 'South Carolina Gamecocks',
    'S Carolina St': 'South Carolina State Bulldogs',
    'S Dakota': 'South Dakota Coyotes',
    'S Dakota St': 'South Dakota State Jackrabbits',
    'S Florida': 'South Florida Bulls',
    'S Illinois': 'Southern Illinois Salukis',
    'S Mississippi': 'Southern Miss Golden Eagles',
    'S Utah': 'Southern Utah Thunderbirds',
    'SC Upstate': 'South Carolina Upstate Spartans',
    'SE Missouri St': 'Southeast Missouri State Redhawks',
    'SF Austin': 'Stephen F. Austin Lumberjacks',
    'SMU': 'SMU Mustangs',
    'Sacramento St': 'Sacramento State Hornets',
    'Sacred Heart': 'Sacred Heart Pioneers',
    'Sam Houston St': 'Sam Houston Bearkats',
    'Samford': 'Samford Bulldogs',
    'San Diego': 'San Diego Toreros',
    'San Diego St': 'San Diego State Aztecs',
    'San Francisco': 'San Francisco Dons',
    'San Jose St': 'San José State Spartans',
    'Santa Clara': 'Santa Clara Broncos',
    'Seattle': 'Seattle U Redhawks',
    'Seton Hall': 'Seton Hall Pirates',
    'Siena': 'Siena Saints',
    'SIU-Edwardsville': 'SIU Edwardsville Cougars',
    'South Carolina': 'South Carolina Gamecocks',
    'Southern': 'Southern Jaguars',
    'Southern U': 'Southern Jaguars',
    'St Bonaventure': 'St. Bonaventure Bonnies',
    'St Francis PA': 'Saint Francis Red Flash',
    "St John's": "St. John's Red Storm",
    "St Joseph's": "Saint Joseph's Hawks",
    "St Peter's": "Saint Peter's Peacocks",
    'St Thomas': 'St. Thomas-Minnesota Tommies',
    'Stanford': 'Stanford Cardinal',
    'Stetson': 'Stetson Hatters',
    'Stonehill': 'Stonehill Skyhawks',
    'Stony Brook': 'Stony Brook Seawolves',
    'Syracuse': 'Syracuse Orange',
    'TCU': 'TCU Horned Frogs',
    'TX A&M-Com': 'East Texas A&M Lions',
    'Tarleton St': 'Tarleton State Texans',
    'Temple': 'Temple Owls',
    'Tenn Tech': 'Tennessee Tech Golden Eagles',
    'Tennessee': 'Tennessee Volunteers',
    'Tennessee St': 'Tennessee State Tigers',
    'Texas': 'Texas Longhorns',
    'Texas A&M': 'Texas A&M Aggies',
    'Texas So': 'Texas Southern Tigers',
    'Texas St': 'Texas State Bobcats',
    'Texas Tech': 'Texas Tech Red Raiders',
    'Toledo': 'Toledo Rockets',
    'Towson': 'Towson Tigers',
    'Troy': 'Troy Trojans',
    'Tulane': 'Tulane Green Wave',
    'Tulsa': 'Tulsa Golden Hurricane',
    'TX-Arlington': 'UT Arlington Mavericks',
    'TX-El Paso': 'UTEP Miners',
    'TX-Rio Grande': 'UT Rio Grande Valley Vaqueros',
    'TX-San Antonio': 'UTSA Roadrunners',
    'UC Davis': 'UC Davis Aggies',
    'UC Irvine': 'UC Irvine Anteaters',
    'UC Riverside': 'UC Riverside Highlanders',
    'UCSD': 'UC San Diego Tritons',
    'UCSB': 'UC Santa Barbara Gauchos',
    'UCLA': 'UCLA Bruins',
    'UConn': 'UConn Huskies',
    'UL Lafayette': "Louisiana Ragin' Cajuns",
    'UL Monroe': 'UL Monroe Warhawks',
    'UMBC': 'UMBC Retrievers',
    'UMKC': 'Kansas City Roos',
    'UMass': 'Massachusetts Minutemen',
    'UMass Lowell': 'UMass Lowell River Hawks',
    'UNLV': 'UNLV Rebels',
    'USC': 'USC Trojans',
    'UT Martin': 'UT Martin Skyhawks',
    'Utah': 'Utah Utes',
    'Utah St': 'Utah State Aggies',
    'Utah Tech': 'Utah Tech Trailblazers',
    'Utah Valley': 'Utah Valley Wolverines',
    'VCU': 'VCU Rams',
    'VMI': 'VMI Keydets',
    'Valparaiso': 'Valparaiso Beacons',
    'Vanderbilt': 'Vanderbilt Commodores',
    'Vermont': 'Vermont Catamounts',
    'Villanova': 'Villanova Wildcats',
    'Virginia': 'Virginia Cavaliers',
    'Virginia Tech': 'Virginia Tech Hokies',
    'W Carolina': 'Western Carolina Catamounts',
    'W Illinois': 'Western Illinois Leathernecks',
    'W Kentucky': 'Western Kentucky Hilltoppers',
    'W Michigan': 'Western Michigan Broncos',
    'Wagner': 'Wagner Seahawks',
    'Wake Forest': 'Wake Forest Demon Deacons',
    'Washington': 'Washington Huskies',
    'Washington St': 'Washington State Cougars',
    'Weber St': 'Weber State Wildcats',
    'West Virginia': 'West Virginia Mountaineers',
    'WI-Green Bay': 'Green Bay Phoenix',
    'WI-Milwaukee': 'Milwaukee Panthers',
    'Wichita St': 'Wichita State Shockers',
    'William & Mary': 'William & Mary Tribe',
    'Winthrop': 'Winthrop Eagles',
    'Wisconsin': 'Wisconsin Badgers',
    'Wm & Mary': 'William & Mary Tribe',
    'Wofford': 'Wofford Terriers',
    'Wright St': 'Wright State Raiders',
    'Wyoming': 'Wyoming Cowboys',
    'Xavier': 'Xavier Musketeers',
    'Yale': 'Yale Bulldogs',
    'Youngstown St': 'Youngstown State Penguins',
    # Variants (alternative BetIQ spellings across seasons)
    'AR-Pine Bluff': 'Arkansas-Pine Bluff Golden Lions',
    'MD-East Shore': 'Maryland Eastern Shore Hawks',
    'IL-Chicago': 'UIC Flames',
    'N Carolina': 'North Carolina Tar Heels',
    'Md-East Shore': 'Maryland Eastern Shore Hawks',
    'C Arkansas': 'Central Arkansas Bears',
    'C Connecticut': 'Central Connecticut Blue Devils',
    'C Michigan': 'Central Michigan Chippewas',
    'CS Bakersfield': 'Cal State Bakersfield Roadrunners',
    'CS Fullerton': 'Cal State Fullerton Titans',
    'CS Northridge': 'Cal State Northridge Matadors',
    'California': 'California Golden Bears',
    'Charleston': 'Charleston Cougars',
    'Coastal Car': 'Coastal Carolina Chanticleers',
    'Detroit Mercy': 'Detroit Mercy Titans',
    'E Carolina': 'East Carolina Pirates',
    'Florida A&M': 'Florida A&M Rattlers',
    'Florida Atlantic': 'Florida Atlantic Owls',
    'Florida Intl': 'Florida International Panthers',
    'G Washington': 'George Washington Revolutionaries',
    'George Mason': 'George Mason Patriots',
    'Georgia So': 'Georgia Southern Eagles',
    'Harvard': 'Harvard Crimson',
    'Hou Christian': 'Houston Christian Huskies',
    'Illinois Chicago': 'UIC Flames',
    'J Madison': 'James Madison Dukes',
    'Kansas City': 'Kansas City Roos',
    'LSU': 'LSU Tigers',
    'Long Beach St': 'Long Beach State Beach',
    'Louisiana': "Louisiana Ragin' Cajuns",
    'Louisiana Tech': 'Louisiana Tech Bulldogs',
    'Loyola Chi': 'Loyola Chicago Ramblers',
    'Loyola Mymt': 'Loyola Marymount Lions',
    'Maryland ES': 'Maryland Eastern Shore Hawks',
    'McNeese': 'McNeese Cowboys',
    'Merrimack': 'Merrimack Warriors',
    'Miami': 'Miami Hurricanes',
    'Middle Tenn': 'Middle Tennessee Blue Raiders',
    'Milwaukee': 'Milwaukee Panthers',
    'Mississippi St': 'Mississippi State Bulldogs',
    'N Alabama': 'North Alabama Lions',
    'N Florida': 'North Florida Ospreys',
    'N Texas': 'North Texas Mean Green',
    'New Haven': 'New Haven Chargers',
    'Nicholls': 'Nicholls Colonels',
    'Pittsburgh': 'Pittsburgh Panthers',
    'SE Louisiana': 'SE Louisiana Lions',
    'SIU Edward': 'SIU Edwardsville Cougars',
    "Saint Joseph's": "Saint Joseph's Hawks",
    'Saint Louis': 'Saint Louis Billikens',
    "Saint Mary's": "Saint Mary's Gaels",
    "Saint Peter's": "Saint Peter's Peacocks",
    'Sam Houston': 'Sam Houston Bearkats',
    'South Dakota': 'South Dakota Coyotes',
    'Southern Miss': 'Southern Miss Golden Eagles',
    'Texas A&M-CC': 'Texas A&M-Corpus Christi Islanders',
    'The Citadel': 'The Citadel Bulldogs',
    'UAB': 'UAB Blazers',
    'UCF': 'UCF Knights',
    'UT Arlington': 'UT Arlington Mavericks',
    'UT Rio Grande': 'UT Rio Grande Valley Vaqueros',
    'UTEP': 'UTEP Miners',
    'UTSA': 'UTSA Roadrunners',
    'W Georgia': 'West Georgia Wolves',
}

# Seasons: BetIQ uses "2023-2024" -> our season_year = 2024
SEASON_MAP = {
    '2023-2024': 2024,
    '2024-2025': 2025,
    '2025-2026': 2026,
}


def main():
    conn = get_connection()

    # Build ESPN name -> id map
    espn_teams = conn.execute('SELECT espn_id, name FROM teams').fetchall()
    espn_name_to_id = {t['name']: t['espn_id'] for t in espn_teams}

    # Build BetIQ name -> ESPN id map
    betiq_to_id = {}
    missing_names = set()
    for betiq_name, espn_name in BETIQ_TO_ESPN_NAME.items():
        eid = espn_name_to_id.get(espn_name)
        if eid:
            betiq_to_id[betiq_name] = eid
        else:
            missing_names.add((betiq_name, espn_name))

    if missing_names:
        print(f"Warning: {len(missing_names)} BetIQ names couldn't map to ESPN IDs:")
        for bn, en in sorted(missing_names):
            print(f"  {bn} -> {en}")

    print(f"Team crosswalk: {len(betiq_to_id)} BetIQ names mapped to ESPN IDs")

    # Load BetIQ data (home rows only to avoid double-counting)
    with open('data/betiq_odds_raw.csv') as f:
        reader = csv.DictReader(f)
        all_rows = [r for r in reader]

    target_seasons = set(SEASON_MAP.keys())
    home_rows = [r for r in all_rows
                 if r['loc'] == 'Home'
                 and r['season'] in target_seasons]

    print(f"BetIQ home rows for target seasons: {len(home_rows)}")

    # Clear existing odds data (replacing scrambled SportsData.io data)
    existing = conn.execute('SELECT COUNT(*) as n FROM game_odds').fetchone()['n']
    if existing > 0:
        conn.execute('DELETE FROM game_odds')
        conn.commit()
        print(f"Cleared {existing} existing odds rows")

    # Build game lookup: (date, home_espn_id, away_espn_id) -> game_id
    games = conn.execute('''
        SELECT game_id, DATE(date) as game_date, home_team_id, away_team_id
        FROM games
        WHERE status = 'STATUS_FINAL'
        AND season_year IN (2024, 2025, 2026)
    ''').fetchall()

    game_lookup = {}
    for g in games:
        key = (g['game_date'], g['home_team_id'], g['away_team_id'])
        game_lookup[key] = g['game_id']

    print(f"ESPN games in lookup: {len(game_lookup)}")

    # Match and insert
    matched = 0
    no_ml = 0
    no_team = 0
    no_game = 0
    inserted = 0

    for row in home_rows:
        home_name = row['team']
        away_name = row['opp']
        date = row['game_date']
        season = SEASON_MAP[row['season']]

        # Get ESPN IDs
        home_id = betiq_to_id.get(home_name)
        away_id = betiq_to_id.get(away_name)

        if not home_id or not away_id:
            no_team += 1
            continue

        # Check moneyline
        ml_str = row.get('money_line')
        if not ml_str or ml_str == 'None':
            no_ml += 1
            continue

        # Find matching game
        game_id = game_lookup.get((date, home_id, away_id))
        if not game_id:
            # Try reversed (neutral site games might have swapped home/away)
            game_id = game_lookup.get((date, away_id, home_id))
            if game_id:
                # Swap perspective
                home_id, away_id = away_id, home_id
                home_name, away_name = away_name, home_name

        if not game_id:
            no_game += 1
            continue

        matched += 1

        # Parse odds
        home_ml = int(ml_str)
        # Get away ML from the matching away row
        away_row = next(
            (r for r in all_rows
             if r['game_date'] == date
             and r['team'] == row['opp']
             and r['opp'] == row['team']
             and r['loc'] == 'Away'),
            None
        )
        away_ml = int(away_row['money_line']) if away_row and away_row.get('money_line') not in (None, 'None', '') else None

        spread_str = row.get('spread')
        spread = float(spread_str) if spread_str and spread_str != 'None' else None

        total_str = row.get('total')
        total = float(total_str) if total_str and total_str != 'None' else None

        home_prob = moneyline_to_prob(home_ml)
        away_prob = moneyline_to_prob(away_ml) if away_ml else None

        conn.execute('''
            INSERT OR IGNORE INTO game_odds
            (game_id, home_moneyline, away_moneyline,
             home_spread, away_spread, over_under,
             home_implied_prob, away_implied_prob,
             num_sportsbooks, odds_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_id,
            home_ml,
            away_ml,
            spread,
            -spread if spread is not None else None,
            total,
            home_prob,
            away_prob,
            1,  # BetIQ doesn't report sportsbook count
            'betiq_teamrankings',
        ))
        inserted += 1

    conn.commit()

    print(f"\nResults:")
    print(f"  {matched} games matched")
    print(f"  {inserted} odds rows inserted")
    print(f"  {no_ml} skipped (no moneyline)")
    print(f"  {no_team} skipped (team not in crosswalk)")
    print(f"  {no_game} skipped (no matching ESPN game)")

    # Verify
    by_season = conn.execute('''
        SELECT g.season_year, COUNT(*) as n
        FROM game_odds o
        JOIN games g ON o.game_id = g.game_id
        GROUP BY g.season_year
        ORDER BY g.season_year
    ''').fetchall()
    print(f"\nBy season:")
    for r in by_season:
        total_games = conn.execute(
            'SELECT COUNT(*) as n FROM games WHERE season_year=? AND status="STATUS_FINAL"',
            (r['season_year'],)
        ).fetchone()['n']
        pct = r['n'] / total_games * 100
        print(f"  {r['season_year']}: {r['n']}/{total_games} games ({pct:.1f}%)")

    conn.close()


if __name__ == '__main__':
    main()
