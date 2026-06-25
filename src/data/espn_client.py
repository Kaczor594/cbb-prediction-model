"""
ESPN API Client for College Basketball Data

Provides access to NCAA Division I men's basketball data including:
- Teams and conferences
- Schedules and game results
- Team statistics
- Player information
"""

import time
from datetime import datetime
from typing import Any, Optional

import requests


class ESPNClient:
    """Client for accessing ESPN's undocumented API for college basketball."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

    def __init__(self, rate_limit_seconds: float = 1.0):
        """
        Initialize ESPN client.

        Args:
            rate_limit_seconds: Minimum seconds between requests
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; CBBPredictionModel/1.0)"
        })
        self.rate_limit_seconds = rate_limit_seconds
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Make a GET request to the ESPN API.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data
        """
        self._rate_limit()
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Teams
    # -------------------------------------------------------------------------

    def get_teams(self, limit: int = 400) -> list[dict]:
        """
        Get all NCAA Division I men's basketball teams.

        Args:
            limit: Maximum number of teams to return

        Returns:
            List of team dictionaries
        """
        data = self._get("teams", params={"limit": limit})
        return data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])

    def get_team(self, team_id: int) -> dict:
        """
        Get detailed information for a specific team.

        Args:
            team_id: ESPN team ID

        Returns:
            Team data dictionary
        """
        data = self._get(f"teams/{team_id}")
        return data.get("team", {})

    def get_team_roster(self, team_id: int) -> list[dict]:
        """
        Get roster for a specific team.

        Args:
            team_id: ESPN team ID

        Returns:
            List of player dictionaries
        """
        data = self._get(f"teams/{team_id}/roster")
        return data.get("athletes", [])

    # -------------------------------------------------------------------------
    # Schedules & Results
    # -------------------------------------------------------------------------

    def get_team_schedule(
        self,
        team_id: int,
        season: Optional[int] = None,
    ) -> list[dict]:
        """
        Get schedule and results for a team.

        Args:
            team_id: ESPN team ID
            season: Season year (e.g., 2024 for 2023-24 season)

        Returns:
            List of game dictionaries
        """
        params = {}
        if season:
            params["season"] = season

        data = self._get(f"teams/{team_id}/schedule", params=params)
        return data.get("events", [])

    def get_scoreboard(
        self,
        date: Optional[str] = None,
        conference_id: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Get scoreboard (games) for a specific date.

        Args:
            date: Date string in YYYYMMDD format (defaults to today)
            conference_id: Filter by conference
            limit: Maximum games to return

        Returns:
            List of game dictionaries
        """
        params = {"limit": limit}
        if date:
            params["dates"] = date
        if conference_id:
            params["groups"] = conference_id

        data = self._get("scoreboard", params=params)
        return data.get("events", [])

    def get_game_summary(self, game_id: str) -> dict:
        """
        Get detailed summary for a specific game.

        Args:
            game_id: ESPN game/event ID

        Returns:
            Game summary data
        """
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary"
        self._rate_limit()
        response = self.session.get(url, params={"event": game_id})
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_team_statistics(
        self,
        team_id: int,
        season: Optional[int] = None,
    ) -> dict:
        """
        Get team statistics.

        Args:
            team_id: ESPN team ID
            season: Season year

        Returns:
            Team statistics dictionary
        """
        params = {}
        if season:
            params["season"] = season

        data = self._get(f"teams/{team_id}/statistics", params=params)
        return data.get("results", {})

    # -------------------------------------------------------------------------
    # Conferences
    # -------------------------------------------------------------------------

    def get_conferences(self) -> list[dict]:
        """
        Get all D1 conferences with their team memberships.

        The ESPN groups endpoint returns top-level groups (e.g., "NCAA Division I")
        with actual conferences nested under "children". Each conference includes
        its member teams.

        For each conference, fetches the numeric ESPN group ID by querying one
        member team's detail endpoint (which includes groups.id).

        Returns:
            List of conference dicts with keys: id, name, abbreviation, team_ids
        """
        data = self._get("groups")
        conferences = []

        for group in data.get("groups", []):
            for conf in group.get("children", []):
                team_ids = [t["id"] for t in conf.get("teams", [])]

                # Get the numeric conference ID from one member team
                conf_id = None
                if team_ids:
                    try:
                        team_data = self.get_team(int(team_ids[0]))
                        conf_id = team_data.get("groups", {}).get("id")
                    except Exception:
                        pass

                conferences.append({
                    "id": conf_id,
                    "name": conf.get("name"),
                    "shortName": conf.get("name", "").replace(" Conference", ""),
                    "abbreviation": conf.get("abbreviation"),
                    "team_ids": team_ids,
                })

        return conferences

    def get_conference_standings(
        self,
        conference_id: int,
        season: Optional[int] = None,
    ) -> dict:
        """
        Get standings for a conference.

        Args:
            conference_id: ESPN conference/group ID
            season: Season year

        Returns:
            Conference standings data
        """
        params = {}
        if season:
            params["season"] = season

        data = self._get(f"groups/{conference_id}/standings", params=params)
        return data

    # -------------------------------------------------------------------------
    # Rankings
    # -------------------------------------------------------------------------

    def get_rankings(self, season: Optional[int] = None) -> list[dict]:
        """
        Get current rankings (AP Poll, Coaches Poll, etc.).

        Args:
            season: Season year

        Returns:
            List of ranking dictionaries
        """
        params = {}
        if season:
            params["season"] = season

        data = self._get("rankings", params=params)
        return data.get("rankings", [])


# -----------------------------------------------------------------------------
# Data Extraction Helpers
# -----------------------------------------------------------------------------

def extract_game_data(game: dict) -> dict:
    """
    Extract relevant fields from an ESPN game/event dictionary.

    Args:
        game: Raw ESPN game data

    Returns:
        Cleaned game data dictionary
    """
    competitions = game.get("competitions", [{}])[0]
    competitors = competitions.get("competitors", [])

    # Find home and away teams
    home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
    away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})

    # Extract venue
    venue = competitions.get("venue", {})

    # Extract odds if available
    odds = competitions.get("odds", [{}])[0] if competitions.get("odds") else {}

    return {
        "game_id": game.get("id"),
        "date": game.get("date"),
        "name": game.get("name"),
        "season": game.get("season", {}).get("year"),
        "season_type": game.get("seasonType", {}).get("type"),
        "week": game.get("week", {}).get("number"),
        "neutral_site": competitions.get("neutralSite", False),
        "conference_competition": competitions.get("conferenceCompetition", False),
        "attendance": competitions.get("attendance"),
        "status": competitions.get("status", {}).get("type", {}).get("name"),
        # Home team
        "home_team_id": home_team.get("team", {}).get("id"),
        "home_team_name": home_team.get("team", {}).get("displayName"),
        "home_team_abbrev": home_team.get("team", {}).get("abbreviation"),
        "home_score": home_team.get("score", {}).get("value") if isinstance(home_team.get("score"), dict) else home_team.get("score"),
        "home_winner": home_team.get("winner"),
        "home_rank": home_team.get("curatedRank", {}).get("current"),
        # Away team
        "away_team_id": away_team.get("team", {}).get("id"),
        "away_team_name": away_team.get("team", {}).get("displayName"),
        "away_team_abbrev": away_team.get("team", {}).get("abbreviation"),
        "away_score": away_team.get("score", {}).get("value") if isinstance(away_team.get("score"), dict) else away_team.get("score"),
        "away_winner": away_team.get("winner"),
        "away_rank": away_team.get("curatedRank", {}).get("current"),
        # Venue
        "venue_id": venue.get("id"),
        "venue_name": venue.get("fullName"),
        "venue_city": venue.get("address", {}).get("city"),
        "venue_state": venue.get("address", {}).get("state"),
        # Odds
        "spread": odds.get("spread"),
        "over_under": odds.get("overUnder"),
    }


def extract_team_data(team: dict) -> dict:
    """
    Extract relevant fields from an ESPN team dictionary.

    Args:
        team: Raw ESPN team data

    Returns:
        Cleaned team data dictionary
    """
    team_info = team.get("team", team)  # Handle nested structure

    return {
        "team_id": team_info.get("id"),
        "name": team_info.get("displayName"),
        "short_name": team_info.get("shortDisplayName"),
        "abbreviation": team_info.get("abbreviation"),
        "nickname": team_info.get("nickname"),
        "location": team_info.get("location"),
        "color": team_info.get("color"),
        "alternate_color": team_info.get("alternateColor"),
        "is_active": team_info.get("isActive"),
        "logo_url": team_info.get("logos", [{}])[0].get("href") if team_info.get("logos") else None,
    }
