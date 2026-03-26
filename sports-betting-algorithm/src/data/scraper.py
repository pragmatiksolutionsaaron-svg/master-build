"""
NCAA Basketball Data Scraper
Fetches real-time team statistics from public sports data sources.
"""

import json
import re
import time
from typing import Optional
from urllib.parse import urljoin

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .team_stats import TeamStats


# ESPN API endpoints (public, no auth required)
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_SCOREBOARD = f"{ESPN_BASE}/scoreboard"
ESPN_RANKINGS = f"{ESPN_BASE}/rankings"
ESPN_TEAMS = f"{ESPN_BASE}/teams"


def fetch_espn_team_stats(team_id: str) -> Optional[dict]:
    """Fetch team statistics from ESPN's public API."""
    if not HAS_REQUESTS:
        return None
    try:
        url = f"{ESPN_TEAMS}/{team_id}/statistics"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [WARN] ESPN fetch failed for team {team_id}: {e}")
        return None


def fetch_espn_scoreboard(dates: Optional[str] = None) -> Optional[dict]:
    """Fetch current/upcoming games from ESPN."""
    if not HAS_REQUESTS:
        return None
    try:
        params = {}
        if dates:
            params['dates'] = dates
        params['groups'] = '100'  # NCAA tournament group
        params['limit'] = '50'
        resp = requests.get(ESPN_SCOREBOARD, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [WARN] ESPN scoreboard fetch failed: {e}")
        return None


def fetch_espn_rankings() -> Optional[dict]:
    """Fetch current rankings from ESPN."""
    if not HAS_REQUESTS:
        return None
    try:
        resp = requests.get(ESPN_RANKINGS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [WARN] ESPN rankings fetch failed: {e}")
        return None


def fetch_team_page(team_id: str) -> Optional[dict]:
    """Fetch a team's summary page data from ESPN."""
    if not HAS_REQUESTS:
        return None
    try:
        url = f"{ESPN_TEAMS}/{team_id}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  [WARN] Team page fetch failed for {team_id}: {e}")
        return None


def parse_espn_stats(raw_stats: dict, team_name: str, seed: int, conference: str) -> TeamStats:
    """Parse ESPN API response into a TeamStats object."""
    stats = TeamStats(name=team_name, seed=seed, conference=conference)

    try:
        splits = raw_stats.get('results', {}).get('stats', {})
        if not splits:
            # Try alternate structure
            splits = raw_stats.get('statistics', {})

        # Extract categories
        categories = {}
        stat_list = splits.get('categories', splits.get('splits', {}).get('categories', []))
        for cat in stat_list:
            cat_name = cat.get('name', cat.get('displayName', ''))
            for stat in cat.get('stats', []):
                key = stat.get('name', stat.get('abbreviation', ''))
                val = stat.get('value', stat.get('displayValue', 0))
                try:
                    categories[key] = float(val)
                except (ValueError, TypeError):
                    categories[key] = val

        # Map to TeamStats fields
        stats.points_per_game = categories.get('avgPoints', categories.get('PPG', 0))
        stats.field_goal_pct = categories.get('fieldGoalPct', categories.get('FG%', 0))
        stats.three_point_pct = categories.get('threePointFieldGoalPct', categories.get('3P%', 0))
        stats.free_throw_pct = categories.get('freeThrowPct', categories.get('FT%', 0))
        stats.assists_per_game = categories.get('avgAssists', categories.get('APG', 0))
        stats.turnovers_per_game = categories.get('avgTurnovers', categories.get('TOPG', 0))
        stats.steals_per_game = categories.get('avgSteals', categories.get('SPG', 0))
        stats.blocks_per_game = categories.get('avgBlocks', categories.get('BPG', 0))
        stats.offensive_rebounds_per_game = categories.get('avgOffensiveRebounds', categories.get('ORPG', 0))
        stats.defensive_rebounds_per_game = categories.get('avgDefensiveRebounds', categories.get('DRPG', 0))

    except Exception as e:
        print(f"  [WARN] Stats parsing incomplete for {team_name}: {e}")

    return stats


class NCAADataCollector:
    """Orchestrates data collection from multiple sources."""

    def __init__(self, rate_limit_delay: float = 1.0):
        self.rate_limit_delay = rate_limit_delay
        self._cache = {}

    def collect_team_data(self, team_id: str, team_name: str,
                          seed: int, conference: str) -> TeamStats:
        """Collect comprehensive data for a single team."""
        cache_key = f"{team_id}_{team_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        print(f"  Fetching data for {team_name} (#{seed} seed)...")

        # Fetch from ESPN
        raw_stats = fetch_espn_team_stats(team_id)
        if raw_stats:
            stats = parse_espn_stats(raw_stats, team_name, seed, conference)
        else:
            stats = TeamStats(name=team_name, seed=seed, conference=conference)

        # Fetch team page for record
        team_page = fetch_team_page(team_id)
        if team_page:
            try:
                team_info = team_page.get('team', {})
                record = team_info.get('record', {}).get('items', [{}])[0]
                summary = record.get('summary', '0-0')
                parts = summary.split('-')
                if len(parts) >= 2:
                    stats.wins = int(parts[0])
                    stats.losses = int(parts[1])
            except Exception:
                pass

        time.sleep(self.rate_limit_delay)
        self._cache[cache_key] = stats
        return stats

    def collect_scoreboard(self, date_str: Optional[str] = None) -> list[dict]:
        """Collect upcoming games from ESPN scoreboard."""
        data = fetch_espn_scoreboard(date_str)
        if not data:
            return []

        games = []
        for event in data.get('events', []):
            game = {
                'id': event.get('id'),
                'name': event.get('name'),
                'date': event.get('date'),
                'status': event.get('status', {}).get('type', {}).get('description'),
            }
            competitions = event.get('competitions', [{}])
            if competitions:
                comp = competitions[0]
                teams = comp.get('competitors', [])
                for team in teams:
                    side = 'home' if team.get('homeAway') == 'home' else 'away'
                    game[f'{side}_team'] = team.get('team', {}).get('displayName')
                    game[f'{side}_seed'] = team.get('curatedRank', {}).get('current')
                    game[f'{side}_score'] = team.get('score')
                    game[f'{side}_id'] = team.get('team', {}).get('id')

                # Odds
                odds = comp.get('odds', [{}])
                if odds:
                    game['spread'] = odds[0].get('details')
                    game['over_under'] = odds[0].get('overUnder')

                game['venue'] = comp.get('venue', {}).get('fullName')

            games.append(game)

        return games
