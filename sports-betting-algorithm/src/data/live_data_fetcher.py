"""
Live Data Fetcher
Pulls real-time NCAA tournament data from ESPN's public API
and attempts to populate team stats automatically.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from .team_stats import TeamStats, Matchup


ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"


class LiveDataFetcher:
    """Fetches live tournament data from ESPN's public API."""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session() if HAS_REQUESTS else None
        if self.session:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (compatible; NCAA-Predictor/1.0)'
            })

    def fetch_tournament_games(self, date_str: Optional[str] = None) -> list[dict]:
        """
        Fetch NCAA tournament games for a given date.
        date_str format: YYYYMMDD
        """
        if not self.session:
            print("  [ERROR] requests library not installed. Run: pip install requests")
            return []

        if date_str is None:
            # Try tomorrow's date
            tomorrow = datetime.now() + timedelta(days=1)
            date_str = tomorrow.strftime("%Y%m%d")

        print(f"  Fetching tournament games for {date_str}...")

        try:
            # Fetch scoreboard
            url = f"{ESPN_BASE}/scoreboard"
            params = {
                'dates': date_str,
                'groups': '100',  # NCAA tournament
                'limit': '50',
            }
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            games = []
            for event in data.get('events', []):
                game = self._parse_event(event)
                if game:
                    games.append(game)

            # If no tournament games found, try without group filter
            if not games:
                print("  No tournament games found with group filter, trying broader search...")
                params.pop('groups', None)
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                for event in data.get('events', []):
                    game = self._parse_event(event)
                    if game:
                        games.append(game)

            print(f"  Found {len(games)} games")
            return games

        except Exception as e:
            print(f"  [ERROR] Failed to fetch games: {e}")
            return []

    def fetch_team_statistics(self, team_id: str, team_name: str) -> dict:
        """Fetch detailed statistics for a team from ESPN."""
        if not self.session:
            return {}

        cache_file = os.path.join(self.cache_dir, f"team_{team_id}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)

        print(f"  Fetching stats for {team_name} (ID: {team_id})...")
        stats = {}

        try:
            # Team summary
            url = f"{ESPN_BASE}/teams/{team_id}"
            resp = self.session.get(url, timeout=15)
            if resp.ok:
                team_data = resp.json()
                stats['team'] = team_data.get('team', {})
            time.sleep(0.5)

            # Team statistics
            url = f"{ESPN_BASE}/teams/{team_id}/statistics"
            resp = self.session.get(url, timeout=15)
            if resp.ok:
                stats['statistics'] = resp.json()
            time.sleep(0.5)

            # Team record
            url = f"{ESPN_BASE}/teams/{team_id}/record"
            resp = self.session.get(url, timeout=15)
            if resp.ok:
                stats['record'] = resp.json()
            time.sleep(0.5)

            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(stats, f, indent=2)

        except Exception as e:
            print(f"  [WARN] Partial data for {team_name}: {e}")

        return stats

    def build_team_stats(self, team_id: str, team_name: str,
                         seed: int, conference: str) -> TeamStats:
        """Build a TeamStats object from live ESPN data."""
        raw = self.fetch_team_statistics(team_id, team_name)
        ts = TeamStats(name=team_name, seed=seed, conference=conference)

        # Parse record
        try:
            team_info = raw.get('team', {})
            record_items = team_info.get('record', {}).get('items', [])
            if record_items:
                summary = record_items[0].get('summary', '0-0')
                parts = summary.split('-')
                ts.wins = int(parts[0])
                ts.losses = int(parts[1]) if len(parts) > 1 else 0
        except Exception:
            pass

        # Parse statistics
        try:
            stat_data = raw.get('statistics', {})
            categories = stat_data.get('results', {}).get('stats', {}).get('categories', [])
            if not categories:
                # Try alternate path
                splits = stat_data.get('statistics', {}).get('splits', {})
                categories = splits.get('categories', [])

            stat_map = {}
            for cat in categories:
                for s in cat.get('stats', []):
                    name = s.get('name', s.get('abbreviation', ''))
                    try:
                        stat_map[name] = float(s.get('value', s.get('displayValue', 0)))
                    except (ValueError, TypeError):
                        stat_map[name] = s.get('displayValue', '')

            # Map ESPN stat names to TeamStats fields
            ts.points_per_game = stat_map.get('avgPoints', stat_map.get('points', 0))
            ts.field_goal_pct = stat_map.get('fieldGoalPct', 0) / 100 if stat_map.get('fieldGoalPct', 0) > 1 else stat_map.get('fieldGoalPct', 0)
            ts.three_point_pct = stat_map.get('threePointFieldGoalPct', 0) / 100 if stat_map.get('threePointFieldGoalPct', 0) > 1 else stat_map.get('threePointFieldGoalPct', 0)
            ts.free_throw_pct = stat_map.get('freeThrowPct', 0) / 100 if stat_map.get('freeThrowPct', 0) > 1 else stat_map.get('freeThrowPct', 0)
            ts.assists_per_game = stat_map.get('avgAssists', stat_map.get('assists', 0))
            ts.turnovers_per_game = stat_map.get('avgTurnovers', stat_map.get('turnovers', 0))
            ts.steals_per_game = stat_map.get('avgSteals', stat_map.get('steals', 0))
            ts.blocks_per_game = stat_map.get('avgBlocks', stat_map.get('blocks', 0))
            ts.offensive_rebounds_per_game = stat_map.get('avgOffensiveRebounds', 0)
            ts.defensive_rebounds_per_game = stat_map.get('avgDefensiveRebounds', 0)
            ts.points_allowed_per_game = stat_map.get('avgPointsAllowed', stat_map.get('pointsAllowed', 0))

        except Exception as e:
            print(f"  [WARN] Stats parsing incomplete for {team_name}: {e}")

        return ts

    def _parse_event(self, event: dict) -> Optional[dict]:
        """Parse a single ESPN event into a structured game dict."""
        try:
            competitions = event.get('competitions', [])
            if not competitions:
                return None

            comp = competitions[0]
            competitors = comp.get('competitors', [])
            if len(competitors) < 2:
                return None

            game = {
                'event_id': event.get('id'),
                'name': event.get('name', ''),
                'short_name': event.get('shortName', ''),
                'date': event.get('date', ''),
                'status': event.get('status', {}).get('type', {}).get('description', ''),
                'venue': comp.get('venue', {}).get('fullName', 'TBD'),
                'notes': [n.get('headline', '') for n in comp.get('notes', [])],
                'teams': [],
            }

            # Odds
            odds_list = comp.get('odds', [])
            if odds_list:
                odds = odds_list[0]
                game['spread_detail'] = odds.get('details', '')
                game['over_under'] = odds.get('overUnder')
                game['spread_value'] = odds.get('spread')
                # Home/away odds
                for ha in odds.get('homeTeamOdds', {}).get('moneyLine', [None]), odds.get('awayTeamOdds', {}).get('moneyLine', [None]):
                    pass  # moneyline parsing varies
                home_odds = odds.get('homeTeamOdds', {})
                away_odds = odds.get('awayTeamOdds', {})
                game['home_moneyline'] = home_odds.get('moneyLine')
                game['away_moneyline'] = away_odds.get('moneyLine')
                game['home_spread_odds'] = home_odds.get('spreadOdds')
                game['away_spread_odds'] = away_odds.get('spreadOdds')

            for team_data in competitors:
                team = {
                    'id': team_data.get('team', {}).get('id'),
                    'name': team_data.get('team', {}).get('displayName', ''),
                    'abbreviation': team_data.get('team', {}).get('abbreviation', ''),
                    'seed': None,
                    'home_away': team_data.get('homeAway', ''),
                    'score': team_data.get('score'),
                    'winner': team_data.get('winner', False),
                }

                # Get seed from curatedRank
                rank = team_data.get('curatedRank', {})
                if rank.get('current'):
                    team['seed'] = rank['current']

                game['teams'].append(team)

            return game

        except Exception as e:
            print(f"  [WARN] Failed to parse event: {e}")
            return None

    def auto_discover_matchups(self, date_str: Optional[str] = None) -> tuple[dict, list]:
        """
        Automatically discover tournament matchups and build team data.
        Returns (teams_dict, matchups_list).
        """
        games = self.fetch_tournament_games(date_str)
        if not games:
            print("  No games found. You may need to provide data manually via config files.")
            return {}, []

        teams = {}
        matchups = []

        for game in games:
            game_teams = game.get('teams', [])
            if len(game_teams) < 2:
                continue

            built_teams = []
            for t in game_teams:
                team_id = t.get('id', '')
                team_name = t.get('name', '')
                seed = t.get('seed') or 0
                if team_name and team_name not in teams:
                    ts = self.build_team_stats(team_id, team_name, seed, '')
                    teams[team_name] = ts
                elif team_name in teams:
                    ts = teams[team_name]
                else:
                    continue
                built_teams.append(ts)

            if len(built_teams) == 2:
                # Determine spread
                spread = None
                over_under = None
                ml_a = None
                ml_b = None

                if game.get('spread_value'):
                    try:
                        spread = float(game['spread_value'])
                    except (ValueError, TypeError):
                        pass
                if game.get('over_under'):
                    try:
                        over_under = float(game['over_under'])
                    except (ValueError, TypeError):
                        pass
                ml_a = game.get('away_moneyline')
                ml_b = game.get('home_moneyline')

                notes = game.get('notes', [])
                round_name = notes[0] if notes else 'NCAA Tournament'

                matchup = Matchup(
                    game_id=game.get('event_id', ''),
                    team_a=built_teams[0],
                    team_b=built_teams[1],
                    round_name=round_name,
                    game_time=game.get('date', 'TBD'),
                    venue=game.get('venue', 'TBD'),
                    spread=spread,
                    over_under=over_under,
                    team_a_moneyline=ml_a,
                    team_b_moneyline=ml_b,
                )
                matchups.append(matchup)

        return teams, matchups
