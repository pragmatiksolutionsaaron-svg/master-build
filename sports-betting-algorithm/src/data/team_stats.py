"""
Team Statistics Data Module
Collects and structures NCAA basketball team statistics from multiple sources.
Uses ESPN, NCAA, and sports-reference style data endpoints.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TeamStats:
    """Comprehensive team statistics for prediction modeling."""
    name: str
    seed: int
    conference: str

    # Record
    wins: int = 0
    losses: int = 0

    # Offensive stats (per game)
    points_per_game: float = 0.0
    field_goal_pct: float = 0.0
    three_point_pct: float = 0.0
    free_throw_pct: float = 0.0
    offensive_rebounds_per_game: float = 0.0
    assists_per_game: float = 0.0
    turnovers_per_game: float = 0.0

    # Defensive stats (per game)
    points_allowed_per_game: float = 0.0
    steals_per_game: float = 0.0
    blocks_per_game: float = 0.0
    defensive_rebounds_per_game: float = 0.0
    opponent_field_goal_pct: float = 0.0
    opponent_three_point_pct: float = 0.0

    # Advanced metrics
    adjusted_offensive_efficiency: float = 0.0  # Points per 100 possessions (adjusted)
    adjusted_defensive_efficiency: float = 0.0  # Points allowed per 100 possessions (adjusted)
    tempo: float = 0.0  # Possessions per 40 minutes
    strength_of_schedule: float = 0.0  # SOS rating
    net_ranking: int = 0  # NCAA NET ranking
    bpi: float = 0.0  # Basketball Power Index
    elo_rating: float = 1500.0  # ELO rating

    # Momentum / recent form
    last_10_wins: int = 0
    last_10_losses: int = 0
    win_streak: int = 0
    tournament_wins: int = 0  # Wins in current tournament

    # Situational
    road_wins: int = 0
    road_losses: int = 0
    neutral_site_wins: int = 0
    neutral_site_losses: int = 0
    vs_top_25_wins: int = 0
    vs_top_25_losses: int = 0

    # Roster metrics
    avg_height_inches: float = 0.0
    experience_score: float = 0.0  # Weighted years of experience
    bench_points_per_game: float = 0.0
    depth_score: float = 0.0  # How deep is the rotation

    @property
    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def scoring_margin(self) -> float:
        return self.points_per_game - self.points_allowed_per_game

    @property
    def net_efficiency(self) -> float:
        return self.adjusted_offensive_efficiency - self.adjusted_defensive_efficiency

    @property
    def turnover_margin(self) -> float:
        return self.steals_per_game - self.turnovers_per_game

    @property
    def momentum_score(self) -> float:
        """Composite momentum metric (0-100)."""
        recent_form = (self.last_10_wins / 10.0) * 40
        streak_bonus = min(self.win_streak * 5, 30)
        tourney_bonus = self.tournament_wins * 10
        return min(recent_form + streak_bonus + tourney_bonus, 100)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['win_pct'] = self.win_pct
        d['scoring_margin'] = self.scoring_margin
        d['net_efficiency'] = self.net_efficiency
        d['turnover_margin'] = self.turnover_margin
        d['momentum_score'] = self.momentum_score
        return d


@dataclass
class Matchup:
    """A single game matchup between two teams."""
    game_id: str
    team_a: TeamStats
    team_b: TeamStats
    round_name: str  # e.g., "Final Four", "Elite Eight"
    game_time: str
    venue: str
    spread: Optional[float] = None  # Positive = team_b favored
    over_under: Optional[float] = None
    team_a_moneyline: Optional[int] = None
    team_b_moneyline: Optional[int] = None


def load_teams_from_json(filepath: str) -> dict[str, TeamStats]:
    """Load team data from a JSON configuration file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    teams = {}
    for team_data in data.get('teams', []):
        name = team_data['name']
        teams[name] = TeamStats(**team_data)
    return teams


def load_matchups_from_json(filepath: str, teams: dict[str, TeamStats]) -> list[Matchup]:
    """Load matchup data from a JSON configuration file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    matchups = []
    for m in data.get('matchups', []):
        team_a = teams.get(m['team_a'])
        team_b = teams.get(m['team_b'])
        if team_a and team_b:
            matchups.append(Matchup(
                game_id=m.get('game_id', f"{m['team_a']}_vs_{m['team_b']}"),
                team_a=team_a,
                team_b=team_b,
                round_name=m.get('round_name', 'Unknown'),
                game_time=m.get('game_time', 'TBD'),
                venue=m.get('venue', 'TBD'),
                spread=m.get('spread'),
                over_under=m.get('over_under'),
                team_a_moneyline=m.get('team_a_moneyline'),
                team_b_moneyline=m.get('team_b_moneyline'),
            ))
    return matchups
