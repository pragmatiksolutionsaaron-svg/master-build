"""
Core Prediction Engine
Combines multiple statistical models to generate game predictions.

Models implemented:
1. KenPom-Style Efficiency Model - Adjusted offensive/defensive efficiency
2. ELO Rating System - Chess-style rating adapted for basketball
3. Momentum & Situational Model - Recent form, tournament experience
4. Composite Bayesian Ensemble - Weighted combination of all models
"""

import math
from dataclasses import dataclass
from typing import Optional

from ..data.team_stats import TeamStats, Matchup


@dataclass
class GamePrediction:
    """Complete prediction for a single game."""
    matchup: Matchup

    # Score predictions
    team_a_predicted_score: float
    team_b_predicted_score: float

    # Win probabilities
    team_a_win_prob: float
    team_b_win_prob: float

    # Model-specific probabilities
    efficiency_prob_a: float
    elo_prob_a: float
    momentum_prob_a: float
    seed_historical_prob_a: float

    # Confidence
    confidence: float  # 0-100 scale
    model_agreement: float  # How much models agree (0-1)

    # Betting analysis
    predicted_spread: float  # Negative = team_a favored
    predicted_total: float
    spread_edge: Optional[float] = None  # Edge vs market spread
    total_edge: Optional[float] = None  # Edge vs market total

    @property
    def predicted_winner(self) -> str:
        if self.team_a_win_prob >= self.team_b_win_prob:
            return self.matchup.team_a.name
        return self.matchup.team_b.name

    @property
    def predicted_margin(self) -> float:
        return abs(self.team_a_predicted_score - self.team_b_predicted_score)

    @property
    def upset_probability(self) -> float:
        """Probability of the higher-seeded (worse) team winning."""
        if self.matchup.team_a.seed > self.matchup.team_b.seed:
            return self.team_a_win_prob
        return self.team_b_win_prob


class EfficiencyModel:
    """
    KenPom-style adjusted efficiency model.

    The gold standard for college basketball prediction. Measures how many
    points a team scores/allows per 100 possessions, adjusted for opponent
    strength. The predicted score is derived from the expected pace of the
    game and each team's efficiency against the other's defense.
    """

    # National average efficiency (points per 100 possessions)
    NATIONAL_AVG_EFFICIENCY = 100.0
    # Average tempo (possessions per 40 min)
    NATIONAL_AVG_TEMPO = 67.5

    def predict(self, team_a: TeamStats, team_b: TeamStats) -> tuple[float, float, float]:
        """
        Returns (team_a_win_prob, predicted_score_a, predicted_score_b)
        """
        # Calculate expected game tempo
        adj_tempo_a = team_a.tempo if team_a.tempo > 0 else self.NATIONAL_AVG_TEMPO
        adj_tempo_b = team_b.tempo if team_b.tempo > 0 else self.NATIONAL_AVG_TEMPO
        expected_possessions = (adj_tempo_a + adj_tempo_b) / 2.0

        # If we have adjusted efficiencies, use them directly
        if team_a.adjusted_offensive_efficiency > 0 and team_b.adjusted_defensive_efficiency > 0:
            off_eff_a = team_a.adjusted_offensive_efficiency
            def_eff_b = team_b.adjusted_defensive_efficiency
            off_eff_b = team_b.adjusted_offensive_efficiency
            def_eff_a = team_a.adjusted_defensive_efficiency
        else:
            # Estimate from raw stats
            off_eff_a = self._estimate_offensive_efficiency(team_a)
            def_eff_a = self._estimate_defensive_efficiency(team_a)
            off_eff_b = self._estimate_offensive_efficiency(team_b)
            def_eff_b = self._estimate_defensive_efficiency(team_b)

        # Expected efficiency for each team in this matchup
        # Formula: Team's offense adjusted for opponent's defense relative to average
        expected_eff_a = off_eff_a + def_eff_b - self.NATIONAL_AVG_EFFICIENCY
        expected_eff_b = off_eff_b + def_eff_a - self.NATIONAL_AVG_EFFICIENCY

        # Convert to predicted scores
        predicted_a = expected_eff_a * (expected_possessions / 100.0)
        predicted_b = expected_eff_b * (expected_possessions / 100.0)

        # Win probability using log5 method with efficiency margin
        margin = predicted_a - predicted_b
        # Standard deviation of scoring margin in college basketball ~11 points
        win_prob_a = self._margin_to_probability(margin, std_dev=11.0)

        return win_prob_a, predicted_a, predicted_b

    def _estimate_offensive_efficiency(self, team: TeamStats) -> float:
        """Estimate adjusted offensive efficiency from box score stats."""
        if team.points_per_game <= 0:
            return self.NATIONAL_AVG_EFFICIENCY

        # Rough estimation using scoring, shooting efficiency, and turnover rate
        base = (team.points_per_game / self.NATIONAL_AVG_TEMPO) * 100

        # Adjust for shooting quality
        fg_bonus = (team.field_goal_pct - 0.44) * 30  # 44% is ~average
        three_bonus = (team.three_point_pct - 0.34) * 20  # 34% is ~average
        ft_bonus = (team.free_throw_pct - 0.70) * 10  # 70% is ~average

        # Adjust for ball control
        to_penalty = (team.turnovers_per_game - 12.5) * -1.5  # 12.5 is ~average
        assist_bonus = (team.assists_per_game - 14.0) * 0.8

        # SOS adjustment
        sos_adj = team.strength_of_schedule * 3.0

        return base + fg_bonus + three_bonus + ft_bonus + to_penalty + assist_bonus + sos_adj

    def _estimate_defensive_efficiency(self, team: TeamStats) -> float:
        """Estimate adjusted defensive efficiency from box score stats."""
        if team.points_allowed_per_game <= 0:
            return self.NATIONAL_AVG_EFFICIENCY

        base = (team.points_allowed_per_game / self.NATIONAL_AVG_TEMPO) * 100

        # Adjust for defensive quality
        opp_fg_bonus = (0.44 - team.opponent_field_goal_pct) * 30
        opp_three_bonus = (0.34 - team.opponent_three_point_pct) * 20
        steal_bonus = (team.steals_per_game - 6.5) * -0.8
        block_bonus = (team.blocks_per_game - 3.5) * -0.6

        sos_adj = team.strength_of_schedule * 3.0

        return base - opp_fg_bonus - opp_three_bonus - steal_bonus - block_bonus + sos_adj

    @staticmethod
    def _margin_to_probability(margin: float, std_dev: float = 11.0) -> float:
        """Convert point margin to win probability using normal CDF."""
        return 0.5 * (1 + math.erf(margin / (std_dev * math.sqrt(2))))


class ELORatingModel:
    """
    ELO rating system adapted for college basketball.

    Each team carries a rating (default 1500). After each game, ratings
    adjust based on actual vs expected outcome. Higher-rated teams are
    expected to beat lower-rated teams. The system naturally accounts for
    strength of schedule through transitive results.
    """

    K_FACTOR = 32  # Rating adjustment speed
    HOME_ADVANTAGE = 0  # Tournament is neutral site

    def predict(self, team_a: TeamStats, team_b: TeamStats) -> float:
        """Returns team_a win probability based on ELO ratings."""
        elo_a = team_a.elo_rating if team_a.elo_rating != 1500 else self._estimate_elo(team_a)
        elo_b = team_b.elo_rating if team_b.elo_rating != 1500 else self._estimate_elo(team_b)

        expected_a = 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
        return expected_a

    def _estimate_elo(self, team: TeamStats) -> float:
        """Estimate ELO from available stats when no ELO is provided."""
        base = 1500.0

        # Win percentage adjustment
        if team.wins + team.losses > 0:
            win_pct = team.win_pct
            base += (win_pct - 0.5) * 400

        # Scoring margin is highly predictive
        if team.scoring_margin != 0:
            base += team.scoring_margin * 10

        # Strength of schedule
        base += team.strength_of_schedule * 50

        # NET ranking (lower is better)
        if team.net_ranking > 0:
            base += max(0, (68 - team.net_ranking)) * 4

        # Seed-based prior (strong predictor in tournament)
        seed_elo = {
            1: 1800, 2: 1720, 3: 1660, 4: 1620,
            5: 1580, 6: 1540, 7: 1510, 8: 1480,
            9: 1470, 10: 1450, 11: 1440, 12: 1420,
            13: 1380, 14: 1350, 15: 1320, 16: 1280,
        }
        seed_prior = seed_elo.get(team.seed, 1500)

        # Blend estimated ELO with seed prior (seed is strong signal)
        return base * 0.6 + seed_prior * 0.4


class MomentumModel:
    """
    Momentum & situational factors model.

    Captures factors that pure efficiency models miss:
    - Recent form (last 10 games)
    - Win streaks
    - Tournament experience
    - Performance in high-pressure situations (road/neutral games)
    - Roster depth and experience
    """

    def predict(self, team_a: TeamStats, team_b: TeamStats) -> float:
        """Returns team_a win probability based on momentum/situational factors."""
        score_a = self._composite_score(team_a)
        score_b = self._composite_score(team_b)

        # Convert to probability
        diff = score_a - score_b
        prob_a = 1.0 / (1.0 + math.exp(-diff / 15.0))
        return prob_a

    def _composite_score(self, team: TeamStats) -> float:
        """Calculate composite momentum/situational score."""
        score = 0.0

        # Recent form (25% weight)
        if team.last_10_wins + team.last_10_losses > 0:
            recent_pct = team.last_10_wins / (team.last_10_wins + team.last_10_losses)
            score += recent_pct * 25
        else:
            score += team.win_pct * 25

        # Win streak bonus (15% weight)
        score += min(team.win_streak * 2.5, 15)

        # Tournament performance (15% weight)
        score += team.tournament_wins * 5

        # Neutral site performance (15% weight)
        neutral_total = team.neutral_site_wins + team.neutral_site_losses
        if neutral_total > 0:
            score += (team.neutral_site_wins / neutral_total) * 15
        else:
            score += team.win_pct * 15

        # Quality wins (15% weight)
        top25_total = team.vs_top_25_wins + team.vs_top_25_losses
        if top25_total > 0:
            score += (team.vs_top_25_wins / top25_total) * 15
        else:
            score += 7.5

        # Experience (10% weight)
        score += min(team.experience_score * 2.5, 10)

        # Depth (5% weight)
        score += min(team.depth_score * 1.25, 5)

        return score


class SeedHistoricalModel:
    """
    Historical seed matchup probabilities.

    Uses decades of NCAA tournament data showing how each seed matchup
    historically plays out. This is a powerful baseline - seeds alone
    predict ~70% of tournament games correctly.
    """

    # Historical win rates for the LOWER seed (better team) in each matchup
    # Source: Aggregated from 1985-2025 NCAA tournament data
    HISTORICAL_MATCHUPS = {
        (1, 16): 0.991,
        (1, 8): 0.800,
        (1, 9): 0.850,
        (1, 4): 0.690,
        (1, 5): 0.720,
        (1, 2): 0.530,
        (1, 3): 0.580,
        (1, 6): 0.680,
        (1, 7): 0.700,
        (1, 10): 0.770,
        (1, 11): 0.780,
        (2, 15): 0.943,
        (2, 7): 0.670,
        (2, 10): 0.680,
        (2, 3): 0.540,
        (2, 6): 0.620,
        (2, 11): 0.650,
        (3, 14): 0.855,
        (3, 6): 0.560,
        (3, 11): 0.600,
        (4, 13): 0.790,
        (4, 5): 0.550,
        (4, 12): 0.640,
        (5, 12): 0.645,
        (5, 4): 0.450,
        (6, 11): 0.625,
        (6, 3): 0.440,
        (7, 10): 0.605,
        (7, 2): 0.330,
        (8, 9): 0.510,
        (8, 1): 0.200,
    }

    def predict(self, team_a: TeamStats, team_b: TeamStats) -> float:
        """Returns team_a win probability based on historical seed matchups."""
        seed_a = team_a.seed
        seed_b = team_b.seed

        if seed_a == seed_b:
            return 0.50

        lower_seed = min(seed_a, seed_b)
        higher_seed = max(seed_a, seed_b)

        # Look up historical probability for lower seed winning
        lower_seed_prob = self.HISTORICAL_MATCHUPS.get(
            (lower_seed, higher_seed), 0.50
        )

        # If no exact match, estimate from seed difference
        if (lower_seed, higher_seed) not in self.HISTORICAL_MATCHUPS:
            seed_diff = higher_seed - lower_seed
            lower_seed_prob = 0.50 + seed_diff * 0.03  # ~3% per seed line

        # Return probability for team_a
        if seed_a <= seed_b:
            return lower_seed_prob
        return 1.0 - lower_seed_prob


class CompositePredictor:
    """
    Bayesian ensemble that combines all models with learned weights.

    Weight assignment rationale:
    - Efficiency (40%): Most predictive single model in college basketball
    - ELO (25%): Strong transitive strength measure, accounts for SOS
    - Seed Historical (20%): Powerful baseline, especially in early rounds
    - Momentum (15%): Captures intangibles, but noisier than other signals
    """

    def __init__(self,
                 efficiency_weight: float = 0.40,
                 elo_weight: float = 0.25,
                 seed_weight: float = 0.20,
                 momentum_weight: float = 0.15):
        self.efficiency_model = EfficiencyModel()
        self.elo_model = ELORatingModel()
        self.momentum_model = MomentumModel()
        self.seed_model = SeedHistoricalModel()

        self.weights = {
            'efficiency': efficiency_weight,
            'elo': elo_weight,
            'seed': seed_weight,
            'momentum': momentum_weight,
        }

    # Tournament round multipliers — later rounds amplify favorite margins
    # Based on 2026 backtest: favorites won by avg 14.8 pts in E8 vs model's 4.1
    ROUND_MARGIN_MULTIPLIER = {
        'First Round': 1.0,
        'Second Round': 1.0,
        'Sweet 16': 1.15,
        'Elite Eight': 1.40,
        'Final Four': 1.50,
        'Championship': 1.50,
    }

    # Tournament total adjustments — defense tightens in later rounds
    # Backtest: model totals ran +17.5 pts high in E8
    ROUND_TOTAL_ADJUSTMENT = {
        'First Round': -2.0,
        'Second Round': -3.0,
        'Sweet 16': -5.0,
        'Elite Eight': -8.0,
        'Final Four': -10.0,
        'Championship': -10.0,
    }

    def _get_round_adjustments(self, round_name: str) -> tuple[float, float]:
        """Get margin multiplier and total adjustment for tournament round."""
        margin_mult = 1.0
        total_adj = 0.0
        for key, val in self.ROUND_MARGIN_MULTIPLIER.items():
            if key.lower() in round_name.lower():
                margin_mult = val
                break
        for key, val in self.ROUND_TOTAL_ADJUSTMENT.items():
            if key.lower() in round_name.lower():
                total_adj = val
                break
        return margin_mult, total_adj

    def predict_game(self, matchup: Matchup) -> GamePrediction:
        """Generate a complete prediction for a single matchup."""
        team_a = matchup.team_a
        team_b = matchup.team_b

        # Run each model
        eff_prob_a, score_a, score_b = self.efficiency_model.predict(team_a, team_b)
        elo_prob_a = self.elo_model.predict(team_a, team_b)
        momentum_prob_a = self.momentum_model.predict(team_a, team_b)
        seed_prob_a = self.seed_model.predict(team_a, team_b)

        # Weighted ensemble
        composite_prob_a = (
            self.weights['efficiency'] * eff_prob_a +
            self.weights['elo'] * elo_prob_a +
            self.weights['seed'] * seed_prob_a +
            self.weights['momentum'] * momentum_prob_a
        )

        # Calibrate — reduced shrinkage (was 0.05, now 0.02)
        # Backtest showed favorites are undervalued with heavy shrinkage
        composite_prob_a = self._calibrate(composite_prob_a, shrinkage=0.02)
        composite_prob_b = 1.0 - composite_prob_a

        # Adjust predicted scores based on composite probability
        eff_margin = score_a - score_b
        composite_margin = self._prob_to_margin(composite_prob_a)

        # Apply tournament round margin multiplier
        # Later rounds: favorites win by MORE than regular-season metrics suggest
        margin_mult, total_adj = self._get_round_adjustments(matchup.round_name)
        composite_margin = composite_margin * margin_mult

        # Blend efficiency scores with composite margin
        avg_total = (score_a + score_b) / 2.0
        adj_score_a = avg_total + composite_margin / 2.0
        adj_score_b = avg_total - composite_margin / 2.0

        # Apply tournament total adjustment (defense tightens in later rounds)
        adj_score_a += total_adj / 2.0
        adj_score_b += total_adj / 2.0

        # Model agreement (how similar are model outputs)
        probs = [eff_prob_a, elo_prob_a, momentum_prob_a, seed_prob_a]
        agreement = 1.0 - self._prob_variance(probs)

        # Confidence: combines model agreement with margin magnitude
        confidence = self._calculate_confidence(composite_prob_a, agreement)

        # Betting edges
        predicted_spread = -(adj_score_a - adj_score_b)  # Negative = team_a favored
        predicted_total = adj_score_a + adj_score_b

        spread_edge = None
        total_edge = None
        if matchup.spread is not None:
            spread_edge = predicted_spread - matchup.spread
        if matchup.over_under is not None:
            total_edge = predicted_total - matchup.over_under

        return GamePrediction(
            matchup=matchup,
            team_a_predicted_score=round(adj_score_a, 1),
            team_b_predicted_score=round(adj_score_b, 1),
            team_a_win_prob=round(composite_prob_a, 4),
            team_b_win_prob=round(composite_prob_b, 4),
            efficiency_prob_a=round(eff_prob_a, 4),
            elo_prob_a=round(elo_prob_a, 4),
            momentum_prob_a=round(momentum_prob_a, 4),
            seed_historical_prob_a=round(seed_prob_a, 4),
            confidence=round(confidence, 1),
            model_agreement=round(agreement, 3),
            predicted_spread=round(predicted_spread, 1),
            predicted_total=round(predicted_total, 1),
            spread_edge=round(spread_edge, 1) if spread_edge is not None else None,
            total_edge=round(total_edge, 1) if total_edge is not None else None,
        )

    @staticmethod
    def _calibrate(prob: float, shrinkage: float = 0.05) -> float:
        """Shrink extreme probabilities toward 50% for calibration."""
        return prob * (1 - shrinkage) + 0.5 * shrinkage

    @staticmethod
    def _prob_to_margin(prob: float, std_dev: float = 11.0) -> float:
        """Convert win probability back to expected scoring margin."""
        if prob <= 0.001:
            return -30.0
        if prob >= 0.999:
            return 30.0
        # Inverse normal CDF approximation
        p = prob
        if p < 0.5:
            p = 1 - p
        t = math.sqrt(-2.0 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
        if prob < 0.5:
            z = -z
        return z * std_dev

    @staticmethod
    def _prob_variance(probs: list[float]) -> float:
        """Calculate variance of probability estimates (0-1 scale)."""
        if not probs:
            return 0.0
        mean = sum(probs) / len(probs)
        variance = sum((p - mean) ** 2 for p in probs) / len(probs)
        # Normalize: max variance for binary probs is 0.25
        return min(variance / 0.25, 1.0)

    @staticmethod
    def _calculate_confidence(prob: float, agreement: float) -> float:
        """Calculate overall prediction confidence (0-100)."""
        # Distance from 50/50 (more decisive = more confident)
        decisiveness = abs(prob - 0.5) * 2  # 0 to 1

        # Blend decisiveness and model agreement
        raw_confidence = decisiveness * 0.6 + agreement * 0.4

        return raw_confidence * 100
