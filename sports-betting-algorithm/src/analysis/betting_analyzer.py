"""
Betting Analysis Engine
Evaluates betting opportunities and generates actionable recommendations.

Analyzes spread, moneyline, and over/under markets to find edges
where our model disagrees with the market, then ranks bets by
expected value and confidence.
"""

import math
from dataclasses import dataclass
from typing import Optional

from ..models.prediction_engine import GamePrediction


@dataclass
class BetRecommendation:
    """A single betting recommendation with reasoning."""
    game_id: str
    bet_type: str  # "spread", "moneyline", "over", "under"
    pick: str  # Team name or Over/Under
    odds_description: str
    edge: float  # Estimated edge (percentage points)
    expected_value: float  # Expected return per unit bet
    confidence: float  # 0-100
    star_rating: int  # 1-5 stars
    reasoning: list[str]  # Detailed reasoning points


class BettingAnalyzer:
    """Analyzes predictions against market lines to find value bets."""

    # Minimum edge required to recommend a bet (accounts for vig/juice)
    MIN_SPREAD_EDGE = 2.0  # points
    MIN_TOTAL_EDGE = 3.0  # points
    MIN_ML_EDGE = 0.05  # 5% probability edge

    # Standard vig assumption
    STANDARD_VIG = 0.0476  # ~4.76% (standard -110/-110)

    def analyze_game(self, prediction: GamePrediction) -> list[BetRecommendation]:
        """Generate all betting recommendations for a single game."""
        bets = []

        # Spread analysis
        spread_bet = self._analyze_spread(prediction)
        if spread_bet:
            bets.append(spread_bet)

        # Moneyline analysis
        ml_bet = self._analyze_moneyline(prediction)
        if ml_bet:
            bets.append(ml_bet)

        # Over/Under analysis
        ou_bet = self._analyze_over_under(prediction)
        if ou_bet:
            bets.append(ou_bet)

        return bets

    def _analyze_spread(self, pred: GamePrediction) -> Optional[BetRecommendation]:
        """Analyze the point spread market."""
        matchup = pred.matchup
        if matchup.spread is None:
            return None

        market_spread = matchup.spread  # Positive = team_b favored
        model_spread = pred.predicted_spread  # Negative = team_a favored

        edge = abs(model_spread - market_spread)
        if edge < self.MIN_SPREAD_EDGE:
            return None

        # Determine which side has value
        reasoning = []
        if model_spread < market_spread:
            # Model favors team_a more than market does
            pick = matchup.team_a.name
            reasoning.append(
                f"Model projects {matchup.team_a.name} spread at {model_spread:+.1f} "
                f"vs market {market_spread:+.1f} ({edge:.1f} pts of edge)"
            )
        else:
            pick = matchup.team_b.name
            reasoning.append(
                f"Model projects {matchup.team_b.name} spread at {-model_spread:+.1f} "
                f"vs market {-market_spread:+.1f} ({edge:.1f} pts of edge)"
            )

        # Add supporting reasoning
        reasoning.extend(self._get_spread_reasoning(pred, pick))

        # Calculate EV (approximate)
        cover_prob = self._edge_to_cover_probability(edge)
        ev = (cover_prob * 0.909) - ((1 - cover_prob) * 1.0)  # -110 odds

        confidence = min(pred.confidence + edge * 2, 100)
        stars = self._calculate_stars(edge, confidence, 'spread')

        return BetRecommendation(
            game_id=matchup.game_id,
            bet_type="spread",
            pick=f"{pick} {market_spread:+.1f}" if pick == matchup.team_b.name
                 else f"{pick} {-market_spread:+.1f}",
            odds_description=f"Spread: {matchup.team_a.name} {-market_spread:+.1f} / {matchup.team_b.name} {market_spread:+.1f}",
            edge=round(edge, 1),
            expected_value=round(ev, 3),
            confidence=round(confidence, 1),
            star_rating=stars,
            reasoning=reasoning,
        )

    def _analyze_moneyline(self, pred: GamePrediction) -> Optional[BetRecommendation]:
        """Analyze the moneyline market."""
        matchup = pred.matchup
        if matchup.team_a_moneyline is None or matchup.team_b_moneyline is None:
            return None

        # Convert American odds to implied probability
        implied_a = self._american_to_implied(matchup.team_a_moneyline)
        implied_b = self._american_to_implied(matchup.team_b_moneyline)

        # Remove vig to get true implied
        total_implied = implied_a + implied_b
        fair_implied_a = implied_a / total_implied
        fair_implied_b = implied_b / total_implied

        # Find edge
        edge_a = pred.team_a_win_prob - fair_implied_a
        edge_b = pred.team_b_win_prob - fair_implied_b

        reasoning = []

        if edge_a > self.MIN_ML_EDGE and edge_a >= edge_b:
            pick = matchup.team_a.name
            edge = edge_a
            ml = matchup.team_a_moneyline
            model_prob = pred.team_a_win_prob
            implied = fair_implied_a
        elif edge_b > self.MIN_ML_EDGE:
            pick = matchup.team_b.name
            edge = edge_b
            ml = matchup.team_b_moneyline
            model_prob = pred.team_b_win_prob
            implied = fair_implied_b
        else:
            return None

        reasoning.append(
            f"Model gives {pick} a {model_prob*100:.1f}% win probability "
            f"vs market-implied {implied*100:.1f}% ({edge*100:.1f}% edge)"
        )
        reasoning.extend(self._get_moneyline_reasoning(pred, pick))

        # Calculate EV
        payout = self._american_to_decimal(ml) - 1
        ev = (model_prob * payout) - ((1 - model_prob) * 1.0)

        confidence = min(pred.confidence + edge * 100, 100)
        stars = self._calculate_stars(edge * 20, confidence, 'moneyline')

        return BetRecommendation(
            game_id=matchup.game_id,
            bet_type="moneyline",
            pick=f"{pick} ML ({ml:+d})",
            odds_description=f"Moneyline: {matchup.team_a.name} ({matchup.team_a_moneyline:+d}) / {matchup.team_b.name} ({matchup.team_b_moneyline:+d})",
            edge=round(edge * 100, 1),
            expected_value=round(ev, 3),
            confidence=round(confidence, 1),
            star_rating=stars,
            reasoning=reasoning,
        )

    def _analyze_over_under(self, pred: GamePrediction) -> Optional[BetRecommendation]:
        """Analyze the over/under (total) market."""
        matchup = pred.matchup
        if matchup.over_under is None:
            return None

        market_total = matchup.over_under
        model_total = pred.predicted_total
        edge = abs(model_total - market_total)

        if edge < self.MIN_TOTAL_EDGE:
            return None

        reasoning = []
        if model_total > market_total:
            pick = "OVER"
            reasoning.append(
                f"Model projects total of {model_total:.1f} vs market line {market_total:.1f} "
                f"({edge:.1f} pts over)"
            )
        else:
            pick = "UNDER"
            reasoning.append(
                f"Model projects total of {model_total:.1f} vs market line {market_total:.1f} "
                f"({edge:.1f} pts under)"
            )

        reasoning.extend(self._get_total_reasoning(pred, pick))

        cover_prob = self._edge_to_cover_probability(edge)
        ev = (cover_prob * 0.909) - ((1 - cover_prob) * 1.0)

        confidence = min(pred.confidence + edge * 1.5, 100)
        stars = self._calculate_stars(edge, confidence, 'total')

        return BetRecommendation(
            game_id=matchup.game_id,
            bet_type="over_under",
            pick=f"{pick} {market_total}",
            odds_description=f"Total: {market_total}",
            edge=round(edge, 1),
            expected_value=round(ev, 3),
            confidence=round(confidence, 1),
            star_rating=stars,
            reasoning=reasoning,
        )

    def _get_spread_reasoning(self, pred: GamePrediction, pick: str) -> list[str]:
        """Generate detailed spread reasoning."""
        reasons = []
        matchup = pred.matchup
        team_a, team_b = matchup.team_a, matchup.team_b

        is_team_a = pick == team_a.name
        picked = team_a if is_team_a else team_b
        opponent = team_b if is_team_a else team_a

        # Efficiency advantage
        if picked.net_efficiency > opponent.net_efficiency:
            reasons.append(
                f"EFFICIENCY EDGE: {picked.name} net efficiency "
                f"({picked.net_efficiency:+.1f}) superior to "
                f"{opponent.name} ({opponent.net_efficiency:+.1f})"
            )

        # Scoring margin
        if picked.scoring_margin > opponent.scoring_margin:
            reasons.append(
                f"SCORING MARGIN: {picked.name} ({picked.scoring_margin:+.1f}) "
                f"outperforms {opponent.name} ({opponent.scoring_margin:+.1f})"
            )

        # Model agreement
        if pred.model_agreement > 0.7:
            reasons.append(
                f"HIGH MODEL AGREEMENT: {pred.model_agreement:.0%} — "
                f"all 4 models align on this pick"
            )
        elif pred.model_agreement < 0.4:
            reasons.append(
                f"CAUTION - LOW MODEL AGREEMENT: {pred.model_agreement:.0%} — "
                f"models disagree, higher variance expected"
            )

        # Momentum
        if picked.momentum_score > opponent.momentum_score + 10:
            reasons.append(
                f"MOMENTUM: {picked.name} ({picked.momentum_score:.0f}) "
                f"trending stronger than {opponent.name} ({opponent.momentum_score:.0f})"
            )

        return reasons

    def _get_moneyline_reasoning(self, pred: GamePrediction, pick: str) -> list[str]:
        """Generate detailed moneyline reasoning."""
        reasons = []
        matchup = pred.matchup
        team_a, team_b = matchup.team_a, matchup.team_b

        is_team_a = pick == team_a.name
        picked = team_a if is_team_a else team_b
        opponent = team_b if is_team_a else team_a

        # Seed advantage or upset case
        if picked.seed < opponent.seed:
            historical_prob = pred.seed_historical_prob_a if is_team_a else (1 - pred.seed_historical_prob_a)
            reasons.append(
                f"SEED ADVANTAGE: #{picked.seed} seed historically wins "
                f"{historical_prob*100:.0f}% vs #{opponent.seed} seeds"
            )
        else:
            reasons.append(
                f"UPSET VALUE: #{picked.seed} vs #{opponent.seed} — "
                f"model sees this as closer than seed lines suggest"
            )

        # Win percentage differential
        if picked.win_pct > opponent.win_pct:
            reasons.append(
                f"RECORD: {picked.name} ({picked.wins}-{picked.losses}, "
                f"{picked.win_pct:.3f}) vs {opponent.name} ({opponent.wins}-{opponent.losses}, "
                f"{opponent.win_pct:.3f})"
            )

        # Quality wins
        if picked.vs_top_25_wins > opponent.vs_top_25_wins:
            reasons.append(
                f"QUALITY WINS: {picked.name} has {picked.vs_top_25_wins} wins vs Top 25 "
                f"compared to {opponent.vs_top_25_wins} for {opponent.name}"
            )

        return reasons

    def _get_total_reasoning(self, pred: GamePrediction, pick: str) -> list[str]:
        """Generate detailed over/under reasoning."""
        reasons = []
        matchup = pred.matchup
        team_a, team_b = matchup.team_a, matchup.team_b

        combined_ppg = team_a.points_per_game + team_b.points_per_game
        combined_allowed = team_a.points_allowed_per_game + team_b.points_allowed_per_game

        if pick == "OVER":
            if combined_ppg > (matchup.over_under or 0):
                reasons.append(
                    f"PACE: Combined scoring avg ({combined_ppg:.1f} PPG) "
                    f"exceeds the line"
                )
            if team_a.tempo > 68 and team_b.tempo > 68:
                reasons.append(
                    f"TEMPO: Both teams play above-average pace "
                    f"({team_a.name}: {team_a.tempo:.1f}, {team_b.name}: {team_b.tempo:.1f})"
                )
        else:
            if combined_allowed < (matchup.over_under or 999):
                reasons.append(
                    f"DEFENSE: Combined defensive avg ({combined_allowed:.1f} allowed) "
                    f"supports the under"
                )
            if team_a.tempo < 67 or team_b.tempo < 67:
                slow = team_a if team_a.tempo <= team_b.tempo else team_b
                reasons.append(
                    f"TEMPO: {slow.name} plays at slow pace ({slow.tempo:.1f} possessions) "
                    f"which should drag the total down"
                )

        # Three-point variance
        avg_3pt = (team_a.three_point_pct + team_b.three_point_pct) / 2
        if avg_3pt > 0.36:
            reasons.append(f"3PT FACTOR: High combined 3PT% ({avg_3pt:.1%}) adds scoring variance")
        elif avg_3pt < 0.32:
            reasons.append(f"3PT FACTOR: Low combined 3PT% ({avg_3pt:.1%}) limits scoring ceiling")

        return reasons

    @staticmethod
    def _american_to_implied(american: int) -> float:
        """Convert American odds to implied probability."""
        if american > 0:
            return 100.0 / (american + 100.0)
        return abs(american) / (abs(american) + 100.0)

    @staticmethod
    def _american_to_decimal(american: int) -> float:
        """Convert American odds to decimal odds."""
        if american > 0:
            return (american / 100.0) + 1.0
        return (100.0 / abs(american)) + 1.0

    @staticmethod
    def _edge_to_cover_probability(edge_points: float) -> float:
        """Convert point edge to approximate cover probability."""
        # Each point of edge ≈ 3% cover probability
        base = 0.50
        return min(base + edge_points * 0.03, 0.85)

    @staticmethod
    def _calculate_stars(edge: float, confidence: float, bet_type: str) -> int:
        """Calculate 1-5 star rating for a bet."""
        score = (edge * 3 + confidence) / 4

        if bet_type == 'spread':
            thresholds = [20, 35, 50, 65]
        elif bet_type == 'moneyline':
            thresholds = [15, 30, 45, 60]
        else:
            thresholds = [20, 35, 50, 65]

        stars = 1
        for t in thresholds:
            if score >= t:
                stars += 1
        return min(stars, 5)

    def rank_bets(self, all_bets: list[BetRecommendation]) -> list[BetRecommendation]:
        """Rank all bets across games by composite score."""
        def sort_key(bet):
            return (bet.star_rating, bet.expected_value, bet.confidence)

        return sorted(all_bets, key=sort_key, reverse=True)
