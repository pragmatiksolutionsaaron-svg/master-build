"""
Report Generator
Produces detailed, human-readable prediction reports with advanced reasoning.
"""

import json
import os
from datetime import datetime
from typing import Optional

from ..models.prediction_engine import GamePrediction
from .betting_analyzer import BetRecommendation


def generate_full_report(
    predictions: list[GamePrediction],
    all_bets: list[BetRecommendation],
    output_dir: str = "output"
) -> str:
    """Generate a comprehensive prediction and betting report."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    lines = []
    lines.append("=" * 80)
    lines.append("  NCAA MARCH MADNESS — PREDICTIVE ANALYTICS REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  Algorithm: Composite Bayesian Ensemble (4 models)")
    lines.append("  Models: Adjusted Efficiency | ELO Rating | Seed Historical | Momentum")
    lines.append("  Weights: 40% | 25% | 20% | 15%")
    lines.append("")

    # ── Executive Summary ──
    lines.append("─" * 80)
    lines.append("  EXECUTIVE SUMMARY")
    lines.append("─" * 80)
    lines.append("")

    if all_bets:
        top_bet = all_bets[0]
        lines.append(f"  TOP PICK: {'★' * top_bet.star_rating}{'☆' * (5 - top_bet.star_rating)} "
                      f"{top_bet.pick}")
        lines.append(f"  Edge: {top_bet.edge:.1f} | EV: {top_bet.expected_value:+.3f} | "
                      f"Confidence: {top_bet.confidence:.0f}/100")
        lines.append("")

    for pred in predictions:
        winner = pred.predicted_winner
        prob = max(pred.team_a_win_prob, pred.team_b_win_prob)
        lines.append(
            f"  {pred.matchup.team_a.name} vs {pred.matchup.team_b.name}: "
            f"{winner} ({prob*100:.1f}%) — "
            f"Projected: {pred.team_a_predicted_score:.0f}-{pred.team_b_predicted_score:.0f}"
        )
    lines.append("")

    # ── Detailed Game-by-Game Analysis ──
    for i, pred in enumerate(predictions):
        lines.append("=" * 80)
        lines.append(f"  GAME {i+1}: {pred.matchup.round_name.upper()}")
        lines.append(f"  {pred.matchup.team_a.name} (#{pred.matchup.team_a.seed}) "
                      f"vs {pred.matchup.team_b.name} (#{pred.matchup.team_b.seed})")
        lines.append(f"  {pred.matchup.game_time} | {pred.matchup.venue}")
        lines.append("=" * 80)
        lines.append("")

        # Score prediction
        lines.append("  ┌─────────────────────────────────────────────────┐")
        lines.append(f"  │  PREDICTED SCORE                                │")
        lines.append(f"  │  {pred.matchup.team_a.name:>20s}  {pred.team_a_predicted_score:5.1f}          │")
        lines.append(f"  │  {pred.matchup.team_b.name:>20s}  {pred.team_b_predicted_score:5.1f}          │")
        lines.append(f"  │  Predicted Total: {pred.predicted_total:5.1f}                       │")
        lines.append(f"  │  Predicted Spread: {pred.predicted_spread:+5.1f}                      │")
        lines.append("  └─────────────────────────────────────────────────┘")
        lines.append("")

        # Win probability breakdown
        lines.append("  WIN PROBABILITY BREAKDOWN:")
        lines.append(f"  {'Model':<25s} {'Team A':>10s} {'Team B':>10s}")
        lines.append(f"  {'─'*25} {'─'*10} {'─'*10}")
        lines.append(f"  {'Efficiency (40%)':<25s} {pred.efficiency_prob_a*100:9.1f}% {(1-pred.efficiency_prob_a)*100:9.1f}%")
        lines.append(f"  {'ELO Rating (25%)':<25s} {pred.elo_prob_a*100:9.1f}% {(1-pred.elo_prob_a)*100:9.1f}%")
        lines.append(f"  {'Seed Historical (20%)':<25s} {pred.seed_historical_prob_a*100:9.1f}% {(1-pred.seed_historical_prob_a)*100:9.1f}%")
        lines.append(f"  {'Momentum (15%)':<25s} {pred.momentum_prob_a*100:9.1f}% {(1-pred.momentum_prob_a)*100:9.1f}%")
        lines.append(f"  {'─'*25} {'─'*10} {'─'*10}")
        lines.append(f"  {'COMPOSITE':<25s} {pred.team_a_win_prob*100:9.1f}% {pred.team_b_win_prob*100:9.1f}%")
        lines.append("")
        lines.append(f"  Model Agreement: {pred.model_agreement:.0%} | Confidence: {pred.confidence:.0f}/100")
        lines.append("")

        # Team comparison
        ta = pred.matchup.team_a
        tb = pred.matchup.team_b
        lines.append("  HEAD-TO-HEAD STATISTICAL COMPARISON:")
        lines.append(f"  {'Metric':<30s} {ta.name:>15s} {tb.name:>15s} {'Edge':>10s}")
        lines.append(f"  {'─'*30} {'─'*15} {'─'*15} {'─'*10}")

        comparisons = [
            ("Record", f"{ta.wins}-{ta.losses}", f"{tb.wins}-{tb.losses}",
             ta.name if ta.win_pct > tb.win_pct else tb.name),
            ("Points Per Game", f"{ta.points_per_game:.1f}", f"{tb.points_per_game:.1f}",
             ta.name if ta.points_per_game > tb.points_per_game else tb.name),
            ("Points Allowed", f"{ta.points_allowed_per_game:.1f}", f"{tb.points_allowed_per_game:.1f}",
             ta.name if ta.points_allowed_per_game < tb.points_allowed_per_game else tb.name),
            ("Scoring Margin", f"{ta.scoring_margin:+.1f}", f"{tb.scoring_margin:+.1f}",
             ta.name if ta.scoring_margin > tb.scoring_margin else tb.name),
            ("FG%", f"{ta.field_goal_pct:.1%}", f"{tb.field_goal_pct:.1%}",
             ta.name if ta.field_goal_pct > tb.field_goal_pct else tb.name),
            ("3PT%", f"{ta.three_point_pct:.1%}", f"{tb.three_point_pct:.1%}",
             ta.name if ta.three_point_pct > tb.three_point_pct else tb.name),
            ("FT%", f"{ta.free_throw_pct:.1%}", f"{tb.free_throw_pct:.1%}",
             ta.name if ta.free_throw_pct > tb.free_throw_pct else tb.name),
            ("Turnovers/Game", f"{ta.turnovers_per_game:.1f}", f"{tb.turnovers_per_game:.1f}",
             ta.name if ta.turnovers_per_game < tb.turnovers_per_game else tb.name),
            ("Steals/Game", f"{ta.steals_per_game:.1f}", f"{tb.steals_per_game:.1f}",
             ta.name if ta.steals_per_game > tb.steals_per_game else tb.name),
            ("Adj Off Efficiency", f"{ta.adjusted_offensive_efficiency:.1f}", f"{tb.adjusted_offensive_efficiency:.1f}",
             ta.name if ta.adjusted_offensive_efficiency > tb.adjusted_offensive_efficiency else tb.name),
            ("Adj Def Efficiency", f"{ta.adjusted_defensive_efficiency:.1f}", f"{tb.adjusted_defensive_efficiency:.1f}",
             ta.name if ta.adjusted_defensive_efficiency < tb.adjusted_defensive_efficiency else tb.name),
            ("SOS", f"{ta.strength_of_schedule:.2f}", f"{tb.strength_of_schedule:.2f}",
             ta.name if ta.strength_of_schedule > tb.strength_of_schedule else tb.name),
            ("Momentum Score", f"{ta.momentum_score:.0f}", f"{tb.momentum_score:.0f}",
             ta.name if ta.momentum_score > tb.momentum_score else tb.name),
        ]

        for label, val_a, val_b, edge_team in comparisons:
            lines.append(f"  {label:<30s} {val_a:>15s} {val_b:>15s} {edge_team:>10s}")

        lines.append("")

        # Advanced reasoning narrative
        lines.append("  ADVANCED REASONING:")
        lines.append("  " + "─" * 50)
        lines.extend(_generate_narrative(pred))
        lines.append("")

        # Game-specific bets
        game_bets = [b for b in all_bets if b.game_id == pred.matchup.game_id]
        if game_bets:
            lines.append("  BETTING RECOMMENDATIONS FOR THIS GAME:")
            for bet in game_bets:
                stars = "★" * bet.star_rating + "☆" * (5 - bet.star_rating)
                lines.append(f"  {stars} {bet.bet_type.upper()}: {bet.pick}")
                lines.append(f"     Edge: {bet.edge:.1f} | EV: {bet.expected_value:+.3f} | "
                              f"Confidence: {bet.confidence:.0f}/100")
                for reason in bet.reasoning:
                    lines.append(f"     • {reason}")
                lines.append("")

    # ── Best Bets Summary ──
    lines.append("=" * 80)
    lines.append("  BEST BETS — RANKED BY EXPECTED VALUE")
    lines.append("=" * 80)
    lines.append("")

    if all_bets:
        for rank, bet in enumerate(all_bets, 1):
            stars = "★" * bet.star_rating + "☆" * (5 - bet.star_rating)
            lines.append(f"  #{rank}  {stars}  {bet.pick}")
            lines.append(f"      Type: {bet.bet_type.upper()} | Edge: {bet.edge:.1f} | "
                          f"EV: {bet.expected_value:+.3f} | Confidence: {bet.confidence:.0f}")
            for reason in bet.reasoning:
                lines.append(f"      → {reason}")
            lines.append("")
    else:
        lines.append("  No bets meet the minimum edge threshold.")
        lines.append("  Consider this a PASS — no actionable edges detected.")
        lines.append("")

    # ── Methodology ──
    lines.append("=" * 80)
    lines.append("  METHODOLOGY & DISCLAIMERS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  This algorithm combines four independent prediction models:")
    lines.append("")
    lines.append("  1. ADJUSTED EFFICIENCY MODEL (40% weight)")
    lines.append("     KenPom-inspired model measuring points per 100 possessions,")
    lines.append("     adjusted for opponent strength. The single most predictive")
    lines.append("     metric in college basketball analytics.")
    lines.append("")
    lines.append("  2. ELO RATING SYSTEM (25% weight)")
    lines.append("     Chess-style rating system adapted for basketball. Naturally")
    lines.append("     accounts for strength of schedule through transitive results.")
    lines.append("     Resistant to recency bias.")
    lines.append("")
    lines.append("  3. SEED HISTORICAL MODEL (20% weight)")
    lines.append("     Leverages 40+ years of NCAA tournament seed matchup data.")
    lines.append("     Seeds predict ~70% of games outright. Particularly valuable")
    lines.append("     as a baseline and for identifying true upset potential.")
    lines.append("")
    lines.append("  4. MOMENTUM & SITUATIONAL MODEL (15% weight)")
    lines.append("     Captures recent form, win streaks, neutral-site performance,")
    lines.append("     quality wins, roster experience, and depth. Adds context that")
    lines.append("     pure number-crunching misses.")
    lines.append("")
    lines.append("  DISCLAIMER: This is a statistical analysis tool for informational")
    lines.append("  purposes. Sports betting involves risk. Past performance does not")
    lines.append("  guarantee future results. Always bet responsibly.")
    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)

    # Save report
    report_path = os.path.join(output_dir, f"prediction_report_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    # Save JSON data
    json_path = os.path.join(output_dir, f"predictions_{timestamp}.json")
    json_data = {
        'generated': datetime.now().isoformat(),
        'predictions': [],
        'best_bets': [],
    }
    for pred in predictions:
        json_data['predictions'].append({
            'game': f"{pred.matchup.team_a.name} vs {pred.matchup.team_b.name}",
            'round': pred.matchup.round_name,
            'predicted_winner': pred.predicted_winner,
            'score': f"{pred.team_a_predicted_score:.0f}-{pred.team_b_predicted_score:.0f}",
            'win_prob': f"{max(pred.team_a_win_prob, pred.team_b_win_prob)*100:.1f}%",
            'confidence': pred.confidence,
            'spread': pred.predicted_spread,
            'total': pred.predicted_total,
        })
    for bet in all_bets:
        json_data['best_bets'].append({
            'rank': all_bets.index(bet) + 1,
            'pick': bet.pick,
            'type': bet.bet_type,
            'edge': bet.edge,
            'ev': bet.expected_value,
            'confidence': bet.confidence,
            'stars': bet.star_rating,
            'reasoning': bet.reasoning,
        })

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n  Report saved to: {report_path}")
    print(f"  JSON data saved to: {json_path}")

    return report


def _generate_narrative(pred: GamePrediction) -> list[str]:
    """Generate an advanced reasoning narrative for a game prediction."""
    lines = []
    ta = pred.matchup.team_a
    tb = pred.matchup.team_b
    winner = pred.predicted_winner
    loser = ta.name if winner == tb.name else tb.name
    w_team = ta if winner == ta.name else tb
    l_team = tb if winner == ta.name else ta
    prob = max(pred.team_a_win_prob, pred.team_b_win_prob)

    # Opening assessment
    if prob > 0.70:
        lines.append(f"  STRONG LEAN: {winner} is the clear favorite here at {prob*100:.1f}%.")
    elif prob > 0.58:
        lines.append(f"  MODERATE LEAN: {winner} has the edge at {prob*100:.1f}%, but "
                      f"{loser} has a legitimate path to victory.")
    else:
        lines.append(f"  TOSS-UP: This is a razor-thin margin. {winner} gets the slight "
                      f"nod at {prob*100:.1f}% but this game could go either way.")

    # Efficiency analysis
    if w_team.net_efficiency > 0 and l_team.net_efficiency > 0:
        eff_diff = w_team.net_efficiency - l_team.net_efficiency
        if eff_diff > 10:
            lines.append(f"  {winner}'s efficiency advantage ({eff_diff:+.1f} net) is substantial "
                          f"and historically translates to tournament success.")
        elif eff_diff > 5:
            lines.append(f"  {winner} holds a meaningful efficiency edge ({eff_diff:+.1f} net), "
                          f"suggesting they produce more points per possession against quality defense.")

    # Offensive/defensive matchup
    if w_team.adjusted_offensive_efficiency > 0 and l_team.adjusted_defensive_efficiency > 0:
        if w_team.adjusted_offensive_efficiency > l_team.adjusted_defensive_efficiency + 5:
            lines.append(f"  KEY MATCHUP: {winner}'s offense (adj. {w_team.adjusted_offensive_efficiency:.1f}) "
                          f"should challenge {loser}'s defense (adj. {l_team.adjusted_defensive_efficiency:.1f}). "
                          f"Expect {winner} to find scoring opportunities.")

    # Defensive identity
    if w_team.adjusted_defensive_efficiency > 0 and w_team.adjusted_defensive_efficiency < 95:
        lines.append(f"  {winner} brings elite defensive efficiency "
                      f"({w_team.adjusted_defensive_efficiency:.1f} adj), which tends to "
                      f"travel well in tournament settings.")

    # Three-point shooting dynamic
    if abs(ta.three_point_pct - tb.three_point_pct) > 0.03:
        better_shooter = ta if ta.three_point_pct > tb.three_point_pct else tb
        lines.append(f"  PERIMETER: {better_shooter.name}'s superior 3PT shooting "
                      f"({better_shooter.three_point_pct:.1%}) adds a ceiling-raiser element. "
                      f"If shots fall, the margin could widen quickly.")

    # Turnover battle
    if abs(ta.turnover_margin - tb.turnover_margin) > 1.5:
        better_ball = ta if ta.turnover_margin > tb.turnover_margin else tb
        lines.append(f"  BALL SECURITY: {better_ball.name} has a significant edge in "
                      f"turnover margin ({better_ball.turnover_margin:+.1f}). In March, "
                      f"possessions are gold — extra possessions compound into points.")

    # Experience & intangibles
    if w_team.experience_score > l_team.experience_score + 0.5:
        lines.append(f"  EXPERIENCE: {winner}'s roster experience "
                      f"({w_team.experience_score:.1f}) could be pivotal in late-game "
                      f"pressure situations.")

    # Momentum narrative
    if w_team.momentum_score > l_team.momentum_score + 15:
        lines.append(f"  MOMENTUM: {winner} enters with significantly stronger momentum "
                      f"({w_team.momentum_score:.0f} vs {l_team.momentum_score:.0f}). "
                      f"Teams riding a hot streak have historically overperformed expectations.")

    # Upset alert
    if pred.upset_probability > 0.35:
        higher_seed = ta if ta.seed > tb.seed else tb
        lines.append(f"  UPSET WATCH: {higher_seed.name} (#{higher_seed.seed} seed) has a "
                      f"{pred.upset_probability*100:.0f}% upset probability. This is elevated "
                      f"and worth monitoring for contrarian value.")

    # Model disagreement
    if pred.model_agreement < 0.5:
        lines.append(f"  VARIANCE WARNING: Models show low agreement ({pred.model_agreement:.0%}). "
                      f"This game has high outcome variance — proceed with smaller bet sizes.")

    return lines
