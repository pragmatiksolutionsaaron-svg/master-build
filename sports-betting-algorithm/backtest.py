#!/usr/bin/env python3
"""
NCAA March Madness 2026 — Model Backtest

Tests the prediction model against actual First Round and Second Round results
from the 2026 NCAA Tournament bracket. Measures:
  - Win prediction accuracy (did we pick the right winner?)
  - Spread accuracy (was our predicted margin close?)
  - Total accuracy (was our projected total close?)
  - Upset detection (did we flag upsets correctly?)

Uses approximate team stats for eliminated teams based on bracket data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.team_stats import TeamStats, Matchup
from src.models.prediction_engine import CompositePredictor


# ============================================================================
#  ACTUAL 2026 NCAA TOURNAMENT RESULTS (from bracket)
# ============================================================================

ACTUAL_RESULTS = [
    # ── EAST REGION — FIRST ROUND ──
    {"team_a": "Duke", "seed_a": 1, "score_a": 71, "team_b": "Siena", "seed_b": 16, "score_b": 55, "round": "R1"},
    {"team_a": "Ohio State", "seed_a": 8, "score_a": 64, "team_b": "TCU", "seed_b": 9, "score_b": 66, "round": "R1"},
    {"team_a": "St. John's", "seed_a": 5, "score_a": 79, "team_b": "Northern Iowa", "seed_b": 12, "score_b": 53, "round": "R1"},
    {"team_a": "Kansas", "seed_a": 4, "score_a": 68, "team_b": "Cal Baptist", "seed_b": 13, "score_b": 60, "round": "R1"},
    {"team_a": "Louisville", "seed_a": 6, "score_a": 83, "team_b": "South Florida", "seed_b": 11, "score_b": 79, "round": "R1"},
    {"team_a": "Michigan State", "seed_a": 3, "score_a": 92, "team_b": "North Dakota State", "seed_b": 14, "score_b": 67, "round": "R1"},
    {"team_a": "UCLA", "seed_a": 7, "score_a": 75, "team_b": "UCF", "seed_b": 10, "score_b": 71, "round": "R1"},
    {"team_a": "UConn", "seed_a": 2, "score_a": 82, "team_b": "Furman", "seed_b": 15, "score_b": 71, "round": "R1"},

    # ── EAST REGION — SECOND ROUND ──
    {"team_a": "Duke", "seed_a": 1, "score_a": 81, "team_b": "TCU", "seed_b": 9, "score_b": 58, "round": "R2"},
    {"team_a": "St. John's", "seed_a": 5, "score_a": 67, "team_b": "Kansas", "seed_b": 4, "score_b": 65, "round": "R2"},
    {"team_a": "Michigan State", "seed_a": 3, "score_a": 77, "team_b": "Louisville", "seed_b": 6, "score_b": 69, "round": "R2"},
    {"team_a": "UConn", "seed_a": 2, "score_a": 73, "team_b": "UCLA", "seed_b": 7, "score_b": 57, "round": "R2"},

    # ── SOUTH REGION — FIRST ROUND ──
    {"team_a": "Florida", "seed_a": 1, "score_a": 114, "team_b": "Prairie View A&M", "seed_b": 16, "score_b": 55, "round": "R1"},
    {"team_a": "Iowa", "seed_a": 9, "score_a": 67, "team_b": "Clemson", "seed_b": 8, "score_b": 61, "round": "R1"},
    {"team_a": "Vanderbilt", "seed_a": 5, "score_a": 78, "team_b": "McNeese", "seed_b": 12, "score_b": 68, "round": "R1"},
    {"team_a": "Nebraska", "seed_a": 4, "score_a": 76, "team_b": "Troy", "seed_b": 13, "score_b": 47, "round": "R1"},
    {"team_a": "VCU", "seed_a": 11, "score_a": 82, "team_b": "North Carolina", "seed_b": 6, "score_b": 78, "round": "R1"},
    {"team_a": "Illinois", "seed_a": 3, "score_a": 105, "team_b": "Penn", "seed_b": 14, "score_b": 70, "round": "R1"},
    {"team_a": "Texas A&M", "seed_a": 10, "score_a": 63, "team_b": "Saint Mary's", "seed_b": 7, "score_b": 50, "round": "R1"},
    {"team_a": "Houston", "seed_a": 2, "score_a": 78, "team_b": "Idaho", "seed_b": 16, "score_b": 47, "round": "R1"},

    # ── SOUTH REGION — SECOND ROUND ──
    {"team_a": "Iowa", "seed_a": 9, "score_a": 73, "team_b": "Florida", "seed_b": 1, "score_b": 72, "round": "R2"},
    {"team_a": "Nebraska", "seed_a": 4, "score_a": 74, "team_b": "Vanderbilt", "seed_b": 5, "score_b": 72, "round": "R2"},
    {"team_a": "Illinois", "seed_a": 3, "score_a": 76, "team_b": "VCU", "seed_b": 11, "score_b": 55, "round": "R2"},
    {"team_a": "Houston", "seed_a": 2, "score_a": 88, "team_b": "Texas A&M", "seed_b": 10, "score_b": 57, "round": "R2"},

    # ── WEST REGION — FIRST ROUND ──
    {"team_a": "Arizona", "seed_a": 1, "score_a": 92, "team_b": "Long Island", "seed_b": 16, "score_b": 58, "round": "R1"},
    {"team_a": "Utah State", "seed_a": 9, "score_a": 76, "team_b": "Villanova", "seed_b": 8, "score_b": 66, "round": "R1"},
    {"team_a": "High Point", "seed_a": 12, "score_a": 83, "team_b": "Wisconsin", "seed_b": 5, "score_b": 82, "round": "R1"},
    {"team_a": "Arkansas", "seed_a": 4, "score_a": 97, "team_b": "Hawaii", "seed_b": 13, "score_b": 78, "round": "R1"},
    {"team_a": "Texas", "seed_a": 11, "score_a": 79, "team_b": "BYU", "seed_b": 6, "score_b": 71, "round": "R1"},
    {"team_a": "Gonzaga", "seed_a": 3, "score_a": 73, "team_b": "Kennesaw State", "seed_b": 14, "score_b": 64, "round": "R1"},
    {"team_a": "Miami FL", "seed_a": 7, "score_a": 80, "team_b": "Missouri", "seed_b": 10, "score_b": 66, "round": "R1"},
    {"team_a": "Purdue", "seed_a": 2, "score_a": 104, "team_b": "Queens", "seed_b": 15, "score_b": 71, "round": "R1"},

    # ── WEST REGION — SECOND ROUND ──
    {"team_a": "Arizona", "seed_a": 1, "score_a": 78, "team_b": "Utah State", "seed_b": 9, "score_b": 66, "round": "R2"},
    {"team_a": "Arkansas", "seed_a": 4, "score_a": 94, "team_b": "High Point", "seed_b": 12, "score_b": 88, "round": "R2"},
    {"team_a": "Texas", "seed_a": 11, "score_a": 74, "team_b": "Gonzaga", "seed_b": 3, "score_b": 68, "round": "R2"},
    {"team_a": "Purdue", "seed_a": 2, "score_a": 79, "team_b": "Miami FL", "seed_b": 7, "score_b": 69, "round": "R2"},

    # ── MIDWEST REGION — FIRST ROUND ──
    {"team_a": "Michigan", "seed_a": 1, "score_a": 101, "team_b": "Howard", "seed_b": 16, "score_b": 80, "round": "R1"},
    {"team_a": "Saint Louis", "seed_a": 9, "score_a": 102, "team_b": "Georgia", "seed_b": 8, "score_b": 77, "round": "R1"},
    {"team_a": "Texas Tech", "seed_a": 5, "score_a": 91, "team_b": "Akron", "seed_b": 12, "score_b": 71, "round": "R1"},
    {"team_a": "Alabama", "seed_a": 4, "score_a": 90, "team_b": "Hofstra", "seed_b": 13, "score_b": 70, "round": "R1"},
    {"team_a": "Tennessee", "seed_a": 6, "score_a": 79, "team_b": "Miami Ohio", "seed_b": 11, "score_b": 56, "round": "R1"},
    {"team_a": "Virginia", "seed_a": 3, "score_a": 82, "team_b": "Wright State", "seed_b": 14, "score_b": 73, "round": "R1"},
    {"team_a": "Kentucky", "seed_a": 7, "score_a": 89, "team_b": "Santa Clara", "seed_b": 10, "score_b": 84, "round": "R1"},
    {"team_a": "Iowa State", "seed_a": 2, "score_a": 108, "team_b": "Tennessee State", "seed_b": 15, "score_b": 74, "round": "R1"},

    # ── MIDWEST REGION — SECOND ROUND ──
    {"team_a": "Michigan", "seed_a": 1, "score_a": 95, "team_b": "Saint Louis", "seed_b": 9, "score_b": 72, "round": "R2"},
    {"team_a": "Alabama", "seed_a": 4, "score_a": 90, "team_b": "Texas Tech", "seed_b": 5, "score_b": 65, "round": "R2"},
    {"team_a": "Tennessee", "seed_a": 6, "score_a": 79, "team_b": "Virginia", "seed_b": 3, "score_b": 72, "round": "R2"},
    {"team_a": "Iowa State", "seed_a": 2, "score_a": 82, "team_b": "Kentucky", "seed_b": 7, "score_b": 74, "round": "R2"},

    # ── FIRST FOUR ──
    {"team_a": "Howard", "seed_a": 16, "score_a": 86, "team_b": "UMBC", "seed_b": 16, "score_b": 83, "round": "FF"},
    {"team_a": "Texas", "seed_a": 11, "score_a": 68, "team_b": "NC State", "seed_b": 11, "score_b": 66, "round": "FF"},
    {"team_a": "Prairie View A&M", "seed_a": 16, "score_a": 55, "team_b": "Lehigh", "seed_b": 16, "score_b": 67, "round": "FF"},
    {"team_a": "Miami Ohio", "seed_a": 11, "score_a": 89, "team_b": "SMU", "seed_b": 11, "score_b": 79, "round": "FF"},
]


def build_team_from_seed(name: str, seed: int, record_str: str = "",
                          ppg: float = 0, opp_ppg: float = 0) -> TeamStats:
    """
    Build approximate TeamStats for a team based on seed and limited info.
    Uses seed-based priors for efficiency metrics when detailed stats unavailable.
    """
    # Parse record if provided
    wins, losses = 20, 12
    if record_str:
        parts = record_str.split("-")
        if len(parts) == 2:
            wins, losses = int(parts[0]), int(parts[1])

    # Estimate PPG from record quality if not provided
    if ppg == 0:
        win_pct = wins / max(wins + losses, 1)
        ppg = 65 + win_pct * 20  # Range ~65–85
    if opp_ppg == 0:
        opp_ppg = ppg - (wins - losses) * 0.4  # rough margin estimate

    # Seed-based efficiency priors (approximate KenPom-style)
    seed_adj_o = {
        1: 128, 2: 125, 3: 123, 4: 120, 5: 118, 6: 116, 7: 114, 8: 112,
        9: 111, 10: 110, 11: 109, 12: 108, 13: 106, 14: 104, 15: 102, 16: 100
    }
    seed_adj_d = {
        1: 90, 2: 92, 3: 93, 4: 94, 5: 95, 6: 96, 7: 97, 8: 98,
        9: 98, 10: 99, 11: 99, 12: 100, 13: 101, 14: 102, 15: 103, 16: 105
    }
    seed_elo = {
        1: 1880, 2: 1830, 3: 1800, 4: 1770, 5: 1750, 6: 1730, 7: 1710, 8: 1690,
        9: 1680, 10: 1670, 11: 1660, 12: 1650, 13: 1620, 14: 1600, 15: 1570, 16: 1540
    }
    seed_sos = {
        1: 0.92, 2: 0.90, 3: 0.88, 4: 0.86, 5: 0.84, 6: 0.82, 7: 0.80, 8: 0.78,
        9: 0.76, 10: 0.74, 11: 0.72, 12: 0.70, 13: 0.65, 14: 0.60, 15: 0.55, 16: 0.50
    }

    adj_o = seed_adj_o.get(seed, 110)
    adj_d = seed_adj_d.get(seed, 98)
    elo = seed_elo.get(seed, 1650)
    sos = seed_sos.get(seed, 0.70)

    win_pct = wins / max(wins + losses, 1)
    last_10_w = min(round(win_pct * 10), 10)

    return TeamStats(
        name=name, seed=seed, conference="Unknown",
        wins=wins, losses=losses,
        points_per_game=ppg, field_goal_pct=0.45, three_point_pct=0.34,
        free_throw_pct=0.73, offensive_rebounds_per_game=10.0,
        assists_per_game=14.0, turnovers_per_game=11.5,
        points_allowed_per_game=opp_ppg, steals_per_game=7.0,
        blocks_per_game=4.0, defensive_rebounds_per_game=25.0,
        opponent_field_goal_pct=0.42, opponent_three_point_pct=0.32,
        adjusted_offensive_efficiency=adj_o, adjusted_defensive_efficiency=adj_d,
        tempo=67.0, strength_of_schedule=sos, net_ranking=seed * 5,
        bpi=adj_o - adj_d, elo_rating=elo,
        last_10_wins=last_10_w, last_10_losses=10 - last_10_w,
        win_streak=max(1, round(win_pct * 5)),
        tournament_wins=0,
        neutral_site_wins=round(win_pct * 6), neutral_site_losses=round((1 - win_pct) * 3),
        vs_top_25_wins=round(win_pct * 8), vs_top_25_losses=round((1 - win_pct) * 5),
        experience_score=3.0, depth_score=3.0, bench_points_per_game=20.0,
    )


# Known team overrides with real stats (Sweet 16 teams)
KNOWN_TEAMS = {}


def get_known_teams():
    """Load real Sweet 16 team data from run_predictions.py"""
    global KNOWN_TEAMS
    if KNOWN_TEAMS:
        return KNOWN_TEAMS

    from run_predictions import build_thursday_data, build_friday_data
    thu_teams, _ = build_thursday_data()
    fri_teams, _ = build_friday_data()
    KNOWN_TEAMS = {**thu_teams, **fri_teams}
    return KNOWN_TEAMS


def get_or_build_team(name: str, seed: int, record: str = "") -> TeamStats:
    """Get real team data if available, otherwise build from seed priors."""
    known = get_known_teams()

    # Try exact match first
    if name in known:
        return known[name]

    # Try fuzzy match (only match if names are very similar, not substrings)
    name_lower = name.lower().replace(".", "").replace("'", "")
    for key, team in known.items():
        key_lower = key.lower().replace(".", "").replace("'", "")
        if name_lower == key_lower:
            return team

    # Build from seed priors
    return build_team_from_seed(name, seed, record)


# Records from the bracket image for all teams
TEAM_RECORDS = {
    "Duke": "32-2", "Siena": "23-1", "Ohio State": "21-12", "TCU": "22-11",
    "St. John's": "28-6", "Northern Iowa": "23-12", "Kansas": "23-10",
    "Cal Baptist": "25-8", "Louisville": "23-10", "South Florida": "25-8",
    "Michigan State": "25-7", "North Dakota State": "27-7", "UCLA": "23-11",
    "UCF": "21-11", "UConn": "29-5", "Furman": "22-12",
    "Florida": "26-7", "Prairie View A&M": "19-17", "Iowa": "21-12",
    "Clemson": "24-10", "Vanderbilt": "26-8", "McNeese": "28-5",
    "Nebraska": "26-6", "Troy": "22-11", "VCU": "27-7",
    "North Carolina": "24-8", "Illinois": "24-8", "Penn": "18-11",
    "Saint Mary's": "27-5", "Texas A&M": "21-11", "Houston": "28-6",
    "Idaho": "21-14",
    "Arizona": "32-2", "Long Island": "24-10", "Villanova": "24-8",
    "Utah State": "28-6", "Wisconsin": "24-10", "High Point": "30-4",
    "Arkansas": "28-8", "Hawaii": "24-8", "BYU": "23-11",
    "Texas": "19-14", "Gonzaga": "30-3", "Kennesaw State": "21-13",
    "Miami FL": "25-8", "Missouri": "20-12", "Purdue": "27-8",
    "Queens": "21-13",
    "Michigan": "31-3", "Howard": "24-10", "Georgia": "22-10",
    "Saint Louis": "28-5", "Texas Tech": "22-10", "Akron": "29-5",
    "Alabama": "23-8", "Hofstra": "24-10", "Tennessee": "22-11",
    "Miami Ohio": "32-1", "Virginia": "29-5", "Wright State": "23-11",
    "Kentucky": "27-13", "Santa Clara": "28-6", "Iowa State": "27-7",
    "Tennessee State": "23-9",
    "UMBC": "24-8", "NC State": "20-13", "Lehigh": "18-16", "SMU": "20-13",
}


def run_backtest():
    """Run the prediction model against all actual tournament results."""
    predictor = CompositePredictor()

    print("=" * 80)
    print("  NCAA 2026 TOURNAMENT — MODEL BACKTEST")
    print("  Testing against actual First Round & Second Round results")
    print("=" * 80)

    results = {
        "total": 0, "correct": 0, "wrong": 0,
        "upsets_actual": 0, "upsets_predicted": 0, "upsets_caught": 0,
        "spread_errors": [], "total_errors": [],
        "by_round": {},
        "details": [],
    }

    for game in ACTUAL_RESULTS:
        team_a_name = game["team_a"]
        team_b_name = game["team_b"]
        seed_a = game["seed_a"]
        seed_b = game["seed_b"]
        actual_score_a = game["score_a"]
        actual_score_b = game["score_b"]
        round_name = game["round"]

        record_a = TEAM_RECORDS.get(team_a_name, "")
        record_b = TEAM_RECORDS.get(team_b_name, "")

        team_a = get_or_build_team(team_a_name, seed_a, record_a)
        team_b = get_or_build_team(team_b_name, seed_b, record_b)

        matchup = Matchup(
            game_id=f"backtest_{team_a_name}_{team_b_name}",
            team_a=team_a,
            team_b=team_b,
            round_name=f"{round_name}",
            game_time="",
            venue="",
        )

        try:
            pred = predictor.predict_game(matchup)
        except Exception as e:
            print(f"  ERROR predicting {team_a_name} vs {team_b_name}: {e}")
            continue

        # Actual results
        actual_winner = team_a_name if actual_score_a > actual_score_b else team_b_name
        actual_margin = abs(actual_score_a - actual_score_b)
        actual_total = actual_score_a + actual_score_b
        is_upset = (actual_winner == team_a_name and seed_a > seed_b) or \
                   (actual_winner == team_b_name and seed_b > seed_a)

        # Model prediction
        pred_winner = pred.predicted_winner
        pred_margin = abs(pred.predicted_spread)
        pred_total = pred.predicted_total
        pred_prob = max(pred.team_a_win_prob, pred.team_b_win_prob)

        # Model predicted upset?
        if pred_winner == team_a.name:
            pred_upset = seed_a > seed_b
        else:
            pred_upset = seed_b > seed_a

        correct = pred_winner == actual_winner
        spread_error = abs(pred_margin - actual_margin)
        total_error = abs(pred_total - actual_total)

        results["total"] += 1
        if correct:
            results["correct"] += 1
        else:
            results["wrong"] += 1

        if is_upset:
            results["upsets_actual"] += 1
            if pred_upset and correct:
                results["upsets_caught"] += 1
        if pred_upset:
            results["upsets_predicted"] += 1

        results["spread_errors"].append(spread_error)
        results["total_errors"].append(total_error)

        if round_name not in results["by_round"]:
            results["by_round"][round_name] = {"total": 0, "correct": 0}
        results["by_round"][round_name]["total"] += 1
        if correct:
            results["by_round"][round_name]["correct"] += 1

        # Status marker
        mark = "✓" if correct else "✗"
        upset_flag = " *** UPSET ***" if is_upset else ""

        results["details"].append({
            "round": round_name, "correct": correct,
            "team_a": team_a_name, "seed_a": seed_a, "score_a": actual_score_a,
            "team_b": team_b_name, "seed_b": seed_b, "score_b": actual_score_b,
            "pred_winner": pred_winner, "pred_prob": pred_prob,
            "pred_margin": pred_margin, "actual_margin": actual_margin,
            "pred_total": pred_total, "actual_total": actual_total,
            "is_upset": is_upset, "mark": mark, "upset_flag": upset_flag,
        })

    # ── PRINT RESULTS ──
    print()

    # Game-by-game results
    current_round = ""
    for d in results["details"]:
        if d["round"] != current_round:
            current_round = d["round"]
            round_labels = {"FF": "FIRST FOUR", "R1": "FIRST ROUND", "R2": "SECOND ROUND"}
            print(f"\n  ── {round_labels.get(current_round, current_round)} ──")

        actual_line = f"#{d['seed_a']} {d['team_a']} {d['score_a']} - #{d['seed_b']} {d['team_b']} {d['score_b']}"
        pred_line = f"Model: {d['pred_winner']} ({d['pred_prob']*100:.0f}%, margin {d['pred_margin']:.1f})"
        print(f"  {d['mark']} {actual_line:<52} {pred_line}{d['upset_flag']}")

    # Summary stats
    print("\n" + "=" * 80)
    print("  BACKTEST SUMMARY")
    print("=" * 80)

    accuracy = results["correct"] / max(results["total"], 1)
    print(f"\n  Overall Accuracy:     {results['correct']}/{results['total']} ({accuracy:.1%})")

    for rnd, data in sorted(results["by_round"].items()):
        rnd_acc = data["correct"] / max(data["total"], 1)
        round_labels = {"FF": "First Four", "R1": "First Round", "R2": "Second Round"}
        label = round_labels.get(rnd, rnd)
        print(f"  {label:20s}   {data['correct']}/{data['total']} ({rnd_acc:.1%})")

    print(f"\n  Actual Upsets:        {results['upsets_actual']}")
    print(f"  Upsets Predicted:     {results['upsets_predicted']}")
    print(f"  Upsets Caught:        {results['upsets_caught']}/{results['upsets_actual']}")

    if results["spread_errors"]:
        avg_spread_err = sum(results["spread_errors"]) / len(results["spread_errors"])
        avg_total_err = sum(results["total_errors"]) / len(results["total_errors"])
        med_spread = sorted(results["spread_errors"])[len(results["spread_errors"]) // 2]
        med_total = sorted(results["total_errors"])[len(results["total_errors"]) // 2]
        print(f"\n  Avg Margin Error:     {avg_spread_err:.1f} pts (median: {med_spread:.1f})")
        print(f"  Avg Total Error:      {avg_total_err:.1f} pts (median: {med_total:.1f})")

    # Confidence calibration
    high_conf = [d for d in results["details"] if d["pred_prob"] >= 0.70]
    med_conf = [d for d in results["details"] if 0.55 <= d["pred_prob"] < 0.70]
    low_conf = [d for d in results["details"] if d["pred_prob"] < 0.55]

    print(f"\n  Confidence Calibration:")
    if high_conf:
        hc_acc = sum(1 for d in high_conf if d["correct"]) / len(high_conf)
        print(f"    High (≥70%):  {sum(1 for d in high_conf if d['correct'])}/{len(high_conf)} ({hc_acc:.1%})")
    if med_conf:
        mc_acc = sum(1 for d in med_conf if d["correct"]) / len(med_conf)
        print(f"    Medium (55-70%): {sum(1 for d in med_conf if d['correct'])}/{len(med_conf)} ({mc_acc:.1%})")
    if low_conf:
        lc_acc = sum(1 for d in low_conf if d["correct"]) / len(low_conf)
        print(f"    Low (<55%):   {sum(1 for d in low_conf if d['correct'])}/{len(low_conf)} ({lc_acc:.1%})")

    # Worst misses
    wrong_games = [d for d in results["details"] if not d["correct"]]
    if wrong_games:
        print(f"\n  WRONG PICKS ({len(wrong_games)}):")
        for d in wrong_games:
            print(f"    #{d['seed_a']} {d['team_a']} {d['score_a']} - "
                  f"#{d['seed_b']} {d['team_b']} {d['score_b']}  "
                  f"(picked {d['pred_winner']} at {d['pred_prob']*100:.0f}%)")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    run_backtest()
