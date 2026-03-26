#!/usr/bin/env python3
"""
NCAA March Madness Prediction Engine — Main Runner

Usage:
    # Auto-discover games from ESPN (live data):
    python run_predictions.py --live

    # Auto-discover games for a specific date:
    python run_predictions.py --live --date 20260327

    # Use manual config files:
    python run_predictions.py --teams config/teams.json --matchups config/matchups.json

    # Use built-in sample data (for testing):
    python run_predictions.py --sample
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.team_stats import TeamStats, Matchup, load_teams_from_json, load_matchups_from_json
from src.models.prediction_engine import CompositePredictor
from src.analysis.betting_analyzer import BettingAnalyzer
from src.analysis.report_generator import generate_full_report


def build_sample_data() -> tuple[dict, list]:
    """
    Build sample tournament data for demonstration.
    Update this with real teams/stats for actual predictions.
    This uses realistic stat profiles based on typical Final Four teams.
    """
    teams = {}

    # These are template team profiles — replace with actual data
    # or use --live mode to pull from ESPN

    teams["Houston"] = TeamStats(
        name="Houston", seed=1, conference="Big 12",
        wins=33, losses=3,
        points_per_game=76.8, field_goal_pct=0.472, three_point_pct=0.352,
        free_throw_pct=0.738, offensive_rebounds_per_game=11.2,
        assists_per_game=14.8, turnovers_per_game=11.1,
        points_allowed_per_game=59.2, steals_per_game=8.1,
        blocks_per_game=4.8, defensive_rebounds_per_game=26.3,
        opponent_field_goal_pct=0.388, opponent_three_point_pct=0.298,
        adjusted_offensive_efficiency=121.5, adjusted_defensive_efficiency=89.2,
        tempo=65.8, strength_of_schedule=0.82, net_ranking=1,
        bpi=28.5, elo_rating=1845,
        last_10_wins=9, last_10_losses=1, win_streak=6, tournament_wins=4,
        neutral_site_wins=8, neutral_site_losses=1,
        vs_top_25_wins=10, vs_top_25_losses=2,
        experience_score=3.2, depth_score=3.5, bench_points_per_game=22.1,
    )

    teams["Duke"] = TeamStats(
        name="Duke", seed=1, conference="ACC",
        wins=32, losses=5,
        points_per_game=82.4, field_goal_pct=0.491, three_point_pct=0.371,
        free_throw_pct=0.762, offensive_rebounds_per_game=10.1,
        assists_per_game=16.2, turnovers_per_game=10.8,
        points_allowed_per_game=66.8, steals_per_game=7.4,
        blocks_per_game=5.2, defensive_rebounds_per_game=24.8,
        opponent_field_goal_pct=0.405, opponent_three_point_pct=0.312,
        adjusted_offensive_efficiency=125.8, adjusted_defensive_efficiency=93.5,
        tempo=69.2, strength_of_schedule=0.78, net_ranking=3,
        bpi=25.2, elo_rating=1820,
        last_10_wins=9, last_10_losses=1, win_streak=8, tournament_wins=4,
        neutral_site_wins=7, neutral_site_losses=2,
        vs_top_25_wins=9, vs_top_25_losses=3,
        experience_score=2.8, depth_score=3.8, bench_points_per_game=25.3,
    )

    teams["Auburn"] = TeamStats(
        name="Auburn", seed=1, conference="SEC",
        wins=31, losses=5,
        points_per_game=80.1, field_goal_pct=0.468, three_point_pct=0.361,
        free_throw_pct=0.745, offensive_rebounds_per_game=10.8,
        assists_per_game=15.1, turnovers_per_game=12.2,
        points_allowed_per_game=64.5, steals_per_game=7.8,
        blocks_per_game=4.1, defensive_rebounds_per_game=25.1,
        opponent_field_goal_pct=0.398, opponent_three_point_pct=0.305,
        adjusted_offensive_efficiency=122.3, adjusted_defensive_efficiency=91.8,
        tempo=68.1, strength_of_schedule=0.85, net_ranking=2,
        bpi=26.8, elo_rating=1830,
        last_10_wins=8, last_10_losses=2, win_streak=3, tournament_wins=4,
        neutral_site_wins=6, neutral_site_losses=2,
        vs_top_25_wins=11, vs_top_25_losses=3,
        experience_score=3.0, depth_score=3.2, bench_points_per_game=20.8,
    )

    teams["Florida"] = TeamStats(
        name="Florida", seed=1, conference="SEC",
        wins=32, losses=6,
        points_per_game=83.5, field_goal_pct=0.485, three_point_pct=0.378,
        free_throw_pct=0.771, offensive_rebounds_per_game=9.8,
        assists_per_game=16.8, turnovers_per_game=11.5,
        points_allowed_per_game=67.2, steals_per_game=7.2,
        blocks_per_game=3.8, defensive_rebounds_per_game=24.2,
        opponent_field_goal_pct=0.412, opponent_three_point_pct=0.318,
        adjusted_offensive_efficiency=126.2, adjusted_defensive_efficiency=95.1,
        tempo=70.5, strength_of_schedule=0.80, net_ranking=5,
        bpi=24.1, elo_rating=1805,
        last_10_wins=8, last_10_losses=2, win_streak=5, tournament_wins=4,
        neutral_site_wins=6, neutral_site_losses=1,
        vs_top_25_wins=8, vs_top_25_losses=4,
        experience_score=3.4, depth_score=3.6, bench_points_per_game=24.5,
    )

    # Sample Final Four matchups
    matchups = [
        Matchup(
            game_id="final_four_1",
            team_a=teams["Auburn"],
            team_b=teams["Houston"],
            round_name="Final Four — National Semifinal",
            game_time="Saturday, March 28, 2026 — 6:09 PM ET",
            venue="Alamodome, San Antonio, TX",
            spread=-2.5,  # Houston favored by 2.5
            over_under=138.5,
            team_a_moneyline=120,
            team_b_moneyline=-140,
        ),
        Matchup(
            game_id="final_four_2",
            team_a=teams["Florida"],
            team_b=teams["Duke"],
            round_name="Final Four — National Semifinal",
            game_time="Saturday, March 28, 2026 — 8:49 PM ET",
            venue="Alamodome, San Antonio, TX",
            spread=1.5,  # Duke favored by 1.5
            over_under=149.0,
            team_a_moneyline=105,
            team_b_moneyline=-125,
        ),
    ]

    return teams, matchups


def run_live_mode(date_str=None):
    """Run predictions using live ESPN data."""
    from src.data.live_data_fetcher import LiveDataFetcher

    print("\n" + "=" * 60)
    print("  NCAA MARCH MADNESS PREDICTION ENGINE")
    print("  Mode: LIVE DATA (ESPN API)")
    print("=" * 60 + "\n")

    fetcher = LiveDataFetcher(cache_dir=os.path.join(
        os.path.dirname(__file__), ".cache"
    ))

    teams, matchups = fetcher.auto_discover_matchups(date_str)

    if not matchups:
        print("\n  No tournament matchups found via live data.")
        print("  Falling back to sample data...")
        print("  TIP: Use --date YYYYMMDD to specify a game date")
        print("  TIP: Or update config/teams.json and config/matchups.json manually\n")
        teams, matchups = build_sample_data()

    return teams, matchups


def run_config_mode(teams_file, matchups_file):
    """Run predictions using manual config files."""
    print("\n" + "=" * 60)
    print("  NCAA MARCH MADNESS PREDICTION ENGINE")
    print("  Mode: MANUAL CONFIG")
    print("=" * 60 + "\n")

    teams = load_teams_from_json(teams_file)
    matchups = load_matchups_from_json(matchups_file, teams)
    return teams, matchups


def run_sample_mode():
    """Run predictions using built-in sample data."""
    print("\n" + "=" * 60)
    print("  NCAA MARCH MADNESS PREDICTION ENGINE")
    print("  Mode: SAMPLE DATA (Update for real predictions)")
    print("=" * 60 + "\n")

    return build_sample_data()


def main():
    parser = argparse.ArgumentParser(
        description="NCAA March Madness Prediction Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_predictions.py --live                    # Auto-fetch from ESPN
  python run_predictions.py --live --date 20260327    # Specific date
  python run_predictions.py --sample                  # Demo with sample data
  python run_predictions.py --teams config/teams.json --matchups config/matchups.json
        """
    )
    parser.add_argument('--live', action='store_true',
                        help='Fetch live data from ESPN API')
    parser.add_argument('--date', type=str, default=None,
                        help='Date for live data (YYYYMMDD format)')
    parser.add_argument('--teams', type=str,
                        help='Path to teams JSON config file')
    parser.add_argument('--matchups', type=str,
                        help='Path to matchups JSON config file')
    parser.add_argument('--sample', action='store_true',
                        help='Use built-in sample data')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for reports')
    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights as JSON: {"efficiency":0.4,"elo":0.25,"seed":0.2,"momentum":0.15}')

    args = parser.parse_args()

    # Determine mode
    if args.live:
        teams, matchups = run_live_mode(args.date)
    elif args.teams and args.matchups:
        teams, matchups = run_config_mode(args.teams, args.matchups)
    elif args.sample:
        teams, matchups = run_sample_mode()
    else:
        # Default to live, fall back to sample
        try:
            teams, matchups = run_live_mode(args.date)
        except Exception:
            print("  Live mode failed, using sample data...")
            teams, matchups = run_sample_mode()

    if not matchups:
        print("\n  ERROR: No matchups to analyze. Exiting.\n")
        sys.exit(1)

    # Parse custom weights
    model_kwargs = {}
    if args.weights:
        try:
            w = json.loads(args.weights)
            model_kwargs = {
                'efficiency_weight': w.get('efficiency', 0.40),
                'elo_weight': w.get('elo', 0.25),
                'seed_weight': w.get('seed', 0.20),
                'momentum_weight': w.get('momentum', 0.15),
            }
        except json.JSONDecodeError:
            print("  [WARN] Invalid weights JSON, using defaults")

    # Initialize models
    predictor = CompositePredictor(**model_kwargs)
    analyzer = BettingAnalyzer()

    # Generate predictions
    print(f"\n  Analyzing {len(matchups)} matchup(s)...\n")

    predictions = []
    all_bets = []

    for matchup in matchups:
        print(f"  → {matchup.team_a.name} vs {matchup.team_b.name} ({matchup.round_name})")

        prediction = predictor.predict_game(matchup)
        predictions.append(prediction)

        bets = analyzer.analyze_game(prediction)
        all_bets.extend(bets)

        # Quick summary
        winner = prediction.predicted_winner
        prob = max(prediction.team_a_win_prob, prediction.team_b_win_prob)
        print(f"    Prediction: {winner} ({prob*100:.1f}%) | "
              f"Score: {prediction.team_a_predicted_score:.0f}-{prediction.team_b_predicted_score:.0f} | "
              f"Confidence: {prediction.confidence:.0f}/100")

    # Rank all bets
    all_bets = analyzer.rank_bets(all_bets)

    # Generate report
    output_dir = os.path.join(os.path.dirname(__file__), args.output)
    report = generate_full_report(predictions, all_bets, output_dir)

    print("\n" + report)

    return predictions, all_bets


if __name__ == "__main__":
    main()
