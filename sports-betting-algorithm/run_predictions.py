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
    Sweet 16 matchups for Friday, March 27, 2026.
    Real data sourced from KenPom and Barttorvik (as of March 25, 2026).
    """
    teams = {}

    # ── EAST REGION ──

    # #1 Duke (34-2) — KenPom #3, Torvik #3
    # KenPom: AdjO 127.37, AdjD 89.51, AdjT 65.6, NetRtg +37.80
    # Torvik: AdjOE 127.1 (#8), AdjDE 91.6 (#2), Barthag .9774
    teams["Duke"] = TeamStats(
        name="Duke", seed=1, conference="ACC",
        wins=34, losses=2,
        points_per_game=82.4, field_goal_pct=0.466, three_point_pct=0.346,
        free_throw_pct=0.752, offensive_rebounds_per_game=10.5,
        assists_per_game=16.2, turnovers_per_game=10.8,
        points_allowed_per_game=62.9, steals_per_game=7.4,
        blocks_per_game=5.2, defensive_rebounds_per_game=26.3,
        opponent_field_goal_pct=0.399, opponent_three_point_pct=0.305,
        adjusted_offensive_efficiency=127.4, adjusted_defensive_efficiency=89.5,
        tempo=65.6, strength_of_schedule=0.88, net_ranking=3,
        bpi=37.8, elo_rating=1870,
        last_10_wins=10, last_10_losses=0, win_streak=12, tournament_wins=2,
        neutral_site_wins=8, neutral_site_losses=1,
        vs_top_25_wins=15, vs_top_25_losses=2,
        experience_score=3.0, depth_score=3.8, bench_points_per_game=25.0,
    )

    # #5 St. John's (30-6) — KenPom #16, Torvik #10
    # KenPom: AdjO 120.3, AdjD 93.5, AdjT 69.6, NetRtg +26.78
    # Torvik: AdjOE 120.3 (#38), AdjDE 93.3 (#8), Barthag .9489
    teams["St. John's"] = TeamStats(
        name="St. John's", seed=5, conference="Big East",
        wins=30, losses=6,
        points_per_game=78.5, field_goal_pct=0.455, three_point_pct=0.332,
        free_throw_pct=0.730, offensive_rebounds_per_game=10.8,
        assists_per_game=14.5, turnovers_per_game=11.2,
        points_allowed_per_game=65.8, steals_per_game=7.8,
        blocks_per_game=4.5, defensive_rebounds_per_game=25.2,
        opponent_field_goal_pct=0.412, opponent_three_point_pct=0.312,
        adjusted_offensive_efficiency=120.3, adjusted_defensive_efficiency=93.5,
        tempo=69.6, strength_of_schedule=0.82, net_ranking=16,
        bpi=26.8, elo_rating=1780,
        last_10_wins=8, last_10_losses=2, win_streak=4, tournament_wins=2,
        neutral_site_wins=5, neutral_site_losses=2,
        vs_top_25_wins=8, vs_top_25_losses=4,
        experience_score=3.2, depth_score=3.0, bench_points_per_game=20.5,
    )

    # #2 UConn (31-5) — KenPom #10, Torvik #9
    # KenPom: AdjO 122.4, AdjD 94.0, AdjT 64.7, NetRtg +28.34
    # Torvik: AdjOE 123.3 (#27), AdjDE 95.3 (#13), Barthag .9511
    teams["UConn"] = TeamStats(
        name="UConn", seed=2, conference="Big East",
        wins=31, losses=5,
        points_per_game=77.2, field_goal_pct=0.463, three_point_pct=0.347,
        free_throw_pct=0.740, offensive_rebounds_per_game=10.2,
        assists_per_game=15.0, turnovers_per_game=12.1,
        points_allowed_per_game=64.5, steals_per_game=7.0,
        blocks_per_game=4.8, defensive_rebounds_per_game=25.5,
        opponent_field_goal_pct=0.408, opponent_three_point_pct=0.307,
        adjusted_offensive_efficiency=122.4, adjusted_defensive_efficiency=94.0,
        tempo=64.7, strength_of_schedule=0.80, net_ranking=10,
        bpi=28.3, elo_rating=1790,
        last_10_wins=8, last_10_losses=2, win_streak=4, tournament_wins=2,
        neutral_site_wins=6, neutral_site_losses=2,
        vs_top_25_wins=9, vs_top_25_losses=3,
        experience_score=3.4, depth_score=3.5, bench_points_per_game=22.0,
    )

    # #3 Michigan State (27-7) — KenPom #9, Torvik #11
    # KenPom: AdjO 123.4, AdjD 94.4, AdjT 66.4, NetRtg +28.97
    # Torvik: AdjOE 123.5 (#25), AdjDE 96.0 (#14), Barthag .9477
    teams["Michigan State"] = TeamStats(
        name="Michigan State", seed=3, conference="Big Ten",
        wins=27, losses=7,
        points_per_game=79.5, field_goal_pct=0.461, three_point_pct=0.365,
        free_throw_pct=0.735, offensive_rebounds_per_game=9.5,
        assists_per_game=16.5, turnovers_per_game=11.8,
        points_allowed_per_game=66.2, steals_per_game=7.2,
        blocks_per_game=3.8, defensive_rebounds_per_game=24.8,
        opponent_field_goal_pct=0.417, opponent_three_point_pct=0.327,
        adjusted_offensive_efficiency=123.4, adjusted_defensive_efficiency=94.4,
        tempo=66.4, strength_of_schedule=0.90, net_ranking=9,
        bpi=29.0, elo_rating=1795,
        last_10_wins=8, last_10_losses=2, win_streak=3, tournament_wins=2,
        neutral_site_wins=5, neutral_site_losses=2,
        vs_top_25_wins=10, vs_top_25_losses=5,
        experience_score=3.5, depth_score=3.4, bench_points_per_game=21.0,
    )

    # ── MIDWEST REGION ──

    # #1 Michigan (33-3) — KenPom #1, Torvik #1
    # KenPom: AdjO 127.76, AdjD 89.92, AdjT 70.9, NetRtg +37.82
    # Torvik: AdjOE 128.7 (#4), AdjDE 92.3 (#4), Barthag .9786
    teams["Michigan"] = TeamStats(
        name="Michigan", seed=1, conference="Big Ten",
        wins=33, losses=3,
        points_per_game=84.8, field_goal_pct=0.477, three_point_pct=0.366,
        free_throw_pct=0.758, offensive_rebounds_per_game=10.0,
        assists_per_game=17.2, turnovers_per_game=11.0,
        points_allowed_per_game=64.5, steals_per_game=7.5,
        blocks_per_game=4.5, defensive_rebounds_per_game=25.8,
        opponent_field_goal_pct=0.395, opponent_three_point_pct=0.308,
        adjusted_offensive_efficiency=127.8, adjusted_defensive_efficiency=89.9,
        tempo=70.9, strength_of_schedule=0.88, net_ranking=1,
        bpi=37.8, elo_rating=1880,
        last_10_wins=9, last_10_losses=1, win_streak=4, tournament_wins=2,
        neutral_site_wins=7, neutral_site_losses=1,
        vs_top_25_wins=16, vs_top_25_losses=2,
        experience_score=3.2, depth_score=3.6, bench_points_per_game=24.0,
    )

    # #4 Alabama (25-9) — KenPom #12, Torvik #14
    # KenPom: AdjO 129.6, AdjD 102.3, AdjT 73.1, NetRtg +27.32
    # Torvik: AdjOE 130.2 (#3), AdjDE 102.2 (#57), Barthag .9421
    teams["Alabama"] = TeamStats(
        name="Alabama", seed=4, conference="SEC",
        wins=25, losses=9,
        points_per_game=87.5, field_goal_pct=0.475, three_point_pct=0.361,
        free_throw_pct=0.745, offensive_rebounds_per_game=10.5,
        assists_per_game=16.8, turnovers_per_game=10.5,
        points_allowed_per_game=74.2, steals_per_game=7.8,
        blocks_per_game=4.0, defensive_rebounds_per_game=24.5,
        opponent_field_goal_pct=0.439, opponent_three_point_pct=0.333,
        adjusted_offensive_efficiency=129.6, adjusted_defensive_efficiency=102.3,
        tempo=73.1, strength_of_schedule=0.92, net_ranking=12,
        bpi=27.3, elo_rating=1770,
        last_10_wins=7, last_10_losses=3, win_streak=2, tournament_wins=2,
        neutral_site_wins=5, neutral_site_losses=3,
        vs_top_25_wins=8, vs_top_25_losses=6,
        experience_score=2.5, depth_score=3.2, bench_points_per_game=22.5,
    )

    # #2 Iowa State (29-7) — KenPom #7, Torvik #7
    # KenPom: AdjO 124.2, AdjD 91.3, AdjT 67.0, NetRtg +32.97
    # Torvik: AdjOE 124.1 (#18), AdjDE 92.8 (#5), Barthag .9660
    teams["Iowa State"] = TeamStats(
        name="Iowa State", seed=2, conference="Big 12",
        wins=29, losses=7,
        points_per_game=78.8, field_goal_pct=0.475, three_point_pct=0.387,
        free_throw_pct=0.742, offensive_rebounds_per_game=9.8,
        assists_per_game=15.5, turnovers_per_game=11.2,
        points_allowed_per_game=63.5, steals_per_game=8.0,
        blocks_per_game=4.2, defensive_rebounds_per_game=25.0,
        opponent_field_goal_pct=0.406, opponent_three_point_pct=0.320,
        adjusted_offensive_efficiency=124.2, adjusted_defensive_efficiency=91.3,
        tempo=67.0, strength_of_schedule=0.86, net_ranking=7,
        bpi=33.0, elo_rating=1810,
        last_10_wins=8, last_10_losses=2, win_streak=3, tournament_wins=2,
        neutral_site_wins=6, neutral_site_losses=2,
        vs_top_25_wins=9, vs_top_25_losses=4,
        experience_score=3.3, depth_score=3.5, bench_points_per_game=21.5,
    )

    # #6 Tennessee (24-11) — KenPom #14, Torvik #12
    # KenPom: AdjO 121.6, AdjD 94.7, AdjT 65.1, NetRtg +26.88
    # Torvik: AdjOE 122.0 (#31), AdjDE 95.1 (#12), Barthag .9462
    teams["Tennessee"] = TeamStats(
        name="Tennessee", seed=6, conference="SEC",
        wins=24, losses=11,
        points_per_game=75.8, field_goal_pct=0.452, three_point_pct=0.340,
        free_throw_pct=0.728, offensive_rebounds_per_game=10.8,
        assists_per_game=14.2, turnovers_per_game=12.0,
        points_allowed_per_game=67.5, steals_per_game=8.5,
        blocks_per_game=4.0, defensive_rebounds_per_game=25.8,
        opponent_field_goal_pct=0.415, opponent_three_point_pct=0.305,
        adjusted_offensive_efficiency=121.6, adjusted_defensive_efficiency=94.7,
        tempo=65.1, strength_of_schedule=0.90, net_ranking=14,
        bpi=26.9, elo_rating=1755,
        last_10_wins=7, last_10_losses=3, win_streak=3, tournament_wins=2,
        neutral_site_wins=4, neutral_site_losses=3,
        vs_top_25_wins=6, vs_top_25_losses=7,
        experience_score=3.4, depth_score=3.2, bench_points_per_game=19.5,
    )

    # ── SWEET 16 MATCHUPS — FRIDAY, MARCH 27, 2026 ──
    matchups = [
        # EAST REGIONAL — Capital One Arena, Washington D.C.
        Matchup(
            game_id="sweet16_east_1",
            team_a=teams["St. John's"],
            team_b=teams["Duke"],
            round_name="Sweet 16 — East Regional",
            game_time="Friday, March 27, 2026 — 7:10 PM ET (CBS)",
            venue="Capital One Arena, Washington D.C.",
            spread=-6.5,  # Duke favored by 6.5
            over_under=142.5,
            team_a_moneyline=240,
            team_b_moneyline=-290,
        ),
        Matchup(
            game_id="sweet16_east_2",
            team_a=teams["Michigan State"],
            team_b=teams["UConn"],
            round_name="Sweet 16 — East Regional",
            game_time="Friday, March 27, 2026 — 9:45 PM ET (CBS)",
            venue="Capital One Arena, Washington D.C.",
            spread=-1.5,  # MSU slight favorite (line varies)
            over_under=138.0,
            team_a_moneyline=110,
            team_b_moneyline=-130,
        ),
        # MIDWEST REGIONAL — United Center, Chicago
        Matchup(
            game_id="sweet16_midwest_1",
            team_a=teams["Alabama"],
            team_b=teams["Michigan"],
            round_name="Sweet 16 — Midwest Regional",
            game_time="Friday, March 27, 2026 — 7:35 PM ET (TBS)",
            venue="United Center, Chicago, IL",
            spread=-10.0,  # Michigan favored by 10
            over_under=174.5,
            team_a_moneyline=370,
            team_b_moneyline=-485,
        ),
        Matchup(
            game_id="sweet16_midwest_2",
            team_a=teams["Tennessee"],
            team_b=teams["Iowa State"],
            round_name="Sweet 16 — Midwest Regional",
            game_time="Friday, March 27, 2026 — 10:10 PM ET (TBS)",
            venue="United Center, Chicago, IL",
            spread=-4.5,  # Iowa State favored by 4.5
            over_under=138.5,
            team_a_moneyline=160,
            team_b_moneyline=-192,
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
