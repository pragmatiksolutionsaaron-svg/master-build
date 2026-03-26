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

    # Use built-in sample data (all Sweet 16 games):
    python run_predictions.py --sample

    # Thursday only (West & South regionals):
    python run_predictions.py --thursday

    # Friday only (East & Midwest regionals):
    python run_predictions.py --friday
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


def build_thursday_data() -> tuple[dict, list]:
    """
    Sweet 16 matchups for Thursday, March 26, 2026 (West & South Regionals).
    Real data sourced from KenPom and Barttorvik.
    """
    teams = {}

    # ── WEST REGION ──

    # #2 Purdue (29-8) — KenPom #8, AdjO #1 (133.5), AdjD #37 (~97.0)
    # Big Ten Tournament champions, 6-game win streak, Braden Smith all-time assists leader
    teams["Purdue"] = TeamStats(
        name="Purdue", seed=2, conference="Big Ten",
        wins=29, losses=8,
        points_per_game=82.0, field_goal_pct=0.522, three_point_pct=0.385,
        free_throw_pct=0.780, offensive_rebounds_per_game=10.2,
        assists_per_game=18.5, turnovers_per_game=10.5,
        points_allowed_per_game=64.0, steals_per_game=6.8,
        blocks_per_game=4.5, defensive_rebounds_per_game=25.5,
        opponent_field_goal_pct=0.420, opponent_three_point_pct=0.320,
        adjusted_offensive_efficiency=133.5, adjusted_defensive_efficiency=97.0,
        tempo=67.5, strength_of_schedule=0.90, net_ranking=8,
        bpi=36.5, elo_rating=1840,
        last_10_wins=9, last_10_losses=1, win_streak=6, tournament_wins=2,
        neutral_site_wins=8, neutral_site_losses=0,
        vs_top_25_wins=10, vs_top_25_losses=5,
        experience_score=3.8, depth_score=3.5, bench_points_per_game=22.0,
    )

    # #11 Texas (21-14) — KenPom ~#45, SEC 9-9
    # First Four survivor, upset BYU and Gonzaga, Sean Miller 9th Sweet 16 trip
    teams["Texas"] = TeamStats(
        name="Texas", seed=11, conference="SEC",
        wins=21, losses=14,
        points_per_game=83.8, field_goal_pct=0.460, three_point_pct=0.340,
        free_throw_pct=0.730, offensive_rebounds_per_game=10.5,
        assists_per_game=12.3, turnovers_per_game=10.7,
        points_allowed_per_game=76.8, steals_per_game=6.5,
        blocks_per_game=3.8, defensive_rebounds_per_game=24.0,
        opponent_field_goal_pct=0.440, opponent_three_point_pct=0.330,
        adjusted_offensive_efficiency=112.0, adjusted_defensive_efficiency=100.0,
        tempo=70.0, strength_of_schedule=0.85, net_ranking=45,
        bpi=12.0, elo_rating=1680,
        last_10_wins=6, last_10_losses=4, win_streak=3, tournament_wins=3,
        neutral_site_wins=4, neutral_site_losses=3,
        vs_top_25_wins=5, vs_top_25_losses=8,
        experience_score=3.2, depth_score=3.0, bench_points_per_game=18.5,
    )

    # #1 Arizona (34-2) — KenPom #3, AdjO #4 (127.9), AdjD #2 (88.8)
    # Big 12 regular season + tournament champs, 17-2 in Q1 games
    teams["Arizona"] = TeamStats(
        name="Arizona", seed=1, conference="Big 12",
        wins=34, losses=2,
        points_per_game=89.9, field_goal_pct=0.547, three_point_pct=0.363,
        free_throw_pct=0.745, offensive_rebounds_per_game=12.5,
        assists_per_game=17.0, turnovers_per_game=11.0,
        points_allowed_per_game=62.5, steals_per_game=7.0,
        blocks_per_game=5.0, defensive_rebounds_per_game=27.0,
        opponent_field_goal_pct=0.390, opponent_three_point_pct=0.311,
        adjusted_offensive_efficiency=127.9, adjusted_defensive_efficiency=88.8,
        tempo=69.5, strength_of_schedule=0.92, net_ranking=3,
        bpi=39.1, elo_rating=1885,
        last_10_wins=10, last_10_losses=0, win_streak=14, tournament_wins=2,
        neutral_site_wins=9, neutral_site_losses=0,
        vs_top_25_wins=16, vs_top_25_losses=2,
        experience_score=3.0, depth_score=3.8, bench_points_per_game=26.0,
    )

    # #4 Arkansas (28-8) — KenPom #16, AdjO #6 (128.5), AdjD #46 (99.5)
    # SEC Tournament champs, Darius Acuff 22.9 PPG (SEC POY + FOY)
    teams["Arkansas"] = TeamStats(
        name="Arkansas", seed=4, conference="SEC",
        wins=28, losses=8,
        points_per_game=89.9, field_goal_pct=0.480, three_point_pct=0.350,
        free_throw_pct=0.740, offensive_rebounds_per_game=11.0,
        assists_per_game=17.0, turnovers_per_game=8.8,
        points_allowed_per_game=80.1, steals_per_game=7.5,
        blocks_per_game=3.5, defensive_rebounds_per_game=24.0,
        opponent_field_goal_pct=0.440, opponent_three_point_pct=0.335,
        adjusted_offensive_efficiency=128.5, adjusted_defensive_efficiency=99.5,
        tempo=72.0, strength_of_schedule=0.88, net_ranking=16,
        bpi=29.0, elo_rating=1790,
        last_10_wins=8, last_10_losses=2, win_streak=5, tournament_wins=2,
        neutral_site_wins=6, neutral_site_losses=2,
        vs_top_25_wins=9, vs_top_25_losses=5,
        experience_score=2.2, depth_score=3.3, bench_points_per_game=23.0,
    )

    # ── SOUTH REGION ──

    # #4 Nebraska (28-6) — KenPom ~#15, AdjO ~118.0, AdjD ~90.5 (#10)
    # Best defensive team in Big Ten, program's first-ever Sweet 16, started 20-0
    teams["Nebraska"] = TeamStats(
        name="Nebraska", seed=4, conference="Big Ten",
        wins=28, losses=6,
        points_per_game=74.5, field_goal_pct=0.455, three_point_pct=0.355,
        free_throw_pct=0.740, offensive_rebounds_per_game=9.5,
        assists_per_game=14.5, turnovers_per_game=10.8,
        points_allowed_per_game=63.0, steals_per_game=7.5,
        blocks_per_game=4.0, defensive_rebounds_per_game=26.0,
        opponent_field_goal_pct=0.400, opponent_three_point_pct=0.300,
        adjusted_offensive_efficiency=118.0, adjusted_defensive_efficiency=90.5,
        tempo=64.5, strength_of_schedule=0.88, net_ranking=15,
        bpi=27.5, elo_rating=1800,
        last_10_wins=8, last_10_losses=2, win_streak=4, tournament_wins=2,
        neutral_site_wins=5, neutral_site_losses=1,
        vs_top_25_wins=10, vs_top_25_losses=4,
        experience_score=3.4, depth_score=3.3, bench_points_per_game=20.0,
    )

    # #9 Iowa (23-12) — KenPom ~#30, AdjO ~117.0, AdjD ~96.0
    # Upset #1 Florida, first Sweet 16 since 1999, first-year coach McCollum
    teams["Iowa"] = TeamStats(
        name="Iowa", seed=9, conference="Big Ten",
        wins=23, losses=12,
        points_per_game=75.0, field_goal_pct=0.450, three_point_pct=0.345,
        free_throw_pct=0.755, offensive_rebounds_per_game=9.8,
        assists_per_game=14.0, turnovers_per_game=11.5,
        points_allowed_per_game=70.0, steals_per_game=7.0,
        blocks_per_game=3.5, defensive_rebounds_per_game=24.5,
        opponent_field_goal_pct=0.425, opponent_three_point_pct=0.325,
        adjusted_offensive_efficiency=117.0, adjusted_defensive_efficiency=96.0,
        tempo=67.0, strength_of_schedule=0.82, net_ranking=30,
        bpi=21.0, elo_rating=1730,
        last_10_wins=7, last_10_losses=3, win_streak=3, tournament_wins=2,
        neutral_site_wins=3, neutral_site_losses=2,
        vs_top_25_wins=5, vs_top_25_losses=7,
        experience_score=3.5, depth_score=3.0, bench_points_per_game=19.0,
    )

    # #2 Houston (30-3) — KenPom #5, AdjO #10 (126.3), AdjD #5 (89.5)
    # Elite defense (61.6 PPG allowed), won first two games by 30+
    teams["Houston"] = TeamStats(
        name="Houston", seed=2, conference="Big 12",
        wins=30, losses=3,
        points_per_game=76.0, field_goal_pct=0.465, three_point_pct=0.355,
        free_throw_pct=0.738, offensive_rebounds_per_game=10.8,
        assists_per_game=14.5, turnovers_per_game=11.0,
        points_allowed_per_game=61.6, steals_per_game=7.8,
        blocks_per_game=5.0, defensive_rebounds_per_game=26.5,
        opponent_field_goal_pct=0.395, opponent_three_point_pct=0.305,
        adjusted_offensive_efficiency=126.3, adjusted_defensive_efficiency=89.5,
        tempo=65.0, strength_of_schedule=0.90, net_ranking=5,
        bpi=36.8, elo_rating=1855,
        last_10_wins=9, last_10_losses=1, win_streak=8, tournament_wins=2,
        neutral_site_wins=7, neutral_site_losses=1,
        vs_top_25_wins=12, vs_top_25_losses=2,
        experience_score=3.3, depth_score=3.6, bench_points_per_game=21.0,
    )

    # #3 Illinois (26-8) — KenPom #6, AdjO #1 (133.9), AdjD #28 (97.0)
    # Historically elite offense, 5 double-figure scorers, 39.2% ORB rate
    teams["Illinois"] = TeamStats(
        name="Illinois", seed=3, conference="Big Ten",
        wins=26, losses=8,
        points_per_game=84.4, field_goal_pct=0.475, three_point_pct=0.370,
        free_throw_pct=0.750, offensive_rebounds_per_game=11.5,
        assists_per_game=16.0, turnovers_per_game=11.2,
        points_allowed_per_game=71.0, steals_per_game=6.5,
        blocks_per_game=4.2, defensive_rebounds_per_game=24.5,
        opponent_field_goal_pct=0.425, opponent_three_point_pct=0.322,
        adjusted_offensive_efficiency=133.9, adjusted_defensive_efficiency=97.0,
        tempo=68.0, strength_of_schedule=0.88, net_ranking=6,
        bpi=36.9, elo_rating=1830,
        last_10_wins=7, last_10_losses=3, win_streak=2, tournament_wins=2,
        neutral_site_wins=5, neutral_site_losses=2,
        vs_top_25_wins=9, vs_top_25_losses=5,
        experience_score=3.0, depth_score=3.4, bench_points_per_game=23.0,
    )

    # ── SWEET 16 MATCHUPS — THURSDAY, MARCH 26, 2026 ──
    matchups = [
        # WEST REGIONAL — SAP Center, San Jose, CA
        Matchup(
            game_id="sweet16_west_1",
            team_a=teams["Texas"],
            team_b=teams["Purdue"],
            round_name="Sweet 16 — West Regional",
            game_time="Thursday, March 26, 2026 — 7:10 PM ET (CBS)",
            venue="SAP Center, San Jose, CA",
            spread=-7.5,  # Purdue favored by 7.5
            over_under=148.5,
            team_a_moneyline=277,
            team_b_moneyline=-361,
        ),
        Matchup(
            game_id="sweet16_west_2",
            team_a=teams["Arkansas"],
            team_b=teams["Arizona"],
            round_name="Sweet 16 — West Regional",
            game_time="Thursday, March 26, 2026 — 9:45 PM ET (CBS)",
            venue="SAP Center, San Jose, CA",
            spread=-8.5,  # Arizona favored by 8.5
            over_under=167.5,
            team_a_moneyline=300,
            team_b_moneyline=-400,
        ),
        # SOUTH REGIONAL — Toyota Center, Houston, TX
        Matchup(
            game_id="sweet16_south_1",
            team_a=teams["Iowa"],
            team_b=teams["Nebraska"],
            round_name="Sweet 16 — South Regional",
            game_time="Thursday, March 26, 2026 — 7:30 PM ET (TBS)",
            venue="Toyota Center, Houston, TX",
            spread=-2.5,  # Nebraska favored by 2.5
            over_under=133.5,
            team_a_moneyline=123,
            team_b_moneyline=-150,
        ),
        Matchup(
            game_id="sweet16_south_2",
            team_a=teams["Illinois"],
            team_b=teams["Houston"],
            round_name="Sweet 16 — South Regional",
            game_time="Thursday, March 26, 2026 — 10:05 PM ET (TBS)",
            venue="Toyota Center, Houston, TX",
            spread=-3.5,  # Houston favored by 3.5
            over_under=139.5,
            team_a_moneyline=140,
            team_b_moneyline=-165,
        ),
    ]

    return teams, matchups


def build_friday_data() -> tuple[dict, list]:
    """
    Sweet 16 matchups for Friday, March 27, 2026 (East & Midwest Regionals).
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
        teams, matchups = build_friday_data()

    return teams, matchups


def build_sample_data() -> tuple[dict, list]:
    """Combine Thursday and Friday Sweet 16 data for full predictions."""
    thu_teams, thu_matchups = build_thursday_data()
    fri_teams, fri_matchups = build_friday_data()
    all_teams = {**thu_teams, **fri_teams}
    all_matchups = thu_matchups + fri_matchups
    return all_teams, all_matchups


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
                        help='Use built-in sample data (all Sweet 16 games)')
    parser.add_argument('--thursday', action='store_true',
                        help='Thursday Mar 26 Sweet 16 (West & South)')
    parser.add_argument('--friday', action='store_true',
                        help='Friday Mar 27 Sweet 16 (East & Midwest)')
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
    elif args.thursday:
        print("\n" + "=" * 60)
        print("  NCAA MARCH MADNESS PREDICTION ENGINE")
        print("  Mode: THURSDAY MAR 26 — West & South Regionals")
        print("=" * 60 + "\n")
        teams, matchups = build_thursday_data()
    elif args.friday:
        print("\n" + "=" * 60)
        print("  NCAA MARCH MADNESS PREDICTION ENGINE")
        print("  Mode: FRIDAY MAR 27 — East & Midwest Regionals")
        print("=" * 60 + "\n")
        teams, matchups = build_friday_data()
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
