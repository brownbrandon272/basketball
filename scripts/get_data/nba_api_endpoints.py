"""
This module contains a dictionary of dictionaries that contain the parameters
"""

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    drafthistory,
    leaguegamelog,
    scoreboardv2,
    leaguestandingsv3,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
)

FIRST_SEASON = 1946  # team_df['year_founded'].min()
LAST_SEASON = 2023

seasons_dict = {
    "name": "seasons",
    "first_season": FIRST_SEASON,
    "last_season": LAST_SEASON,
}

players_dict = {
    "name": "players",
    "endpoint": lambda kwargs: players.get_players(),
    "main_loop_parameter": None,
    "parent": None,
    "parent_col": None,
    "csv": "players_data.csv",
}

teams_dict = {
    "name": "teams",
    "endpoint": lambda kwargs: teams.get_teams(),
    "main_loop_parameter": None,
    "parent": None,
    "parent_col": None,
    "csv": "teams_data.csv",
}

draft_history_dict = {
    "name": "draft_history",
    "endpoint": lambda kwargs: drafthistory.DraftHistory(**kwargs).draft_history,
    "main_loop_parameter": None,
    "parent": None,
    "parent_col": None,
    "csv": "draft_history_data.csv",
}

league_game_log_dict = {
    "name": "league_game_log",
    "endpoint": lambda kwargs: leaguegamelog.LeagueGameLog(**kwargs).league_game_log,
    "main_loop_parameter": "season",
    "parent": teams_dict,
    "parent_col": "year_founded",
    "additional_parameters": {
        "season_type_all_star": ["Regular Season", "Pre Season", "Playoffs", "All-Star"]
    },
    "csv": "game_log_data.csv",
}

scoreboard_dict = {
    "name": "scoreboard",
    "endpoint": lambda kwargs: scoreboardv2.ScoreboardV2(**kwargs).game_header,
    "main_loop_parameter": "season",
    "parent": seasons_dict,
    "parent_col": None,
    "csv": "scoreboard_data.csv",
}

league_standings_dict = {
    "name": "league_standings",
    "endpoint": lambda kwargs: leaguestandingsv3.LeagueStandingsV3(**kwargs).standings,
    "main_loop_parameter": "season",
    "parent": seasons_dict,
    "parent_col": None,
    "csv": "league_standings_data.csv",
}

box_score_traditional_dict = {
    "name": "box_score_traditional",
    "endpoint": lambda kwargs: boxscoretraditionalv2.BoxScoreTraditionalV2(
        **kwargs
    ).player_stats,
    "main_loop_parameter": "game_id",
    "parent": scoreboard_dict,
    "parent_col": "GAME_ID",
    "csv": "box_score_traditional_data.csv",
}

box_score_advanced_dict = {
    "name": "box_score_advanced",
    "endpoint": lambda kwargs: boxscoreadvancedv2.BoxScoreAdvancedV2(
        **kwargs
    ).player_stats,
    "main_loop_parameter": "game_id",
    "parent": scoreboard_dict,
    "parent_col": "GAME_ID",
    "csv": "box_score_advanced_data.csv",
}
