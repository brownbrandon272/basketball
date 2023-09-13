"""
This module contains a dictionary of dictionaries that contain the parameters
"""

import pandas as pd
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    drafthistory,
    leaguegamelog,
    scoreboardv2,
    leaguestandings,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
)
from nba_api.stats.library.parameters import SeasonAll, SeasonYear, Season

FIRST_SEASON = 1946  # team_df['year_founded'].min()

SEASONS_DICT = {
    "name": "seasons",
    "first_season": FIRST_SEASON,
    "current_season": Season.current_season,  # '{}-{}'.format(current_season_year, str(current_season_year + 1)[2:])
    "current_season_year": SeasonYear.current_season_year,
    "all_seasons": SeasonAll.all,  # difficult to use, not sure if i'm using this anywhere right now
}

PLAYERS_DICT = {
    "name": "players",
    "endpoint": lambda: pd.DataFrame(players.get_players()),
    "csv": "players_data.csv",
}

TEAMS_DICT = {
    "name": "teams",
    "endpoint": lambda: pd.DataFrame(teams.get_teams()),
    "csv": "teams_data.csv",
}

DRAFT_HISTORY_DICT = {
    "name": "draft_history",
    "endpoint": lambda: drafthistory.DraftHistory().draft_history.get_data_frame(),
    "csv": "draft_history_data.csv",
}

LEAGUE_STANDINGS_DICT = {
    "name": "league_standings",
    "endpoint": lambda kwargs: leaguestandings.LeagueStandings(
        **kwargs
    ).standings.get_data_frame(),
    "main_loop_parameter": "season",
    "parent": SEASONS_DICT,
    "current_col": "season",
    "main_col_dtype": "int",
    "csv": "league_standings_data.csv",
}

LEAGUE_GAME_LOG_DICT = {
    "name": "league_game_log",
    "endpoint": lambda kwargs: leaguegamelog.LeagueGameLog(
        **kwargs
    ).league_game_log.get_data_frame(),
    "main_loop_parameter": "season",
    "parent": SEASONS_DICT,
    "current_col": "season",
    "main_col_dtype": "int",
    "additional_parameters": {
        "season_type_all_star": ["Regular Season", "Pre Season", "Playoffs", "All-Star"]
    },
    "csv": "game_log_data.csv",
}

SCOREBOARD_DICT = {
    "name": "scoreboard",
    "endpoint": lambda kwargs: scoreboardv2.ScoreboardV2(
        **kwargs
    ).game_header.get_data_frame(),
    "main_loop_parameter": "game_date",
    "parent": LEAGUE_GAME_LOG_DICT,
    "parent_col": "GAME_DATE",
    "current_col": "GAME_DATE_EST",
    "main_col_dtype": "datetime64[ns]",
    "main_col_transform": lambda x: x.strftime("%Y-%m-%d"),
    "save_transform": {
        "GAME_DATE_EST": lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S")
    },
    "csv": "scoreboard_data.csv",
}

BOX_SCORE_TRADITIONAL_DICT = {
    "name": "box_score_traditional",
    "endpoint": lambda kwargs: boxscoretraditionalv2.BoxScoreTraditionalV2(
        **kwargs
    ).player_stats.get_data_frame(),
    "main_loop_parameter": "game_id",
    "parent": SCOREBOARD_DICT,
    "parent_col": "GAME_ID",
    "current_col": "GAME_ID",
    "main_col_dtype": "int",
    "main_col_transform": lambda x: str(x).zfill(10),
    "csv": "box_score_traditional_data.csv",
}

BOX_SCORE_ADVANCED_DICT = {
    "name": "box_score_advanced",
    "endpoint": lambda kwargs: boxscoreadvancedv2.BoxScoreAdvancedV2(
        **kwargs
    ).player_stats.get_data_frame(),
    "main_loop_parameter": "game_id",
    "parent": SCOREBOARD_DICT,
    "parent_col": "GAME_ID",
    "current_col": "GAME_ID",
    "main_col_dtype": "int",
    "main_col_transform": lambda x: str(x).zfill(10),
    "csv": "box_score_advanced_data.csv",
}

LIST_OF_ALL_ENDPOINT_DICTS = [
    PLAYERS_DICT,
    TEAMS_DICT,
    DRAFT_HISTORY_DICT,
    LEAGUE_STANDINGS_DICT,
    LEAGUE_GAME_LOG_DICT,
    SCOREBOARD_DICT,
    BOX_SCORE_TRADITIONAL_DICT,
    BOX_SCORE_ADVANCED_DICT,
]

LIST_OF_DATE_ENDPOINT_DICTS = [
    LEAGUE_STANDINGS_DICT,
    LEAGUE_GAME_LOG_DICT,
    SCOREBOARD_DICT,
    BOX_SCORE_TRADITIONAL_DICT,
    BOX_SCORE_ADVANCED_DICT,
]
