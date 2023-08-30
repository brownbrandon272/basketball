"""
get nba api data
"""

import typing as t
from datetime import datetime
from itertools import product
import pandas as pd

from scripts.get_data.call_api import retry_api_call
from scripts.get_data.nba_api_endpoints import SEASONS_DICT, LIST_OF_ENDPOINT_DICTS


TEST_LOOPS = 2


def retrieve_data(
    all_dates: bool = False, test: bool = False, specific_endpoints: t.List[str] = None
):
    ## Raise error if any of the endpoints are not in LIST_OF_ENDPOINT_DICTS
    endpoint_names = [
        endpoint_dict.get("name") for endpoint_dict in LIST_OF_ENDPOINT_DICTS
    ]
    for endpoint in specific_endpoints:
        if endpoint not in endpoint_names:
            raise ValueError(
                f"{endpoint} is not a valid endpoint\nAvailable endpoints: {endpoint_names}"
            )

    # if specific_endpoints was passed as None or had all its values removed
    if not specific_endpoints:
        specific_endpoints = LIST_OF_ENDPOINT_DICTS

    for endpoint_dict in specific_endpoints:
        query_api = NBAAPI(endpoint_dict, all_dates, test)
        query_api.fetch_data()
    return


class NBAAPI:
    def __init__(self, endpoint_dict, all_dates: bool = False, test: bool = False):
        self.data_path = "../data_temp/"
        self.start_season = (
            SEASONS_DICT.get("first_season")
            if all_dates
            else SEASONS_DICT.get("current_season_year")
        )
        self.test = test

        self.name = endpoint_dict.get("name")
        self.endpoint = endpoint_dict.get("endpoint")
        self.main_loop_parameter = endpoint_dict.get("main_loop_parameter")
        self.parent_dict = endpoint_dict.get("parent")
        self.parent_col = endpoint_dict.get("parent_col")
        self.current_csv_path = self.data_path + endpoint_dict.get("csv")
        self.parent_csv_path = (
            (self.data_path + self.parent_dict.get("csv")) if self.parent_dict else None
        )
        self.main_col_dtype = endpoint_dict.get("main_col_dtype")
        self.additional_parameters = endpoint_dict.get("additional_parameters")
        self.current_col = endpoint_dict.get("current_col")

        self.parent_csv_data = self._load_parent_csv()
        self.current_csv_data = self._load_current_csv()
        return

    def _load_current_csv(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(self.current_csv_path)
            data[self.current_col] = data[self.current_col].astype(self.main_col_dtype)
            return data
        except FileNotFoundError:
            print(
                f"No existing CSV file found for {self.name} endpoint. A new CSV file will be created."
            )
            return None

    def _load_parent_csv(self) -> pd.DataFrame:
        if self.parent_dict is None:
            return None

        try:
            data = pd.read_csv(self.parent_csv_path)
            data[self.parent_col] = data[self.parent_col].astype(self.main_col_dtype)
            return data
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Parent CSV file not found: {self.parent_csv_path}"
            ) from exc
        except Exception as exc:
            raise Exception(
                f"Error occurred while reading parent CSV file: {self.parent_csv_path}\n{exc}"
            ) from exc

    def fetch_data(self):
        ## TODO: add logic to check if data is already up to date
        if self.parent_dict is None:
            data = self._call_endpoint()
            self._save_data(data)
            self._printer(f"Data saved for {self.name}")
            return

        iterable = self._get_iterable()

        for i, loop_value in enumerate(iterable):
            if self.test and (i >= TEST_LOOPS):
                break

            if self.additional_parameters is None:
                param_dict = {self.main_loop_parameter: loop_value}
                data = self._call_endpoint(param_dict=param_dict)
            else:
                for param_dict in self._get_distinct_loop_combos(loop_value):
                    if i == 0:
                        data = self._call_endpoint(param_dict)
                    data = pd.concat(
                        [data, self._call_endpoint(param_dict)], ignore_index=True
                    )

            self._save_data(data)
            self._printer(i, f"Data saved for {self.name}, {param_dict}")
        return

    def _get_distinct_loop_combos(self, main_loop_value) -> t.List[t.Dict[str, t.Any]]:
        all_params_dict = self.additional_parameters
        all_params_dict[self.main_loop_parameter] = [main_loop_value]

        keys, values = all_params_dict.keys(), all_params_dict.values()

        return [dict(zip(keys, combination)) for combination in product(*values)]

    def _call_endpoint(self, param_dict: t.Dict[str, t.Any] = None) -> pd.DataFrame:
        # Call the endpoint with the current loop_value
        def api_call():
            if param_dict is None:
                return self.endpoint()
            return self.endpoint(**param_dict)

        data_frame = retry_api_call(api_call)
        return data_frame

    def _save_data(self, data):
        # Append 'scoreboard_data' to the existing CSV and export to a new CSV file
        self.current_csv_data = pd.concat(
            [self.current_csv_data, data], ignore_index=True
        )
        self.current_csv_data.to_csv(self.current_csv_path, index=False)
        return

    def _get_iterable(self) -> t.Iterable[t.Any]:
        if self.parent_dict is None:
            raise Exception(
                f"Parent dictionary not found for {self.name} endpoint. Unable to get iterable."
            )
        if self.parent_dict.get("name") == "seasons":
            return range(self.start_season, SEASONS_DICT["current_season_year"] + 1)
        return self._get_loop_values()

    def _get_loop_values(self) -> t.List[t.Any]:
        try:
            unique_loop_values = self.parent_csv_data[self.parent_col].unique().tolist()
            max_saved_loop_value = self.current_csv_data[self.main_loop_parameter].max()
            if max_saved_loop_value is None:
                print(f"Getting all unique values of {self.parent_col}")
                return unique_loop_values
            print(f"Filtering game dates to those greater than {max_saved_loop_value}")
            return unique_loop_values[unique_loop_values > max_saved_loop_value]
        except Exception as exc:
            raise Exception(
                f"Error occurred while getting unique values of {self.parent_col} from {self.parent_csv_path}\n{exc}"
            ) from exc

    def _printer(self, i: int = 0, verbage: str = ""):
        if self.test or (i % 5 == 0):
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{i}][{current_time}] {verbage}")
        return


if __name__ == "__main__":
    retrieve_data(
        all_dates=False,
        test=True,
        specific_endpoints=[],
    )


# ## static NBA APIs
# # Find player ID
# player_df = pd.DataFrame(players.get_players())
# print(player_df.sample(5))
# print(player_df.shape)
# # player_df.to_csv(data_path + 'players.csv', index=False)

# # Find team ID
# team_df = pd.DataFrame(teams.get_teams())
# print(team_df.head())
# print(team_df.shape)
# # team_df.to_csv(data_path + 'teams.csv', index=False)

# FIRST_SEASON = 1946  # team_df['year_founded'].min()
# LAST_SEASON = 2022

# data_path = "../data/"


# ## NBA APIs - No parameters needed
# def get_draft_history(season_year=None):
#     if season_year is None:
#         return drafthistory.DraftHistory().draft_history.get_data_frame()
#     return drafthistory.DraftHistory(
#         season_year_nullable=season_year
#     ).draft_history.get_data_frame()


# ## APIs by season
# # Get seasons of game log data
# def get_leaguegamelog_data(year, season_type_all_star):
#     lg = leaguegamelog.LeagueGameLog(
#         season=str(year), season_type_all_star=season_type_all_star
#     )
#     data_frame = lg.league_game_log.get_data_frame()
#     data_frame["SeasonTypeAllStar"] = season_type_all_star
#     return data_frame


# def get_leaguestandings_data(year):
#     ls = leaguestandingsv3.LeagueStandingsV3(season=str(year))
#     data_frame = ls.standings.get_data_frame()
#     return data_frame


# def fetch_game_data():
#     game_log_data = pd.DataFrame()
#     season_types = ["Regular Season", "Pre Season", "Playoffs", "All-Star"]

#     for i, year in enumerate(range(FIRST_SEASON, LAST_SEASON + 1)):
#         if TEST and (i >= TEST_LOOPS):
#             break
#         print(f"Getting data for {year}")
#         for season_type in season_types:
#             game_log_data = pd.concat(
#                 [game_log_data, get_leaguegamelog_data(year, season_type)],
#                 ignore_index=True,
#             )

#     # Export 'game_log_data' to a single CSV file
#     game_log_data.to_csv(data_path + "game_log_data.csv", index=False)
#     print("Game Log Data exported to game_log_data.csv")
#     return


# ## APIs - game_date
# def fetch_scoreboard_data(game_date) -> t.Tuple[pd.DataFrame, list]:
#     def api_call():
#         return scoreboardv2.ScoreboardV2(
#             game_date=game_date
#         ).game_header.get_data_frame()

#     data = retry_api_call(api_call)
#     game_ids = data["GAME_ID"].unique().tolist()
#     return data, game_ids


# ## APIs - game_id
# def fetch_boxscoretraditional_data(game_ids):
#     player_stats_data = pd.DataFrame()

#     def api_call(game_id):
#         return boxscoretraditionalv2.BoxScoreTraditionalV2(
#             game_id=game_id
#         ).player_stats.get_data_frame()

#     for i, game_id in enumerate(game_ids):
#         data = retry_api_call(lambda: api_call(game_id))
#         if i == 0:
#             player_stats_data = data
#         else:
#             player_stats_data = pd.concat([player_stats_data, data], ignore_index=True)

#     return player_stats_data


# def fetch_boxscoreadvanced_data(game_ids):
#     player_stats_data = pd.DataFrame()

#     def api_call(game_id):
#         return boxscoreadvancedv2.BoxScoreAdvancedV2(
#             game_id=game_id
#         ).player_stats.get_data_frame()

#     for i, game_id in enumerate(game_ids):
#         data = retry_api_call(lambda: api_call(game_id))
#         if i == 0:
#             player_stats_data = data
#         else:
#             player_stats_data = pd.concat([player_stats_data, data], ignore_index=True)

#     return player_stats_data


# def fetch_data():
#     game_log_data = pd.read_csv(data_path + "game_log_data.csv")
#     distinct_game_dates = np.sort(game_log_data["GAME_DATE"].unique())

#     ## Get distinct_game_dates that are greater than get_max_game_date()
#     distinct_game_dates = filter_game_dates(distinct_game_dates)

#     for i, game_date in enumerate(distinct_game_dates):
#         if TEST and (i >= TEST_LOOPS):
#             break

#         # Fetch and scoreboard data for the current game date
#         scoreboard_data, game_ids = fetch_scoreboard_data(game_date)

#         # Fetch and save player stats data for the current game date
#         box_score_traditional_data = fetch_boxscoretraditional_data(game_ids)

#         save_data(scoreboard_data, box_score_traditional_data)
#         printer(
#             i=i,
#             verbage=f"Scoreboard and Box Score Data saved for {game_date}, game_ids: {game_ids}",
#         )
#     return


# def printer(i: int, verbage: str):
#     if TEST or (i % 5 == 0):
#         current_time = datetime.now().strftime("%H:%M:%S")
#         print(f"[{i}][{current_time}] {verbage}")
#     return


# def filter_game_dates(distinct_game_dates: list):
#     try:
#         scoreboard_data = pd.read_csv(data_path + "scoreboard_data.csv")
#         max_game_date = scoreboard_data["GAME_DATE_EST"].max()
#         print(f"Filtering game dates to those greater than {max_game_date}")
#         return distinct_game_dates[distinct_game_dates > max_game_date]
#     except FileNotFoundError:
#         print("No existing scoreboard_data.csv file found. Fetching all game dates.")
#         return distinct_game_dates


# def save_data(scoreboard_data, box_score_traditional_data):
#     # Read the existing scoreboard CSV if it exists
#     try:
#         existing_scoreboard_data = pd.read_csv(data_path + f"scoreboard_data.csv")
#     except FileNotFoundError:
#         existing_scoreboard_data = pd.DataFrame()

#     # Read the existing box score traditional CSV if it exists
#     try:
#         existing_box_score_traditional_data = pd.read_csv(
#             data_path + f"box_score_traditional_data.csv"
#         )
#     except FileNotFoundError:
#         existing_box_score_traditional_data = pd.DataFrame()

#     # Append 'scoreboard_data' to the existing CSV and export to a new CSV file
#     updated_scoreboard_data = pd.concat(
#         [existing_scoreboard_data, scoreboard_data], ignore_index=True
#     )
#     updated_scoreboard_data.to_csv(data_path + f"scoreboard_data.csv", index=False)

#     # Append 'box_score_traditional_data' to the existing CSV and export to a new CSV file
#     updated_box_score_traditional_data = pd.concat(
#         [existing_box_score_traditional_data, box_score_traditional_data],
#         ignore_index=True,
#     )
#     updated_box_score_traditional_data.to_csv(
#         data_path + f"box_score_traditional_data.csv", index=False
#     )
#     return


# # Fetch data and save to CSV files
# fetch_data()
