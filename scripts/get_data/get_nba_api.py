"""
This script retrieves data from the NBA API and saves it to a CSV file.
The NBA API is a RESTful API that provides data for NBA games, players, teams, etc.
"""

import typing as t
from datetime import datetime
from itertools import product
import pandas as pd
import numpy as np

from scripts.get_data.call_api import retry_api_call
from scripts.get_data.nba_api_endpoints import SEASONS_DICT, LIST_OF_ENDPOINT_DICTS


TEST_LOOPS = 2


def retrieve_data(
    all_dates: bool = False,
    test: bool = False,
    specific_endpoint_names: t.List[str] = None,
):
    # if specific_endpoint_names is None then retrieve all endpoints
    if not specific_endpoint_names:
        _list_of_endpoint_dicts = LIST_OF_ENDPOINT_DICTS
    else:
        ## Check if passed endpoints are valid, and if so add their dictionaries to _list_of_endpoint_dicts
        _list_of_endpoint_dicts = []
        ## Raise error if any of the endpoints are not in LIST_OF_ENDPOINT_DICTS
        dict_of_endpoints = {
            endpoint_dict.get("name"): endpoint_dict
            for endpoint_dict in LIST_OF_ENDPOINT_DICTS
        }

        for endpoint_name in specific_endpoint_names:
            if endpoint_name not in dict_of_endpoints.keys():
                raise ValueError(
                    f"{endpoint_name} is not a valid endpoint\nAvailable endpoints: {dict_of_endpoints.keys()}"
                )
            _list_of_endpoint_dicts.append(dict_of_endpoints.get(endpoint_name))

    for endpoint_dict in _list_of_endpoint_dicts:
        query_api = NBAAPI(endpoint_dict=endpoint_dict, all_dates=all_dates, test=test)
        query_api.fetch_data()
    return


class NBAAPI:
    def __init__(
        self,
        endpoint_dict,
        all_dates: bool = False,
        test: bool = False,
        data_path: str = None,
    ):
        """
        NBA API class to retrieve data from the NBA API and save it to a CSV file

        Parameters
        ----------
        endpoint_dict : dict
            Dictionary containing the parameters for the endpoint
        all_dates : bool, optional
            If True, retrieve all dates for the endpoint, by default False
        test : bool, optional
            If True, only retrieve a small subset of data, by default False
        data_path : str, optional
            Path to where the data is stored, by default None
        """
        self.data_path = "./data/" if data_path is None else data_path
        self.start_season = (
            SEASONS_DICT.get("first_season")
            if all_dates
            else SEASONS_DICT.get("current_season_year")
        )
        self.test = test

        self.name = endpoint_dict.get("name")
        self.endpoint = endpoint_dict.get("endpoint")
        self.main_loop_parameter = endpoint_dict.get("main_loop_parameter")
        self.current_csv_path = self.data_path + endpoint_dict.get("csv")
        self.main_col_dtype = endpoint_dict.get("main_col_dtype")
        self.additional_parameters = endpoint_dict.get("additional_parameters")
        self.current_col = endpoint_dict.get("current_col")
        self.main_col_transform = endpoint_dict.get("main_col_transform")
        self.save_transform = endpoint_dict.get("save_transform")

        self.current_csv_data = self._load_current_csv()

        self.parent_col = endpoint_dict.get("parent_col")
        self.parent_dict = endpoint_dict.get("parent")
        if self.parent_dict and self.parent_dict.get("csv"):
            self.parent_csv_path = (
                (self.data_path + self.parent_dict.get("csv"))
                if (self.parent_dict and self.parent_dict.get("csv"))
                else None
            )
            self.parent_csv_data = self._load_parent_csv()
        else:
            self.parent_csv_path = None
            self.parent_csv_data = pd.DataFrame()

        return

    def _load_current_csv(self) -> pd.DataFrame:
        """
        Load the current CSV file, if it exists
        """
        try:
            data = pd.read_csv(self.current_csv_path)
            if self.main_col_dtype:
                data[self.current_col] = data[self.current_col].astype(
                    self.main_col_dtype
                )
            return data
        except FileNotFoundError:
            print(
                f"No existing CSV file found for {self.name} endpoint. A new CSV file will be created."
            )
            return pd.DataFrame()
        except Exception as exc:
            raise Exception(
                f"Error occurred while reading current CSV file: {self.current_csv_path}\n{exc}"
            ) from exc

    def _load_parent_csv(self) -> pd.DataFrame:
        """
        Load the parent CSV file, if it exists
        """
        if self.parent_dict is None:
            return pd.DataFrame()

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
        """
        Retrieve data from the NBA API and save it to a CSV file
        """
        ## TODO: add logic to check if data is already up to date
        if self.parent_dict is None:
            data = self._call_endpoint()
            self._save_data(data)
            self._printer(verbage=f"Data saved for {self.name}")
            return

        iterable = self._get_iterable()

        for i, loop_value in enumerate(iterable):
            if self.test and (i >= TEST_LOOPS):
                break

            if self.additional_parameters is None:
                param_dict = {self.main_loop_parameter: loop_value}
                data = self._call_endpoint(param_dict=param_dict)
            else:
                for j, param_dict in enumerate(
                    self._get_distinct_loop_combos(loop_value)
                ):
                    if j == 0:
                        data = self._call_endpoint(param_dict)
                    else:
                        data = pd.concat(
                            [data, self._call_endpoint(param_dict)], ignore_index=True
                        )
            if data.empty:
                self._printer(i=i, verbage=f"No data for {self.name}, {param_dict}")
                continue
            self._save_data(data)
            self._printer(i=i, verbage=f"Data saved for {self.name}, {param_dict}")
        return

    def _get_distinct_loop_combos(self, main_loop_value) -> t.List[t.Dict[str, t.Any]]:
        """
        Get all distinct combinations of the additional parameters

        Parameters
        ----------
        main_loop_value : int
            The current value of the main loop parameter

        Returns
        -------
        list
            List of dictionaries containing the distinct combinations of the additional parameters
        """
        all_params_dict = self.additional_parameters
        all_params_dict[self.main_loop_parameter] = [main_loop_value]

        keys, values = all_params_dict.keys(), all_params_dict.values()

        return [dict(zip(keys, combination)) for combination in product(*values)]

    def _call_endpoint(self, param_dict: t.Dict[str, t.Any] = None) -> pd.DataFrame:
        # Call the endpoint with the current loop_value
        def api_call():
            if param_dict is None:
                return self.endpoint()
            return self.endpoint(param_dict)

        data_frame = retry_api_call(
            api_call, endpoint=self.endpoint, param_dict=param_dict
        )
        data_frame = self._add_params(data_frame, param_dict)
        return data_frame

    def _save_data(self, data: pd.DataFrame):
        if self.save_transform:
            for col, transform in self.save_transform.items():
                data[col] = data[col].apply(transform)
        if self.current_csv_data.empty:
            data.to_csv(self.current_csv_path, index=False)
            self.current_csv_data = self._load_current_csv()
            return
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
        loop_values = self._get_loop_values()
        if self.current_csv_data.empty:
            print(f"Getting all unique values of {self.parent_col}")
        else:
            max_saved_loop_value = self.current_csv_data[self.current_col].max()
            print(f"Filtering game dates to those greater than {max_saved_loop_value}")
            loop_values = loop_values[loop_values > max_saved_loop_value]

        if self.main_col_transform:
            loop_values = pd.Series(loop_values).apply(self.main_col_transform)
        return np.sort(loop_values)

    def _get_loop_values(self):
        try:
            if self.parent_dict.get("name") == "seasons":
                loop_values = np.array(
                    range(self.start_season, SEASONS_DICT["current_season_year"] + 1)
                )
            else:
                loop_values = np.array(self.parent_csv_data[self.parent_col].unique())
            return loop_values
        except Exception as exc:
            raise Exception(
                f"Error occurred while getting unique values of {self.parent_col} from {self.parent_csv_path}\n{exc}"
            ) from exc

    @staticmethod
    def _add_params(
        data: pd.DataFrame, param_dict: t.Dict[str, t.Any] = None
    ) -> pd.DataFrame:
        if param_dict is None:
            return data
        for key, value in param_dict.items():
            if key not in data.columns:
                data[key] = value
        return data

    def _printer(self, i: int = 0, verbage: str = ""):
        if self.test or (i % 5 == 0):
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{i}][{current_time}] {verbage}")
        return


if __name__ == "__main__":
    retrieve_data(
        all_dates=False,
        test=True,
        specific_endpoint_names=[],
    )
