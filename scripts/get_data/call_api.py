"""
This module contains a function to retry an API call in case of failure.
"""
import random
import time
import pandas as pd
from requests.exceptions import ReadTimeout
from urllib3.exceptions import ProtocolError
from json.decoder import JSONDecodeError

MAX_RETRIES = 5
WAIT_TIME = [5, 5, 5, 20, 120]


def retry_api_call(api_call, max_retries=MAX_RETRIES, param_dict=None):
    retries = 0
    while retries < max_retries:
        try:
            data = api_call()
            # Add a random delay between 0.1 and 0.15 second
            delay = random.uniform(0.1, 0.15)
            time.sleep(delay)
            return data
        except (ConnectionError, ProtocolError, ReadTimeout) as exc:
            print(f"Error occurred: {exc}. param_dict: {param_dict}")
            retries += 1
            print(f"Retrying... (Attempt {retries}/{max_retries})")
            time.sleep(WAIT_TIME[retries - 1])
        except JSONDecodeError as exc:
            err_msg = f"Error occurred: {exc}"
            print(f"{err_msg}. param_dict: {param_dict}")
            param_dict["error"] = err_msg
            return pd.DataFrame(index=[0], data=param_dict)
        except IndexError as exc:
            print(
                f"Index Error occurred, possibly meaning that this API has no data for these params: {exc}"
            )
            print(f"param_dict: {param_dict}")
            return pd.DataFrame()
        except Exception as exc:
            print(f"Unknown error occurred: {exc}")
            retries += 1
            print(f"Retrying... (Attempt {retries}/{max_retries})")
            time.sleep(WAIT_TIME[retries - 1])
    raise ConnectionError(f"API call failed after {max_retries} retries.")
