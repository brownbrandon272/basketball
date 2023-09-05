"""
This module contains a function to retry an API call in case of failure.
"""
import random
import time
import pandas as pd
from requests.exceptions import ReadTimeout
from urllib3.exceptions import ProtocolError
from json.decoder import JSONDecodeError

MAX_RETRIES = 3
WAIT_TIME = 40


def retry_api_call(api_call, max_retries=MAX_RETRIES, endpoint=None, param_dict=None):
    retries = 0
    while retries < max_retries:
        try:
            data = api_call()
            # Add a random delay between 0.5 and 1 second
            delay = random.uniform(0.5, 1.0)
            time.sleep(delay)
            return data
        except (ConnectionError, ProtocolError, ReadTimeout, JSONDecodeError) as exc:
            print(f"Error occurred: {exc}. param_dict: {param_dict}")
            retries += 1
            print(f"Retrying... (Attempt {retries}/{max_retries})")
            time.sleep(WAIT_TIME)
        except IndexError as exc:
            print(
                f"Index Error occurred, possibly meaning that this API has no data for these params: {exc}"
            )
            print(f"param_dict: {param_dict}")
            return pd.DataFrame()
        except Exception as exc:
            raise exc
    raise ConnectionError(f"API call failed after {max_retries} retries.")
