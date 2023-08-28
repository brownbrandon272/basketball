import time
from requests.exceptions import ConnectionError, ReadTimeout
from urllib3.exceptions import ProtocolError

MAX_RETRIES = 3
WAIT_TIME = 5


def retry_api_call(api_call, max_retries=MAX_RETRIES):
    retries = 0
    while retries < max_retries:
        try:
            return api_call()
        except (ConnectionError, ProtocolError, ReadTimeout) as exc:
            print(f"Error occurred: {exc}")
            retries += 1
            print(f"Retrying... (Attempt {retries}/{max_retries})")
            time.sleep(WAIT_TIME)
    raise ConnectionError(f"API call failed after {max_retries} retries.")