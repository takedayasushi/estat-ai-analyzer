import time

import requests

# e-Stat API must be called over HTTPS: the appId is a credential and must
# never travel over plain HTTP.
BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json"

REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
# e-Stat allows up to 100,000 records per getStatsData request; larger
# results are continued with startPosition / NEXT_KEY pagination.
PAGE_LIMIT = 100000
MAX_PAGES = 100  # safety cap against runaway pagination loops


def _get_json(url: str, params: dict) -> dict:
    """GET JSON with a timeout and retry (exponential backoff) on
    transient network errors and 5xx responses."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status is None or status < 500:
                raise
            last_exc = e
        if attempt < MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
    raise last_exc


def _check_result(data: dict, root_key: str):
    """Validate the RESULT block shared by all e-Stat endpoints.
    Returns the RESULT dict; raises ValueError on API errors."""
    result_info = data.get(root_key, {}).get('RESULT', {})
    status_code = result_info.get('STATUS')
    if status_code == 101:
        # STATUS 101 means "Successfully finished, but no data found".
        return result_info
    if status_code != 0 and status_code is not None:
        error_msg = result_info.get('ERROR_MSG', 'Unknown APIs Error')
        raise ValueError(f"e-Stat API Error: {error_msg}")
    return result_info


def search_stats_list(stats_field: str, app_id: str, search_word: str = ""):
    """
    Search for statistical tables based on a statsField (Category) and an optional searchWord.
    """
    if not app_id:
        raise ValueError("e-Stat Application ID is missing. Please set it in the Settings menu.")

    url = f"{BASE_URL}/getStatsList"
    params = {
        "appId": app_id,
        "statsField": stats_field,
    }
    if search_word.strip():
        params["searchWord"] = search_word.strip()
    data = _get_json(url, params)

    result_info = _check_result(data, 'GET_STATS_LIST')
    if result_info.get('STATUS') == 101:
        return []

    # Safely extract list of tables
    try:
        tables = data['GET_STATS_LIST']['DATALIST_INF']['TABLE_INF']
        # If there's only one result, API might return it as a dict instead of list
        if isinstance(tables, dict):
            tables = [tables]
        return tables
    except KeyError:
        return []


def get_meta_info(stats_data_id: str, app_id: str):
    """
    Retrieve only the MetaData (schema/dimensions) for a statistical table without downloading the data.
    """
    if not app_id:
        raise ValueError("e-Stat Application ID is missing. Please set it in the Settings menu.")

    url = f"{BASE_URL}/getMetaInfo"
    params = {
        "appId": app_id,
        "statsDataId": stats_data_id,
    }
    data = _get_json(url, params)
    _check_result(data, 'GET_META_INFO')
    return data


def get_stats_data(stats_data_id: str, app_id: str, filter_params: dict = None):
    """
    Retrieve actual statistical data for a specific table ID.
    filter_params allows narrowing down data via cdCat01, cdArea, cdTime etc.

    Follows NEXT_KEY pagination so results larger than one page
    (100,000 records) are fetched in full instead of being silently
    truncated; the VALUE lists of all pages are merged into the first
    page's response structure.
    """
    if not app_id:
        raise ValueError("e-Stat Application ID is missing. Please set it in the Settings menu.")

    url = f"{BASE_URL}/getStatsData"
    params = {
        "appId": app_id,
        "statsDataId": stats_data_id,
        "metaGetFlg": "Y",
        "cntGetFlg": "N",
        "limit": PAGE_LIMIT,
    }

    # Add optional filters (e.g., {'cdArea': '13000', 'cdTime': '2020'})
    if filter_params:
        params.update(filter_params)

    merged = None
    merged_values = []
    start_position = None

    for _ in range(MAX_PAGES):
        page_params = dict(params)
        if start_position is not None:
            page_params["startPosition"] = start_position
        data = _get_json(url, page_params)

        result_info = _check_result(data, 'GET_STATS_DATA')
        if result_info.get('STATUS') == 101:
            # 101 means no data found
            return merged if merged is not None else {}

        if merged is None:
            merged = data
        try:
            values = data['GET_STATS_DATA']['STATISTICAL_DATA']['DATA_INF']['VALUE']
            if isinstance(values, dict):
                values = [values]
            merged_values.extend(values)
        except KeyError:
            pass

        # RESULT_INF lives under STATISTICAL_DATA in the getStatsData response.
        stat_data = data.get('GET_STATS_DATA', {}).get('STATISTICAL_DATA', {})
        next_key = stat_data.get('RESULT_INF', {}).get('NEXT_KEY')
        if not next_key:
            break
        start_position = next_key
    else:
        raise ValueError("e-Stat API Error: pagination did not finish within the safety limit.")

    if merged is not None:
        try:
            merged['GET_STATS_DATA']['STATISTICAL_DATA']['DATA_INF']['VALUE'] = merged_values
        except KeyError:
            pass
    return merged if merged is not None else {}
