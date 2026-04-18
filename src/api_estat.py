import requests

BASE_URL = "http://api.e-stat.go.jp/rest/3.0/app/json"

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
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Error checking from API response
    result_info = data.get('GET_STATS_LIST', {}).get('RESULT', {})
    status_code = result_info.get('STATUS')
    
    if status_code == 101:
        # STATUS 101 means "Successfully finished, but no data found". 
        return []
    elif status_code != 0 and status_code is not None:
        error_msg = result_info.get('ERROR_MSG', 'Unknown APIs Error')
        raise ValueError(f"e-Stat API Error: {error_msg}")
    
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
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    result_info = data.get('GET_META_INFO', {}).get('RESULT', {})
    status_code = result_info.get('STATUS')
    
    if status_code != 0 and status_code is not None:
        error_msg = result_info.get('ERROR_MSG', 'Unknown APIs Error')
        raise ValueError(f"e-Stat Meta API Error: {error_msg}")
        
    return data

def get_stats_data(stats_data_id: str, app_id: str, filter_params: dict = None):
    """
    Retrieve actual statistical data for a specific table ID.
    filter_params allows narrowing down data via cdCat01, cdArea, cdTime etc.
    """
    if not app_id:
        raise ValueError("e-Stat Application ID is missing. Please set it in the Settings menu.")

    url = f"{BASE_URL}/getStatsData"
    params = {
        "appId": app_id,
        "statsDataId": stats_data_id,
        "metaGetFlg": "Y",
        "cntGetFlg": "N"
    }
    
    # Add optional filters (e.g., {'cdArea': '13000', 'cdTime': '2020'})
    if filter_params:
        params.update(filter_params)

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Error checking from API response
    result_info = data.get('GET_STATS_DATA', {}).get('RESULT', {})
    status_code = result_info.get('STATUS')
    
    if status_code == 101:
        # 101 means no data found
        return {}
    elif status_code != 0 and status_code is not None:
        error_msg = result_info.get('ERROR_MSG', 'Unknown APIs Error')
        raise ValueError(f"e-Stat API Error: {error_msg}")
        
    return data
