import pandas as pd

def parse_estat_json_to_dataframe(json_data):
    """
    Parses e-Stat API JSON response into a flattened Pandas DataFrame.
    Maps internal @id codes (like cat01) to actual Japanese @name (like 年齢).
    Safely handles masked data ('-', '***', 'X').
    """
    try:
        # Extract metadata
        class_objects = json_data['GET_STATS_DATA']['STATISTICAL_DATA']['CLASS_INF']['CLASS_OBJ']
        
        # Create mappings
        dimension_maps = {}      # internal dim_code -> name dictionary
        column_name_maps = {}    # @id ('cat01') -> @name ('年齢')
        
        for obj in class_objects:
            obj_id = obj['@id']
            obj_name = obj.get('@name', obj_id) # Use id as fallback
            column_name_maps[obj_id] = obj_name
            
            classes = obj.get('CLASS', [])
            if isinstance(classes, dict):
                classes = [classes]
            
            mapping = {cls['@code']: cls.get('@name', cls['@code']) for cls in classes}
            dimension_maps[obj_id] = mapping

        # Extract actual data values
        raw_values = json_data['GET_STATS_DATA']['STATISTICAL_DATA']['DATA_INF']['VALUE']
        if isinstance(raw_values, dict):
            raw_values = [raw_values]
            
        # Build flattened records
        records = []
        for val in raw_values:
            record = {}
            
            # The numerical value itself. Handle typical e-Stat masks like '-', '...', 'X'
            raw_val_str = str(val.get('$', '')).strip()
            try:
                record['value'] = float(raw_val_str)
            except ValueError:
                record['value'] = None  # Missing/Masked data
                
            # units
            record['unit'] = val.get('@unit', '')
            
            # Map dimensions (like @tab, @cat01, @time) to Japanese column names
            for key, code in val.items():
                if key.startswith('@') and key not in ['@unit']:
                    dim_id = key[1:] # e.g. '@time' -> 'time'
                    if dim_id in dimension_maps:
                        japanese_col_name = column_name_maps.get(dim_id, dim_id)
                        record[japanese_col_name] = dimension_maps[dim_id].get(code, code)
                        
            records.append(record)
            
        df = pd.DataFrame(records)
        return df
        
    except Exception as e:
        print(f"Error parsing e-Stat data: {e}")
        return pd.DataFrame() # Return empty on failure
