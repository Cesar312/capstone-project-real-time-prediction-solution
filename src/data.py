from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from pdb import set_trace as stop

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR

def download_one_file_of_raw_data(year: int, month: int) -> Path:
    
    '''
    Downloads PARQUET file with histroical taxi rides for specified year and month
    '''

    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        open(path, 'wb').write(response.content)
        return path
    else:
        raise Exception(f'Failed to download the file {year}-{month:02d}')

def validate_raw_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    
    '''
    Remove records outside of date range
    '''
    
    current_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= current_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]

    return rides

def load_raw_data(year: int, months: Optional[List[int]] = None) -> pd.DataFrame:

    '''
    Loads raw data from local storage or downloads it from the cloud into a DataFrame

    Args: 
    year: int - year to load data for
    months: List[int] - list of months to load data for. If None, all months are loaded

    Returns:
    pd.DataFrame - DataFrame with raw data
        - pickup_datetime: datetime - pickup time
        - pickup_location_id: int - pickup location ID
    '''

    rides = pd.DataFrame()

    if months is None:
        # Download all months
        months = list(range(1, 13))
    elif isinstance(months, int):
        # Download specified month
        months = [months]

    for month in months:
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                # Download the file
                print(f'Downloading the file {year}-{month:02d}')
                download_one_file_of_raw_data(year, month)
            except:
                print(f'Failed to download the file {year}-{month:02d}')
                continue
        else:
            print(f'{year}-{month:02d} file is already in local storage')

        # Load the file to DataFrame
        rides_one_month = pd.read_parquet(local_file)

        # Rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        rides_one_month.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime', 
            'PULocationID': 'pickup_location_id'}, 
            inplace=True)
        
        # Validate the data
        rides_one_month = validate_raw_data(rides_one_month, year, month)

        # Append to the main DataFrame
        rides = pd.concat([rides, rides_one_month])

    if rides.empty:
        # If no data was loaded, return an empty DataFrame
        return pd.DataFrame()
    else:
        # Keep only the columns we need
        rides = rides[['pickup_datetime', 'pickup_location_id']]
        return rides
    
def add_missing_slots(ts_data: pd.DataFrame) -> pd.DataFrame:
    
    '''
    Add missing slots to the time series data and fill them with 0
    '''

    location_ids = range(1, ts_data['pickup_location_id'].max() + 1)
    
    full_range = pd.date_range(ts_data['pickup_hour'].min(), ts_data['pickup_hour'].max(), freq='H')
    output = pd.DataFrame()
    for location_id in tqdm(location_ids):
        
        # Filter the rides for the current location
        ts_data_i = ts_data.loc[ts_data['pickup_location_id'] == location_id, ['pickup_hour', 'rides']]

        if ts_data_i.empty:
            #
            ts_data_i = pd.DataFrame.from_dict([{'pickup_hour': ts_data['pickup_hour'].max(), 'rides': 0}])

        # Add missing slots and fill with 0 
        # stackoverflow.com/questions/a/19324591
        ts_data_i.set_index('pickup_hour', inplace=True)
        ts_data_i.index = pd.DatetimeIndex(ts_data_i.index)
        ts_data_i = ts_data_i.reindex(full_range, fill_value=0)

        # Reset the index and add the location_id
        ts_data_i['pickup_location_id'] = location_id

        output = pd.concat([output, ts_data_i])

    # Reset the index and rename the columns
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})

    return output

def fetch_ride_events_from_dw(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    
    '''
    Fetch ride events from the data warehouse by sampling from 52 weeks ago
    '''

    from_date_ = from_date - timedelta(days=7*52)
    to_date_ = to_date - timedelta(days=7*52)
    print(f'Fetching data from {from_date} to {to_date}')

    if (from_date_.year == to_date_.year) and (from_date_.month == to_date_.month):
        # If the range is within one month, load the data for that month
        rides = load_raw_data(year=from_date_.year, months=from_date_.month)
        rides = rides[rides.pickup_datetime >= from_date_]
        rides = rides[rides.pickup_datetime < to_date_]

    else:
        # If the range spans two months, load the data for both months and concatenate
        rides = load_raw_data(year=from_date_.year, months=from_date_.month)
        rides = rides[rides.pickup_datetime >= from_date_]
        rides_2 = load_raw_data(year=to_date_.year, months=to_date_.month)
        rides_2 = rides_2[rides_2.pickup_datetime < to_date_]
        rides = pd.concat([rides, rides_2])

    # 
    #
    rides['pickup_datetime'] += timedelta(days=7*52)

    rides.sort_values(by=['pickup_location_id', 'pickup_datetime'], inplace=True)

    return rides

def transform_raw_data_into_timeseries_data(rides: pd.DataFrame) -> pd.DataFrame:
    
    '''
    Transform raw data into time series data
    '''

    # Aggregate the data
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    agg_rides = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0: 'rides'}, inplace=True)

    # Add missing slots
    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots

def get_cutoff_indices_features_targets(data: pd.DataFrame, input_sequence_length: int, step_size: int) -> list:
    
    '''
    Get the indices to split the data into features and targets
    '''

    stop_position = len(data) -1

    #
    subsequent_first_index = 0
    subsequent_middle_index = input_sequence_length
    subsequent_last_index = input_sequence_length + 1
    indices = []

    while subsequent_last_index < stop_position:
        indices.append((subsequent_first_index, subsequent_middle_index, subsequent_last_index))
        subsequent_first_index += step_size
        subsequent_middle_index += step_size
        subsequent_last_index += step_size

    return indices 

def transform_timeseries_data_into_features_targets(
        ts_data: pd.DataFrame, 
        input_sequence_length: int, 
        step_size: int
        ) -> Tuple[pd.DataFrame, pd.Series]:
    
    '''
    Transform time series data into features and targets to train ML models
    '''

    assert set(ts_data.columns) == {'pickup_hour', 'rides', 'pickup_location_id'}

    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame()
    targets = pd.DataFrame()

    for location_id in tqdm(location_ids):

        # 
        ts_data_one_location = ts_data.loc[ts_data.pickup_location_id == location_id, ['pickup_hour', 'rides']].sort_values(by='pickup_hour')

        # 
        indices = get_cutoff_indices_features_targets(ts_data_one_location, input_sequence_length, step_size)

        # 
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples, input_sequence_length), dtype=np.float32)
        y = np.ndarray(shape=(n_examples), dtype=np.float32)
        pickup_hours = []

        for i, idx in enumerate(indices):
            x[i, :] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values[0]
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])

        #
        features_one_location = pd.DataFrame(x, columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_sequence_length))])
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id

        # 
        targets_one_location = pd.DataFrame(y, columns=[f'target_rides_next_hour'])

        # 
        features = pd.concat([features, features_one_location])
        targets = pd.concat([targets, targets_one_location])

    features.reset_index(drop=True, inplace=True)
    targets.reset_index(drop=True, inplace=True)

    return features, targets['target_rides_next_hour']