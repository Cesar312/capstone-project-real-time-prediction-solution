import os
import sys
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import hopsworks

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import src.config as config
from src.feature_store_api import get_feature_store, get_or_create_feature_view
from src.config import FEATURE_VIEW_METADATA

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:

    '''
    Get predictions from a trained model
    '''

    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)

    return results

def load_batch_of_features_from_feature_store(current_date: pd.Timestamp) -> pd.DataFrame:

    '''
    Load a batch of features from the feature store

    Args:
        current_date: pd.Timestamp: current date

    Returns:
        features: pd.DataFrame: DataFrame with 4 columns:
            - rides
            - pickup_hour
            - pickup_location_id
            - pickup_ts
    '''
    
    n_features = config.N_FEATURES

    feature_view = get_or_create_feature_view(FEATURE_VIEW_METADATA)

    # Fetch data from feature store
    fetch_data_from = current_date - timedelta(days=28)
    fetch_data_to = current_date - timedelta(hours=1)

    # Get the time-series data with +- margin
    ts_data = feature_view.get_batch_data(
        start_time = fetch_data_from - timedelta(days=1),
        end_time = fetch_data_to + timedelta(days=1)
    )

    # Filter the data
    pickup_ts_from = int(fetch_data_from.timestamp() * 1000)
    pickup_ts_to = int(fetch_data_to.timestamp() * 1000)
    ts_data = ts_data[ts_data.pickup_ts.between(pickup_ts_from, pickup_ts_to)]

    # Sort the data
    ts_data.sort_values(by=['pickup_location_id', 'pickup_hour'], inplace=True)

    # Check if the data is complete
    location_ids = ts_data.pickup_location_id.unique()
    assert len(ts_data) == config.N_FEATURES * len(location_ids), 'Time-series data is not complete. Verify the feature pipeline is operational.'

    # Transpose time-series data into feature vector for each location
    x = np.ndarray(shape=(len(location_ids), n_features), dtype=np.float32)
    for i, location_id in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_id, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i, :] = ts_data_i['rides'].values

    # Create a DataFrame with the features from numpy array
    features = pd.DataFrame(x, columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))])
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids
    features.sort_values(by=['pickup_location_id'], inplace=True)

    return features


def load_model_from_registry():

    ''''
    Add text
    '''

    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=config.MODEL_NAME,
        version=config.MODEL_VERSION
        )
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir, 'model.pkl'))

    return model

def load_predictions_from_feature_store(
        from_pickup_hour: datetime,
        to_pickup_hour: datetime
        ) -> pd.DataFrame:
    
    '''
    Load predictions from the feature store and retrieve model

    Args:
        from_pickup_hour: datetime: start date
        to_pickup_hour: datetime: end date

    Returns:
        predictions: pd.DataFrame: DataFrame with the following columns:
            - pickup_hour
            - pickup_location_id
            - predicted_demand
    '''

    from src.config import FEATURE_VIEW_PREDICTIONS_METADATA
    from src.feature_store_api import get_or_create_feature_view

    # Get pointer to the feature view
    predictions_feature_view = get_or_create_feature_view(FEATURE_VIEW_PREDICTIONS_METADATA)

    # Fetch predictions from the feature view
    print(f'Fetching predictions from Pick-up Hours between {from_pickup_hour} and {to_pickup_hour}')
    predictions = predictions_feature_view.get_batch_data(
        start_date=from_pickup_hour - timedelta(days=1),
        end_date=to_pickup_hour + timedelta(days=1)
    )

    # Verify datetimes are UTC format
    predictions['pickup_hour'] = pd.to_datetime(predictions['pickup_hour'], utc=True)
    from_pickup_hour = pd.to_datetime(from_pickup_hour, utc=True)
    to_pickup_hour = pd.to_datetime(to_pickup_hour, utc=True)

    # Filter the predictions
    predictions = predictions[predictions.pickup_hour.between(from_pickup_hour, to_pickup_hour)]

    # Sort the predictions
    predictions.sort_values(by=['pickup_hour', 'pickup_location_id'], inplace=True)

    return predictions