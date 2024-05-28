import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from dotenv import load_dotenv

from src.feature_store_api import FeatureGroupConfig, FeatureViewConfig
from src.paths import PARENT_DIR

load_dotenv(PARENT_DIR / '.env')

try:
    HOPSWORKS_PROJECT_NAME = os.environ['HOPSWORKS_PROJECT_NAME']
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']

except:
    raise Exception('Create .env file on the project root')

FEATURE_GROUP_NAME = 'timeseries_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name='timeseries_hourly_feature_group', 
    version=1,
    description='Timeseries data with hourly frequency', 
    primary_key=['pickup_location_id', 'pickup_ts'], 
    event_time='pickup_ts',
    online_enabled=True
)

FEATURE_VIEW_NAME = 'timeseries_hourly_feature_view'
FEATURE_VIEW_VERSION = 1
FEATURE_VIEW_METADATA = FeatureViewConfig(
    name='timeseries_hourly_feature_view',
    version=1,
    feature_group=FEATURE_GROUP_METADATA
)

MODEL_NAME = 'taxi_demand_prediction'
MODEL_VERSION = 1

FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group'
FEATURE_GROUP_MODEL_PREDICTIONS_VERSION = 1
FEATURE_GROUP_PREDICTIONS_METADATA = FeatureGroupConfig(
    name='model_predictions_feature_group',
    version=1,
    description='Predictions by production model',
    primary_key=['pickup_location_id', 'pickup_ts'],
    event_time='pickup_ts'
)

FEATURE_VIEW_MODEL_PREDICTIONS = 'model_predictions_feature_view'
FEATURE_VIEW_MODEL_VERSION = 1
FEATURE_VIEW_PREDICTIONS_METADATA = FeatureViewConfig(
    name='model_predictions_feature_view',
    version=1,
    feature_group=FEATURE_GROUP_PREDICTIONS_METADATA
)

MONITORING_FEATURE_VIEW_NAME = 'monitoring_feature_view'
MONITORING_FEATURE_VIEW_VERSION = 1

# Number of historical values to use for prediction
N_FEATURES = 24 * 28

# Number of trials by Optuna to find the best hyperparameters
N_HYPERPARAMETER_SEARCH_TRIALS = 10

# Maximum Mean Absolute Error allowed for the model
MAX_MAE = 30.0