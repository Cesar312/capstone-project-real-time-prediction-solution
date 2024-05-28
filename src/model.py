import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline

import lightgbm as lgb

def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:

    '''
    Add columns with average rides from 7, 14, 21 and 28 days ago
    '''

    X['average_rides_last_4_weeks'] = 0.25*(X[f'rides_previous_{7*24}_hour'] + \
                                            X[f'rides_previous_{14*24}_hour'] + \
                                            X[f'rides_previous_{21*24}_hour'] + \
                                            X[f'rides_previous_{28*24}_hour'])

    return X

class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):

    '''
    Add temporal features to the dataset: hour of the day and day of the week
    Drop the pickup_hour column
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X_ = X.copy()

        # Add hour and day of week
        X_['hour'] = X_['pickup_hour'].dt.hour
        X_['day_of_week'] = X_['pickup_hour'].dt.dayofweek

        return X_.drop(columns=['pickup_hour'])
    
def get_pipeline(**hyperparameters) -> Pipeline:

    '''
    Create a pipeline with the following steps:
    - Add feature average rides from the last 4 weeks
    - Add temporal features
    - Train a LightGBM model with the hyperparameters passed as arguments
    '''

    add_feature_average_rides_last_4_weeks = FunctionTransformer(average_rides_last_4_weeks, validate=False)
    add_temporal_features = TemporalFeaturesEngineer()

    return make_pipeline(add_feature_average_rides_last_4_weeks, add_temporal_features, lgb.LGBMRegressor(**hyperparameters))