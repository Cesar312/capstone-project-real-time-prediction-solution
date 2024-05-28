import os
import sys

from typing import List, Optional
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from src.paths import VISUALIZATIONS_DIR

def plot_one_sample(
        example_id: int,
        features: pd.DataFrame,
        targets: Optional[pd.Series] = None,
        predictions: Optional[pd.Series] = None,
        display_title: Optional[bool] = True
        ):
    
    '''
    Plot the time series of a single example.
    '''

    features_ = features.iloc[example_id]

    if targets is not None:
        target_ = targets.iloc[example_id]
    else:
        target_ = None

    ts_columns = [c for c in features.columns if c.startswith('rides_previous_')]
    ts_values = [features_[c] for c in ts_columns] + [target_]
    ts_dates = pd.date_range(
        features_['pickup_hour'] - timedelta(hours=len(ts_columns)),
        features_['pickup_hour'],
        freq='H'
    )

    # plot the time series
    title = f'Pickup Hour={features_["pickup_hour"]}, location_id={features_["pickup_location_id"]}' if display_title else None
    fig = px.line(
        x=ts_dates,
        y=ts_values,
        template='plotly_dark',     # default for this project
        # template='ggplot2',
        markers=True,
        title=title,
    )

    if targets is not None:
        # green dot for the target
        fig.add_scatter(
            x=[ts_dates[-1]],
            y=[target_],
            line_color='lime',
            mode='markers',
            marker_size=10,
            name='actual value'
        )
    
    if predictions is not None:
        # red X for the prediction
        prediction_ = predictions.iloc[example_id]
        fig.add_scatter(
            x=ts_dates[-1:],
            y=[prediction_],
            line_color='crimson',
            mode='markers',
            marker_symbol='x',
            marker_size=15,
            name='prediction'
        )

    return fig

def plot_timeseries(
        ts_data: pd.DataFrame,
        locations: Optional[List[int]] = None
        ):
    
    '''
    Plot the time-series data.
    '''

    ts_data_to_plot = ts_data[ts_data.pickup_location_id.isin(locations)] if locations else ts_data

    fig = px.line(
        ts_data,
        x='pickup_hour',
        y='rides',
        color='pickup_location_id',
        template='none',
        )
    
    fig.show()

def plot_actual_predicted_scatter(y_test, predictions, residuals, output_dir, title, file_name):
    
    '''
    Scatter plot of the actual vs predicted values
    '''

    absolute_residuals = np.abs(residuals)

    plt.figure(figsize=(10, 6)) 
    sns.scatterplot(x=y_test, y=predictions, hue=absolute_residuals, palette='coolwarm', alpha=0.6)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend().remove()  # Remove the legend

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / file_name)
    
    plt.show()
