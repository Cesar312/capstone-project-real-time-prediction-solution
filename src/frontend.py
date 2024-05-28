import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import pydeck as pdk
import requests
import streamlit as st
import zipfile

from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from src.inference import load_batch_of_features_from_feature_store
from src.inference import load_predictions_from_feature_store
from src.paths import DATA_DIR
from src.plot import plot_one_sample

st.set_page_config(layout="wide")

# Set the title of the web app
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor('H')
st.title(f'Taxi Demand Prediction')
st.header(f'{current_date} UTC')

progress_bar = st.sidebar.header(f'Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 3

def load_tlc_shape_data_file() -> gpd.GeoDataFrame:

    
    '''
    Add text here
    '''

    # Download the zip file
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    response = requests.get(URL)
    path = DATA_DIR / f'taxi_zones.zip'

    if response.status_code == 200:
        open(path, 'wb').write(response.content)
    else:
        raise Exception(f'Failed to download file from {URL}')
    
    # Unzip the file
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')

    # Load the shape file
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('EPSG:4326')

@st.cache_data
def _load_batch_of_features_from_feature_store(current_date: datetime) -> pd.DataFrame:

    '''
    Add text here
    '''

    return load_batch_of_features_from_feature_store(current_date)

@st.cache_data
def _load_predictions_from_feature_store(from_pickup_hour: datetime, to_pickup_hour: datetime) -> pd.DataFrame:

    '''
    Add text here
    '''

    return load_predictions_from_feature_store(from_pickup_hour, to_pickup_hour)

with st.spinner(text='Downloading shape file to plot taxi zones'):
    geo_df = load_tlc_shape_data_file()
    st.sidebar.write('Shape file downloaded successfully')
    progress_bar.progress(1/N_STEPS)

with st.spinner(text='Loading predictions model from feature store'):
    predictions_df = _load_predictions_from_feature_store(
        from_pickup_hour = current_date - timedelta(hours=1), 
        to_pickup_hour = current_date
    )
    st.sidebar.write('Model loaded successfully')
    progress_bar.progress(2/N_STEPS)

# 
next_hour_predictions_ready = False if predictions_df[predictions_df.pickup_hour == current_date].empty else True
previous_hour_predictions_ready = False if predictions_df[predictions_df.pickup_hour == (current_date - timedelta(hours=1))].empty else True

if next_hour_predictions_ready:
    #
    predictions_df = predictions_df[predictions_df.pickup_hour == current_date]

elif previous_hour_predictions_ready:
    #
    predictions_df = predictions_df[predictions_df.pickup_hour == (current_date - timedelta(hours=1))]
    current_date = current_date - timedelta(hours=1)
    st.subheader('The most recent data is not yet ready. Using last hour predictions')

else:
    raise Exception('Features are not ready for the last two hours. Confirm if feature pipeline is operational')

with st.spinner(text='Preparing data to plot'):

    def pseudocolor(value, min_value, max_value, startcolor, stopcolor):

        '''
        Add text here
        '''

        f = float(value - min_value) / (max_value - min_value)
        
        return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))
    
    df = pd.merge(geo_df,
                  predictions_df,
                  right_on='pickup_location_id', 
                  left_on='LocationID', 
                  how='inner')
    
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    progress_bar.progress(3/N_STEPS)

with st.spinner(text='Generate New York City map'):

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,
        longitude=-73.9712,
        zoom=10,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        'GeoJsonLayer',
        df,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color='fill_color',
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True
    )

    tooltip = {'html': '<b>Zone:</b> [{LocationID}]{zone} <br /><b>Predicted Rides:</b> {predicted_demand}'}

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)
    progress_bar.progress(4/N_STEPS)

with st.spinner(text='Fetch features from previous run'):
    features_df = _load_batch_of_features_from_feature_store(current_date)
    st.sidebar.write('Inference features loaded successfully')
    progress_bar.progress(5/N_STEPS)

with st.spinner(text='Plotting time-series data'):

    predictions_df = df
    row_indices = np.argsort(predictions_df['predicted_demand'].values)[::-1]
    n_to_plot = 10

    #
    for row_id in row_indices[:n_to_plot]:

        location_id = predictions_df['pickup_location_id'].iloc[row_id]
        loacation_name = predictions_df['zone'].iloc[row_id]
        st.header(f'Location ID: {location_id} - {loacation_name}')


        prediction = predictions_df['predicted_demand'].iloc[row_id]
        st.metric(label='Predicted Rides', value=int(prediction))

        # Plot predictions
        fig = plot_one_sample(
            example_id=row_id,
            features_df=features_df,
            targets=predictions_df['predicted_demand'],
            predictions=pd.Series(predictions_df['predicted_demand']),
            display_title=False
        )

        st.plotly_chart(fig, theme='streamlit', use_container_width=True, width=1000)

    progress_bar.progress(6/N_STEPS)