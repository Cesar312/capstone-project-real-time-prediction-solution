import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from typing import List, Optional
from dataclasses import dataclass

import hsfs
import hopsworks

import src.config as config
from src.logger import get_logger

logger = get_logger()

@dataclass
class FeatureGroupConfig:
    name: str
    version: int
    description: str
    primary_key: List[str]
    event_time: str
    online_enabled: Optional[bool] = False

@dataclass
class FeatureViewConfig:
    name: str
    version: int
    feature_group: FeatureGroupConfig

def get_feature_store() -> hsfs.feature_store.FeatureStore:

    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )

    return project.get_feature_store()

def get_feature_group(name: str, version: Optional[int] = 1) -> hsfs.feature_group.FeatureGroup:
    
    '''
    Connects to the feature store and retrieves pointer to a feature group

    Args:
        name: str: name of the feature group
        version: int: version of the feature group

    Returns:
        feature_group: hsfs.feature_group.FeatureGroup: pointer to the feature group
    '''

    return get_feature_store().get_feature_group(name=name, version=version)

def get_or_create_feature_group(
        feature_group_metadata: FeatureGroupConfig
        ) -> hsfs.feature_group.FeatureGroup:
    
    '''
    Connect to the feature store and retrieves pointer to a feature group. If the feature group does not exist, create it.

    Args:
        feature_group_metadata: FeatureGroupConfig: metadata for the feature group

    Returns:
        feature_group: hsfs.feature_group.FeatureGroup: pointer to the feature group
    '''

    return get_feature_store().get_or_create_feature_group(
        name=feature_group_metadata.name,
        version=feature_group_metadata.version,
        description=feature_group_metadata.description,
        primary_key=feature_group_metadata.primary_key,
        event_time=feature_group_metadata.event_time,
        online_enabled=feature_group_metadata.online_enabled
    )

def get_or_create_feature_view(
        feature_view_metadata: FeatureViewConfig
        ) -> hsfs.feature_view.FeatureView:
    
    '''
    Connect to the feature store and retrieves pointer to a feature view. If the feature view does not exist, create it.

    Args:
        feature_view_metadata: FeatureViewConfig: metadata for the feature view

    Returns:
        feature_view: hsfs.feature_view.FeatureView: pointer to the feature view
    '''

    # Get pointer to the feature store
    feature_store = get_feature_store()

    # Get pointer to the feature group
    feature_group = feature_store.get_feature_group(
        name=feature_view_metadata.feature_group.name,
        version=feature_view_metadata.feature_group.version
    )

    # Create the feature view
    try:
        feature_store.create_feature_view(
            name=feature_view_metadata.name,
            version=feature_view_metadata.version,
            features=feature_group.primary_key,
            query=feature_group.select_all()
        )
    except:
        logger.info(f"Feature view {feature_view_metadata.name} already exists")

    # 
    feature_store = get_feature_store()
    feature_view = feature_store.get_feature_view(
        name=feature_view_metadata.name,
        version=feature_view_metadata.version
    )

    return feature_view