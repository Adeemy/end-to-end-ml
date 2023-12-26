"""
Creates a feature store in local path. ML
platforms on Azure and AWS provides feature
store capability, but in this project feature
store is created in local path.
"""

import os
import sys
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, String

sys.path.append(str(Path(__file__).parent.resolve().parent.parent))

from feature_store.utils.config import Config

#################################
# Specify required column names by data type
config = Config(
    config_path=str(Path(__file__).parent.resolve().parent.parent.parent)
    + "/config/feature_store/config.yml"
)
PRIMARY_KEY = config.params["data"]["params"]["pk_col_name"]
CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]

# TTL duration
ttl_duration_in_days = timedelta(days=180)

# Specify path to features and target
feat_path_source = (
    os.path.abspath("..") + "/feature_repo/data/raw_dataset_features.parquet"
)
target_path_source = (
    os.path.abspath("..") + "/feature_repo/data/raw_dataset_target.parquet"
)

#################################
# Define an entity for encounters
patient = Entity(
    name="patient",
    join_keys=[PRIMARY_KEY],
    description="Patient ID",
)

# Define the source of training set features
feat_source = FileSource(
    path=feat_path_source,
    timestamp_field="event_timestamp",
)

# Define a view for training set features
_feat_view = FeatureView(
    name="features_view",
    ttl=ttl_duration_in_days,
    entities=[patient],
    source=feat_source,
    online=True,
    schema=[
        Field(name=PRIMARY_KEY, dtype=String),
        Field(name="BMI", dtype=Float32),
        Field(name="PhysHlth", dtype=Float32),
        Field(name="Age", dtype=String),
        Field(name="HighBP", dtype=String),
        Field(name="HighChol", dtype=String),
        Field(name="CholCheck", dtype=String),
        Field(name="Smoker", dtype=String),
        Field(name="Stroke", dtype=String),
        Field(name="HeartDiseaseorAttack", dtype=String),
        Field(name="PhysActivity", dtype=String),
        Field(name="Fruits", dtype=String),
        Field(name="Veggies", dtype=String),
        Field(name="HvyAlcoholConsump", dtype=String),
        Field(name="AnyHealthcare", dtype=String),
        Field(name="NoDocbcCost", dtype=String),
        Field(name="GenHlth", dtype=String),
        Field(name="MentHlth", dtype=String),
        Field(name="DiffWalk", dtype=String),
        Field(name="Sex", dtype=String),
        Field(name="Education", dtype=String),
        Field(name="Income", dtype=String),
    ],
    tags={"patient": "population_health"},
    description="Patient level features.",
)

#################################
# Define the source of training set target
target_source = FileSource(
    path=target_path_source,
    timestamp_field="event_timestamp",
)

# Define a view for training set target
_target_view = FeatureView(
    name="target_view",
    ttl=ttl_duration_in_days,
    entities=[patient],
    source=target_source,
    schema=[
        Field(name=PRIMARY_KEY, dtype=String),
        Field(name=CLASS_COL_NAME, dtype=String),
    ],
    online=True,
    tags={"patient": "population_health"},
    description="Patient level target value.",
)
