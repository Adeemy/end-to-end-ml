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

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, String

sys.path.append(str(Path(__file__).parent.resolve().parent.parent.parent))

from src.feature_store.utils.config import Config

#################################
# Specify required column names by data type
config = Config(
    config_path=str(Path(__file__).parent.resolve().parent.parent.parent)
    + "/config/feature_store/config.yml"
)
PRIMARY_KEY = config.params["data"]["params"]["pk_col_name"]
CLASS_COL_NAME = config.params["data"]["params"]["class_col_name"]
EVENT_TIMESTAMP_COL_NAME = config.params["data"]["params"]["event_timestamp_col_name"]

# TTL duration
ttl_duration_in_days = timedelta(days=180)

# Specify path to features and target
feat_path_source = (
    f"{os.path.abspath('..')}/feature_repo/data/preprocessed_dataset_features.parquet"
)
target_path_source = (
    f"{os.path.abspath('..')}/feature_repo/data/preprocessed_dataset_target.parquet"
)

#################################
# Define an entity for encounters
patient = Entity(
    name="patient",
    join_keys=[PRIMARY_KEY],
    value_type=ValueType.STRING,
    description="Patient ID",
)

# Define the source of training set features
feat_source = FileSource(
    path=feat_path_source,
    timestamp_field=EVENT_TIMESTAMP_COL_NAME,
)

# Define a view for training set features
_feat_view = FeatureView(
    name="features_view",
    ttl=ttl_duration_in_days,
    entities=[patient],
    source=feat_source,
    online=True,
    schema=[
        Field(
            name=PRIMARY_KEY, dtype=String, description="Uniquely identifies each row."
        ),
        Field(name="BMI", dtype=Float32, description="Patient Body Mass Index."),
        Field(
            name="PhysHlth",
            dtype=Float32,
            description="Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days.",
        ),
        Field(
            name="Age",
            dtype=String,
            description="13-level age category (_AGEG5YR see codebook) 1 = 18-24 9 = 60-64 13 = 80 or older.",
        ),
        Field(name="HighBP", dtype=String, description="High blood pressure (Yes/No)."),
        Field(name="HighChol", dtype=String, description="High cholestrol (Yes/No)."),
        Field(
            name="CholCheck",
            dtype=String,
            description="cholesterol check in 5 years (Yes/No).",
        ),
        Field(
            name="Smoker",
            dtype=String,
            description="Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] (Yes/No).",
        ),
        Field(
            name="Stroke",
            dtype=String,
            description="(Ever told) you had a stroke (Yes/No).",
        ),
        Field(
            name="HeartDiseaseorAttack",
            dtype=String,
            description="Coronary heart disease (CHD) or myocardial infarction (MI) (Yes/No).",
        ),
        Field(
            name="PhysActivity",
            dtype=String,
            description="Physical activity in past 30 days - not including job (Yes/No)",
        ),
        Field(
            name="Fruits",
            dtype=String,
            description="Consume Fruit 1 or more times per day (Yes/No)",
        ),
        Field(
            name="Veggies",
            dtype=String,
            description="Consume vegetables 1 or more times per day (Yes/No).",
        ),
        Field(
            name="HvyAlcoholConsump",
            dtype=String,
            description="Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) (Yes/No).",
        ),
        Field(
            name="AnyHealthcare",
            dtype=String,
            description="Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. (Yes/No).",
        ),
        Field(
            name="NoDocbcCost",
            dtype=String,
            description="Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? (Yes/No).",
        ),
        Field(
            name="GenHlth",
            dtype=String,
            description="Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor.",
        ),
        Field(
            name="MentHlth",
            dtype=String,
            description="Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days.",
        ),
        Field(
            name="DiffWalk",
            dtype=String,
            description="Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days.",
        ),
        Field(name="Sex", dtype=String, description="0 = female 1 = male."),
        Field(
            name="Education",
            dtype=String,
            description="Education level (EDUCA see codebook) scale 1-6 1 = Never attended school or only kindergarten 2 = Grades 1 through 8 (Elementary) 3 = Grades 9 through 11 (Some high school) 4 = Grade 12 or GED (High school graduate) 5 = College 1 year to 3 years (Some college or technical school) 6 = College 4 years or more (College graduate).",
        ),
        Field(
            name="Income",
            dtype=String,
            description="Income scale (INCOME2 see codebook) scale 1-8 1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more.",
        ),
    ],
    tags={"patient": "population_health"},
    description="Patient level features.",
)

#################################
# Define the source of training set target
target_source = FileSource(
    path=target_path_source,
    timestamp_field=EVENT_TIMESTAMP_COL_NAME,
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
