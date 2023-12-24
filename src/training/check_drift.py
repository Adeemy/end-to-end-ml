# pylint: disable-all

"""
This script checks if there is data drift in prepared dataset
and model drift through monitoring the performance of champion
model.
"""

import os

# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import (
#     precision_score,
#     recall_score,
#     f1_score,
#     fbeta_score,
#     roc_auc_score,
# )
import subprocess

##########################
import sys
from datetime import datetime  # , timedelta

import pandas as pd
from pytz import timezone

##########################
champion_model_predictions_dataset_name = "champ_model_preds"
champion_predictions_dataset_version = "latest"
actual_class_col_name = "Diabetes_binary"
predicted_class_col_name = "PredictedClass"
predicted_score_col_name = "PredictedClassProb"
positive_class_label = "Diabetic"  # Minority class label
date_column_name = ""
primary_key_cols_names = ["ID"]
# train_test_sets_prep_script = 'split_data.py' # Use in dev when running code in vs code
model_train_script = "./submit_train.py"  # Use in dev when running code in vs code

# Data drift threshold value [0, 100] for email alerting and triggering retraining experiments
# Note: sometimes drift in the independent variables may not necessarily affect the model's
# performance because P(Y|X) didn't significantly change. Thus, there may not be a need to
# trigger training experiment because of that. However, if there singificant data drift
# (e.g., > 40), it may be approperiate to proactively trigger a training experiment in
# anticipation that P(Y|X) will change and the model performance will deteriorate in the near
# future. So make sure to set this value reasonably high enough to avoid triggering relatively
# expensive training experiment needlessly.
data_drift_threshold = 40

# Model drift threshold value [0, 100] for triggering retraining experiments
# Note: this is the value of the primary performance metric (primary_perf_metric) like
# sensitivity or f1-score that champion model performance shouldn't deteriorate below on specific
# period of time (e.g., per week). To ensure this value is calculated based on proper sample size
# and to prevent triggering retraining experiments based on model drift value on small sample size,
# the model_drfit_min_sample_size must be set to a reasonable number like depending on business requirement.
primary_perf_metric = (
    "precision"  # Can be "recall_score_binary", "Precision", "F1-Score", or "AUC Score"
)
secondary_perf_metric = "recall"
model_drift_threshold = 0.60  # Minimum value established based on business requirements
model_drfit_min_sample_size = 100

############
# Pass variable in runtime to pipeline to trigger training job
print("##vso[task.setvariable variable=DataModelDriftChecked;isOutput=true;]True")
