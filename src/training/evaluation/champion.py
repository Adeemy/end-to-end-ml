"""
Champion model management utilities. It selects, calibrates, logs, and registers the
best performing model from candidate models.

The best candidate is selected based on validation performance using experiment trackers.
The selected model is then calibrated using validation data and registered as the champion
model in the experiment tracking backend after evaluating it on the test set to ensure it
meets deployment criteria.
"""

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.training.tracking.experiment_tracker import ExperimentTracker


class ModelChampionManager:
    """Manages all champion model related activities, such as selecting the best
    performer from candidate models, calibrating the best model, and registering
    it in the workspace.

    Attributes:
        champ_model_name (str): name of the champion model.
        tracker (ExperimentTracker): Experiment tracker for model registration.
    """

    def __init__(
        self, champ_model_name: str, tracker: Optional[ExperimentTracker] = None
    ) -> None:
        """Initialize ModelChampionManager.

        Args:
            champ_model_name: Name of the champion model.
            tracker: Optional experiment tracker for model registration.
        """
        self.champ_model_name = champ_model_name
        self.tracker = tracker

    def select_best_performer(
        self,
        trackers: dict[str, ExperimentTracker],
        comparison_metric: str,
    ) -> str:
        """Selects the best performer from all challenger models using their trackers.

        Args:
            trackers: Dictionary mapping model names to their experiment trackers.
            comparison_metric: Metric name to compare models.

        Returns:
            best_challenger_name: Name of the best challenger model.
        """

        exp_scores = {}
        for model_name, tracker in trackers.items():
            metric_value = tracker.get_metric(comparison_metric)
            if metric_value is not None:
                exp_scores[model_name] = metric_value

        if not exp_scores:
            raise ValueError(
                f"No valid scores found for metric '{comparison_metric}' across models."
            )

        # Select the best performer (highest score)
        best_challenger_name = max(exp_scores, key=exp_scores.get)

        return best_challenger_name

    @staticmethod
    def calibrate_pipeline(
        valid_features: pd.DataFrame,
        valid_class: np.ndarray,
        fitted_pipeline: Pipeline,
        cv_folds: int = 5,
    ) -> Pipeline:
        """Takes a fitted pipeline and returns a calibrated pipeline.

        Args:
            valid_features (np.ndarray): Validation features.
            valid_class (np.ndarray): Validation class labels.
            fitted_pipeline (Pipeline): Fitted pipeline on the training set.
            cv_folds (int): Number of cross-validation folds for calibration.

        Returns:
            calib_pipeline (Pipeline): Calibrated pipeline.

        Raises:
            ValueError: If the classifier in the fitted pipeline is not fitted.
        """

        # Extract preprocessor, selector, and classifier from the fitted pipeline
        preprocessor = fitted_pipeline.named_steps.get("preprocessor")
        selector = fitted_pipeline.named_steps.get("selector")
        model = fitted_pipeline.named_steps.get("classifier")

        if not hasattr(model, "classes_"):
            raise ValueError("The classifier in the fitted pipeline is not fitted.")

        # Transform the validation set using the preprocessor and selector steps
        valid_features_transformed = fitted_pipeline.named_steps[
            "preprocessor"
        ].transform(valid_features)
        valid_features_transformed = fitted_pipeline.named_steps["selector"].transform(
            valid_features_transformed
        )

        # Calibrate the fitted model using the transformed validation set
        calibrator = CalibratedClassifierCV(
            estimator=model,
            method=("isotonic" if len(valid_class) > 1000 else "sigmoid"),
            cv=cv_folds,
        )
        calibrator.fit(valid_features_transformed, valid_class)

        # Create a new pipeline with the calibrated classifier
        calib_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("selector", selector),
                ("classifier", calibrator),
            ]
        )

        return calib_pipeline

    def log_and_register_champ_model(
        self,
        local_path: str,
        pipeline: Pipeline,
    ) -> None:
        """Logs and registers champion model using the experiment tracker.

        Args:
            local_path: Local path to save champion model.
            pipeline: Fitted pipeline.

        Raises:
            ValueError: If no tracker is configured.
        """

        if self.tracker is None:
            raise ValueError(
                "No experiment tracker configured. Cannot log or register model."
            )

        if not os.path.exists(local_path):
            os.makedirs(local_path)

        model_path = f"{local_path}/{self.champ_model_name}.pkl"
        joblib.dump(pipeline, model_path)

        self.tracker.log_model(
            name=self.champ_model_name,
            file_or_folder=model_path,
            overwrite=False,
        )
        self.tracker.register_model(model_name=self.champ_model_name)
        self.tracker.end()
