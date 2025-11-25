"""
Study logging utilities for Optuna hyperparameter optimization studies.

It defines the `StudyLogger` class responsible for logging Optuna study results
to experiment tracking backends like mlflow or Comet ML.
"""

import re
from pathlib import PosixPath

import optuna

from src.training.tracking.experiment_tracker import ExperimentTracker
from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class StudyLogger:
    """Logs Optuna study results to experiment tracking backends.

    Single Responsibility: Handle logging of hyperparameter optimization studies.
    """

    @staticmethod
    def log_study_trials(
        tracker: ExperimentTracker,
        study: optuna.study.Study,
        classifier_name: str,
        artifacts_path: str,
        fbeta_score_beta: float,
    ) -> None:
        """Logs Optuna study results to Comet experiment.

        Args:
            tracker: Experiment tracker object.
            study: Optuna study object.
            classifier_name: Name of the classifier.
            artifacts_path: Path to save study artifacts.
            fbeta_score_beta: Beta value for fbeta score.
        """
        study_results = study.trials_dataframe()
        study_results.rename(
            columns={"value": f"f_{fbeta_score_beta}_score"}, inplace=True
        )
        study_results.rename(columns=lambda x: re.sub("params_", "", x), inplace=True)

        csv_path = f"{artifacts_path}/study_{classifier_name}.csv"
        study_results.to_csv(csv_path, index=False)

        tracker.log_asset(
            file_path=csv_path,
            file_name=f"study_{classifier_name}",
        )
        logger.info("Logged study trials for %s", classifier_name)
