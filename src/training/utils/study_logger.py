"""
Study logging utilities for Optuna hyperparameter optimization studies.
"""

import re
from pathlib import PosixPath

import optuna
from comet_ml import Experiment

from src.training.utils.experiment import ExperimentManager
from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class StudyLogger:
    """Logs Optuna study results to Comet experiment.

    Single Responsibility: Handle logging of hyperparameter optimization studies.
    """

    @staticmethod
    def log_study_trials(
        experiment: Experiment,
        study: optuna.study.Study,
        classifier_name: str,
        artifacts_path: str,
        fbeta_score_beta: float,
    ) -> None:
        """Logs Optuna study results to Comet experiment.

        Args:
            experiment: Comet experiment object.
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

        ExperimentManager.log_asset(
            experiment=experiment,
            file_path=csv_path,
            file_name=f"study_{classifier_name}",
        )
        logger.info("Logged study trials for %s", classifier_name)
