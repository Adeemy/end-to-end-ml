"""
Model hyperparameter optimization utilities.
"""

from pathlib import PosixPath
from typing import Callable

import numpy as np
import optuna
import optuna_distributed
import pandas as pd
from dask.distributed import Client
from numpy.typing import ArrayLike
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.training.utils.experiment_tracker import ExperimentTracker
from src.utils.logger import get_console_logger

module_name: str = PosixPath(__file__).stem
logger = get_console_logger(module_name)


class ModelOptimizer:
    """A class to optimize model hyperparameters. It requires supplying preprocessed
    versions of the train and validation features to avoid fitting the whole pipeline
    in each objective function call during hyperparameters optimization.

    Attributes:
        tracker (ExperimentTracker): Experiment tracker for logging metrics and artifacts.
        train_features_preprocessed (pd.DataFrame): preprocessed train features.
        train_class (np.ndarray): train class labels.
        valid_features_preprocessed (pd.DataFrame): preprocessed validation features.
        valid_class (np.ndarray): validation class labels.
        n_features (int): number of features in the data.
        model (Callable): model object.
        fbeta_score_beta (float): beta value for fbeta score.
        encoded_pos_class_label (int): encoded positive class label.
        is_voting_ensemble (bool): whether the model is a voting ensemble or not.
        classifier_name (str): name of the classifier.
    """

    # List of supported models in this class
    # Note: this private variable shouldn't be mutated outside the class. It
    # should be updated when a new model is added to the class, which requires
    # adding its search space definition to generate_trial_params method.
    _supported_models = (
        "LogisticRegression",
        "RandomForestClassifier",
        "LGBMClassifier",
        "XGBClassifier",
    )

    def __init__(
        self,
        tracker: ExperimentTracker,
        train_features_preprocessed: pd.DataFrame,
        train_class: np.ndarray,
        valid_features_preprocessed: pd.DataFrame,
        valid_class: np.ndarray,
        n_features: int,
        model: Callable,
        search_space_params: dict,
        fbeta_score_beta: float = 1.0,
        encoded_pos_class_label: int = 1,
        is_voting_ensemble: bool = False,
    ) -> None:
        """Creates a ModelOptimizer instance.

        Args:
            tracker: Experiment tracker for logging.
            train_features_preprocessed: Preprocessed train features.
            train_class: Train class labels.
            valid_features_preprocessed: Preprocessed validation features.
            valid_class: Validation class labels.
            n_features: Number of features in the data.
            model: Model object.
            search_space_params: Hyperparameter search space.
            fbeta_score_beta: Beta value for fbeta score.
            encoded_pos_class_label: Encoded positive class label.
            is_voting_ensemble: Whether the model is a voting ensemble.

        Raises:
            AssertionError: if the specified model name is not supported.
        """
        self.tracker = tracker
        self.train_features_preprocessed = train_features_preprocessed
        self.train_class = train_class
        self.valid_features_preprocessed = valid_features_preprocessed
        self.valid_class = valid_class
        self.n_features = n_features
        self.model = model
        self.search_space_params = search_space_params
        self.fbeta_score_beta = fbeta_score_beta
        self.encoded_pos_class_label = encoded_pos_class_label
        self.is_voting_ensemble = is_voting_ensemble
        self.classifier_name = self.model.__class__.__name__

        if not self.is_voting_ensemble:
            assert (
                self.classifier_name in self._supported_models
            ), f"Supported models are: {self._supported_models}. Got {self.classifier_name}!"

    def generate_trial_params(self, trial: optuna.trial.Trial) -> dict:
        """Samples model parameters values from search space as specified in
        config file.

        Args:
            trial (optuna.trial.Trial): an optuna trial object.

        Returns:
            params (dict): a dictionary of model parameters and their values.
        """

        params = {}
        for param, values in self.search_space_params.items():
            if isinstance(values[0], list):
                params[param] = trial.suggest_categorical(param, values[0])
            elif isinstance(values[0], int):
                params[param] = trial.suggest_int(param, int(values[0]), int(values[1]))
            else:
                params[param] = trial.suggest_float(
                    param, float(values[0]), float(values[1]), log=bool(values[2])
                )

        return params

    def calc_perf_metrics(
        self,
        true_class: ArrayLike,
        pred_class: ArrayLike,
    ) -> pd.DataFrame:
        """Calculates different performance metrics for binary classification models.

        Args:
            true_class (ArrayLike): true class label.
            pred_class (ArrayLike): predicted class label not probability.

        Returns:
            performance_metrics (pd.DataFrame): a dataframe with metric name and score columns.
        """

        cal_metrics = [
            ("accuracy", accuracy_score(true_class, pred_class)),
            (
                "precision",
                precision_score(
                    true_class,
                    pred_class,
                ),
            ),
            ("recall", recall_score(true_class, pred_class)),
            ("f1", f1_score(true_class, pred_class)),
            (
                f"f_{self.fbeta_score_beta}_score",
                fbeta_score(
                    true_class,
                    pred_class,
                    beta=self.fbeta_score_beta,
                ),
            ),
            (
                "roc_auc",
                roc_auc_score(true_class, pred_class, average=None),
            ),
        ]

        performance_metrics = pd.DataFrame(cal_metrics, columns=["Metric", "Score"])

        return performance_metrics

    def objective_function(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        """Objective function that evaluates the provided hyperparameters for a
        specified model, where the search metric being optimized is fbeta score.
        A trial hyperparameters are sampled from the search space using generate_trial_params
        method and then the model is fitted on training set and evaluated on the
        validation set.

        Args:
            trial (optuna.trial.Trial): an optuna trial object.

        Returns:
            valid_score (float): validation score.
        """

        # Define parameters search space
        params = self.generate_trial_params(trial=trial)

        # Fit model and calculate training score
        self.model.set_params(**params)
        self.model.fit(self.train_features_preprocessed, self.train_class)

        # Evaluate model on training and validation set
        # Note: default threshold of 0.5 is used for positive class but
        # other htreshold values can be used, which is problem-dependent.
        pred_train_preds = self.model.predict(self.train_features_preprocessed)
        pred_valid_preds = self.model.predict(self.valid_features_preprocessed)
        train_scores = self.calc_perf_metrics(
            true_class=self.train_class,
            pred_class=pred_train_preds,
        )
        valid_scores = self.calc_perf_metrics(
            true_class=self.valid_class,
            pred_class=pred_valid_preds,
        )

        train_score = train_scores.loc[
            train_scores["Metric"] == f"f_{self.fbeta_score_beta}_score", "Score"
        ].iloc[0]
        valid_score = valid_scores.loc[
            valid_scores["Metric"] == f"f_{self.fbeta_score_beta}_score", "Score"
        ].iloc[0]

        self.tracker.log_metric(name="training_score", value=float(train_score))
        self.tracker.log_metric(name="validation_score", value=float(valid_score))

        # Return the validation score to ensure it's used for model selection
        return -valid_score

    def tune_model(
        self,
        max_search_iters: int = 100,
        model_opt_timeout_secs: int = 180,
    ) -> optuna.study.Study:
        """Performs hyperparameters optimization using Optuna package.

        Args:
            max_search_iters (int): maximum number of search iterations.
            model_opt_timeout_secs (int): maximum time in seconds to optimize model.

        Returns:
            study (optuna.study.Study): optuna study object.
        """

        # Turn off optuna log notes
        # Note: uncomment this during dev to see warnings.
        optuna.logging.set_verbosity(optuna.logging.WARN)

        # A callback function to output a log only when the best value is updated
        # Note: this callback may show incorrect values when optimizing an objective
        # function with n_jobs > 1
        def logging_callback(study, frozen_trial):
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            if previous_best_value != study.best_value:
                study.set_user_attr("previous_best_value", study.best_value)
                logger.info(
                    "\nTrial %d finished, best value: %d hyperparameters: %s.",
                    int(frozen_trial.number),
                    frozen_trial.value,
                    frozen_trial.params,
                )

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(0.1 * max_search_iters),
            warn_independent_sampling=False,
            multivariate=True,
            constant_liar=True,
        )
        study = optuna.create_study(sampler=sampler)

        logger.info(
            """\n
        ----------------------------------------------------------------
        --- Hyperparameter Optimization of %s Starts ...
        ----------------------------------------------------------------\n""",
            self.classifier_name,
        )

        study.optimize(
            func=self.objective_function,
            n_trials=max_search_iters,
            timeout=model_opt_timeout_secs,
            gc_after_trial=False,  # Set to True if memory consumption increases over several trials
            callbacks=[logging_callback],
        )

        return study

    def tune_model_in_parallel(
        self,
        max_search_iters: int = 100,
        n_parallel_jobs: int = 1,
        model_opt_timeout_secs: int = 180,
    ) -> optuna.study.Study:
        """Tunes model hyperparameters using Optuna package.

        Args:
            max_search_iters (int): maximum number of search iterations.
            n_parallel_jobs (int): number of parallel jobs.
            model_opt_timeout_secs (int): maximum time in seconds to optimize model.

        Returns:
            study (optuna.study.Study): optuna study object.
        """

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(0.1 * max_search_iters),
            warn_independent_sampling=False,
            multivariate=True,
            constant_liar=True,
        )
        client = Client()
        study = optuna_distributed.from_study(
            optuna.create_study(sampler=sampler), client=client
        )

        study.optimize(
            direction="minimize",
            func=self.objective_function,
            n_trials=max_search_iters,
            n_jobs=n_parallel_jobs,
            timeout=model_opt_timeout_secs,
        )

        # Shutdown Dask cluster
        client.shutdown()

        return study

    @staticmethod
    def create_pipeline(
        preprocessor_step: ColumnTransformer,
        selector_step: VarianceThreshold,
        model: Callable,
    ) -> Pipeline:
        """Creates a pipeline including data prep steps and fitted model.

        Args:
            preprocessor_step (ColumnTransformer): data preprocessing step.
            selector_step (VarianceThreshold): feature selection step.
            model (Callable): model object.

        Returns:
            pipeline (Pipeline): pipeline including data prep steps and fitted model.
        """

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor_step),
                ("selector", selector_step),
                ("classifier", model),
            ]
        )

        return pipeline

    def fit_pipeline(
        self,
        train_features: pd.DataFrame,
        preprocessor_step: ColumnTransformer,
        selector_step: VarianceThreshold,
        model: Callable,
    ) -> Pipeline:
        """Fits a pipeline including model with data preprocessing steps.

        Args:
            train_features (pd.DataFrame): train features.
            preprocessor_step (ColumnTransformer): data preprocessing step.
            selector_step (VarianceThreshold): feature selection step.
            model (Callable): model object.

        Returns:
            pipeline (Pipeline): fitted pipeline.
        """

        # Fit a pipeline
        pipeline = self.create_pipeline(
            preprocessor_step=preprocessor_step,
            selector_step=selector_step,
            model=model,
        )
        pipeline.fit(train_features, self.train_class)

        return pipeline
