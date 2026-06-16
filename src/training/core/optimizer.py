"""
Model hyperparameter optimization utilities. It defines the ModelOptimizer class
that encapsulates the logic for tuning model hyperparameters using Optuna. The class
requires preprocessed training and validation features to avoid redundant pipeline
fitting during optimization.

Classes:
    ModelOptimizer: A class to optimize model hyperparameters using Optuna.

Design Decision:
    The ModelOptimizer class is designed to require preprocessed training and validation
    features. This design choice is made to enhance efficiency during hyperparameter
    optimization, as it avoids the overhead of fitting the entire data preprocessing
    pipeline in each trial.

    The optimization process leverages Optuna's TPE sampler with multivariate sampling
    and constant liar strategy to effectively explore the hyperparameter space, especially
    in parallel execution scenarios. This approach helps mitigate redundant sampling and
    promotes a more diverse exploration of hyperparameters.
"""

from pathlib import PosixPath
from typing import Callable, Optional

import numpy as np
import optuna
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.training.evaluation import metrics
from src.training.schemas import SupportedModelsConfig
from src.training.tracking.experiment_tracker import ExperimentTracker
from src.utils.logger import get_logger

module_name: str = PosixPath(__file__).stem
logger = get_logger(module_name)

# Metrics for which a *lower* value is better, used to derive the Optuna study
# direction so a lower-is-better metric (e.g. log_loss) is not optimized
# backwards. Sourced from the shared metrics module (single source of truth).
LOWER_IS_BETTER_METRICS: frozenset = metrics.LOWER_IS_BETTER_METRICS


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
        optimization_metric (str): metric the search optimizes (config comparison_metric).
        random_seed (Optional[int]): seed for the sampler to make the search reproducible.
        classifier_name (str): name of the classifier.
    """

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
        supported_models: SupportedModelsConfig,
        registered_model_name: str,
        fbeta_score_beta: float = 1.0,
        encoded_pos_class_label: int = 1,
        is_voting_ensemble: bool = False,
        optimization_metric: str = "fbeta_score",
        random_seed: Optional[int] = None,
        task_type: str = "binary",
        cv_folds: int = 1,
    ) -> None:
        """Creates a ModelOptimizer instance.

        Args:
            tracker: Experiment tracker for logging.
            train_features_preprocessed: Preprocessed train features.
            train_class: Train class labels.
            valid_features_preprocessed: Preprocessed validation features.
            valid_class: Validation class labels.
            n_features: Number of features in the data.
            model: Model instance.
            search_space_params: Hyperparameter search space.
            supported_models: SupportedModelsConfig instance.
            registered_model_name: Registry name for this model (e.g., 'logistic-regression').
            fbeta_score_beta: Beta value for fbeta score.
            encoded_pos_class_label: Encoded positive class label.
            is_voting_ensemble: Whether the model is a voting ensemble.
            optimization_metric: Metric the search optimizes. Mirrors the config
                ``comparison_metric`` so tuning and champion selection agree.
            random_seed: Seed passed to the Optuna sampler for reproducible searches.

        Raises:
            ValueError: if the specified model name is not supported.
        """
        self.tracker = tracker
        self.train_features_preprocessed = train_features_preprocessed
        self.train_class = train_class
        self.valid_features_preprocessed = valid_features_preprocessed
        self.valid_class = valid_class
        self.n_features = n_features
        self.model = model
        self.search_space_params = search_space_params
        self.supported_models = supported_models
        self.registered_model_name = registered_model_name
        self.fbeta_score_beta = fbeta_score_beta
        self.encoded_pos_class_label = encoded_pos_class_label
        self.is_voting_ensemble = is_voting_ensemble
        self.optimization_metric = optimization_metric
        self.random_seed = random_seed
        # Task type drives which metric set is used and (for classification)
        # whether folds are stratified. cv_folds > 1 enables CV inside the
        # objective so each trial is scored as the mean over folds rather than a
        # single noisy holdout point.
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.classifier_name = self.model.__class__.__name__

        if not self.is_voting_ensemble and not self.supported_models.is_supported(
            self.registered_model_name
        ):
            raise ValueError(
                f"Unsupported model: {self.registered_model_name}. "
                f"Supported models are: {self.supported_models.models}"
            )

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
        pred_proba: ArrayLike = None,
    ) -> pd.DataFrame:
        """Calculates different performance metrics for binary classification models.

        Args:
            true_class (ArrayLike): true class label.
            pred_class (ArrayLike): predicted class label not probability.
            pred_proba (ArrayLike): predicted probability of the positive class. Required
                to compute ROC-AUC correctly; if omitted, ROC-AUC is skipped.

        Returns:
            performance_metrics (pd.DataFrame): a dataframe with metric name and score columns.
        """

        rows = metrics.compute_metrics(
            task_type=self.task_type,
            y_true=true_class,
            y_pred=pred_class,
            proba=pred_proba,
            fbeta_beta=self.fbeta_score_beta,
        )
        return metrics.rows_to_dataframe(rows)

    def _pos_class_proba(self, features: ArrayLike) -> np.ndarray:
        """Returns the predicted probability of the positive class.

        The positive-class column is located via ``model.classes_`` rather than
        assuming index 1, so the result is correct regardless of how the label
        encoder ordered the classes.

        Args:
            features (ArrayLike): preprocessed features to score.

        Returns:
            np.ndarray: positive-class probabilities (one value per row).
        """

        proba = self.model.predict_proba(features)
        pos_col = int(
            np.where(self.model.classes_ == self.encoded_pos_class_label)[0][0]
        )
        return proba[:, pos_col]

    def _proba_for_metrics(self, features: ArrayLike) -> Optional[np.ndarray]:
        """Returns the probabilities the metric functions expect for the task.

        Binary -> positive-class probabilities (1-D); multi-class -> the full
        ``(n_samples, n_classes)`` matrix; regression -> None.

        Args:
            features (ArrayLike): features to score.

        Returns:
            Optional[np.ndarray]: probabilities, or None when not applicable.
        """
        if self.task_type == metrics.REGRESSION or not hasattr(
            self.model, "predict_proba"
        ):
            return None
        if self.task_type == metrics.MULTICLASS:
            return self.model.predict_proba(features)
        return self._pos_class_proba(features)

    def _fit_score_fold(
        self,
        train_features: ArrayLike,
        train_target: np.ndarray,
        valid_features: ArrayLike,
        valid_target: np.ndarray,
        metric_name: str,
    ) -> float:
        """Fits the model on one fold and returns the optimization metric.

        Args:
            train_features: Fold training features (preprocessed).
            train_target: Fold training labels/targets.
            valid_features: Fold validation features (preprocessed).
            valid_target: Fold validation labels/targets.
            metric_name: Metric row name to extract.

        Returns:
            float: the metric value on the fold's validation split.
        """
        self.model.fit(train_features, train_target)
        pred = self.model.predict(valid_features)
        proba = self._proba_for_metrics(valid_features)
        scores = self.calc_perf_metrics(
            true_class=valid_target, pred_class=pred, pred_proba=proba
        )
        row = scores.loc[scores["Metric"] == metric_name, "Score"]
        return float(row.iloc[0]) if not row.empty else float("-inf")

    def _metric_row_name(self) -> str:
        """Resolves the configured optimization metric to its row name in the
        ``calc_perf_metrics`` output.

        The config name ``fbeta_score`` maps to ``f_{beta}_score`` (matching how
        ``evaluate.py`` and the evaluator name it); for multi-class tasks the
        resolver applies macro averaging (e.g. ``roc_auc`` -> ``roc_auc_macro``).

        Returns:
            str: metric name as it appears in the metrics dataframe.
        """

        return metrics.selection_metric_row_name(
            self.optimization_metric, self.task_type, self.fbeta_score_beta
        )

    @property
    def optimization_direction(self) -> str:
        """Optuna study direction derived from the configured metric.

        Returns:
            str: ``"minimize"`` for lower-is-better metrics, else ``"maximize"``.
        """

        if self.optimization_metric in LOWER_IS_BETTER_METRICS:
            return "minimize"
        return "maximize"

    def obj_func(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        """Objective function that evaluates the provided hyperparameters for a
        specified model. The metric being optimized is the configured
        ``optimization_metric`` (defaults to fbeta score), so the search optimizes
        the same metric champion selection later ranks on.
        A trial hyperparameters are sampled from the search space using generate_trial_params
        method and then the model is fitted on training set and evaluated on the
        validation set.

        Args:
            trial (optuna.trial.Trial): an optuna trial object.

        Returns:
            valid_score (float): the configured optimization metric. With CV
                enabled this is the mean across folds; otherwise it is the single
                validation-split score.
        """

        # Define parameters search space
        params = self.generate_trial_params(trial=trial)
        self.model.set_params(**params)
        metric_name = self._metric_row_name()

        # CV path: score the trial as the mean of the optimization metric over
        # stratified folds of the train+valid pool, so trials are ranked on a
        # variance-reduced estimate instead of one noisy holdout. The per-trial
        # std is stored so champion selection can apply a 1-SE rule.
        if self.cv_folds and self.cv_folds > 1:
            return self._cv_objective(trial, metric_name)

        # Single-holdout path (cv_folds <= 1): preserves the original behaviour.
        self.model.fit(self.train_features_preprocessed, self.train_class)
        pred_train_preds = self.model.predict(self.train_features_preprocessed)
        pred_valid_preds = self.model.predict(self.valid_features_preprocessed)
        pred_train_proba = self._proba_for_metrics(self.train_features_preprocessed)
        pred_valid_proba = self._proba_for_metrics(self.valid_features_preprocessed)
        train_scores = self.calc_perf_metrics(
            true_class=self.train_class,
            pred_class=pred_train_preds,
            pred_proba=pred_train_proba,
        )
        valid_scores = self.calc_perf_metrics(
            true_class=self.valid_class,
            pred_class=pred_valid_preds,
            pred_proba=pred_valid_proba,
        )

        train_score = train_scores.loc[
            train_scores["Metric"] == metric_name, "Score"
        ].iloc[0]
        valid_score = valid_scores.loc[
            valid_scores["Metric"] == metric_name, "Score"
        ].iloc[0]

        self.tracker.log_metric(
            name=f"train_{metric_name}", value=float(train_score), step=trial.number
        )
        self.tracker.log_metric(
            name=f"valid_{metric_name}", value=float(valid_score), step=trial.number
        )

        # Return the validation score to ensure it's used for model selection
        return valid_score

    def _cv_objective(self, trial: optuna.trial.Trial, metric_name: str) -> float:
        """Scores the current trial via stratified K-fold CV on the train+valid pool.

        Args:
            trial: The active Optuna trial (used to record the per-fold std).
            metric_name: Optimization metric row name to average over folds.

        Returns:
            float: mean of the optimization metric across folds.
        """
        pool_features = pd.concat(
            [self.train_features_preprocessed, self.valid_features_preprocessed],
            axis=0,
            ignore_index=True,
        )
        pool_target = np.concatenate(
            [np.asarray(self.train_class), np.asarray(self.valid_class)]
        )

        if self.task_type == metrics.REGRESSION:
            splitter = KFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
            )
            fold_iter = splitter.split(pool_features)
        else:
            splitter = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed
            )
            fold_iter = splitter.split(pool_features, pool_target)

        fold_scores = [
            self._fit_score_fold(
                pool_features.iloc[train_idx],
                pool_target[train_idx],
                pool_features.iloc[valid_idx],
                pool_target[valid_idx],
                metric_name,
            )
            for train_idx, valid_idx in fold_iter
        ]

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        trial.set_user_attr("cv_std", std_score)
        trial.set_user_attr("cv_folds", int(self.cv_folds))

        self.tracker.log_metric(
            name=f"valid_{metric_name}", value=mean_score, step=trial.number
        )
        self.tracker.log_metric(
            name=f"valid_{metric_name}_std", value=std_score, step=trial.number
        )

        return mean_score

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
        def logging_callback(
            study: optuna.study.Study, frozen_trial: optuna.trial.FrozenTrial
        ) -> None:
            """Logs only when the best value is updated during hyperparameter optimization.

            Args:
                study: Optuna study object.
                frozen_trial: Optuna frozen trial object.
            """
            previous_best_value = study.user_attrs.get("previous_best_value", None)
            if previous_best_value != study.best_value:
                study.set_user_attr("previous_best_value", study.best_value)
                logger.info(
                    "\nTrial %d finished, best value: %d hyperparameters: %s.",
                    int(frozen_trial.number),
                    frozen_trial.value,
                    frozen_trial.params,
                )

        # Define the sampler for hyperparameter optimization
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(
                0.1 * max_search_iters
            ),  # Warm-up trials that use random sampling
            multivariate=True,
            constant_liar=True,
            seed=self.random_seed,  # makes the search reproducible when set
        )
        study = optuna.create_study(
            sampler=sampler, direction=self.optimization_direction
        )

        print(
            """\n
        ----------------------------------------------------------------
        --- Hyperparameter Optimization of %s Starts ...
        ----------------------------------------------------------------\n""",
            self.classifier_name,
        )

        study.optimize(
            func=self.obj_func,
            n_trials=max_search_iters,
            timeout=model_opt_timeout_secs,
            gc_after_trial=False,  # Set to True if memory consumption increases over several trials
            callbacks=[logging_callback],
        )

        return study

    def tune_model_in_parallel(
        self,
        max_search_iters: int = 100,
        n_parallel_jobs: int = 2,
        model_opt_timeout_secs: int = 180,
    ) -> optuna.study.Study:
        """Performs hyperparameters optimization using Optuna package in parallel. If
        Dask distributed client is available, it distributes trials across many physical
        workers in the cluster. Otherwise, it distributes work among available CPU cores
        by using multiprocessing.

        Note:
            The TPE sampler is inherently sequential, using the history of completed
            trials to suggest the next hyperparameters. In parallel execution,
            concurrent trials are not yet completed, which can lead to redundant
            sampling due to incomplete information. To mitigate this, `constant_liar=True`
            is used in the TPESampler. This strategy assigns a temporary "poor" objective
            value to running trials, discouraging other workers from exploring the same
            hyperparameter space immediately. This promotes exploration and prevents
            workers from chasing the same local optima simultaneously, though it may
            slightly alter the optimization trajectory compared to a purely sequential
            run.

        Args:
            max_search_iters (int): maximum number of search iterations.
            n_parallel_jobs (int): number of parallel jobs.
            model_opt_timeout_secs (int): maximum time in seconds to optimize model.

        Returns:
            study (optuna.study.Study): optuna study object.
        """

        # Imported lazily so the distributed extra is optional: serial search
        # (the default) does not require dask/optuna-distributed.
        try:
            import optuna_distributed  # pylint: disable=import-outside-toplevel
            from dask.distributed import (
                Client,  # pylint: disable=import-outside-toplevel
            )
        except ImportError as exc:
            raise ImportError(
                "Parallel optimization requires the 'distributed' extra. "
                "Install it with: uv pip install -e '.[distributed]'"
            ) from exc

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=int(
                0.1 * max_search_iters
            ),  # Warm-up trials that use random sampling
            warn_independent_sampling=False,  # Disable warning for independent sampling
            multivariate=True,  # Enable multivariate sampling to consider interactions between hyperparameters
            constant_liar=True,  # to mitigate redundant sampling in parallel execution
            seed=self.random_seed,  # makes the search reproducible when set
        )
        client = Client()
        study = optuna_distributed.from_study(
            optuna.create_study(sampler=sampler, direction=self.optimization_direction),
            client=client,
        )

        study.optimize(
            func=self.obj_func,
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
