# Enhancement Plan: rigorous evaluation, task-type extensibility, and de-bloating

> Implementation step 0: copy this file to `docs/enhancement-plan.md` (uncommitted) as the user requested the plan live in `docs/`. Plan mode restricts edits to this plans file, so the `docs/` copy is the first action after approval. Nothing is committed until the user explicitly approves a commit.

## Context

This repo trains, selects, calibrates, and serves classification models that feed revenue-generating, client-facing products. The single most important property is that **offline model selection and the reported headline metrics accurately predict production behavior**. A multi-perspective review (data scientist, ML engineer, code reviewer) plus direct code verification found that this property is currently violated in a few specific, fixable ways, that the codebase is hard-coded to binary classification in several spots, and that there is real duplication and dead code to remove. The work below fixes the rigor gaps, makes binary and multi-class first-class while leaving a clean seam for regression, and cuts duplication without adding a new layer of abstraction bloat.

### Verified findings (with the agent claims I checked and rejected)

I read the live code paths rather than trusting the review verbatim. Corrections to avoid acting on wrong claims:

- **No training/serving preprocessing skew.** The champion artifact is a `CalibratedClassifierCV` wrapping the full `Pipeline[preprocessor, selector, classifier]` (`champion.py:133-141`). Inference calls `model.predict_proba(raw_df)` (`helpers.py:77`, `api_server.py:162`), so preprocessing runs at serving. The "preprocessor not applied at inference" claim is false.
- **The two tracking files are not duplicates.** `experiment.py` holds `ExperimentManager` (experiment creation, model registration, project init); `experiment_tracker.py` holds `ExperimentTracker` (logging primitives). Both are used; neither will be deleted.
- **Inference already uses `model.classes_` + a persisted threshold sidecar** (`helpers.py:73-82`), and `optimizer._pos_class_proba` locates the positive column via `classes_`. The "hard-coded positive index" and "logisticregression vs logistic_regression drift" issues are already fixed. The remaining real config drift is `included_models` vs `includedmodels` (below).
- `.env`, `.coverage`, and `mlruns/` are **not** git-tracked; only data parquet/db files, two model pkls, and one notebook pkl are tracked (12 parquet/pkl + 2 db). Repo-hygiene work targets those tracked files only.

## Scope decisions (from the user)

1. Selection generalization: implement **both** a stratified K-fold CV path and a repeated-holdout path behind a config switch, **defaulting to CV** (`cv_folds > 1` => CV).
2. Regression: **architecturally extensible** this pass (explicit `task_type`, a task-strategy seam, a `RegressionEvaluator` with regression metrics), without fully wiring an end-to-end regression training/serving path. Binary and multi-class must work and be tested.
3. Repo hygiene: **untrack** data/artifacts, keep a small committed sample (or synthetic generator) so CI/tests still run, and tighten `.gitignore`.
4. Metrics: **add PR-AUC (average precision), Brier score, log loss, and ECE** on calibration and test; keep F-beta(0.5) as the default selection metric but make it switchable, and **decouple the optimization metric from the selection metric** to break the tuning/selection circularity.

## Workstream 1 - Evaluation and selection rigor (highest priority)

The deliverable: the numbers we report and gate on are produced by the exact artifact we deploy, with variance-aware selection.

1. **Evaluate the deployed artifact on the untouched test set.** In `src/training/evaluation/orchestrator.py::run_evaluation_workflow`, reorder so the champion is calibrated on the calibration split and its operating threshold is resolved **before** test evaluation, then evaluate that calibrated pipeline at the persisted threshold on test (currently `evaluate_on_test_set` at lines 731-734 runs the uncalibrated pipeline at argmax/0.5, while the deployed model is calibrated + tuned-threshold). The deployment gate (lines 753-783) must use the same calibrated, thresholded test score. Pass the resolved `decision_threshold` into `evaluate_test_set_only` instead of the hard-coded `0.5` default (`evaluator.py:877-909`).
2. **CV-based, variance-aware selection.** Wire the currently dead `cross_val_folds` (config:89, `TrainParams.cross_val_folds`). In `core/optimizer.py::obj_func`, when `cv_folds > 1`, score each trial with stratified K-fold CV on the train+valid pool and return the mean; expose per-fold std. Champion selection in `orchestrator._select_champion_in_process` ranks on the CV mean and applies a **1-SE rule** (prefer the simplest model within one standard error of the best) to avoid selecting on noise. When `cv_folds <= 1`, fall back to repeated-holdout over seeds for a confidence interval. Log mean and std to the tracker.
3. **Decouple optimization metric from selection metric.** Add `selection_metric` to config/`TrainParams` (default = `comparison_metric`), so tuning can optimize one metric while selection ranks on another, removing the current circularity where both use the same metric on the same split (`optimizer._metric_row_name`, `evaluate.py:130-137`).
4. **Expanded metric set** in the shared metrics module (Workstream 3): PR-AUC / average precision, Brier score, log loss, plus ECE reported on both the calibration split and the test set (today ECE is computed only on valid in `trainer.py:268-274` and never logged as a tracked metric). Add these to what `tracker.log_metrics` receives.
5. **Threshold tuning robustness.** Keep tuning on the calibration split (`champion.tune_decision_threshold`) but guard it to binary only and document that multi-class keeps argmax (already partially handled at `orchestrator.py:390-393`).

## Workstream 2 - Task-type abstraction: binary, multi-class, regression seam

Principle: one pipeline, a small injected task strategy. Do **not** create per-task orchestrator/predictor/threshold packages (that would add the bloat the user wants removed). Extend the existing factory pattern in `evaluator.py::create_model_evaluator`.

1. **Explicit `task_type` config** (`binary | multiclass | regression`) in `training-config.yml` and `TrainParams`, replacing the implicit "count unique classes" dispatch (`evaluator.py:1456`) which can misroute (e.g. a test fold missing a class, or a low-cardinality regression target). The factory dispatches on `task_type`; class-count becomes a validation cross-check, not the router.
2. **Fix the multi-class selection bug (real, verified).** `orchestrator._score_pipeline_on_valid` looks up the row named by `comparison_metric_name` (e.g. `f_0.5_score`, `roc_auc`), but the multi-class evaluator only emits suffixed names (`f_0.5_score_macro/micro/weighted`, `roc_auc_macro`). The lookup returns empty -> `-inf` for every candidate -> selection silently falls back to the first candidate, and the gate's `test_{metric}` lookup then raises. Resolve the metric name through an averaging-aware helper keyed on `task_type` (e.g. multi-class default = `f_{beta}_score_macro`). This makes multi-class selection actually work end to end.
3. **`RegressionEvaluator`** added alongside the binary/multi-class evaluators implementing the same `ModelEvaluator` interface, with MAE, RMSE, R2, MAPE, and residual plots; `predict_proba`/threshold/calibration paths are skipped for regression. Factory returns it for `task_type == "regression"`. Training/serving wiring for regression is left as documented TODOs (extensible, not fully built) per scope.
4. **Task-aware pipeline assembly.** Gate label encoding, calibration, threshold tuning, and `predict_proba`-based metrics on "is classification" so the regression seam does not execute classification-only code. Centralize the binary `_get_pred_class` (`evaluator.py:712`, currently returns literal `0/1`) to emit the model's own class labels.

## Workstream 3 - De-duplication, dead code, and config hardening

Cut duplication and dead weight; this is a first-class goal, not cleanup-on-the-side.

1. **Collapse triplicated `calc_perf_metrics`** into one `src/training/evaluation/metrics.py` (or a function module) used by `core/optimizer.py:168-215`, `evaluation/evaluator.py:552-599` (binary) and `:973-1074` (multi-class). Delete the duplicated copy in `notebooks/utils.py` (which additionally has a real bug: it passes hard labels to `roc_auc_score`). Notebook imports from the shared module.
2. **Fix config drift + dead config.** `build_training_config` reads `params.get("included_models", {})` (`schemas.py:338`) but the YAML key is `includedmodels` (config:176) and `train.py` reads `config.params["includedmodels"]` directly, so the `included_models` dataclass is always defaults and never used. Rename to match and make `train.py` consume the dataclass, or remove the dead dataclass. Wire or remove `cross_val_folds` (Workstream 1 wires it).
3. **Config schema validation.** `map_to_dataclass` silently drops unknown keys; `check_params` runs (via `config_loader.py:51`) but is partial and has wrong nested-key references in its error branches (e.g. `self.params['data']['params']['split_rand_seed']`). Tighten: add `task_type`, `selection_metric`, `cv_folds`, `deployment_score_thresh`, and threshold keys to validation; fix the broken error-branch keys; add a check that every YAML section maps to a known dataclass field (catch future drift). Pydantic is optional; fixing/extending `check_params` plus a key-coverage assertion is the lower-bloat path and is the default unless the user prefers pydantic.
4. **Remove dead/commented code:** commented-out test bodies in `tests/test_training/test_job.py` and `tests/test_training/test_training_model.py`; the unused `ModelEvaluator` class in `notebooks/utils.py` (keep the genuine EDA helpers); any manager `log_*` methods on `ExperimentManager` confirmed unused after the metrics consolidation.

## Workstream 4 - Correctness and robustness

1. **Stop in-place overwrite of split parquet.** `train.py:357-383` rewrites `train/valid/test/calib.parquet` with feature-selected, label-encoded data (non-idempotent: a second run reads already-mutated splits) and mutates feature frames via `train_set[class] = ...`. Write encoded splits to distinct filenames (e.g. `*_encoded.parquet`) that `evaluate.py` reads, leaving the canonical splits from `split_data.py` intact.
2. **Narrow swallowed failures.** The broad `except Exception` in `trainer.run_training_experiment` (`trainer.py:387`) drops a failed model to `None` and silently shrinks the candidate set, which can let a degraded set produce a "champion". Narrow the catch, log loudly, and make a total wipeout fail fast. Same review for the feature-importance and MLflow-register broad excepts.
3. **Deployment gate hygiene.** Keep the explicit gate but ensure it reads the calibrated/thresholded test score (Workstream 1) and that a missing metric is a hard error (already raises at `orchestrator.py:757`).

## Workstream 5 - Repo hygiene (per user: untrack + CI sample)

1. `git rm --cached` the tracked data/artifacts: `src/feature/feature_repo/data/*.parquet` and `*.db`, `src/inference/artifacts/batch_predictions.parquet`, `src/training/artifacts/*.pkl`, `notebooks/.eda_figures/basline_model.pkl`. Add matching `.gitignore` rules (the current file has parquet commented in to allow CI to read local data).
2. Keep CI green without the bulk: commit a **small sampled** `inference`/`train` parquet (or have `generate_initial_data.py` produce a deterministic synthetic sample), and point the smoke/integration tests and CI at the sample. Verify `.env`/`.coverage` remain ignored (they already are).
3. Update `.github/workflows/*` and the `Makefile` data steps if they assume the full committed parquet.

## Workstream 6 - Serving robustness (lower priority)

`api_server.py:62-79` loads the model at import time; if registry and local both fail the process cannot start, and `/` is the only status route. Add lazy load with cache + reload-on-failure and a `/health` route reporting model load state and source. This is opt-in within this pass; flag if it should be deferred.

## Workstream 7 - Tests and verification

- Unit: shared metrics module (binary, multi-class, regression), averaging-aware selection-metric resolver, 1-SE selection rule, task_type factory dispatch, threshold-on-test wiring.
- Regression-guard tests: a synthetic 3-class dataset runs selection -> calibration -> test -> gate without falling back to the first candidate (locks in the multi-class fix); a test asserting reported test metrics equal metrics recomputed from the saved champion at its persisted threshold (locks in Workstream 1).
- Keep/trim the existing `tests/test_training` calibration/threshold/split tests; they already assert disjoint splits and prefit calibration.

## Explicitly NOT doing (rejected review claims)

- Not deleting `experiment.py` (it is not a duplicate of `experiment_tracker.py`).
- Not "adding the preprocessor at inference" (already present via the saved pipeline).
- Not creating per-task `orchestrators/`, `predictors/`, `thresholds/` packages (bloat); using one pipeline + injected task strategy instead.
- Not migrating to DVC in this pass (out of scope; sampling covers CI).

## Critical files

- Evaluation/selection: `src/training/evaluation/orchestrator.py`, `src/training/evaluation/evaluator.py`, `src/training/evaluation/champion.py`, `src/training/evaluate.py`, `src/training/core/optimizer.py`, `src/training/core/trainer.py`.
- New: `src/training/evaluation/metrics.py` (shared metrics), `RegressionEvaluator` in `evaluator.py`.
- Config: `src/config/training-config.yml`, `src/training/schemas.py`, `src/utils/config_loader.py`.
- Data/idempotency: `src/training/train.py`, `src/training/split_data.py`.
- Hygiene: `.gitignore`, `.github/workflows/`, `Makefile`, tracked data/artifact paths.
- Tests: `tests/test_training/*`, `notebooks/utils.py`.

## Verification

1. `make lint` and `make test` (pytest + coverage) green, including the new task-type and rigor tests.
2. End-to-end on a small sample: `make split_data train evaluate` with `task_type: binary`, then repeat with a synthetic `task_type: multiclass` config; confirm selection does not fall back to the first candidate and the gate uses the calibrated/thresholded test score.
3. Assert parity: reported `test_*` metrics equal metrics recomputed from the saved `champion_model.pkl` at its `*_metadata.json` threshold.
4. `make predict_batch` and `make test_api_cli` still produce predictions from the saved champion (serving path unchanged in behavior).
5. `git status` shows data/artifacts untracked and `.gitignore` updated; CI workflow still runs against the sample.
