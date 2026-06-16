[![CI/CD](https://github.com/Adeemy/end-to-end-ml/actions/workflows/main.yml/badge.svg)](https://github.com/Adeemy/end-to-end-ml/actions/workflows/main.yml)[![python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.99.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![codecov](https://codecov.io/gh/Adeemy/end-to-end-ml/graph/badge.svg?token=LO67YZIGXR)](https://codecov.io/gh/Adeemy/end-to-end-ml)

# End-to-end ML

An end-to-end ML project for tabular data covering the full lifecycle of a classifier: data exploration, preprocessing, feature engineering, model selection, training, evaluation, and deployment.

The project uses the Diabetes Health Indicators dataset from [UCI](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (patient demographics, lab results, and self-reported health history) to predict whether a patient is diabetic. It applies ML engineering practices (modular code, tests, logging, YAML configuration, version control) and tools including pandas, scikit-learn, [Feast](https://feast.dev) (feature store), [Optuna](https://optuna.org) (hyperparameter search), experiment tracking with [MLflow](https://mlflow.org) (default) or [Comet](https://www.comet.com/site/), and Docker.

## Key Features

- **Config-driven models**: add or remove a model by editing the `models:` list in `config/training-config.yml`; no code change.
- **Hyperparameter search**: per-model Optuna search spaces, optionally with stratified K-fold CV.
- **Variance-aware selection**: champion chosen by a 1-SE rule over candidate scores, then calibrated and gated on a held-out test set.
- **Experiment tracking**: MLflow (default, local file store) or Comet ML, switchable in config.
- **Serving**: CLI batch scoring (`predict.py`) and a FastAPI REST service (`api_server.py`), containerized with Docker.
- **Reproducible environment**: dev container and `uv`-managed dependencies.

## Project Structure

- **`notebooks/`**: exploratory data analysis and baseline model development.
- **`config/`**: YAML configuration for the feature store and training.
- **`src/feature/`**: data ingestion, preprocessing, and the Feast feature store.
- **`src/training/`**: training, hyperparameter search, the model factory, evaluation, and champion selection.
- **`src/inference/`**: serving - CLI batch scoring (`predict.py`) and the REST API (`api_server.py`).
- **`src/utils/`**: shared logging, paths, and config loading.
- **`tests/`**: unit tests for the feature, training, and inference components.

```
.
├── config/                feature-store-config.yml, training-config.yml, logging.conf
├── Makefile, pyproject.toml
└── src/
    ├── feature/           generate_initial_data.py, prep_data.py, feature_repo/ (Feast)
    ├── training/          train.py, evaluate.py, split_data.py
    │   ├── core/          optimizer.py, trainer.py, ensemble.py, model_factory.py
    │   ├── evaluation/    orchestrator.py, evaluator.py, champion.py, selector.py
    │   └── tracking/      experiment.py, experiment_tracker.py
    ├── inference/         predict.py, api_server.py, utils/model.py, Dockerfile
    └── utils/             config_loader.py, logger.py, path.py
```

### ML pipeline flow

1. **Split** (`split_data.py`): load from the feature store and create train/validation/calibration/test splits.
2. **Train** (`train.py`): per-model Optuna search, fit, and register each enabled model with the tracker.
3. **Evaluate** (`evaluate.py`): rank candidates on the validation set, calibrate the best, evaluate on the held-out test set, and register it as champion if it clears the deployment gate.
4. **Serve**: load the champion for batch scoring or via the REST API.

## Environment setup

The project uses [uv](https://github.com/astral-sh/uv) to manage dependencies (defined in `pyproject.toml`). A dev container is provided for a reproducible environment and is the recommended path; it requires [Docker](https://docs.docker.com/engine/install/). To set up a local virtual environment instead, run from the project root:

    python3.14 -m venv ml_env
    source ml_env/bin/activate
    make install

### Experiment tracker credentials

MLflow is the default tracker and needs no configuration; it logs to a local `./mlruns` file store. To use Comet ML instead, set `experiment_tracker: "comet"` in `config/training-config.yml` and provide:

    COMET_API_KEY=your_comet_api_key
    ENABLE_COMET_LOGGING=true

`ENABLE_COMET_LOGGING` ensures `comet_ml` is imported early enough for auto-logging. Copy `.env_template` to `.env` and add your keys (`cp .env_template .env`).

## Configuring models

The set of models to train is the `models:` list in `config/training-config.yml`. Each entry is self-contained and resolved by the model factory (`src/training/core/model_factory.py`), which imports the estimator class and instantiates it. **Adding a model is config-only** - append an entry with any scikit-learn-compatible classifier; no code change is needed.

```yaml
models:
  - name: "lightgbm"                       # registered model name + experiment key
    estimator: "lightgbm.LGBMClassifier"   # importable class path, resolved via importlib
    enabled: true                          # whether this model is trained
    params:                                # fixed estimator kwargs
      objective: "binary"
      class_weight: "balanced"
    search_space_params:                   # Optuna search space
      max_depth: [3, 8, false]             #   numeric: [min, max, log_scale]
      learning_rate: [1e-4, 1e-1, true]
      n_estimators: [20, 100, false]
  - name: "xgboost"
    estimator: "xgboost.XGBClassifier"
    enabled: false
    params:
      objective: "binary:logistic"
      scale_pos_weight: "${scale_pos_weight}"  # runtime placeholder: train-set neg/pos ratio
    search_space_params:
      criterion: [["gini", "entropy"], false]  # categorical: [[choices], false]
```

- Set `enabled: true`/`false` to include or exclude a model from training.
- A param value of the form `"${scale_pos_weight}"` is substituted at runtime with the train-set negative/positive ratio (class-imbalance weighting).
- `ensemble: {enabled: true}` trains a soft-voting ensemble over the enabled base models. `modelregistry` holds the ensemble and champion registered names.

## Running the pipeline

Each stage is a `make` target (also wired into GitHub Actions):

| Command | What it does |
|---|---|
| `make gen_init_data` | Import the raw UCI dataset and hold out a 5% inference set. |
| `make prep_data` | Preprocess data before feature-store ingestion. |
| `make setup_feast` | Set up the Feast feature store. |
| `make split_data` | Create train/validation/calibration/test splits from the feature store. |
| `make train` | **Training only**: Optuna search, fit, and register each enabled model. Requires existing splits; does not select or deploy a champion. |
| `make evaluate` | Select the champion among the most recent runs, calibrate it, evaluate on the test set, and deploy it if it clears the gate. |
| `make submit_train` | **Full pipeline**: `prep_data` → `split_data` → `train` → `evaluate`. |

### `train` vs `evaluate` vs `submit_train`

- **`make train`** runs the training stage alone. It assumes the data splits already exist (from `make split_data`) and stops after fitting and registering the enabled models. It does no model selection.
- **`make evaluate`** runs the selection-and-deployment stage on its own. By default it evaluates the most recent training run, discovered live from the tracking store (the latest registered version of each enabled model). To evaluate a specific MLflow run instead:

      make evaluate RUN_ID=<mlflow_run_id>

- **`make submit_train`** chains the data, training, and evaluation stages end-to-end (`prep_data → split_data → train → evaluate`), taking preprocessed data through to a deployed champion in one command.

### Browsing experiments

Launch the MLflow UI to view and compare training and evaluation runs:

    make view_mlflow          # serves http://localhost:8080
    make view_mlflow PORT=8088 # override the port

Use this target, not a bare `mlflow ui`. MLflow 3.x refuses the local `./mlruns`
file store in "maintenance mode" unless `MLFLOW_ALLOW_FILE_STORE=true` is set;
`make view_mlflow` sets it and points at this repo's `mlruns`, so it works right
after a clone and a training run.

## Champion selection and deployment gate

`make evaluate` ranks candidate models on the validation set, selects the champion using a 1-SE rule (the simplest model within one standard error of the best), calibrates it on the calibration split, and evaluates the calibrated model on the untouched test set. The champion is deployed only if its test score clears `deployment_score_thresh` in the config. When deployed, it is:

- registered with the tracker under `champion_model_name` (default `champion_model`), and
- saved to `src/training/artifacts/champion_model.pkl` with its decision threshold in `champion_model_metadata.json`.

The saved champion is a calibrated pipeline that includes preprocessing, so inference takes raw input and applies the same transformations used in training.

## Inference and serving

Serving loads the champion from the tracker registry, falling back to the local `champion_model.pkl` if the registry lookup misses.

**Batch scoring (CLI):**

    make test_api_cli                  # score the built-in sample rows
    make predict_batch                 # score src/feature/feature_repo/data/inference.parquet
    make predict_batch_custom INPUT_FILE=path/to/input.parquet OUTPUT_FILE=path/to/output.parquet

**REST API (FastAPI):**

    make start_api_server              # serve at http://localhost:8000 (docs at /docs)
    make test_api_with_sample          # POST sample rows to a running server
    make test_api_full                 # start server -> test health + /predict -> stop
    make help_api                      # list the API/CLI testing commands

**Containerized model** (example image tag):

    docker pull ghcr.io/adeemy/end-to-end-ml:<image-tag>

## Testing and code quality

    make test     # pytest + coverage
    make lint     # pylint
    make format   # isort + black
