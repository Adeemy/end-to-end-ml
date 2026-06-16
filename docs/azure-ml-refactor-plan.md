# Azure ML refactor plan

Run this project's training and evaluation on an **Azure ML workspace** (jobs on
managed compute, tracked in the workspace), keep ingesting the **UCI** dataset
(no SQL), and serve features from an **Azure ML managed feature store** instead
of the local Feast store. Patterns mirror `orlando_cash_forecast_ts`, adapted to
the v2 SDK and this project's data.

## Repo vs branch

One repo, on a branch (`azure-ml`), as an **additive execution backend** - not a
fork. The `src/` training/evaluation/serving code is reused as the Azure ML job
entrypoints. The data source (local Feast vs Azure feature store) and the MLflow
tracking URI are gated behind config, so the same code runs locally and on Azure.
Merge to `main` once stable; Azure stays an optional backend (local is the
default). A separate repo would only be warranted for org/governance isolation or
if the Azure variant diverges in data or models.

## What carries over for free

The project is already MLflow-first. Azure ML *is* MLflow: pointing
`mlflow.set_tracking_uri(...)` at the workspace makes the existing run logging,
metrics, `log_model` + signature, and `register_model` write into the workspace
with no logic change. The config-driven models, 1-SE selection, calibration,
deployment gate, and champion logic all transfer as the in-cluster job logic.

## orlando patterns mirrored

- A local **submit script** vs a cluster **entrypoint**: `submit_train.py`
  ensures compute + environment and submits a job that runs `train.py` on the
  cluster; `submit_evaluate.py` runs `evaluate.py`.
- **Auth**: a Service Principal (tenant/app id), the secret pulled from Key Vault
  via the cluster's managed identity; ids from `.env`.
- **Compute**: an `AmlCompute` cluster, create-or-get, config-driven.
- **Environment**: a conda spec + Dockerfile pushed to ACR, registered as an
  Azure ML Environment with hash-based rebuild.
- **Models**: `mlflow.sklearn.log_model(..., signature=...)` + `register_model`
  with tags; champion chosen on a test metric.
- **CI/CD**: a pipeline that runs the submit script in the ACR container.

orlando reads from Snowflake/SQL and uses plain Tabular Datasets (no feature
store); this plan drops SQL and uses the Azure ML managed feature store instead.

## Phase 0 - decisions to lock first (blocking)

1. **SDK: v2 (`azure-ai-ml`)**, not orlando's legacy v1 (`azureml-core`). The
   managed feature store is v2-only and v1 is deprecated. Mirror orlando's
   *structure*, implement with `MLClient` + `command()` jobs.
2. **Python on the cluster: 3.11/3.12**, not 3.14. `azure-ai-ml` and
   `azureml-featurestore` target 3.11/3.12; 3.14 wheels are unlikely. Local dev
   stays 3.14; the Azure *job* image pins a supported Python. Validate before
   building images.

## Phase 1 - Azure connectivity (`src/azureml/`)

- `client.py`: build `MLClient` from a Service Principal (`ClientSecretCredential`)
  or `DefaultAzureCredential`; subscription/resource-group/workspace from `.env`;
  the SP secret resolved from Key Vault. A `workspace_mlflow_uri()` helper returns
  the workspace tracking URI for the entrypoints.
- Add an `azureml:` section to `config/training-config.yml` (workspace + compute +
  environment + feature-store + dataset names) and `AzureMLConfig` to `schemas.py`.
- Azure deps go in a new `[project.optional-dependencies] azure` extra, so the
  default `make install` and the test suite never import them.

## Phase 2 - compute + environment

- `compute.py`: create-or-get an `AmlCompute` cluster (name, vm size, min/max
  nodes, idle scaledown, `SystemAssigned` identity for Key Vault).
- `environment.py`: register an Azure ML `Environment` from `azureml/conda.yml` +
  `azureml/Dockerfile`, with a content hash so it rebuilds only on change.
- `azureml/conda.yml` + `azureml/Dockerfile`: Python 3.11/3.12 + the project
  dependencies (minus 3.14-only pins) + `azureml-mlflow`.

## Phase 3 - data (no SQL)

`gen_init_data` (UCI download) and `prep_data` are unchanged. `prep_data` writes
the prepared, UCI-derived features to the feature store's offline source
(ADLS/Blob), instead of (or in addition to) the local Feast materialization.

## Phase 4 - Azure feature store (replaces Feast)

- One-time infra: an Azure ML managed **feature store**, an **entity** (the row
  key), and a **feature set** (a transformation spec over the prepared data)
  materialized to the offline store.
- `feature_store.py`: retrieve training features with `get_offline_features(...)`
  on an observation/entity frame - the direct analog of Feast's
  `get_historical_features`.
- A `feature_backend: feast | azureml` config switch keeps the existing Feast
  path working locally; the Azure path imports azure libs lazily.
- *Lighter alternative*: keep Feast but point it at Azure providers (Blob offline,
  Redis online). Heavier-but-Azure-native is the managed feature store; pick per
  how much Azure-native tooling is wanted.

## Phase 5 - job submission

- `submit_train.py` / `submit_evaluate.py`: build the client, ensure compute +
  environment, then submit a `command()` job that runs the existing `train.py` /
  `evaluate.py` on the cluster, with MLflow pointed at the workspace and features
  read from the feature store.
- Make targets `azure_submit_train` / `azure_submit_evaluate`.

## Phase 6 - serving (online endpoint)

Deploy the champion as an Azure ML **managed online endpoint** reusing the
`predict.py` logic, reading online features from the feature store, alongside the
existing local FastAPI service.

## Phase 7 - CI/CD

An `azure-pipelines.yml` (or GitHub Action) that builds/pushes the image to ACR
and runs the submit scripts, mirroring orlando's `train_pipeline.yml`.

## Risks to resolve early

- v2 SDK + Python 3.11/3.12 on the cluster (3.14 incompatibility).
- Managed feature store setup overhead for a static UCI dataset (the lighter
  Feast-on-Azure path is the fallback).
- Azure SDK code in this branch is written against the documented v2 APIs but is
  **not runnable locally** (no Azure subscription, no 3.14 azure wheels); it needs
  validation against a real workspace and the installed `azure-ai-ml` version.
