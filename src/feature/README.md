# Feature Engineering Module

Data preprocessing, feature engineering, and feature store management for the ML pipeline.

## Quick Commands

```bash
make gen_init_data   # Generate raw and inference (simulates production data) datasets
make prep_data       # Preprocess and transform data
make setup_feast     # Initialize feature store
make split_data      # Create train/validation/test splits
```

## Pipeline Overview

1. **Data Generation** → Load raw data and create inference set
2. **Preprocessing** → Clean, transform, and validate data quality
3. **Feature Store** → Register features with Feast for consistent serving
4. **Data Splitting** → Create stratified train/validation/test splits

## Key Files

- **`generate_initial_data.py`**: Creates raw dataset for training and inference set (5% holdout) to simulate production data
- **`prep_data.py`**: Preprocesses and transforms raw data before feature store ingestion
- **`feature_repo/define_feature.py`**: Feast feature definitions and schema
- **`utils/data.py`**: Main data preprocessing classes (`TrainingDataPrep`)
- **`utils/prep.py`**: Data preparation utilities and transformations

## Feature Engineering

The `TrainingDataPrep` class handles:
- **Numerical features**: Scaling (Standard/Robust/MinMax), imputation
- **Categorical features**: One-hot encoding, missing value handling
- **Feature selection**: Variance threshold filtering
- **Data validation**: Type checking and quality constraints

## Feast Feature Store

### Setup Commands
```bash
make setup_feast          # Complete setup (teardown + apply)
make teardown_feast       # Remove existing feature store
make init_feast           # Apply feature definitions
make show_feast_entities  # List registered entities
make show_feast_views     # List feature views
make show_feast_ui        # Launch web UI (experimental)
```

### Directory Structure
```
feature_repo/
├── feature_store.yaml    # Feast configuration
├── define_feature.py     # Feature definitions
└── data/                 # Feature data storage
    ├── raw_dataset.parquet
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
```
