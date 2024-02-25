import pytest

from src.feature_store.utils.config import Config
from src.utils.path import PARENT_DIR


def test_config_init():
    """Tests if Config class init method returns the correct
    config yaml file (config file exists)."""

    config_path = f"{str(PARENT_DIR)}/config/feature_store_config.yml"
    config = Config(config_path)
    assert config.config_path == config_path


def test_config_init_file_not_found():
    """Tests if Config class init method raises FileNotFoundError if
    config file doesn't exist."""

    incorrect_config_path = f"{str(PARENT_DIR)}/config/nonexistent_config.yml"
    with pytest.raises(FileNotFoundError):
        Config(incorrect_config_path)


def test_config_check_params():
    """Tests if Config class _check_params method doesn't raise any
    exceptions when all required values are present."""

    params = {
        "description": "Sample description",
        "data": {
            "params": {
                "raw_dataset_source": "raw_data_source",
                "pk_col_name": "id",
                "num_col_names": "num_col",
                "cat_col_names": "cat_col",
                "uci_raw_data_num": 123,
            }
        },
    }
    Config._check_params(params)  # Should not raise any exceptions


def test_config_check_params_missing_raw_dataset_source():
    """Tests if Config class _check_params method raises ValueError
    when raw_dataset_source is not specified."""

    params = {"raw_dataset_source": "none"}
    with pytest.raises(AssertionError):
        Config._check_params(params)


def test_config_check_params_missing_pk_col_name():
    """Tests if Config class _check_params method raises ValueError
    when pk_col_name is not specified."""

    params = {
        "description": "Sample description",
        "data": {
            "params": {
                "raw_dataset_source": "raw_data_source",
                "pk_col_name": "none",
            }
        },
    }
    with pytest.raises(ValueError):
        Config._check_params(params)


def test_config_check_params_missing_num_and_cat_col_names():
    """Tests if Config class _check_params method raises ValueError
    when neither numerical nor categorical column names are specified."""

    params = {
        "description": "Sample description",
        "data": {
            "params": {
                "raw_dataset_source": "raw_data_source",
                "pk_col_name": "id",
                "num_col_names": "none",
                "cat_col_names": "none",
            }
        },
    }
    with pytest.raises(ValueError):
        Config._check_params(params)


def test_config_check_params_missing_uci_raw_data_num():
    """Tests if Config class _check_params method raises ValueError
    when uci_raw_data_num is not an integer."""

    params = {
        "description": "Sample description",
        "data": {
            "params": {
                "raw_dataset_source": "raw_data_source",
                "pk_col_name": "id",
                "num_col_names": "num_col_name",
                "cat_col_names": "cat_col_name",
                "uci_raw_data_num": "none",
            }
        },
    }
    with pytest.raises(ValueError):
        Config._check_params(params)
