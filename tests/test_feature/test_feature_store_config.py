"""
Test functions for the Config class in the feature store
module src/feature/schemas.py.
"""

import pytest

from src.feature.schemas import Config
from src.utils.path import PARENT_DIR


@pytest.fixture(scope="module")
def config_fixture():
    """Fixture to provide a valid Config instance and its path."""
    config_path = f"{str(PARENT_DIR)}/config/feature-store-config.yml"
    config = Config(config_path)
    return config, config_path


def test_config_init(config_fixture):  # pylint: disable=redefined-outer-name
    """Tests if Config class init method correctly initializes the config path."""
    config, config_path = config_fixture
    assert config.config_path == config_path


def test_config_init_file_not_found():
    """Tests if Config class init method raises FileNotFoundError when the config file doesn't exist."""
    incorrect_config_path = f"{str(PARENT_DIR)}/config/nonexistent_config.yml"
    with pytest.raises(FileNotFoundError):
        Config(incorrect_config_path)


def test_config_check_params_valid(
    config_fixture,
):  # pylint: disable=redefined-outer-name
    """Tests if Config class check_params method doesn't raise any exceptions when all required values are present."""
    config, _ = config_fixture
    # This should not raise any exceptions
    config.check_params()


def test_config_check_params_missing_description():
    """Tests if Config class check_params method raises ValueError when 'description' is missing."""
    config = Config(config_path=f"{str(PARENT_DIR)}/config/feature-store-config.yml")
    config.params.pop("description", None)  # Remove 'description' key
    with pytest.raises(KeyError, match="description is not included in config file"):
        config.check_params()


def test_config_check_params_missing_data():
    """Tests if Config class check_params method raises ValueError when 'data' is missing."""
    config = Config(config_path=f"{str(PARENT_DIR)}/config/feature-store-config.yml")
    config.params.pop("data", None)  # Remove 'data' key
    with pytest.raises(KeyError, match="data is not included in config file"):
        config.check_params()
