"""
Extracts preprocessed data from feature store,i.e., features and
class labels, and creates data splits for model training.
"""

import argparse
import logging
import logging.config
import os
import sys
from pathlib import Path

from azureml.core import Dataset, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from dotenv import load_dotenv

# To import modules from the parent directory in Azure compute cluster
root_dir = Path(__name__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.training.utils.config import Config
from src.training.utils.helpers import (
    import_preprocessed_data,
    prepare_data_splits,
    split_data,
)
from src.utils.logger import get_console_logger

load_dotenv()


def main(
    config_yaml_path: str,
    logger: logging.Logger,
) -> None:
    """Splits dataset into train and test sets.

    Args:
        config_yaml_path (str): path to the config yaml file.
        logger (logging.Logger): logger object.
    """

    logger.info("Directory of training config file: %s", config_yaml_path)

    config = Config(config_path=config_yaml_path)
    train_data_name = config.params["azure_datasets"]["train_data_name"]
    train_data_desc = config.params["azure_datasets"]["train_data_desc"]
    train_data_tags = config.params["azure_datasets"]["train_data_tags"]

    validation_data_name = config.params["azure_datasets"]["validation_data_name"]
    validation_data_desc = config.params["azure_datasets"]["validation_data_desc"]
    validation_data_tags = config.params["azure_datasets"]["validation_data_tags"]

    test_data_name = config.params["azure_datasets"]["test_data_name"]
    test_data_desc = config.params["azure_datasets"]["test_data_desc"]
    test_data_tags = config.params["azure_datasets"]["test_data_tags"]

    # Connect to the training workspace
    sp_authentication = ServicePrincipalAuthentication(
        tenant_id=os.environ["TENANT_ID"],
        service_principal_id=os.environ["APP_REGISTRATION_ID"],
        service_principal_password=os.environ["SP_PWD"],
    )
    ws = Workspace(
        os.environ["SUBSCRIPTION_ID"],
        os.environ["RESOURCE_GROUP_NAME"],
        os.environ["AML_WORKSPACE_NAME"],
        auth=sp_authentication,
    )

    # Extract preprocessed data from feature store
    preprocessed_data = import_preprocessed_data(
        config_file_path=config_yaml_path,
        workspace=ws,
    )

    # Split preprocessed data into train and test sets
    training_set, testing_set = split_data(
        preprocessed_data=preprocessed_data,
        config_yaml_path=config_yaml_path,
    )

    # Prepare training and testing sets
    train_set, valid_set, test_set = prepare_data_splits(
        training_set=training_set,
        testing_set=testing_set,
        config_yaml_path=config_yaml_path,
    )

    # Register data splits in Azure ML workspace to reuse in model training
    Dataset.Tabular.register_pandas_dataframe(
        dataframe=train_set,
        target=ws.get_default_datastore(),
        name=train_data_name,
        description=train_data_desc,
        tags=train_data_tags,
        show_progress=True,
    )

    Dataset.Tabular.register_pandas_dataframe(
        dataframe=valid_set,
        target=ws.get_default_datastore(),
        name=validation_data_name,
        description=validation_data_desc,
        tags=validation_data_tags,
        show_progress=True,
    )

    Dataset.Tabular.register_pandas_dataframe(
        dataframe=test_set,
        target=ws.get_default_datastore(),
        name=test_data_name,
        description=test_data_desc,
        tags=test_data_tags,
        show_progress=True,
    )

    logger.info("Train, validation, and test sets created.")


###########################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml_path",
        type=str,
        default="./config.yml",
        help="Path to the configuration yaml file.",
    )
    parser.add_argument(
        "--logger_path",
        type=str,
        default="./logger.conf",
        help="Path to the logger configuration file.",
    )

    args = parser.parse_args()

    # Get the logger objects by name
    console_logger = get_console_logger("split_data_logger")

    console_logger.info(
        "Splitting Preprocessed Data into Train and Test Sets Starts ..."
    )

    main(
        config_yaml_path=args.config_yaml_path,
        logger=console_logger,
    )
