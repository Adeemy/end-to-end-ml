"""
Preprocesses and transforms raw data before saving it in the feature store.
"""

import argparse
import logging
import logging.config
import os

from azureml.core import Dataset, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from dotenv import load_dotenv

from src.feature_store.utils.config import Config
from src.feature_store.utils.helpers import (
    import_raw_data,
    preprocess_data,
    transform_data,
)
from src.utils.logger import create_console_logger

# import sys
# sys.path.append("/workspaces/end-to-end-ml/")


load_dotenv()


#################################
def main(config_yaml_path: str, logger: logging.Logger) -> None:
    """Imports raw data to be preprocessed and transformed.

    Args:
        config_yaml_path (str): path to the config yaml file.
        logger (logging.Logger): logger object.
    """

    logger.info(
        "Directory of data perprocessing and transformation config file: %s",
        config_yaml_path,
    )

    # Get configuration parameters
    config = Config(config_path=config_yaml_path)

    preprocessed_data_name = config.params["azure_datasets"]["preprocessed_data_name"]
    preprocessed_data_desc = config.params["azure_datasets"]["preprocessed_data_desc"]
    preprocessed_data_tags = config.params["azure_datasets"]["preprocessed_data_tags"]

    # Connect to the training workspace
    # sp_authentication = ServicePrincipalAuthentication(
    #     tenant_id=os.environ["TENANT_ID"],
    #     service_principal_id=os.environ["APP_REGISTRATION_ID"],
    #     service_principal_password=os.environ["SP_PWD"],
    # )
    # ws = Workspace(
    #     os.environ["SUBSCRIPTION_ID"],
    #     os.environ["RESOURCE_GROUP_NAME"],
    #     os.environ["AML_WORKSPACE_NAME"],
    #     auth=sp_authentication,
    # )
    ws = Workspace.from_config()

    #################################
    raw_dataset = import_raw_data(config_yaml_path=config_yaml_path, ws=ws)
    logger.info("Raw dataset was imported.")

    # Apply required preprocessing on raw dataset
    # Note: preprocessing and transofmration stepd applied
    # here include mapping values and defining column data
    # types, i.e., doesn't cause data leakage. Hence,
    # transformed dataset can be split into train and test sets.
    preprocessed_dataset, data_preprocessor = preprocess_data(
        raw_dataset=raw_dataset, config_yaml_path=config_yaml_path
    )
    logger.info("Raw dataset was preprocessed.")

    transformed_data = transform_data(
        preprocessed_dataset=preprocessed_dataset,
        data_preprocessor=data_preprocessor,
        config_yaml_path=config_yaml_path,
    )
    logger.info("Preprocessed dataset was transformed.")

    Dataset.Tabular.register_pandas_dataframe(
        dataframe=transformed_data,
        target=ws.get_default_datastore(),
        name=preprocessed_data_name,
        description=preprocessed_data_desc,
        tags=preprocessed_data_tags,
        show_progress=True,
    )

    logger.info("Prepared dataset was registered in Azure ML workspace %s.", ws.name)


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
    console_logger = create_console_logger("prep_data_logger")

    console_logger.info("Transforming for Feature Store Starts ...")

    main(config_yaml_path=args.config_yaml_path, logger=console_logger)
