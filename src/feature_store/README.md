## Using Feast for Feature Store

Feature store is a centralized place to store and serve features for machine learning. [Feast](https://feast.dev/), an open source feature store for machine learning, is used in this project to manage and serve features consistently across offline training and online inference environments.

Feast can handle the ingestion of feature data from both batch and streaming sources. It also manages both warehouse and serving databases for historical and the latest data.

<!-- <figure>
<img
  src="./img/feast_workflow.png"
  alt="feast workflow"
  title="feast workflow"
  style="display: inline-block; margin: 0 auto; max-width: 600px">
<figcaption align = "center">Source: [Feast](https://feast.dev/)</figcaption>
</figure> -->

! [feast workflow] (…/…/img/feast_workflow.png)

## Data Transformation

Data transformation (preprocessing and feature engineering) is applied before ingesting it by the feature store as Feast does not support transformations natively and also to have consistent feature values across training and serving. The data preparation script (prep_data.py) transforms data into fresh features and target that are stored in local path (preprocessed_dataset_features.parquet and preprocessed_dataset_target.parquet). Note that raw and transformed data can be sourced and stored in a database that is compatible with Feast (see [Feast Data Sources](https://docs.feast.dev/reference/data-sources)). These files are then ingested by feature store and served for training or inference. Any changes needed for data preprocessing or transformation should be applied in prep_data.py script and prep.py utility module and then the feature definition can be changed in define_feature.py.

## Feature Store Setup

The feature store setup consists of the following steps:

- Define feature definitions in Python files under the `feature_repo` directory. Each feature definition specifies the name, data type, description, and source of the feature.

- Apply the feature definitions to the feature store using the `make init_feast` command. This command registers feature definition and create storage infrastructure (online_store.db in this project). Running this command returns the following ouput:

        Created entity patient
        Created feature view features_view
        Created feature view target_view

        Created sqlite table feature_store_features_view
        Created sqlite table feature_store_target_view

- The list of entities in the feature store registry can be shown using `make show_feast_entities`, which returns the following ouput:

        NAME     DESCRIPTION    TYPE
        patient  Patient ID     ValueType.STRING

- The views created in the feature store registry can be shown using `make show_feast_views`. The output should be:

        NAME           ENTITIES     TYPE
        features_view  {'patient'}  FeatureView
        target_view    {'patient'}  FeatureView

- The feature store can be explored using the web UI (experimental) by running the `make show_feast_ui` command. This will launch a browser window that shows the list of features, their statistics, and lineage.

- Build training datasets by specifying the features and entities of interest, and the time range and point-in-time correctness of the data. This can be done using the `get_historical_features` or `get_online_features` methods of the `FeatureStore` class in Python SDK.

- Existing feature store registry can be deleted by running the `make teardown_feast`.

Finally, to setup the feature store in one command, run the following command `make setup_feast`.
