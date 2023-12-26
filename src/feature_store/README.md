- To initiate feature store:

    cd /workspaces/end-to-end-ml/src/feature_store/feature_repo  # Navigate to root

    feast apply  # Registers feature definition and create storage infrastructure.

Output:

    Created entity patient
    Created feature view features_view
    Created feature view target_view

    Created sqlite table feature_store_features_view
    Created sqlite table feature_store_target_view

- To list entities in registry:

    feast entities list

output: 

    NAME     DESCRIPTION    TYPE
    patient  Patient ID     ValueType.UNKNOWN

- To view views in registry:
    feast feature-views list

output:

    NAME           ENTITIES     TYPE
    features_view  {'patient'}  FeatureView
    target_view    {'patient'}  FeatureView


- To delete existing feature store registry:

    feast teardown