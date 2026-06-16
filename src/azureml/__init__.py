"""Azure ML execution backend (optional).

Submits this project's training and evaluation as Azure ML jobs, tracks them in
the workspace via MLflow, and serves features from an Azure ML managed feature
store. Additive to the local pipeline and gated behind config; importing this
package requires the `azure` optional dependencies (see pyproject.toml). The
local Feast / file-store path does not import anything here.
"""
