"""
ML model inference module for real-time and batch predictions.

This module provides both REST API and command-line interfaces for serving
trained machine learning models. Supports multiple experiment tracking
backends (MLflow, Comet ML) with fallback mechanisms for robust production
deployment.

Main Components:
    api_server.py: FastAPI REST service for real-time predictions
    predict.py: CLI tool for batch predictions from parquet files
    utils/: Model loading strategies and helper utilities
"""
