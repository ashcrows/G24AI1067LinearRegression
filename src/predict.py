#!/usr/bin/env python3
"""
Load trained LinearRegression model and run inference on California Housing test set. Print a few predictions.
"""

import logging
from joblib import load
from sklearn.datasets import fetch_california_housing
import numpy as np
from pathlib import Path


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def main():
    setup_logging()

    # Load model
    model_path = Path("artifacts") / "model.joblib"
    model = load(model_path)
    logging.info(f"Loaded model from: {model_path.resolve()}")

    # Load data
    X, y = fetch_california_housing(return_X_y=True)

    # Predict on 5 samples
    preds = model.predict(X[:5])
    logging.info("Sample Predictions:")
    for i, (pred, actual) in enumerate(zip(preds, y[:5])):
        logging.info(f"Sample {i+1}: Prediction = {pred:.2f}, Actual = {actual:.2f}")


if __name__ == "__main__":
    main()
