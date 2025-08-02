#!/usr/bin/env python3
"""
Train a LinearRegression model on the California Housing dataset, print R² and MSE, and serialize the fitted model to disk.
"""

import logging
import os
from pathlib import Path

from joblib import dump
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_data():
    logging.info("Loading California Housing dataset")
    data = fetch_california_housing()
    return data.data, data.target


def train_model(X, y):
    logging.info("Training LinearRegression model")
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    logging.info("Evaluating model on training data")
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)
    logging.info(f"R2 score: {r2:.4f}")
    logging.info(f"Mean Squared Error: {mse:.4f}")
    return r2, mse


def save_model(model, output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    model_path = Path(output_dir) / "model.joblib"
    dump(model, model_path)
    logging.info(f"Model serialized to: {model_path.resolve()}")
    return model_path


def main():
    setup_logging()
    X, y = load_data()
    model = train_model(X, y)
    evaluate_model(model, X, y)
    save_model(model)


if __name__ == "__main__":
    main()
