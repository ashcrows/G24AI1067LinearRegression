import logging
import os
from pathlib import Path
import numpy as np
from joblib import dump, load
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def load_data():
    """Fetch California Housing dataset"""
    return fetch_california_housing(return_X_y=True)


def evaluate_model(model, X, y):
    """Return R2 and MSE"""
    preds = model.predict(X)
    return r2_score(y, preds), mean_squared_error(y, preds)


def save_model(model, path="artifacts/model.joblib"):
    os.makedirs(Path(path).parent, exist_ok=True)
    dump(model, path)
    logging.info(f"Model saved at {Path(path).resolve()}")


def load_model(path="artifacts/model.joblib"):
    model = load(path)
    logging.info(f"Model loaded from {Path(path).resolve()}")
    return model


def save_dict(data: dict, filename: str):
    """Save dictionary to artifacts directory"""
    os.makedirs("artifacts", exist_ok=True)
    path = Path("artifacts") / filename
    dump(data, path)
    logging.info(f"Saved: {path.resolve()}")
