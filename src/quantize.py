#!/usr/bin/env python3
"""
Quantize a trained LinearRegression model's parameters, store both raw and quantized values, and perform test inference using de-quantized weights.
"""

import logging
import numpy as np
import os
from pathlib import Path
from joblib import load, dump
from sklearn.datasets import fetch_california_housing


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )


def quantize_to_uint8(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scale = (max_val - min_val) / 255 if max_val != min_val else 1e-6
    quantized = np.round((array - min_val) / scale).astype(np.uint8)
    return quantized, scale, min_val


def dequantize_uint8(q_array, scale, min_val):
    return q_array.astype(np.float32) * scale + min_val


def save_parameters(data, filename):
    os.makedirs("artifacts", exist_ok=True)
    path = Path("artifacts") / filename
    dump(data, path)
    logging.info(f"Saved: {path.resolve()}")


def main():
    setup_logging()

    # Load model
    model_path = Path("artifacts") / "model.joblib"
    model = load(model_path)
    logging.info(f"Loaded trained model from {model_path}")

    # Extract parameters
    coef = model.coef_
    intercept = model.intercept_

    raw_params = {
        "coef": coef,
        "intercept": intercept
    }
    save_parameters(raw_params, "unquant_params.joblib")

    # Quantize coef and intercept
    q_coef, coef_scale, coef_min = quantize_to_uint8(coef)
    q_intercept, int_scale, int_min = quantize_to_uint8(np.array([intercept]))

    quant_params = {
        "coef": q_coef,
        "intercept": q_intercept,
        "scales": {"coef": coef_scale, "intercept": int_scale},
        "mins": {"coef": coef_min, "intercept": int_min}
    }
    save_parameters(quant_params, "quant_params.joblib")

    # Load test data
    X, y = fetch_california_housing(return_X_y=True)

    # Dequantize and infer
    deq_coef = dequantize_uint8(q_coef, coef_scale, coef_min)
    deq_intercept = dequantize_uint8(q_intercept, int_scale, int_min)[0]

    predictions = np.dot(X, deq_coef) + deq_intercept

    # Compute basic R2 for verification
    from sklearn.metrics import r2_score
    r2 = r2_score(y, predictions)
    logging.info(f"R2 score using dequantized weights: {r2:.4f}")


if __name__ == "__main__":
    main()
