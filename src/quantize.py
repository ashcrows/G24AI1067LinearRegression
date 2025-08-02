import numpy as np
from utils import setup_logging, load_data, load_model, evaluate_model, save_dict
from sklearn.metrics import r2_score


def quantize_to_uint8(array):
    min_val = np.min(array)
    max_val = np.max(array)
    scale = (max_val - min_val) / 255 if max_val != min_val else 1e-6
    quantized = np.round((array - min_val) / scale).astype(np.uint8)
    return quantized, scale, min_val


def dequantize_uint8(q_array, scale, min_val):
    return q_array.astype(np.float32) * scale + min_val


def main():
    setup_logging()

    model = load_model()
    coef = model.coef_
    intercept = model.intercept_

    # Save unquantized parameters
    raw_params = {"coef": coef, "intercept": intercept}
    save_dict(raw_params, "unquant_params.joblib")

    # Quantize
    q_coef, coef_scale, coef_min = quantize_to_uint8(coef)
    q_intercept, int_scale, int_min = quantize_to_uint8(np.array([intercept]))

    quant_params = {
        "coef": q_coef,
        "intercept": q_intercept,
        "scales": {"coef": coef_scale, "intercept": int_scale},
        "mins": {"coef": coef_min, "intercept": int_min}
    }
    save_dict(quant_params, "quant_params.joblib")

    # Dequantized inference
    X, y = load_data()
    deq_coef = dequantize_uint8(q_coef, coef_scale, coef_min)
    deq_intercept = dequantize_uint8(q_intercept, int_scale, int_min)[0]

    preds = np.dot(X, deq_coef) + deq_intercept
    r2 = r2_score(y, preds)
    print(f"R2 score using dequantized weights: {r2:.4f}")

if __name__ == "__main__":
    main()
