# G24AI1067LinearRegression
# California Housing Regression MLOps Pipeline

This repository contains an end-to-end MLOps pipeline for training, quantizing,
and deploying a LinearRegression model on the California Housing dataset.

## Contents

- `src/train.py`: trains the model, prints R² & loss, serializes to `model.joblib`  
- `src/quantize.py`: extracts & quantizes parameters, serializes to `unquant_params.joblib` & `quant_params.joblib`  
- `src/predict.py`: loads a model and prints sample predictions  
- `src/utils.py`: shared load/save and metric functions  
- `tests/test_train.py`: unit tests for data loading & model training  
- `Dockerfile`: containerizes the pipeline  
- `.github/workflows/ci.yml`: GitHub Actions CI/CD  

## Getting Started

1. Create & activate a virtual environment  
2. `pip install -r requirements.txt`  
3. `pytest`  
4. `python src/train.py`  
5. `python src/quantize.py`  
6. Build & run the Docker image to smoke-test
