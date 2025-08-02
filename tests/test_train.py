import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.datasets import fetch_california_housing


@pytest.fixture(scope="module")
def data():
    X, y = fetch_california_housing(return_X_y=True)
    return X, y


def test_data_loading(data):
    X, y = data
    assert X.shape[0] > 0, "Feature data should not be empty"
    assert y.shape[0] > 0, "Target data should not be empty"
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"


def test_model_training(data):
    X, y = data
    model = LinearRegression()
    model.fit(X, y)

    # Check model instance
    assert isinstance(model, LinearRegression), "Model should be LinearRegression instance"

    # Check model is trained
    assert hasattr(model, "coef_"), "Trained model must have coef_ attribute"

    # Check R2 score
    r2 = r2_score(y, model.predict(X))
    print(f"R2 score: {r2}")
    assert r2 > 0.5, "R2 score should be greater than 0.5"
