import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils import load_data


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_data_loading(data):
    X, y = data
    assert X.shape[0] > 0
    assert y.shape[0] > 0
    assert X.shape[0] == y.shape[0]


def test_model_training(data):
    X, y = data
    model = LinearRegression()
    model.fit(X, y)

    assert isinstance(model, LinearRegression)
    assert hasattr(model, "coef_")

    r2 = r2_score(y, model.predict(X))
    assert r2 > 0.5, f"Expected R² > 0.5, got {r2}"
