from sklearn.linear_model import LinearRegression
from utils import setup_logging, load_data, evaluate_model, save_model


def main():
    setup_logging()
    X, y = load_data()

    model = LinearRegression()
    model.fit(X, y)

    r2, mse = evaluate_model(model, X, y)
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")

    save_model(model)


if __name__ == "__main__":
    main()
