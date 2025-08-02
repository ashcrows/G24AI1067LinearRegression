from utils import setup_logging, load_data, load_model


def main():
    setup_logging()
    model = load_model()
    X, y = load_data()

    preds = model.predict(X[:5])
    print("Sample Predictions:")
    for i, (p, actual) in enumerate(zip(preds, y[:5])):
        print(f"Sample {i+1}: Prediction = {p:.2f}, Actual = {actual:.2f}")


if __name__ == "__main__":
    main()
