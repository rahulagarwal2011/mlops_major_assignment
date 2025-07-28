from utils import load_model, load_data, evaluate_predictions, print_predictions

if __name__ == "__main__":
    model = load_model()
    X_test, y_test = load_data(return_test_only=True)
    predictions = model.predict(X_test)
    evaluate_predictions(y_test, predictions)
    print_predictions(y_test, predictions, top_n=10)