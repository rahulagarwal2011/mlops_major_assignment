from utils import load_data, save_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from tabulate import tabulate

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    table = [
        ["r² score", f"{r2:.4f}"],
        ["mean squared error", f"{mse:.4f}"]
    ]
    print(tabulate(table, headers=["metric", "value"], tablefmt="github"))
    return r2, mse

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    r2, mse = evaluate_model(model, X_test, y_test)
    save_model(model)