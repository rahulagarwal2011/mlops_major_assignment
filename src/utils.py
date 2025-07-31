import joblib
import numpy as np
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error   
from tabulate import tabulate

def load_data(return_test_only=False, test_size=0.2, seed=42):
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=seed
    )
    if return_test_only:
        return X_test, y_test
    return X_train, X_test, y_train, y_test


def load_model(path="model.joblib"):
    print("\n" + "=" * 60)
    print("loading trained model and test data")
    print("=" * 60)
    model = joblib.load(path)
    print(f"model loaded successfully from '{path}'")
    return model

def save_model(model, path="model.joblib"):
    joblib.dump(model, path)
    print(f"model saved to {path}")

def evaluate_predictions(y_true, y_pred, top_n=10):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print("=" * 60)
    print("making predictions and evaluating performance")
    print("=" * 60)
    print(f"r² score                        : {r2:.4f}")
    print(f"mean squared error              : {mse:.4f}")
    print(f"mean absolute error             : {mae:.4f}")
    print(f"mean absolute percentage error  : {mape:.4f}")
    print(f"total test samples              : {len(y_true)}")
    return r2, mse

def print_predictions(y_true, y_pred, top_n=10):
    print("\n" + "=" * 60)
    print(f"first {top_n} rows predictions")
    print("=" * 60)
    print(f"{'index':<6} {'actual':<10} {'predicted':<12} {'error'}")
    print("-" * 45)
    for i in range(top_n):
        actual = round(y_true[i], 2)
        pred = round(y_pred[i], 2)
        diff = round(abs(actual - pred), 2)
        print(f"{i:<6} {actual:<10} {pred:<12} {diff}")

def save_params(params, filename):
    joblib.dump(params, filename)
    print(f"saved to {filename}")

def get_file_size_kb(path):
    return round(os.path.getsize(path) / 1024, 1)

def quantize_array(arr):
    minimum = arr.min()
    scale = 255 / (arr.max() - minimum + 1e-8)
    quant = ((arr - minimum) * scale).astype(np.uint8)
    return quant, minimum, scale

def dequantize_array(quant, minimum, scale):
    return quant.astype(np.float32) / scale + minimum

def quantize_array_16bit(arr):
    minimum = arr.min()
    scale = 65535 / (arr.max() - minimum + 1e-8)
    quant = ((arr - minimum) * scale).astype(np.uint16)
    return quant, minimum, scale

def dequantize_array_16bit(quant, minimum, scale):
    return quant.astype(np.float32) / scale + minimum

def print_metric_table(metrics):
    print("\\nmodel accuracy and loss\\n")
    print(tabulate(metrics, headers=["metric", "original model", "quantized model"], tablefmt="github"))