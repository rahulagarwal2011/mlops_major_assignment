import sys
import os
import joblib
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from train import load_data, train_model, evaluate_model

def test_data_loading():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Data not loaded properly"
    assert y_train.shape[0] > 0 and y_test.shape[0] > 0, "Labels not loaded properly"

def test_model_creation():
    model = LinearRegression()
    assert isinstance(model, LinearRegression), "Model is not an instance of LinearRegression"

def test_model_training_and_coef():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    assert hasattr(model, "coef_"), "Model does not have coefficients after training"
    assert model.coef_.shape[0] == X_train.shape[1], "Incorrect number of coefficients"

def test_model_r2_threshold():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    r2, _ = evaluate_model(model, X_test, y_test)
    assert r2 > 0.5, f"R² Score is too low: {r2:.4f}"
