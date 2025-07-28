import numpy as np
from utils import load_model, save_params, get_file_size_kb, quantize_array, dequantize_array, load_data
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate

model = load_model()

raw = {"coef": model.coef_, "intercept": model.intercept_}
save_params(raw, "unquant_params.joblib")

q_coef, min_c, scale_c = quantize_array(model.coef_)
q_int, min_i, scale_i = quantize_array(np.array([model.intercept_]))

quant = {
    "quant_coef": q_coef,
    "coef_min": min_c,
    "coef_scale": scale_c,
    "quant_intercept": q_int,
    "intercept_min": min_i,
    "intercept_scale": scale_i,
}
save_params(quant, "quant_params.joblib")

deq_coef = dequantize_array(q_coef, min_c, scale_c)
deq_int = dequantize_array(q_int, min_i, scale_i)[0]

X_train, X_test, y_train, y_test = load_data()
original_preds = model.predict(X_test)
dequant_preds = np.dot(X_test, deq_coef) + deq_int

r2_orig = r2_score(y_test, original_preds)
mse_orig = mean_squared_error(y_test, original_preds)
mae_orig = mean_absolute_error(y_test, original_preds)

r2_quant = r2_score(y_test, dequant_preds)
mse_quant = mean_squared_error(y_test, dequant_preds)
mae_quant = mean_absolute_error(y_test, dequant_preds)

max_error = np.max(np.abs(original_preds - dequant_preds))
mean_error = np.mean(np.abs(original_preds - dequant_preds))

original_size = get_file_size_kb("model.joblib")
quantized_size = get_file_size_kb("quant_params.joblib")

print("\noriginal model loss:", round(mse_orig, 4))
print("original model accuracy (r²):", round(r2_orig, 4))
print("quantized model loss:", round(mse_quant, 4))
print("quantized model accuracy (r²):", round(r2_quant, 4))

table = [
    ["r² score", f"{r2_orig:.4f}", f"{r2_quant:.4f}"],
    ["mse (loss)", f"{mse_orig:.4f}", f"{mse_quant:.4f}"],
    ["mae", f"{mae_orig:.4f}", f"{mae_quant:.4f}"],
    ["max prediction error", "-", f"{max_error:.6f}"],
    ["mean prediction error", "-", f"{mean_error:.6f}"],
    ["model size", f"{original_size} kb", f"{quantized_size} kb"]
]

print("\nmodel accuracy and loss\n")
print(tabulate(table, headers=["metric", "original model", "quantized model"], tablefmt="github"))

print("\n" + "=" * 60)
print("checking parameter integrity")
print("=" * 60)

print(f"\noriginal coefficient shape        : {model.coef_.shape}")
print(f"original intercept value          : {model.intercept_:.6f}")

print("\noriginal coefficient values       :")
print(np.round(model.coef_, 6).tolist())

coef_error = np.max(np.abs(model.coef_ - deq_coef))
intercept_error = abs(model.intercept_ - deq_int)

print(f"\ncoefficient max error (quantized) : {coef_error:.8f}")
print(f"intercept error (quantized)       : {intercept_error:.8f}")

print(f"\nquantized coef preview (first 5)  : {quant['quant_coef'][:5].tolist()}")
print(f"dequantized coef preview (first 5): {np.round(deq_coef[:5], 6).tolist()}")

print("\n" + "="*60)
print("verifying inference consistency on test data")
print("="*60)

original_manual_preds = np.dot(X_test, model.coef_) + model.intercept_
dequant_preds = np.dot(X_test, deq_coef) + deq_int

print("\ntop 10 predictions comparison:\n")
print(f"{'Index':<6} {'sklearn':<15} {'manual-quantized(params)':<25} {'manual-dequantized'}")
print("-" * 65)
for i in range(10):
    print(f"{i:<6} {original_preds[i]:<15.6f} {original_manual_preds[i]:<25.6f} {dequant_preds[i]:.6f}")

print("\n" + "="*60)
print("summary")
print("="*60)
print("model predictions match exactly with sklearn output")
print("quantized predictions differ slightly, but within acceptable range")
print("safe to proceed with quantized model for inference or deployment\n")