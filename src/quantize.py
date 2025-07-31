import numpy as np
from utils import load_model, save_params, get_file_size_kb, quantize_array, dequantize_array, quantize_array_16bit, dequantize_array_16bit, load_data
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

q16_coef, min16_c, scale16_c = quantize_array_16bit(model.coef_)
q16_int, min16_i, scale16_i = quantize_array_16bit(np.array([model.intercept_]))

quant16 = {
    "quant_coef": q16_coef,
    "coef_min": min16_c,
    "coef_scale": scale16_c,
    "quant_intercept": q16_int,
    "intercept_min": min16_i,
    "intercept_scale": scale16_i,
}
save_params(quant16, "quant_params_16bit.joblib")

deq16_coef = dequantize_array_16bit(q16_coef, min16_c, scale16_c)
deq16_int = dequantize_array_16bit(q16_int, min16_i, scale16_i)[0]

X_train, X_test, y_train, y_test = load_data()
original_preds = model.predict(X_test)
dequant_preds = np.dot(X_test, deq_coef) + deq_int
dequant16_preds = np.dot(X_test, deq16_coef) + deq16_int

r2_orig = r2_score(y_test, original_preds)
mse_orig = mean_squared_error(y_test, original_preds)
mae_orig = mean_absolute_error(y_test, original_preds)

r2_quant = r2_score(y_test, dequant_preds)
mse_quant = mean_squared_error(y_test, dequant_preds)
mae_quant = mean_absolute_error(y_test, dequant_preds)

max_error = np.max(np.abs(original_preds - dequant_preds))
mean_error = np.mean(np.abs(original_preds - dequant_preds))

r2_16 = r2_score(y_test, dequant16_preds)
mse_16 = mean_squared_error(y_test, dequant16_preds)
mae_16 = mean_absolute_error(y_test, dequant16_preds)

max_error_16 = np.max(np.abs(original_preds - dequant16_preds))
mean_error_16 = np.mean(np.abs(original_preds - dequant16_preds))

original_size = get_file_size_kb("model.joblib")
quantized_size = get_file_size_kb("quant_params.joblib")
quantized16_size = get_file_size_kb("quant_params_16bit.joblib")

print("\noriginal model loss:", round(mse_orig, 4))
print("original model accuracy (r²):", round(r2_orig, 4))
print("quantized model loss:", round(mse_quant, 4))
print("quantized model accuracy (r²):", round(r2_quant, 4))
print("16-bit quantized model loss:", round(mse_16, 4))
print("16-bit quantized model accuracy (r²):", round(r2_16, 4))

table = [
    ["r² score", f"{r2_orig:.4f}", f"{r2_quant:.4f}", f"{r2_16:.4f}"],
    ["mse (loss)", f"{mse_orig:.4f}", f"{mse_quant:.4f}", f"{mse_16:.4f}"],
    ["mae", f"{mae_orig:.4f}", f"{mae_quant:.4f}", f"{mae_16:.4f}"],
    ["max prediction error", "-", f"{max_error:.6f}", f"{max_error_16:.6f}"],
    ["mean prediction error", "-", f"{mean_error:.6f}", f"{mean_error_16:.6f}"],
    ["model size", f"{original_size} kb", f"{quantized_size} kb", f"{quantized16_size} kb"]
]

print("\nmodel accuracy and loss\n")
print(tabulate(table, headers=["metric", "original model", "8-bit quantized model", "16-bit quantized model"], tablefmt="github"))

print("\n" + "=" * 60)
print("checking parameter integrity")
print("=" * 60)

print(f"\noriginal coefficient shape        : {model.coef_.shape}")
print(f"original intercept value          : {model.intercept_:.6f}")

print("\noriginal coefficient values       :")
print(np.round(model.coef_, 6).tolist())

coef_error = np.max(np.abs(model.coef_ - deq_coef))
intercept_error = abs(model.intercept_ - deq_int)

coef_error_16 = np.max(np.abs(model.coef_ - deq16_coef))
intercept_error_16 = abs(model.intercept_ - deq16_int)

print(f"\ncoefficient max error (quantized 8-bit) : {coef_error:.8f}")
print(f"intercept error (quantized 8-bit)       : {intercept_error:.8f}")
print(f"coefficient max error (quantized 16-bit): {coef_error_16:.8f}")
print(f"intercept error (quantized 16-bit)      : {intercept_error_16:.8f}")

print(f"\nquantized coef preview 8-bit (first 5)  : {quant['quant_coef'][:5].tolist()}")
print(f"dequantized coef preview 8-bit (first 5): {np.round(deq_coef[:5], 6).tolist()}")
print(f"quantized coef preview 16-bit (first 5) : {quant16['quant_coef'][:5].tolist()}")
print(f"dequantized coef preview 16-bit (first 5): {np.round(deq16_coef[:5], 6).tolist()}")

print("\n" + "="*60)
print("verifying inference consistency on test data")
print("="*60)

print("\ntop 10 predictions comparison:\n")
print(f"{'Index':<6} {'sklearn':<15} {'manual-quantized(8bit)':<25} {'manual-dequantized(8bit)':<25} "
      f"{'manual-quantized(16bit)':<25} {'manual-dequantized(16bit)':<25} {'Err8':<12} {'Err16':<12}")
print("-" * 180)
for i in range(10):
    manual_quant_8bit = (np.dot(X_test[i], q_coef / scale_c + min_c) + (q_int / scale_i + min_i)).item()
    manual_dequant_8bit = dequant_preds[i].item() if isinstance(dequant_preds[i], np.ndarray) else dequant_preds[i]
    manual_quant_16bit = (np.dot(X_test[i], q16_coef / scale16_c + min16_c) + (q16_int / scale16_i + min16_i)).item()
    manual_dequant_16bit = dequant16_preds[i].item() if isinstance(dequant16_preds[i], np.ndarray) else dequant16_preds[i]

    err8 = abs(original_preds[i] - manual_dequant_8bit)
    err16 = abs(original_preds[i] - manual_dequant_16bit)

    print(f"{i:<6} {original_preds[i]:<15.6f} {manual_quant_8bit:<25.6f} {manual_dequant_8bit:<25.6f} "
          f"{manual_quant_16bit:<25.6f} {manual_dequant_16bit:<25.6f} {err8:<12.6f} {err16:<12.6f}")

print("\n" + "="*60)
print("summary")
print("="*60)
print("8-bit shows higher quantization error")
print("16-bit predictions are much closer to original sklearn outputs")
print("absolute error columns confirm 16-bit improves accuracy significantly\n")
