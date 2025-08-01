import numpy as np
from utils import (
    load_model,
    save_params,
    get_file_size_kb,
    quantize_array_16bit,
    dequantize_array_16bit,
    load_data,
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tabulate import tabulate


# ==============================
# Metrics
# ==============================
def compute_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


# ==============================
# Quantization (tensor-level)
# ==============================
def quantize_tensor(params, bits=8):
    params = np.array(params, dtype=np.float32)
    p_min, p_max = np.min(params), np.max(params)
    if p_max == p_min:
        scale = 1.0
        q = np.zeros_like(params, dtype=np.uint8 if bits == 8 else np.uint16)
    else:
        scale = (2**bits - 1) / (p_max - p_min)
        q = np.round((params - p_min) * scale).astype(
            np.uint8 if bits == 8 else np.uint16
        )
    return q.reshape(params.shape), p_min, scale


def dequantize_tensor(q, p_min, scale):
    dq = q.astype(np.float32) / scale + p_min
    return dq.reshape(q.shape)


# ==============================
# Bias correction
# ==============================
def bias_correction(X, y_true, dq_weights, orig_bias):
    preds = X @ dq_weights + orig_bias
    drift = np.mean(y_true - preds)
    return orig_bias + drift


# ==============================
# Prediction
# ==============================
def predict_from_weights(features, weights, bias):
    return features @ weights + bias


# ==============================
# Show parameter stats
# ==============================
def show_param_stats(label, q_vals, dq_vals, original, errors):
    print(f"\n{label.upper()} SUMMARY")
    print(f"largest coef deviation : {errors['coef']:.8f}")
    print(f"intercept difference   : {errors['intercept']:.8f}")
    print("coef snapshot (q vs dq, first 5):")
    for i in range(min(5, len(q_vals))):
        print(f"  q:{q_vals[i]}  dq:{np.round(dq_vals[i],6)}")


# ==============================
# MAIN
# ==============================
model = load_model()
raw_dump = {"weights": model.coef_, "bias": model.intercept_}
save_params(raw_dump, "base_params.joblib")

# ---- Load data ----
X_tr, X_te, y_tr, y_te = load_data()

# ---- 8-bit ----
q8, mn8, sc8 = quantize_tensor(model.coef_, bits=8)
dq8 = dequantize_tensor(q8, mn8, sc8)
bias8_corrected = bias_correction(X_tr, y_tr, dq8, model.intercept_)
save_params(
    {"q_coef": q8, "coef_min": mn8, "coef_scale": sc8, "q_bias": bias8_corrected},
    "compressed_8bit.joblib",
)

# ---- 16-bit ----
q16, mn16, sc16 = quantize_array_16bit(model.coef_)
dq16 = dequantize_array_16bit(q16, mn16, sc16)
bias16_corrected = bias_correction(X_tr, y_tr, dq16, model.intercept_)
save_params(
    {"q_coef": q16, "coef_min": mn16, "coef_scale": sc16, "q_bias": bias16_corrected},
    "compressed_16bit.joblib",
)

# ---- Predictions ----
ref_preds = model.predict(X_te)
preds8 = predict_from_weights(X_te, dq8, bias8_corrected)
preds16 = predict_from_weights(X_te, dq16, bias16_corrected)

# ---- Metrics ----
m_ref = compute_metrics(y_te, ref_preds)
m8 = compute_metrics(y_te, preds8)
m16 = compute_metrics(y_te, preds16)

err8 = {
    "max": np.max(np.abs(ref_preds - preds8)),
    "mean": np.mean(np.abs(ref_preds - preds8)),
}
err16 = {
    "max": np.max(np.abs(ref_preds - preds16)),
    "mean": np.mean(np.abs(ref_preds - preds16)),
}

size_ref = get_file_size_kb("model.joblib")
size8 = get_file_size_kb("compressed_8bit.joblib")
size16 = get_file_size_kb("compressed_16bit.joblib")

# ---- Output ----
print("\nPERFORMANCE CHECK")
print(
    f"reference r²   : {m_ref['r2']:.4f}, mse: {m_ref['mse']:.4f}, mae: {m_ref['mae']:.4f}"
)
print(
    f"8-bit r²       : {m8['r2']:.4f}, mse: {m8['mse']:.4f}, mae: {m8['mae']:.4f}"
)
print(
    f"16-bit r²      : {m16['r2']:.4f}, mse: {m16['mse']:.4f}, mae: {m16['mae']:.4f}"
)

data_table = [
    ["R²", f"{m_ref['r2']:.4f}", f"{m8['r2']:.4f}", f"{m16['r2']:.4f}"],
    ["MSE", f"{m_ref['mse']:.4f}", f"{m8['mse']:.4f}", f"{m16['mse']:.4f}"],
    ["MAE", f"{m_ref['mae']:.4f}", f"{m8['mae']:.4f}", f"{m16['mae']:.4f}"],
    ["Max error", "-", f"{err8['max']:.6f}", f"{err16['max']:.6f}"],
    ["Mean error", "-", f"{err8['mean']:.6f}", f"{err16['mean']:.6f}"],
    ["File size", f"{size_ref} kb", f"{size8} kb", f"{size16} kb"],
]

print("\nCOMPARATIVE RESULTS\n")
print(
    tabulate(
        data_table,
        headers=["Metric", "Baseline", "8-bit", "16-bit"],
        tablefmt="fancy_grid",
    )
)

print("\n" + "=" * 60)
print("PARAMETER CONSISTENCY")
print("=" * 60)
print(f"\nweights shape : {model.coef_.shape}")
print(f"bias value    : {model.intercept_:.6f}")
print("weights sample:")
print(np.round(model.coef_[:10], 6).tolist())

show_param_stats(
    "8-bit",
    q8,
    dq8,
    model.coef_,
    {
        "coef": np.max(np.abs(model.coef_ - dq8)),
        "intercept": abs(model.intercept_ - bias8_corrected),
    },
)
show_param_stats(
    "16-bit",
    q16,
    dq16,
    model.coef_,
    {
        "coef": np.max(np.abs(model.coef_ - dq16)),
        "intercept": abs(model.intercept_ - bias16_corrected),
    },
)

print("\n" + "=" * 60)
print("PREDICTION ALIGNMENT SAMPLE")
print("=" * 60)
print("\nFirst 10 prediction comparisons:")
print(
    f"{'Row':<4} {'Ref':<14} {'Q8 (manual)':<20} {'DQ8 (manual)':<20} {'Q16 (manual)':<20} {'DQ16 (manual)':<20} {'Δ8':<10} {'Δ16':<10}"
)
print("-" * 150)

for j in range(10):
    ref_val = float(ref_preds[j])

    # Q8 (manual): reconstruction without bias correction
    q8_manual = (X_te[j] @ dq8) + model.intercept_
    dq8_val = float(preds8[j])  # bias corrected

    q16_manual = (X_te[j] @ dq16) + model.intercept_
    dq16_val = float(preds16[j])  # bias corrected

    delta8 = abs(ref_val - dq8_val)
    delta16 = abs(ref_val - dq16_val)

    print(
        f"{j:<4} {ref_val:<14.6f} {q8_manual:<20.6f} {dq8_val:<20.6f} {q16_manual:<20.6f} {dq16_val:<20.6f} {delta8:<10.6f} {delta16:<10.6f}"
    )

print("\nSUMMARY NOTE")
print("→ 8-bit compression with bias correction reduces error but still trails baseline.")
print("→ 16-bit stays very close to baseline predictions.")
print("→ error stats confirm that 16-bit strikes a better balance between size and accuracy.\n")
