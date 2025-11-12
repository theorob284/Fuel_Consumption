# baseline_algorithm.py
from __future__ import annotations
import os, glob
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

def run_baselines(input_folder: str | Path = "fuel_data_with_fuelrate", sample_frac = 0.10, svr_sample = 50_000, save_path: Optional[Path] = None,) -> Dict[str, Any]:

    input_folder = str(input_folder)
    files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No processed data found in '{input_folder}'")

    # Load and combine
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    data = pd.concat(dfs, ignore_index=True)

    # Column names (both variants supported)
    if "speed_kmh" in data.columns:
        speed_col = "speed_kmh"
    elif "TRACKS.MUNIC.SPEED_MERGED (km/h)" in data.columns:
        speed_col = "TRACKS.MUNIC.SPEED_MERGED (km/h)"
    else:
        raise KeyError("No speed column found.")

    if "fuel_ml_per_s" in data.columns:
        fuel_col = "fuel_ml_per_s"
    elif "TRACKS.MUNIC.FUEL_RATE (ml/s)" in data.columns:
        fuel_col = "TRACKS.MUNIC.FUEL_RATE (ml/s)"
    else:
        raise KeyError("No fuel-rate column found.")

    # Clean + light filter
    data = data.dropna(subset=[speed_col, fuel_col])
    data = data[(data[speed_col] > 5) & (data[fuel_col] > 0) & (data[fuel_col] < 10)]

    # Downsample for speed
    if 0 < sample_frac < 1:
        data = data.sample(frac=sample_frac, random_state=42)

    # Light smoothing to reduce sensor jitter
    data = data.sort_values(speed_col)
    data[speed_col] = data[speed_col].rolling(3, min_periods=1).mean()
    data[fuel_col]  = data[fuel_col].rolling(3, min_periods=1).mean()

    X = data[[speed_col]].values
    y = data[fuel_col].values

    # ---------- Linear ----------
    lin_reg = LinearRegression().fit(X, y)
    y_pred_lin = lin_reg.predict(X)

    # ---------- Polynomial (deg=3) ----------
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    poly_reg = LinearRegression().fit(X_poly, y)
    y_pred_poly = poly_reg.predict(X_poly)

    # ---------- SVR (RBF) on a subset ----------
    sample_for_svr = data.sample(n=min(svr_sample, len(data)), random_state=42)
    X_svr = sample_for_svr[[speed_col]].values
    y_svr = sample_for_svr[fuel_col].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_svr)
    y_scaled = scaler_y.fit_transform(y_svr.reshape(-1, 1)).ravel()

    svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.01).fit(X_scaled, y_scaled)
    y_pred_svr = scaler_y.inverse_transform(svr.predict(X_scaled).reshape(-1, 1)).ravel()

    # ---------- Metrics ----------
    def m(name, y_true, y_hat):
        return dict(name=name, r2=float(r2_score(y_true, y_hat)), mse=float(mean_squared_error(y_true, y_hat)))
    metrics = {"Linear": m("Linear", y, y_pred_lin), "Polynomial (deg=3)": m("Polynomial (deg=3)", y, y_pred_poly), "SVR (RBF, sample)": m("SVR (RBF, sample)", y_svr, y_pred_svr)}

    # ---------- Plot ----------
    speeds = np.linspace(data[speed_col].min(), data[speed_col].max(), 200).reshape(-1, 1)
    plt.figure(figsize=(10, 6))
    plt.scatter(data[speed_col], y, alpha=0.25, s=8, label="actual consumption")

    plt.plot(speeds, lin_reg.predict(speeds), color="red",   label="Linear")
    plt.plot(speeds, poly_reg.predict(poly.transform(speeds)), color="orange", label="Polynomial (degree 3)")
    plt.plot(speeds, scaler_y.inverse_transform(svr.predict(scaler_X.transform(speeds)).reshape(-1, 1)), color="green", label="SVR (RBF, sample)")

    plt.xlabel("speed [km/h]"), plt.ylabel("consumption [ml/s]"), plt.title("Comparison of different models"), plt.ylim(0, 10),plt.xlim(0, 160), plt.ylabel("consumption [ml/s]"), plt.legend(), plt.grid(True), plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    return {"speed_col": speed_col, "fuel_col": fuel_col, "metrics": metrics, "models": { "linear": lin_reg, "poly_features": poly,  "poly_reg": poly_reg, "svr_scaler_X": scaler_X, "svr_scaler_y": scaler_y, "svr": svr,},"plot_path": str(save_path) if save_path else None, "rows_used": int(len(data)), "svr_rows": int(len(sample_for_svr))}
