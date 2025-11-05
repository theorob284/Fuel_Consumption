# fuel_rate_rf_model.py
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ----- column names -----
SPEED_COL  = "TRACKS.MUNIC.SPEED_MERGED (km/h)"
TARGET_COL = "TRACKS.MUNIC.FUEL_RATE (ml/s)"
RPM_COL_CANDIDATES = [
    "TRACKS.MUNIC.MDI_OBD_RPM (rpm)", "RPM", "TRACKS.MUNIC.RPM", "MDI_RPM"
]

# ----- defaults (can be overridden by caller) -----
ROLL_WIN = 5
N_ESTIMATORS = 400
RANDOM_STATE = 42

def _find_rpm_column(df: pd.DataFrame):
    for c in RPM_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Keep only needed columns plus optional time/RPM
    keep = [SPEED_COL, TARGET_COL] + [c for c in df.columns if c.lower() == "time" or c in RPM_COL_CANDIDATES]
    df = df[keep].copy().dropna(subset=[SPEED_COL, TARGET_COL])

    # Sort by time if available
    time_col = next((c for c in df.columns if c.lower() == "time"), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col)
        dt = df[time_col].diff().dt.total_seconds().clip(lower=1)
    else:
        dt = pd.Series(1.0, index=df.index)

    # Numeric
    df[SPEED_COL]  = pd.to_numeric(df[SPEED_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Engineered features
    df["acceleration"] = df[SPEED_COL].diff() / dt
    df["speed_roll"]   = df[SPEED_COL].rolling(ROLL_WIN, min_periods=1, center=True).mean()
    df["acc_roll"]     = df["acceleration"].rolling(ROLL_WIN, min_periods=1, center=True).mean()
    df["speed_sq"]     = df[SPEED_COL] ** 2

    rpm_col = _find_rpm_column(df)
    feats = [SPEED_COL, "acceleration", "speed_roll", "acc_roll", "speed_sq"]

    if rpm_col:
        df[rpm_col] = pd.to_numeric(df[rpm_col], errors="coerce")
        df["rpm"] = df[rpm_col]
        df["rpm_speed_ratio"] = df["rpm"] / df[SPEED_COL].replace(0, np.nan)
        df["rpm_speed_ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
        feats += ["rpm", "rpm_speed_ratio"]

    df = df.dropna(subset=feats + [TARGET_COL])

    return df, feats

def _temporal_split(df: pd.DataFrame, features: list[str], test_size=0.2):
    n = len(df)
    cut = int((1 - test_size) * n)
    X_train = df.iloc[:cut][features]
    y_train = df.iloc[:cut][TARGET_COL]
    X_test  = df.iloc[cut:][features]
    y_test  = df.iloc[cut:][TARGET_COL]
    return X_train, X_test, y_train, y_test

def train_rf(
    input_csv: Path,
    model_out: Path,
    plots: bool = True,
    n_estimators: int = N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
):
    """Train RF on the given CSV and save model. Returns metrics dict."""
    df = pd.read_csv(input_csv, low_memory=False)
    df, feature_cols = _build_features(df)

    X_train, X_test, y_train, y_test = _temporal_split(df, feature_cols, test_size=0.2)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # metrics (RMSE computed manually for sklearn compat)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)

    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # save model
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "features": feature_cols}, model_out)

    if plots:
        # Predicted vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, s=8, alpha=0.5)
        plt.xlabel("Actual fuel rate (ml/s)")
        plt.ylabel("Predicted fuel rate (ml/s)")
        plt.title("Random Forest — Predicted vs Actual")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

        # Residuals
        plt.figure(figsize=(8, 5))
        residuals = y_test - y_pred
        plt.hist(residuals, bins=40)
        plt.xlabel("Residual (Actual - Predicted) ml/s")
        plt.ylabel("Count")
        plt.title("Residuals Distribution")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    summary = {
        "features": feature_cols,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "importances": importances,
        "model_path": str(model_out),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return summary
