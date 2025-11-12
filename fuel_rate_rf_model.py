# fuel_rate_rf_model.py
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time

# column names
SPEED_COL  = "TRACKS.MUNIC.SPEED_MERGED (km/h)"
TARGET_COL = "TRACKS.MUNIC.FUEL_RATE (ml/s)"
RPM_COL_CANDIDATES = ["TRACKS.MUNIC.MDI_OBD_RPM (rpm)", "RPM", "TRACKS.MUNIC.RPM", "MDI_RPM"]

def _build_features(df: pd.DataFrame):
    #Builds predictive features for fuel rate modelling
    df = df.copy()

    # Core numeric columns
    df[SPEED_COL] = pd.to_numeric(df[SPEED_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    # Sort by time to preserve sequence
    df = df.sort_values("time").reset_index(drop=True)

    # Compute acceleration (delta speed / delta time)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["dt"] = df["time"].diff().dt.total_seconds().fillna(0).replace(0, np.nan)
    df["acceleration"] = df[SPEED_COL].diff() / df["dt"]
    df["acceleration"].fillna(0, inplace=True)

    # Rolling statistics for smoothing (3-sec window)
    df["speed_roll"] = df[SPEED_COL].rolling(window=3, min_periods=1).mean()
    df["acc_roll"] = df["acceleration"].rolling(window=3, min_periods=1).mean()

    # NEW — engineered features
    df["speed_sq"] = df[SPEED_COL] ** 2
    df["accel_sq"] = df["acceleration"] ** 2
    df["speed_change"] = df[SPEED_COL].diff().fillna(0)
    df["speed_x_acc"] = df[SPEED_COL] * df["acceleration"]

    # Drop NaN target rows
    df = df.dropna(subset=[TARGET_COL])
    feature_cols = [SPEED_COL,"acceleration","speed_roll","acc_roll","speed_sq","accel_sq","speed_change","speed_x_acc",]
    return df, feature_cols

def _temporal_split(df: pd.DataFrame, features: list[str], test_size=0.2):
    n = len(df)
    cut = int((1 - test_size) * n)
    X_train = df.iloc[:cut][features]
    y_train = df.iloc[:cut][TARGET_COL]
    X_test  = df.iloc[cut:][features]
    y_test  = df.iloc[cut:][TARGET_COL]
    return X_train, X_test, y_train, y_test

def train_rf(input_csv: Path,model_out: Path,plots: bool = True,n_estimators: int = 100,random_state: int = 42 ):
    df = pd.read_csv(input_csv, low_memory=False)
    df, feature_cols = _build_features(df)

    X_train, X_test, y_train, y_test = _temporal_split(df, feature_cols, test_size=0.2)
    # downcast to reduce RAM & speed
    X_train = X_train.astype("float32")
    X_test  = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test  = y_test.astype("float32")

    # training values
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,  # switch to -1 if you have plenty of RAM
        random_state=random_state,
        verbose=1,
    )

    print(f"\nTraining RF on {len(X_train):,} rows / {len(feature_cols)} features with {n_estimators} trees…")
    t0 = time.time()
    rf.fit(X_train, y_train)
    print(f"Training complete in {time.time() - t0:.1f}s")

    # Evaluation
    y_pred = rf.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Always save into models/ directory
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_out = models_dir / Path(model_out).name
    joblib.dump({"model": rf, "features": feature_cols}, model_out)
    print(f"Model saved to: {model_out}")

    #build prediction for curve
    def _curve_features(speed_vals_kmh: np.ndarray) -> pd.DataFrame:
        cur = pd.DataFrame({SPEED_COL: speed_vals_kmh})
        cur["acceleration"] = 0.0
        cur["speed_roll"] = cur[SPEED_COL]
        cur["acc_roll"] = 0.0
        cur["speed_sq"] = cur[SPEED_COL] ** 2
        # ensure exact order/columns model expects
        return cur.reindex(columns=feature_cols, fill_value=0.0).astype("float32")

    # overlay graph plot
    speed_range = np.arange(0, 151).astype(float)  # 0 -> 150 inclusive
    pred_curve = rf.predict(_curve_features(speed_range))

    # real data: keep within 0..150, bin by 1 km/h and smooth lightly
    plot_df = df[[SPEED_COL, TARGET_COL]].dropna().copy()
    plot_df["speed_bin"] = plot_df[SPEED_COL].round().astype(int)
    plot_df = plot_df[(plot_df["speed_bin"] >= 0) & (plot_df["speed_bin"] <= 150)] # ------------------ CHANGE HERE TO LIMIT SPEED USED (MAKE SURE TO LIMITE PLOTTED AS WELL (LINE 150)

    real_avg = (plot_df.groupby("speed_bin")[TARGET_COL].mean().reset_index().sort_values("speed_bin"))
    # 3-point centered rolling mean for a clean line
    real_avg["smooth"] = real_avg[TARGET_COL].rolling(3, center=True, min_periods=1).mean()
    plt.figure(figsize=(12, 6))
    # raw scatter, smooth average + model pred curve
    plt.scatter(plot_df[SPEED_COL], plot_df[TARGET_COL], s=5, alpha=0.02, color="steelblue", label="Raw data")
    plt.plot(real_avg["speed_bin"], real_avg["smooth"],color="tab:red", linewidth=2, label="Real average (smoothed)")
    plt.plot(speed_range, pred_curve,color="orange", linewidth=2, label="Model prediction")

    # labels / limits / legend
    plt.xlabel("Speed (km/h)"), plt.ylabel("Fuel Rate (ml/s)"), plt.title("Fuel vs Speed — Real Data vs Model")
    plt.xlim(0, 150) # ---------------- CHENGE HERE TO LIMITE SPEED SHOWN ON GRAPH
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left")

    overlay_path = model_out.parent / f"{model_out.stem}_overlay.png"
    plt.savefig(overlay_path, dpi=300)
    plt.close()
    print(f"Overlay curve saved to: {overlay_path}")

    # ---------------- RETURN SUMMARY ----------------
    return {"features": feature_cols,"r2": r2,"mae": mae,"rmse": rmse,"importances": importances,"model_path": str(model_out),"overlay_path": str(overlay_path),"n_train": len(X_train),"n_test": len(X_test),}