from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt


# Column names (aligned with merge_speed.py defaults)
COL_SPEED_MERGED = "TRACKS.MUNIC.SPEED_MERGED (km/h)"
COL_SPEED_GPS = "TRACKS.MUNIC.GPS_SPEED (km/h)"
COL_SPEED_OBD = "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"
COL_FUEL_CUM_ML = "TRACKS.MUNIC.MDI_OBD_FUEL (ml)"
COL_TIME = "time"


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _compute_fuel_rate_ml_per_s(df: pd.DataFrame) -> pd.Series:
    if COL_FUEL_CUM_ML not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    time_s = _ensure_datetime(df[COL_TIME]).astype("int64") // 10**9
    fuel = pd.to_numeric(df[COL_FUEL_CUM_ML], errors="coerce")

    dt = pd.Series(np.diff(time_s, prepend=time_s.iloc[0]), index=df.index).astype(float)
    d_fuel = pd.Series(np.diff(fuel, prepend=fuel.iloc[0]), index=df.index).astype(float)

    # Reset detection: negative jumps -> set to NaN for that step
    d_fuel[d_fuel < 0] = np.nan

    # Avoid division by zero or tiny dt
    dt[dt <= 0] = np.nan
    rate = d_fuel / dt

    # Basic cleaning of outliers
    rate[(rate < 0) | (rate > 50.0)] = np.nan  # 50 ml/s is extremely high
    return rate


def _build_speed_features(df: pd.DataFrame) -> pd.DataFrame:
    if COL_SPEED_MERGED in df.columns:
        speed = pd.to_numeric(df[COL_SPEED_MERGED], errors="coerce")
    else:
        gps = pd.to_numeric(df.get(COL_SPEED_GPS), errors="coerce")
        obd = pd.to_numeric(df.get(COL_SPEED_OBD), errors="coerce")
        speed = gps.combine_first(obd)

    # If many NaNs, align by nearest timestamp within a small tolerance
    # 1) build a table of available speed observations
    try:
        time_dt = _ensure_datetime(df[COL_TIME])
        base = pd.DataFrame({"__time__": time_dt, "__speed__": speed})
        have_speed = base.dropna(subset=["__speed__"]).sort_values("__time__")
        if len(have_speed) > 0:
            need_speed = base[["__time__"]].sort_values("__time__")
            asof = pd.merge_asof(
                need_speed,
                have_speed[["__time__", "__speed__"]],
                on="__time__",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=2),
            )
            # fill missing with nearest timestamp speed when available
            nearest_speed = asof.set_index(base.index)["__speed__"]
            speed = speed.fillna(nearest_speed)
    except Exception:
        pass

    features = pd.DataFrame(index=df.index)
    features["speed_kmh"] = speed
    features["speed_ms"] = speed * (1000.0 / 3600.0)
    features["acc_ms2"] = features["speed_ms"].diff()
    features["speed_sq"] = np.square(features["speed_ms"])  # proxy for aerodynamic drag

    # Rolling statistics (use small windows assuming ~1 Hz; robust to NaNs)
    for w in (3, 5, 9):
        features[f"speed_ms_mean_{w}"] = features["speed_ms"].rolling(w, min_periods=1).mean()
        features[f"speed_ms_std_{w}"] = features["speed_ms"].rolling(w, min_periods=2).std()
        features[f"acc_ms2_mean_{w}"] = features["acc_ms2"].rolling(w, min_periods=1).mean()

    return features


def _prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df = df.sort_values(COL_TIME)

    y = _compute_fuel_rate_ml_per_s(df)
    X = _build_speed_features(df)

    data = pd.concat([X, y.rename("fuel_rate_ml_s")], axis=1)
    data = data.dropna(subset=["fuel_rate_ml_s"])  # require target
    data = data.dropna(axis=0, how="any")  # drop rows with feature NaNs (simple baseline)

    y_clean = data["fuel_rate_ml_s"]
    X_clean = data.drop(columns=["fuel_rate_ml_s"])  # keep feature names for interpretability
    return X_clean, y_clean


def _downsample_stride(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    if stride <= 1:
        return df
    return df.iloc[::stride].reset_index(drop=True)


def _read_csv_with_source(path: Path) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, low_memory=False)
    df[COL_TIME] = _ensure_datetime(df[COL_TIME])
    source = path.stem
    return df, source


def load_concatenated_dataset(input_dir: Path, stride: int = 1) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.csv"))
    frames: List[pd.DataFrame] = []
    groups: List[str] = []

    for f in files:
        try:
            df, source = _read_csv_with_source(f)
        except Exception:
            continue

        df = _downsample_stride(df.sort_values(COL_TIME), stride)
        X, y = _prepare_dataset(df)
        if len(X) == 0:
            continue

        X["__source__"] = source
        frames.append(pd.concat([X, y.rename("__target__")], axis=1))
        groups.extend([source] * len(X))

    if not frames:
        raise RuntimeError("No valid data loaded. Check columns and input directory.")

    data_all = pd.concat(frames, axis=0, ignore_index=True)
    y_all = data_all.pop("__target__")
    groups_array = np.array(groups)
    X_all = data_all.drop(columns=["__source__"])  # remove helper column
    return X_all, y_all, groups_array


def _prepare_dataset_with_time(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = df.copy()
    df = df.sort_values(COL_TIME)

    y = _compute_fuel_rate_ml_per_s(df)
    X = _build_speed_features(df)
    t = _ensure_datetime(df[COL_TIME])

    data = pd.concat([t.rename("__time__"), X, y.rename("fuel_rate_ml_s")], axis=1)
    data = data.dropna(subset=["fuel_rate_ml_s"])  # require target
    data = data.dropna(axis=0, how="any")

    t_clean = data.pop("__time__")
    y_clean = data.pop("fuel_rate_ml_s")
    X_clean = data
    return X_clean, y_clean, t_clean


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> Dict[str, Dict[str, float]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RF": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = float(r2_score(y_test, pred))
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    return results


def run_speed_only_experiment(
    input_dir: Path = Path("fuel_data_speed_merged"),
    strides: Optional[List[int]] = None,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    if strides is None:
        strides = [1, 2, 5, 10]

    summary: Dict[int, Dict[str, Dict[str, float]]] = {}
    for s in strides:
        X, y, groups = load_concatenated_dataset(input_dir, stride=s)
        results = train_and_evaluate_models(X, y, groups)
        summary[s] = results
    return summary


def plot_summary(summary: Dict[int, Dict[str, Dict[str, float]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    strides = sorted(summary.keys())
    models = sorted(next(iter(summary.values())).keys())

    metrics = ["MAE", "RMSE", "R2"]
    for metric in metrics:
        plt.figure(figsize=(7, 4))
        for m in models:
            ys = [summary[s][m][metric] for s in strides]
            plt.plot(strides, ys, marker="o", label=m)
        plt.xlabel("Stride")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Stride")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = output_dir / f"summary_{metric.lower()}_vs_stride.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def main():
    input_dir = Path("fuel_data_speed_merged")
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir.resolve()}")
        return

    strides = [1, 2, 5, 10]
    summary = run_speed_only_experiment(input_dir=input_dir, strides=strides)

    print("\nSpeed-only models (fuel_rate ml/s) with stride downsampling:")
    for s in strides:
        print(f"\nStride = {s}")
        for model_name, metrics in summary[s].items():
            mae = metrics["MAE"]
            rmse = metrics["RMSE"]
            r2 = metrics["R2"]
            print(f"  {model_name:6s} | MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

    # Save plots
    plot_summary(summary, output_dir=Path("figures"))


if __name__ == "__main__":
    main()


