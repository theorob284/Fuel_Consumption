from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse helpers from training pipeline
from fuel_cons_algorithm import (
    COL_TIME,
    COL_SPEED_MERGED,
    COL_SPEED_GPS,
    COL_SPEED_OBD,
    _ensure_datetime,
    _downsample_stride,
    _prepare_dataset_with_time,
    _build_speed_features,
    _compute_fuel_rate_ml_per_s,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


def _train_test_time_split(X: pd.DataFrame, y: pd.Series, t: pd.Series, test_ratio: float = 0.2, use_all: bool = False) -> Tuple:
    n = len(X)
    if n == 0:
        raise RuntimeError("No samples after preprocessing for plotting.")
    if use_all:
        # use entire sequence as "test" part for visualization
        return (
            X.iloc[0:0], X,
            y.iloc[0:0], y,
            t.iloc[0:0], t,
        )
    split = max(1, int(n * (1 - test_ratio)))
    return (
        X.iloc[:split], X.iloc[split:],
        y.iloc[:split], y.iloc[split:],
        t.iloc[:split], t.iloc[split:],
    )


def plot_timeseries(
    input_csv: Path,
    stride: int = 1,
    smooth_window: int = 15,
    output_dir: Path = Path("figures"),
    test_ratio: float = 0.2,
    use_all: bool = False,
    scatter_alpha: float = 0.5,
    scatter_size: int = 6,
) -> Path:
    input_csv = Path(input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    df[COL_TIME] = _ensure_datetime(df[COL_TIME])
    df = _downsample_stride(df.sort_values(COL_TIME), stride)

    X, y, t = _prepare_dataset_with_time(df)
    X_train, X_test, y_train, y_test, t_train, t_test = _train_test_time_split(X, y, t, test_ratio=test_ratio, use_all=use_all)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RF": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    preds: Dict[str, np.ndarray] = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds[name] = mdl.predict(X_test)

    # Smooth ground truth for a clean reference line
    y_test_smooth = pd.Series(y_test.values).rolling(smooth_window, min_periods=1).mean().values

    # Plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    # Scatter of ground truth on test region
    plt.scatter(t_test, y_test, s=scatter_size, alpha=scatter_alpha, label="Truth (points)")
    # Smoothed truth line
    plt.plot(t_test, y_test_smooth, color="black", linewidth=1.5, label=f"Truth MA({smooth_window})")
    # Model lines
    for name, yhat in preds.items():
        plt.plot(t_test, yhat, linewidth=1.5, label=name)

    plt.xlabel("Time")
    plt.ylabel("Fuel rate (ml/s)")
    plt.title(f"Fuel rate vs Time | {input_csv.stem} | stride={stride}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = output_dir / f"timeseries_{input_csv.stem}_stride{stride}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _process_for_overlay(input_csv: Path, stride: int = 1, test_ratio: float = 0.2, use_all: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    df = pd.read_csv(input_csv, low_memory=False)
    if COL_TIME not in df.columns:
        raise RuntimeError(f"missing column '{COL_TIME}'")
    df[COL_TIME] = _ensure_datetime(df[COL_TIME])
    df = _downsample_stride(df.sort_values(COL_TIME), stride)

    # Try build dataset; if empty, emit diagnostics
    try:
        X, y, t = _prepare_dataset_with_time(df)
    except Exception as e:
        raise RuntimeError(f"prepare failed: {e}")

    if len(X) == 0 or len(y) == 0:
        cols = set(map(str, df.columns))
        hints = []
        if 'TRACKS.MUNIC.SPEED_MERGED (km/h)' not in cols and (
            'TRACKS.MUNIC.GPS_SPEED (km/h)' not in cols and 'TRACKS.MUNIC.MDI_OBD_SPEED (km/h)' not in cols
        ):
            hints.append('no speed columns')
        if 'TRACKS.MUNIC.MDI_OBD_FUEL (ml)' not in cols:
            hints.append('no cumulative fuel column')
        # basic non-na counts
        y_raw = pd.to_numeric(df.get('TRACKS.MUNIC.MDI_OBD_FUEL (ml)'), errors='coerce')
        speed_raw = None
        if 'TRACKS.MUNIC.SPEED_MERGED (km/h)' in cols:
            speed_raw = pd.to_numeric(df['TRACKS.MUNIC.SPEED_MERGED (km/h)'], errors='coerce')
        else:
            gps = pd.to_numeric(df.get('TRACKS.MUNIC.GPS_SPEED (km/h)'), errors='coerce')
            obd = pd.to_numeric(df.get('TRACKS.MUNIC.MDI_OBD_SPEED (km/h)'), errors='coerce')
            if gps is not None or obd is not None:
                speed_raw = (gps if gps is not None else pd.Series(dtype=float)).combine_first(
                    obd if obd is not None else pd.Series(dtype=float)
                )
        if y_raw is not None:
            hints.append(f"fuel_non_na={int(y_raw.notna().sum())}")
        if speed_raw is not None:
            hints.append(f"speed_non_na={int(pd.Series(speed_raw).notna().sum())}")
        raise RuntimeError("empty after preprocessing; " + ", ".join(hints))
    X_train, X_test, y_train, y_test, t_train, t_test = _train_test_time_split(X, y, t, test_ratio=test_ratio, use_all=use_all)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "RF": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    preds: Dict[str, np.ndarray] = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds[name] = mdl.predict(X_test)

    t_rel_s = (t_test - t_test.iloc[0]).dt.total_seconds().to_numpy()
    return t_rel_s, y_test.to_numpy(), preds


def plot_overlay(
    inputs: List[Path],
    stride: int = 1,
    output_dir: Path = Path("figures"),
    title: str = "Overlay: Fuel rate vs Relative Time",
    bin_seconds: int = 1,
    test_ratio: float = 0.2,
    use_all: bool = False,
    scatter_alpha: float = 0.15,
    scatter_size: int = 5,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all series
    all_t: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_preds: Dict[str, List[np.ndarray]] = {"Linear": [], "Ridge": [], "RF": []}

    for p in inputs:
        try:
            t_rel, y_test, preds = _process_for_overlay(p, stride=stride, test_ratio=test_ratio, use_all=use_all)
            all_t.append(t_rel)
            all_y.append(y_test)
            for k in all_preds.keys():
                all_preds[k].append(preds[k])
        except Exception:
            continue

    if not all_t:
        raise RuntimeError("No valid series for overlay plot.")

    # Scatter all truth points
    plt.figure(figsize=(11, 5))
    for t_rel, y_test in zip(all_t, all_y):
        plt.scatter(t_rel, y_test, s=scatter_size, alpha=scatter_alpha, color="#1f77b4")

    # Build mean lines by integer-second bins
    def mean_line(series_list: List[np.ndarray], t_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        bins: Dict[int, List[float]] = {}
        for t_rel, vals in zip(t_list, series_list):
            if len(t_rel) != len(vals):
                m = min(len(t_rel), len(vals))
                t_rel = t_rel[:m]
                vals = vals[:m]
            secs = (np.round(t_rel / bin_seconds) * bin_seconds).astype(int)
            for s, v in zip(secs, vals):
                if np.isfinite(v):
                    bins.setdefault(int(s), []).append(float(v))
        if not bins:
            return np.array([]), np.array([])
        xs = np.array(sorted(bins.keys()))
        ys = np.array([np.mean(bins[x]) for x in xs])
        return xs, ys

    # Truth mean line
    xs_truth, ys_truth = mean_line(all_y, all_t)
    if len(xs_truth) > 0:
        plt.plot(xs_truth, ys_truth, color="black", linewidth=2.0, label="Truth mean")

    # Model mean lines
    colors = {"Linear": "#ff7f0e", "Ridge": "#2ca02c", "RF": "#d62728"}
    for name, series_list in all_preds.items():
        xs_m, ys_m = mean_line(series_list, all_t)
        if len(xs_m) > 0:
            plt.plot(xs_m, ys_m, linewidth=1.8, label=f"{name} mean", color=colors.get(name))

    plt.xlabel("Relative time (s)")
    plt.ylabel("Fuel rate (ml/s)")
    plt.title(title + f" | stride={stride}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = output_dir / f"timeseries_overlay_stride{stride}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_speed_curve(
    input_csv: Path,
    stride: int = 1,
    output_dir: Path = Path("figures"),
    use_all: bool = True,
) -> Path:
    """Scatter: speed_kmh vs fuel_rate, with Linear/Ridge/RF curves."""
    df = pd.read_csv(input_csv, low_memory=False)
    if COL_TIME not in df.columns:
        raise RuntimeError(f"missing column '{COL_TIME}'")
    df[COL_TIME] = _ensure_datetime(df[COL_TIME])
    df = _downsample_stride(df.sort_values(COL_TIME), stride)

    # Build only what's needed: speed_kmh and fuel_rate_ml_s
    feats = _build_speed_features(df)
    y_rate = _compute_fuel_rate_ml_per_s(df)
    data = pd.concat([feats[["speed_kmh"]], y_rate.rename("fuel_rate_ml_s")], axis=1)
    data = data.dropna(subset=["speed_kmh", "fuel_rate_ml_s"])  # keep intersection only
    if len(data) == 0:
        # Diagnostics to help user
        cols = set(map(str, df.columns))
        hints = []
        if COL_SPEED_MERGED not in cols and (COL_SPEED_GPS not in cols and COL_SPEED_OBD not in cols):
            hints.append('no speed columns')
        if 'TRACKS.MUNIC.MDI_OBD_FUEL (ml)' not in cols:
            hints.append('no cumulative fuel column')
        raise RuntimeError("empty after preprocessing (speed/fuel no overlap); " + ", ".join(hints))

    speed = data[["speed_kmh"]].to_numpy()
    target = data["fuel_rate_ml_s"].to_numpy()

    # Fit models: Linear, Ridge, RandomForest on 1-D speed
    lin_reg = LinearRegression().fit(speed, target)
    ridge = Ridge(alpha=1.0, random_state=42).fit(speed, target)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(speed, target)

    # Domain for curves
    s_min, s_max = np.nanmin(speed), np.nanmax(speed)
    speeds = np.linspace(s_min, s_max, 200).reshape(-1, 1)

    # Predictions
    y_lin = lin_reg.predict(speeds)
    y_ridge = ridge.predict(speeds)
    y_rf = rf.predict(speeds)

    # Plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(speed, target, alpha=0.3, s=10, label="Truth")
    plt.plot(speeds, y_lin, color="red", linewidth=2.5, zorder=3, label="Linear")
    plt.plot(speeds, y_ridge, color="orange", linestyle="--", linewidth=2.0, zorder=2, label="Ridge")
    plt.plot(speeds, y_rf, color="green", linewidth=2.0, zorder=1, label="RF")
    plt.xlabel("Speed [km/h]")
    plt.ylabel("Fuel rate [ml/s]")
    plt.title(f"Speed vs Fuel rate | {input_csv.stem} | stride={stride}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = output_dir / f"speed_curve_{input_csv.stem}_stride{stride}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_speed_curve_overlay(
    inputs: List[Path],
    stride: int = 1,
    output_dir: Path = Path("figures"),
    scatter_alpha: float = 0.25,
    scatter_size: int = 6,
) -> Path:
    """Combine many CSVs on one scatter (speed_kmh vs fuel_rate) and fit global Linear/Ridge/RF curves."""
    all_rows: List[pd.DataFrame] = []
    for p in inputs:
        try:
            df = pd.read_csv(p, low_memory=False)
            if COL_TIME not in df.columns:
                continue
            df[COL_TIME] = _ensure_datetime(df[COL_TIME])
            df = _downsample_stride(df.sort_values(COL_TIME), stride)
            feats = _build_speed_features(df)
            y_rate = _compute_fuel_rate_ml_per_s(df)
            data = pd.concat([feats[["speed_kmh"]], y_rate.rename("fuel_rate_ml_s")], axis=1)
            data = data.dropna(subset=["speed_kmh", "fuel_rate_ml_s"])  # keep valid pairs
            if len(data) > 0:
                all_rows.append(data)
        except Exception:
            continue

    if not all_rows:
        raise RuntimeError("No valid points across inputs for speed-curve overlay.")

    data_all = pd.concat(all_rows, axis=0, ignore_index=True)
    speed = data_all[["speed_kmh"]].to_numpy()
    target = data_all["fuel_rate_ml_s"].to_numpy()

    # Fit models
    lin_reg = LinearRegression().fit(speed, target)
    ridge = Ridge(alpha=1.0, random_state=42).fit(speed, target)
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1).fit(speed, target)

    s_min, s_max = np.nanmin(speed), np.nanmax(speed)
    speeds = np.linspace(s_min, s_max, 200).reshape(-1, 1)
    y_lin = lin_reg.predict(speeds)
    y_ridge = ridge.predict(speeds)
    y_rf = rf.predict(speeds)

    # Plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.scatter(speed, target, alpha=scatter_alpha, s=scatter_size, label="Truth (all files)")
    plt.plot(speeds, y_lin, color="red", linewidth=2.5, zorder=3, label="Linear")
    plt.plot(speeds, y_ridge, color="orange", linestyle="--", linewidth=2.0, zorder=2, label="Ridge")
    plt.plot(speeds, y_rf, color="green", linewidth=2.0, zorder=1, label="RF")
    plt.xlabel("Speed [km/h]")
    plt.ylabel("Fuel rate [ml/s]")
    plt.title(f"Speed vs Fuel rate | overlay | stride={stride}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = output_dir / f"speed_curve_overlay_stride{stride}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot time series predictions for CSV file(s) or a directory")
    parser.add_argument("inputs", nargs="+", type=str, help="One or more CSV files, or a directory to process all *_merged.csv")
    parser.add_argument("--stride", type=int, default=1, help="Downsample stride (default 1)")
    parser.add_argument("--smooth_window", type=int, default=15, help="Moving average window for truth line")
    parser.add_argument("--out", type=str, default="figures", help="Output directory for figures")
    parser.add_argument("--overlay", action="store_true", help="If set, plot all inputs on a single figure with mean lines")
    parser.add_argument("--speed_curve", action="store_true", help="If set, draw speed vs fuel rate with model curves for a single CSV")
    parser.add_argument("--speed_curve_overlay", action="store_true", help="If set, combine many CSVs into one speed-fuel scatter with global model curves")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio for split (unused if --use_all)")
    parser.add_argument("--use_all", action="store_true", help="Use the whole series for plotting (no train/test split)")
    parser.add_argument("--bin_seconds", type=int, default=1, help="Bin size (seconds) when computing mean lines in overlay")
    parser.add_argument("--scatter_alpha", type=float, default=0.15, help="Scatter alpha for truth points")
    parser.add_argument("--scatter_size", type=int, default=5, help="Scatter size for truth points")
    args = parser.parse_args()

    inputs: list[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            inputs.extend(sorted(p.glob("*_merged.csv")))
        else:
            inputs.append(p)

    if not inputs:
        raise SystemExit("No input CSVs found.")

    if args.speed_curve_overlay:
        out_path = plot_speed_curve_overlay(
            inputs=inputs,
            stride=args.stride,
            output_dir=Path(args.out),
            scatter_alpha=args.scatter_alpha,
            scatter_size=args.scatter_size,
        )
        print(f"Saved speed-curve-overlay: {out_path}")
    elif args.speed_curve:
        if len(inputs) != 1:
            raise SystemExit("--speed_curve expects exactly one input CSV")
        out_path = plot_speed_curve(
            input_csv=inputs[0],
            stride=args.stride,
            output_dir=Path(args.out),
            use_all=args.use_all,
        )
        print(f"Saved speed-curve: {out_path}")
    elif args.overlay and len(inputs) >= 1:
        try:
            out_path = plot_overlay(
                inputs=inputs,
                stride=args.stride,
                output_dir=Path(args.out),
                bin_seconds=args.bin_seconds,
                test_ratio=args.test_ratio,
                use_all=args.use_all,
                scatter_alpha=args.scatter_alpha,
                scatter_size=args.scatter_size,
            )
            print(f"Saved overlay: {out_path}")
        except Exception as e:
            print(f"Overlay failed: {e}")
    else:
        for inp in inputs:
            try:
                out_path = plot_timeseries(
                    input_csv=inp,
                    stride=args.stride,
                    smooth_window=args.smooth_window,
                    output_dir=Path(args.out),
                    test_ratio=args.test_ratio,
                    use_all=args.use_all,
                    scatter_alpha=args.scatter_alpha,
                    scatter_size=args.scatter_size,
                )
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Skip {inp}: {e}")


if __name__ == "__main__":
    main()


