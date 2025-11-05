from pathlib import Path
import pandas as pd
import numpy as np

# ======== SETTINGS ========
TIME_COL  = "time"
FUEL_COL  = "TRACKS.MUNIC.MDI_OBD_FUEL (ml)"
RATE_COL  = "TRACKS.MUNIC.FUEL_RATE (ml/s)"

TARGET_INTERVAL = 60
TOLERANCE = 5
MIN_INTERVAL = TARGET_INTERVAL - TOLERANCE
MAX_INTERVAL = TARGET_INTERVAL + TOLERANCE

OUTLIER_METHOD = "cap"
PERCENTILE_FOR_CAP = 0.99
ABS_MAX_CAP = 10.0
ABS_PHYS_MAX = 50.0
# ===========================================


def _compute_precise_pairs(df: pd.DataFrame) -> pd.Series:
    """Compute ml/s only between directly adjacent 1-minute-apart readings."""
    df["_t"] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[FUEL_COL] = pd.to_numeric(df[FUEL_COL], errors="coerce")

    fuel_pts = df[df[FUEL_COL].notna()][["_t", FUEL_COL]].reset_index(drop=True)
    fuel_pts["time_diff_s"] = fuel_pts["_t"].diff().dt.total_seconds()
    fuel_pts["fuel_diff_ml"] = fuel_pts[FUEL_COL].diff()
    fuel_pts["segment_rate"] = fuel_pts["fuel_diff_ml"] / fuel_pts["time_diff_s"]

    valid_mask = fuel_pts["time_diff_s"].between(MIN_INTERVAL, MAX_INTERVAL, inclusive="both")
    valid_segments = fuel_pts.loc[valid_mask, ["_t", "segment_rate"]].reset_index(drop=True)

    rate = pd.Series(index=df.index, dtype="float64")

    for i in range(len(valid_segments)):
        cur_time = valid_segments.iloc[i]["_t"]
        pos = fuel_pts.index[fuel_pts["_t"] == cur_time][0]
        if pos == 0:
            continue
        prev_time = fuel_pts.iloc[pos - 1]["_t"]
        rate_value = valid_segments.iloc[i]["segment_rate"]

        mask = (df["_t"] >= prev_time) & (df["_t"] < cur_time)
        rate.loc[mask] = rate_value

    rate = pd.to_numeric(rate, errors="coerce")
    rate[(rate < 0) | (rate > ABS_PHYS_MAX)] = np.nan
    return rate


def _fix_outliers(rate: pd.Series) -> pd.Series:
    """Cap or interpolate extreme rate values."""
    r = rate.copy()
    finite = r[np.isfinite(r)]
    if len(finite) == 0:
        return r

    if OUTLIER_METHOD == "cap":
        cap_value = min(float(finite.quantile(PERCENTILE_FOR_CAP)), ABS_MAX_CAP)
        r[r > cap_value] = cap_value
        return r

    if OUTLIER_METHOD == "interpolate":
        r[(r < 0) | (r > ABS_MAX_CAP) | ~np.isfinite(r)] = np.nan
        r = r.interpolate(method="linear", limit_direction="both")
        return r

    return r


def add_fuel_rate(input_csv_path: Path, output_csv_path: Path):
    """Adds TRACKS.MUNIC.FUEL_RATE (ml/s) and removes rows without it."""
    df = pd.read_csv(input_csv_path, low_memory=False)

    for c in (TIME_COL, FUEL_COL):
        if c not in df.columns:
            raise KeyError(f"Missing required column '{c}' in {input_csv_path.name}")

    raw_rate = _compute_precise_pairs(df)
    cleaned_rate = _fix_outliers(raw_rate)
    df[RATE_COL] = cleaned_rate

    # Remove rows without valid rate
    df = df[df[RATE_COL].notna()].copy()
    df.drop(columns=["_t"], inplace=True, errors="ignore")

    # Save to file
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    finite = df[RATE_COL].replace([np.inf, -np.inf], np.nan).dropna()
    pct99 = float(finite.quantile(0.99)) if len(finite) else float("nan")
    print(f"✅ Saved cleaned file: {output_csv_path.name}")
    print(f"   Rows kept: {len(df)} | Mean: {finite.mean():.2f} ml/s | "
          f"Median: {finite.median():.2f} | Max: {finite.max():.2f} | p99: {pct99:.2f}")
    print(f"   Only directly adjacent fuel pairs {MIN_INTERVAL}–{MAX_INTERVAL} s apart were used.")
