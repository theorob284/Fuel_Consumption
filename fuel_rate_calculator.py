from pathlib import Path
import pandas as pd
import numpy as np

INPUT_DIR  = Path("fuel_data_speed_merged")    
OUTPUT_DIR = Path("fuel_data_with_fuelrate")   

# Column names
TIME_COL_PRIMARY   = "time"
TIME_COL_FALLBACKS = ["received_at"]

SPEED_COL = "TRACKS.MUNIC.SPEED_MERGED (km/h)"    
FUEL_CUM_COL = "TRACKS.MUNIC.MDI_OBD_FUEL (ml)"      
RATE_COL  = "TRACKS.MUNIC.FUEL_RATE (ml/s)"    

TARGET_INTERVAL = 60
TOLERANCE = 5
MIN_INTERVAL = TARGET_INTERVAL - TOLERANCE
MAX_INTERVAL = TARGET_INTERVAL + TOLERANCE

# Outlier handling (cap after computing)
PERCENTILE_FOR_CAP = 0.99
ABS_MAX_CAP = 10.0  
ABS_PHYS_MAX = 50.0   

def _pick_time_column(df: pd.DataFrame) -> str:
    if TIME_COL_PRIMARY in df.columns:
        return TIME_COL_PRIMARY
    for c in TIME_COL_FALLBACKS:
        if c in df.columns:
            return c
    raise KeyError(f"No time-like column found. Expected '{TIME_COL_PRIMARY}' or one of {TIME_COL_FALLBACKS}.")

def _compute_adjacent_minute_rates(df: pd.DataFrame, time_col: str, weight_power: float = 1.0,  ) -> pd.Series: # float 1 → proportional to speed; try 2 to stress aero drag 
    # Parse time / cumulative fuel / speed
    t = pd.to_datetime(df[time_col], errors="coerce")
    f = pd.to_numeric(df[FUEL_CUM_COL], errors="coerce")
    s = pd.to_numeric(df[SPEED_COL], errors="coerce").clip(lower=0).fillna(0)

    fuel_pts = (pd.DataFrame({"_idx": df.index, "_t": t, "_f": f}).dropna(subset=["_t", "_f"]).reset_index(drop=True))
    if len(fuel_pts) < 2:
        return pd.Series(index=df.index, dtype="float64")

    # Time / fuel diffs between consecutive valid points
    fuel_pts["dt"] = fuel_pts["_t"].diff().dt.total_seconds()
    fuel_pts["dfuel"] = fuel_pts["_f"].diff()
    fuel_pts["r_seg"] = fuel_pts["dfuel"] / fuel_pts["dt"]  # ml/s
    # Keep only ~1 minute segments
    valid = fuel_pts["dt"].between(MIN_INTERVAL, MAX_INTERVAL, inclusive="both")
    valid_idx = valid[valid].index

    out = pd.Series(index=df.index, dtype="float64")

    for i in valid_idx:
        # segment is (i-1) → i
        if i == 0:
            continue
        prev_t = fuel_pts.loc[i - 1, "_t"]
        cur_t  = fuel_pts.loc[i, "_t"]
        seg_dt = float(fuel_pts.loc[i, "dt"])          # seconds
        seg_r  = float(fuel_pts.loc[i, "r_seg"])       # ml/s
        if not np.isfinite(seg_dt) or seg_dt <= 0 or not np.isfinite(seg_r):
            continue

        # rows inside this segment [prev_t, cur_t)
        mask = (t >= prev_t) & (t < cur_t)
        if not mask.any():
            continue
        idx = df.index[mask]

        # per-row durations dt_i (last row reaches the segment end)
        times = t.loc[idx]
        dt = (times.shift(-1) - times).dt.total_seconds().fillna(0)
        # extend the last row to segment end
        last_extra = (cur_t - times.iloc[-1]).total_seconds()
        dt.iloc[-1] = max(0.0, last_extra)

        total_dt = float(dt.sum())
        if total_dt <= 0:
            out.loc[idx] = seg_r  # fallback: constant within segment
            continue
    
        weights = (s.loc[idx].to_numpy() ** weight_power) * dt.to_numpy()
        sum_w = float(np.sum(weights))
        if sum_w <= 0:
            out.loc[idx] = seg_r
            continue

        # total fuel for the segment and per-row instantaneous rate
        F = seg_r * seg_dt  # ml
        with np.errstate(divide='ignore', invalid='ignore'):
            ml_per_s = (F * (weights / sum_w)) / dt.to_numpy()

        # assign
        out.loc[idx] = ml_per_s

    # sanity check / clean impossible values
    out = pd.to_numeric(out, errors="coerce")
    out[(out < 0) | (out > ABS_PHYS_MAX)] = np.nan
    return out

def _cap_outliers(rate: pd.Series) -> pd.Series:
    r = pd.to_numeric(rate, errors="coerce")
    r[r < 0] = np.nan
    finite = r.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return r

    p99 = float(finite.quantile(PERCENTILE_FOR_CAP))
    # allow a bit above p99 but never past a hard absolute cap
    soft_cap = p99 * 1.5
    hard_cap = min(soft_cap, ABS_MAX_CAP)
    return r.clip(lower=0, upper=hard_cap)


def add_fuel_rate_for_file(input_csv: Path, output_csv: Path, *, weight_power: float = 1.0, ) -> bool:  # float 1: proportional to speed; try 2 to stress aero drag
    df = pd.read_csv(input_csv, low_memory=False)
    # Ensure the two required columns exist
    required = {FUEL_CUM_COL, SPEED_COL}
    if not required.issubset(df.columns):
        print(f"Skipping {input_csv.name} – missing one of {required}.")
        return False

    # Choose a time column
    try:
        time_col = _pick_time_column(df)
    except KeyError as e:
        print(f"Skipping {input_csv.name} – {e}")
        return False

    # Compute rate with speed-weighted redistribution
    raw_rate = _compute_adjacent_minute_rates(df, time_col=time_col, weight_power=weight_power)
    # Cap outliers and drop NaNs
    cleaned_rate = _cap_outliers(raw_rate)
    df[RATE_COL] = cleaned_rate
    df = df[df[RATE_COL].notna()].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # simple progress signal
    finite = df[RATE_COL].replace([np.inf, -np.inf], np.nan).dropna()
    p99 = float(finite.quantile(PERCENTILE_FOR_CAP)) if not finite.empty else float("nan")
    print(f"{input_csv.name} → {output_csv.name} | rows kept: {len(df):,} | p99≈{p99:.2f} ml/s")
    return True


def batch_process_all(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(input_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {input_dir}")
        return

    processed = skipped = 0
    for f in csvs:
        out = output_dir / f"{f.stem}_fuelrate.csv"
        try:
            ok = add_fuel_rate_for_file(f, out)
            processed += int(ok)
            skipped   += int(not ok)
        except Exception as e:
            skipped += 1
            print(f"Error on {f.name}: {e}")

    print(f"\nDone. Processed: {processed} | Skipped: {skipped} | "f"Input: {input_dir} | Output: {output_dir}")