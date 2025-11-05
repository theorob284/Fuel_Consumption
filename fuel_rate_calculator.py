from pathlib import Path
import pandas as pd
import numpy as np

# ========= SETTINGS =========
INPUT_DIR  = Path("fuel_data_speed_merged")     # where the merged CSVs live
OUTPUT_DIR = Path("fuel_data_with_fuelrate")    # where to write *_fuelrate.csv

# Column names
TIME_COL_PRIMARY   = "time"
TIME_COL_FALLBACKS = ["received_at"]

SPEED_COL = "TRACKS.MUNIC.SPEED_MERGED (km/h)"        # required (feature)
FUEL_CUM_COL = "TRACKS.MUNIC.MDI_OBD_FUEL (ml)"       # required (cumulative fuel)
RATE_COL  = "TRACKS.MUNIC.FUEL_RATE (ml/s)"           # new column to create

# Adjacent interval rule (≈ 1 minute, ±5 s)
TARGET_INTERVAL = 60
TOLERANCE       = 5
MIN_INTERVAL    = TARGET_INTERVAL - TOLERANCE
MAX_INTERVAL    = TARGET_INTERVAL + TOLERANCE

# Outlier handling (cap after computing)
PERCENTILE_FOR_CAP = 0.99
ABS_MAX_CAP        = 10.0     # hard cap after percentile
ABS_PHYS_MAX       = 50.0     # physically impossible; drop to NaN
# ===========================


def _pick_time_column(df: pd.DataFrame) -> str:
    """Return a usable timestamp column name."""
    if TIME_COL_PRIMARY in df.columns:
        return TIME_COL_PRIMARY
    for c in TIME_COL_FALLBACKS:
        if c in df.columns:
            return c
    raise KeyError(
        f"No time-like column found. Expected '{TIME_COL_PRIMARY}' or one of {TIME_COL_FALLBACKS}."
    )


def _compute_adjacent_minute_rates(df: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Compute ml/s only between directly-adjacent ~1-minute-apart fuel readings
    from the cumulative fuel column (FUEL_CUM_COL). Assign the segment's rate
    to all rows between those two timestamps.
    """
    # Prepare time & fuel points
    t = pd.to_datetime(df[time_col], errors="coerce")
    f = pd.to_numeric(df[FUEL_CUM_COL], errors="coerce")

    fuel_pts = pd.DataFrame({"_t": t, "_f": f}).dropna().reset_index()  # keep original row index
    if len(fuel_pts) < 2:
        return pd.Series(index=df.index, dtype="float64")

    # Consecutive diffs
    fuel_pts["time_diff_s"]  = fuel_pts["_t"].diff().dt.total_seconds()
    fuel_pts["fuel_diff_ml"] = fuel_pts["_f"].diff()
    fuel_pts["segment_rate"] = fuel_pts["fuel_diff_ml"] / fuel_pts["time_diff_s"]

    # Keep only directly-adjacent ~1-minute segments
    valid = fuel_pts["time_diff_s"].between(MIN_INTERVAL, MAX_INTERVAL, inclusive="both")
    valid_segments = fuel_pts.loc[valid, ["index", "_t", "segment_rate"]].reset_index(drop=True)

    # Assign piecewise-constant rate to rows between prev_t and cur_t
    rate = pd.Series(index=df.index, dtype="float64")
    for i in range(len(valid_segments)):
        cur_pos = valid_segments.loc[i, "index"]
        pos_in_fuel_pts = fuel_pts.index[fuel_pts["index"] == cur_pos][0]
        if pos_in_fuel_pts == 0:
            continue
        prev_t = fuel_pts.loc[pos_in_fuel_pts - 1, "_t"]
        cur_t  = valid_segments.loc[i, "_t"]
        seg_rate = valid_segments.loc[i, "segment_rate"]

        mask = (t >= prev_t) & (t < cur_t)
        rate.loc[mask] = seg_rate

    # Clean impossible values
    rate = pd.to_numeric(rate, errors="coerce")
    rate[(rate < 0) | (rate > ABS_PHYS_MAX)] = np.nan
    return rate


def _cap_outliers(rate: pd.Series) -> pd.Series:
    finite = rate.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return rate
    cap_value = min(float(finite.quantile(PERCENTILE_FOR_CAP)), ABS_MAX_CAP)
    capped = rate.copy()
    capped[capped > cap_value] = cap_value
    return capped


def add_fuel_rate_for_file(input_csv: Path, output_csv: Path) -> bool:
    """
    Process one CSV:
    - Require BOTH required columns: cumulative fuel + merged speed
    - Compute per-second fuel rate from adjacent ~1-minute pairs
    - Drop rows without a valid rate
    - Save to output_csv
    Returns True if processed, False if skipped.
    """
    df = pd.read_csv(input_csv, low_memory=False)

    # ✅ Minimal-change guard: ensure the two required columns exist
    required = {FUEL_CUM_COL, SPEED_COL}
    if not required.issubset(df.columns):
        print(f"⏭️  Skipping {input_csv.name} — missing one of {required}.")
        return False

    # Choose time column
    try:
        time_col = _pick_time_column(df)
    except KeyError as e:
        print(f"⏭️  Skipping {input_csv.name} — {e}")
        return False

    # Compute rates with strict adjacent 1-minute logic and cap outliers
    raw_rate = _compute_adjacent_minute_rates(df, time_col)
    cleaned_rate = _cap_outliers(raw_rate)

    # Attach, keep only rows with valid rate
    df[RATE_COL] = cleaned_rate
    df = df[df[RATE_COL].notna()].copy()

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    finite = df[RATE_COL].replace([np.inf, -np.inf], np.nan).dropna()
    p99 = float(finite.quantile(0.99)) if not finite.empty else float("nan")
    print(
        f"✅ {input_csv.name} → {output_csv.name} | rows kept: {len(df)} | "
        f"mean: {finite.mean():.2f} ml/s | p99: {p99:.2f}"
    )
    return True


def batch_process_all(input_dir: Path = INPUT_DIR, output_dir: Path = OUTPUT_DIR):
    """
    Process all CSVs in input_dir. Each file must contain BOTH:
      - TRACKS.MUNIC.MDI_OBD_FUEL (ml)
      - TRACKS.MUNIC.SPEED_MERGED (km/h)
    Writes <stem>_fuelrate.csv to output_dir and removes rows without a rate.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(input_dir.glob("*.csv"))
    if not csvs:
        print(f"⚠️ No CSV files found in {input_dir}")
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
            print(f"❌ Error on {f.name}: {e}")

    print(
        f"\nDone. Processed: {processed} | Skipped: {skipped} | "
        f"Input: {input_dir} | Output: {output_dir}"
    )


if __name__ == "__main__":
    batch_process_all()
