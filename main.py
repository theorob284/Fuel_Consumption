# main.py
from __future__ import annotations

import gc
import os
import shutil
import sys
from pathlib import Path
import pandas as pd

# Local modules
import merge_speed
import fuel_rate_calculator
import fuel_rate_rf_model
import baseline_algorithm



# Paths 
RAW_DIR           = Path("fuel_data")               # your original raw CSVs (per trip)
SPEED_MERGED_DIR  = Path("fuel_data_speed_merged")      # output of merge_speed
FUELRATE_DIR      = Path("fuel_data_with_fuelrate")     # output of fuel_rate_calculator
MODELS_DIR        = Path("models")
COMBINED_CSV_PATH = FUELRATE_DIR / "_combined_all_fuelrate.csv"

SPEED_GPS_COL     = "TRACKS.MUNIC.GPS_SPEED (km/h)"
SPEED_OBD_COL     = "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"
SPEED_MERGED_COL  = "TRACKS.MUNIC.SPEED_MERGED (km/h)"
FUEL_RATE_COL     = "TRACKS.MUNIC.FUEL_RATE (ml/s)"

# Merge speeds

def merge_all():
    # This merges GPS and OBD speed columns for ALL CSVs in RAW_DIR → SPEED_MERGED_DIR.
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SPEED_MERGED_DIR.mkdir(parents=True, exist_ok=True)

    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {RAW_DIR}")
        return

    print(f"Merging {len(csvs)} file(s) from {RAW_DIR} → {SPEED_MERGED_DIR}")
    for src in csvs:
        dst = SPEED_MERGED_DIR / f"{src.stem}_merged.csv"
        try:
            merge_speed.merge_speed_columns(
                input_csv_path=src,
                output_csv_path=dst,
                gps_col=SPEED_GPS_COL,
                obd_col=SPEED_OBD_COL,
                merged_col=SPEED_MERGED_COL,
                drop_original=True,
            )
            print(f"  --> {src.name} → {dst.name}")
        except Exception as e:
            print(f"  --> {src.name}: {e}")


# Compute fuel rate for all files

def compute_fuelrate_all():
    # Compute fuel rate for ALL merged files in SPEED_MERGED_DIR → FUELRATE_DIR.

    print("...Computing fuel rate for all merged files…")
    fuel_rate_calculator.batch_process_all(
        input_dir=SPEED_MERGED_DIR,
        output_dir=FUELRATE_DIR
    )

    os.chdir(".")
    if os.path.exists(SPEED_MERGED_DIR):
        shutil.rmtree(SPEED_MERGED_DIR, ignore_errors=True)
        print("Deleted temp folder of merged speeds")


# combine processed data
def _combine_processed(sample_frac: float | None = 0.02) -> Path:
    #Concatenate all processed CSVs in FUELRATE_DIR into a single CSV.

    files = sorted(FUELRATE_DIR.glob("*.csv"))
    files = [f for f in files if f.name != COMBINED_CSV_PATH.name]
    if not files:
        raise FileNotFoundError(f"No processed CSVs found in {FUELRATE_DIR}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        # keep only rows with both merged speed and fuel rate
        if SPEED_MERGED_COL in df.columns and FUEL_RATE_COL in df.columns:
            frames.append(df[[c for c in df.columns]])
    combined = pd.concat(frames, ignore_index=True)

    if sample_frac and 0 < sample_frac < 1:
        combined = combined.sample(frac=sample_frac, random_state=42)
        print(f"Using sampled dataset size: {len(combined):,}")

    COMBINED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_CSV_PATH, index=False)
    print(f"Combined dataset written: {COMBINED_CSV_PATH}")
    return COMBINED_CSV_PATH

# Train Random Forest
def train_rf_all():
    print("Preparing training data from ALL processed files...")
    combined_csv = _combine_processed(sample_frac=0.20)  # tweak fraction if you like

    summary = fuel_rate_rf_model.train_rf(
        input_csv=combined_csv,
        model_out= "fuel_rate_rf_all.pkl",
        plots=False,            
        n_estimators=100,
        random_state=42,
    )

    print("\n=== Random Forest Summary ===")
    print(f"Train rows: {summary['n_train']} | Test rows: {summary['n_test']}\nFeatures : {summary['features']}\nR²       : {summary['r2']:.3f}\nMAE      : {summary['mae']:.3f} ml/s\nRMSE     : {summary['rmse']:.3f} ml/s")
    print("Top importances:\n", summary["importances"].head(10), f"Model saved → {summary['model_path']}\nOverlay   → {summary['overlay_path']}")

#Baselines (Linear/Poly/SVR)

def run_baselines_all():
    #This runs the baseline algorithms
    print("Running baseline models (Linear / Polynomial / SVR)…")
    out_img = MODELS_DIR / "baseline_comparison.png"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    result = baseline_algorithm.run_baselines(
        input_folder=FUELRATE_DIR,
        sample_frac=0.50, # use 10% for speed; increase if you want
        svr_sample=50_000,
        save_path=out_img,
        show=True # set False to only save the figure
    )

    print("\n=== Baseline Model Metrics ===")
    for name, m in result["metrics"].items():
        print(f"{name:>22} | R²: {m['r2']:.3f}  MSE: {m['mse']:.3f}")
    print(f"Rows used: {result['rows_used']:,} (SVR trained on {result['svr_rows']:,})")
    print(f"Plot saved → {out_img}")


def main():
    print("Choose an option:\n1) Merge speeds + compute fuel_rate (NOTE: required before using algorithms \n2) Train Random Forest on ALL processed files\n3) Baseline models (Linear/Poly/SVR)")
    choice = input(">> ").strip()

    if choice == "1":
        merge_all()
        compute_fuelrate_all()
    elif choice == "2":
        train_rf_all()
    elif choice == "3":
        run_baselines_all()
    else:
        print("Please enter 1, 2, 3, or 4.")
        sys.exit(1)


if __name__ == "__main__":
    main()
