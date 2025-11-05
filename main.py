# main.py
from pathlib import Path
import sys
import pandas as pd

import merge_speed
import fuel_rate_calculator
import fuel_rate_rf_model


# --- Folders / paths (edit if needed) ---
RAW_DIR       = Path("fuel_data")
MERGED_DIR    = Path("fuel_data_speed_merged")
FUELRATE_DIR  = Path("fuel_data_with_fuelrate")
MODEL_PATH    = Path("models/fuel_rate_rf_all.pkl")


def merge_all():
    """Merge GPS + OBD speed into one column for every CSV in RAW_DIR."""
    if not RAW_DIR.exists():
        print(f"Folder not found: {RAW_DIR.resolve()}")
        return

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print(f"No CSV files in {RAW_DIR}")
        return

    print(f"Merging speed columns for {len(files)} file(s) → {MERGED_DIR}")
    for src in files:
        dst = MERGED_DIR / f"{src.stem}_merged.csv"
        try:
            merge_speed.merge_speed_columns(
                input_csv_path=src,
                output_csv_path=dst,
                gps_col=merge_speed.GPS_COL_DEFAULT,
                obd_col=merge_speed.OBD_COL_DEFAULT,
                merged_col=merge_speed.MERGED_COL_DEFAULT,
                drop_original=True,
            )
            print(f"{src.name} → {dst.name}")
        except Exception as e:
            print(f"{src.name}: {e}")
    print("Merge complete.\n")


def compute_fuelrate_all():
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    FUELRATE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Computing fuel rate for files in {MERGED_DIR} → {FUELRATE_DIR}")
    fuel_rate_calculator.batch_process_all(input_dir=MERGED_DIR, output_dir=FUELRATE_DIR)
    print("Fuel-rate computation complete.\n")


def train_rf_all():

    files = sorted(FUELRATE_DIR.glob("*_fuelrate.csv"))
    if not files:
        print(f"No '*_fuelrate.csv' files in {FUELRATE_DIR}. Run option 2 first.")
        return

    combined_csv = FUELRATE_DIR / "_combined_all_fuelrate.csv"
    print(f"Preparing training data from {len(files)} file(s)...")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["trip_id"] = f.stem  # keep source info
            frames.append(df)
        except Exception as e:
            print(f" kipping {f.name}: {e}")

    if not frames:
        print("No usable data to train on.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sample(frac=0.05, random_state=42)  # 5% CHANGE AMOUNT OF VALUE TESTED HERE !!!!
    print(f"Using sampled dataset size: {len(combined):,}")
    print(f"Rows in combined dataset: {len(combined):,}")
    print(f"Columns: {list(combined.columns)}")
    combined.to_csv(combined_csv, index=False)
    print(f"Combined dataset written: {combined_csv}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = fuel_rate_rf_model.train_rf(
        input_csv=combined_csv,
        model_out=MODEL_PATH,
        plots=False,
        n_estimators=100,
        random_state=42,
    )

    print("\n=== Random Forest Summary ===\n")
    print(f"Train rows: {summary['n_train']} | Test rows: {summary['n_test']}")
    print(f"Features : {summary['features']}")
    print(f"R²       : {summary['r2']:.3f}")
    print(f"MAE      : {summary['mae']:.3f} ml/s")
    print(f"RMSE     : {summary['rmse']:.3f} ml/s")
    print("Top importances:\n", summary["importances"].head(10))
    print(f"Model saved → {summary['model_path']}\n")


def main():
    print("Choose an option:")
    print("1) Merge speed for ALL raw files")
    print("2) Compute fuel rate for ALL merged files")
    print("3) Train Random Forest on ALL processed files")

    choice = input("> ").strip()
    if choice == "1":
        merge_all()
    elif choice == "2":
        compute_fuelrate_all()
    elif choice == "3":
        train_rf_all()
    else:
        print("Please enter 1, 2, or 3.")
        sys.exit(1)


if __name__ == "__main__":
    main()
