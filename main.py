# main.py
from common_imports import *

# Local modules
import merge_speed
import fuel_rate_calculator
import fuel_rate_rf_model
import baseline_algorithm

# Combine processed data 
def _combine_processed(sample_frac: float | None = 0.02) -> Path:
    files = sorted(FUELRATE_DIR.glob("*.csv"))
    files = [f for f in files if f.name != COMBINED_CSV_PATH.name]
    if not files:
        raise FileNotFoundError(f"No processed CSVs found in {FUELRATE_DIR}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        # keep only rows with both merged speed and fuel rate
        if SPEED_COL in df.columns and TARGET_COL in df.columns:
            frames.append(df)
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
    combined_csv = _combine_processed(sample_frac=0.60)  # how much data is used

    summary = fuel_rate_rf_model.train_rf(  # RF parameters here - This is where you can play with rf factors
        input_csv=combined_csv,
        model_out="fuel_rate_rf_all.pkl",
        plots=False,
        n_estimators=400,
        random_state=42,
    )

    print(f"\n=== Random Forest Summary ===\nTrain rows: {summary['n_train']} | Test rows: {summary['n_test']}\nFeatures : {summary['features']}\nR²       : {summary['r2']:.3f}\nMAE      : {summary['mae']:.3f} ml/s\nRMSE     : {summary['rmse']:.3f} ml/s")
    
#Baselines (Linear/Poly/SVR)
def run_baselines_all():
    print("Running baseline models (Linear / Polynomial / SVR)…")
    out_img = MODELS_DIR / "baseline_comparison.png"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # This is where you change parameters of the baseline algorithm
    result = baseline_algorithm.run_baselines(
        input_folder=FUELRATE_DIR,
        sample_frac=0.50,    # 50% for better fit; lower for speed
        svr_sample=50_000,
        save_path=out_img,
    )

    print("\n=== Baseline Model Metrics ===")
    for name, m in result["metrics"].items():
        print(f"{name:>22} | R²: {m['r2']:.3f}  MSE: {m['mse']:.3f}")
    print(f"Rows used: {result['rows_used']:,}\n(SVR trained on {result['svr_rows']:,})\nPlot saved → {out_img}")

# Merge speeds
def merge_all():
    merge_speed.merge_speed_all(input_dir=RAW_DIR,output_dir=SPEED_MERGED_DIR)

def main():
    print("Choose an option:\n1) Merge speeds + compute fuel_rate (NOTE: required before using algorithms)\n2) Train Random Forest on ALL processed files\n3) Baseline models (Linear/Poly/SVR)")
    choice = input(">> ").strip()

    if choice == "1":
        merge_all()
        fuel_rate_calculator.compute_fuelrate_all()
    elif choice == "2":
        train_rf_all()
    elif choice == "3":
        run_baselines_all()
    else:
        print("Please enter 1, 2, or 3.")
        sys.exit(1)

if __name__ == "__main__":
    main()
