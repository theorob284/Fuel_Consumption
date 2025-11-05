from pathlib import Path
import sys
import merge_speed  # imports your existing script
import fuel_rate_calculator
import fuel_rate_rf_model

def main():
    """
    Main entrypoint — merges all CSV files in 'fuel_data/' into 'fuel_data_merged/'.
    """
    User_program_run = int(input("Before running any Algorithms, make sure speed is merged, then the fuel rate is calculated\n------------------------\n1) Merge speed\n2) Calculate Fuel Rate 1\n3) Linear Regression\n"))

    if User_program_run == 1:
        input_dir = Path("fuel_data")
        output_dir = Path("fuel_data_speed_merged")

        # Make sure the input folder exists
        if not input_dir.exists():
            print(f"❌ Input folder not found: {input_dir.resolve()}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(input_dir.glob("*.csv"))
        if not csv_files:
            print(f"⚠️ No CSV files found in {input_dir}")
            return

        print(f"📁 Found {len(csv_files)} CSV files in {input_dir}")
        print(f"➡️ Output will be saved in {output_dir}")

        for csv_path in csv_files:
            target_path = output_dir / f"{csv_path.stem}_merged.csv"
            try:
                merge_speed.merge_speed_columns(
                    input_csv_path=csv_path,
                    output_csv_path=target_path,
                    gps_col=merge_speed.GPS_COL_DEFAULT,
                    obd_col=merge_speed.OBD_COL_DEFAULT,
                    merged_col=merge_speed.MERGED_COL_DEFAULT,
                    drop_original=True,  # set False if you want to keep the original speed columns
                )
                print(f"✅ Merged: {csv_path.name}")
            except Exception as e:
                print(f"❌ Failed on {csv_path.name}: {e}")

        print("\n All files processed!")
    elif User_program_run == 2:
        input_file = Path("fuel_data_speed_merged/863609060548678_merged.csv")
        output_file = input_file.with_name("863609060548678_fuelrate.csv")

        fuel_rate_calculator.add_fuel_rate(input_file, output_file)
        print("✅ Fuel rate calculation complete.\n")
    
    elif User_program_run == 3:
        # Train RF on the cleaned file produced by step 3
        input_file = Path("863609060548678_fuelrate.csv")
        model_out  = Path("models/fuel_rate_rf.pkl")

        summary = fuel_rate_rf_model.train_rf(
            input_csv=input_file,
            model_out=model_out,
            plots=True,            # set False if running headless
            n_estimators=400,
            random_state=42,
        )

        print("\n=== Random Forest Summary ===")
        print(f"Train size: {summary['n_train']}, Test size: {summary['n_test']}")
        print(f"Features : {summary['features']}")
        print(f"R²       : {summary['r2']:.3f}")
        print(f"MAE      : {summary['mae']:.3f} ml/s")
        print(f"RMSE     : {summary['rmse']:.3f} ml/s")
        print("Top importances:\n", summary["importances"].head(10))
        print(f"Model saved → {summary['model_path']}\n")


if __name__ == "__main__":
    main()
