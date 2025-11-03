from pathlib import Path
import sys
import merge_speed  # imports your existing script

def main():
    """
    Main entrypoint — merges all CSV files in 'fuel_data/' into 'fuel_data_merged/'.
    """
    input_dir = Path("fuel_data")
    output_dir = Path("fuel_data_merged")

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

    print("\n🎉 All files processed!")


if __name__ == "__main__":
    main()
