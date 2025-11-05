import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


GPS_COL_DEFAULT = "TRACKS.MUNIC.GPS_SPEED (km/h)"
OBD_COL_DEFAULT = "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"
MERGED_COL_DEFAULT = "TRACKS.MUNIC.SPEED_MERGED (km/h)"

def merge_speed_columns(input_csv_path: Path,output_csv_path: Optional[Path] = None,gps_col: str = GPS_COL_DEFAULT,obd_col: str = OBD_COL_DEFAULT,merged_col: str = MERGED_COL_DEFAULT,drop_original: bool = True,) -> Path:

    if output_csv_path is None:
        output_csv_path = input_csv_path.with_name(f"{input_csv_path.stem}_merged{input_csv_path.suffix}")

    df = pd.read_csv(input_csv_path, low_memory=False)

    missing = [c for c in [gps_col, obd_col] if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. Expected '{gps_col}' and '{obd_col}'."
        )

    gps_series = pd.to_numeric(df[gps_col], errors="coerce")
    obd_series = pd.to_numeric(df[obd_col], errors="coerce")

    df[merged_col] = gps_series.combine_first(obd_series).round(2)

    # NEW: drop the originals if requested
    if drop_original:
        df.drop(columns=[gps_col, obd_col], inplace=True, errors="ignore")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
        )
    )
    parser.add_argument("input_csv",type=Path,nargs="?",default=Path("fuel_data"),help="Input CSV file path (optional)")
    parser.add_argument("--output",type=Path,default=Path("fuel_data_speed_merged"),help="Output CSV path (optional, defaults to add _merged to input name)")
    parser.add_argument("--gps-col",type=str,default="TRACKS.MUNIC.GPS_SPEED (km/h)", help=f"GPS speed column name (default: {GPS_COL_DEFAULT})")
    parser.add_argument("--obd-col",type=str,default="TRACKS.MUNIC.MDI_OBD_SPEED (km/h)",help=f"OBD speed column name (default: {OBD_COL_DEFAULT})")
    parser.add_argument("--merged-col",type=str,default="TRACKS.MUNIC.SPEED_MERGED (km/h)",help=f"New merged column name (default: {MERGED_COL_DEFAULT})")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--keep-originals",action="store_true",help="Keep the original GPS/OBD speed columns (default is to drop them).")
    group.add_argument("--drop-originals",action="store_true",help="Explicitly drop the original GPS/OBD speed columns.")

    args = parser.parse_args()

    input_path: Path = args.input_csv
    output_base: Path = args.output

    # If input is a directory: batch process all CSVs inside (non-recursive)
    if input_path.is_dir():
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in directory: {input_path}")
            return
        output_base.mkdir(parents=True, exist_ok=True)
        print(f"Merging {len(csv_files)} file(s) into: {output_base}")
        for csv_path in csv_files:
            target_path = output_base / f"{csv_path.stem}_merged{csv_path.suffix}"
            try:
                out = merge_speed_columns(input_csv_path=csv_path,output_csv_path=target_path,gps_col=args.gps_col,obd_col=args.obd_col,merged_col=args.merged_col,)
                print(f"Written: {out}")
            except KeyError as e:
                print(f"Skipped {csv_path.name}: {e}")
            except Exception as e:
                print(f"Skipped {csv_path.name}: {e}")
        return

    # Single-file: ensure output is a file path; if directory-like, write inside it
    target_output = output_base
    if target_output is None:
        target_output = input_path.with_name(f"{input_path.stem}_merged{input_path.suffix}")
    elif target_output.suffix.lower() != ".csv":
        target_output.mkdir(parents=True, exist_ok=True)
        target_output = target_output / f"{input_path.stem}_merged{input_path.suffix}"

    out = merge_speed_columns(input_csv_path=input_path,output_csv_path=target_output,gps_col=args.gps_col,obd_col=args.obd_col,merged_col=args.merged_col,drop_original=drop_original,)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()


