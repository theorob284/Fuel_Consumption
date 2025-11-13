# merge_speed.py
from common_imports import *

def merge_speed_file(input_csv: Path,output_csv: Optional[Path] = None,gps_col: str = GPS_COL,obd_col: str = OBD_COL,merged_col: str = MERGED_COL,drop_original: bool = True) -> Path:
    input_csv = Path(input_csv)

    if output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_merged{input_csv.suffix}")
    else:
        output_csv = Path(output_csv)

    df = pd.read_csv(input_csv, low_memory=False)

    # Basic column check
    missing = [c for c in (gps_col, obd_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}.\nExpected '{gps_col}' and '{obd_col}'.")

    # Convert to numeric and combine
    gps = pd.to_numeric(df[gps_col], errors="coerce")
    obd = pd.to_numeric(df[obd_col], errors="coerce")

    # Prefer GPS, fall back to OBD
    df[merged_col] = gps.combine_first(obd).round(2)

    if drop_original:
        df.drop(columns=[gps_col, obd_col], inplace=True, errors="ignore")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv

def merge_speed_all(input_dir: Path,output_dir: Path,gps_col: str = GPS_COL,obd_col: str = OBD_COL,merged_col: str = MERGED_COL,drop_original: bool = True,) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(input_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {input_dir}")
        return

    print(f"Merging {len(csvs)} file(s) from {input_dir} → {output_dir}")
    for src in csvs:
        dst = output_dir / f"{src.stem}_merged{src.suffix}"
        try:
            merge_speed_file(input_csv=src,output_csv=dst,gps_col=gps_col,obd_col=obd_col,merged_col=merged_col,drop_original=drop_original,)
            print(f"{src.name} → {dst.name}")
        except Exception as e:
            print(f"{src.name}: {e}")
