from pathlib import Path
from typing import Optional
import pandas as pd

# Public defaults for column names so callers (e.g., main.py) can import them
GPS_COL_DEFAULT = "TRACKS.MUNIC.GPS_SPEED (km/h)"
OBD_COL_DEFAULT = "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"
MERGED_COL_DEFAULT = "TRACKS.MUNIC.SPEED_MERGED (km/h)"


def merge_speed_columns(
    input_csv_path: Path,
    output_csv_path: Optional[Path] = None,
    gps_col: str = GPS_COL_DEFAULT,
    obd_col: str = OBD_COL_DEFAULT,
    merged_col: str = MERGED_COL_DEFAULT,
    drop_original: bool = True,
) -> Path:
    """
    Read a CSV, merge GPS/OBD speed into a single column, save, and return output path.

    Exclusion logic (matches original behavior):
    - If either the GPS or OBD column is missing, raise KeyError. Caller can skip this file.

    Merge rule:
    - Prefer GPS where present; otherwise use OBD.
    - Final merged values are rounded to 2 decimal places.
    """
    input_csv_path = Path(input_csv_path)
    if output_csv_path is None:
        output_csv_path = input_csv_path.with_name(f"{input_csv_path.stem}_merged{input_csv_path.suffix}")
    else:
        output_csv_path = Path(output_csv_path)

    # Read CSV
    df = pd.read_csv(input_csv_path, low_memory=False)

    # Exclusion rule: require both columns
    missing = [c for c in (gps_col, obd_col) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Expected '{gps_col}' and '{obd_col}'.")

    # Combine (prefer GPS), coerce to numeric, and round to 2 decimals
    gps = pd.to_numeric(df[gps_col], errors="coerce")
    obd = pd.to_numeric(df[obd_col], errors="coerce")
    df[merged_col] = gps.combine_first(obd).round(2)

    # Optionally drop the originals
    if drop_original:
        df.drop(columns=[gps_col, obd_col], inplace=True, errors="ignore")

    # Write result
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return output_csv_path
