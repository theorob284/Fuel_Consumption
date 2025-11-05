import pandas as pd
import numpy as np
import os
import glob

input_folder = "fuel_data_speed_merged"
output_folder = "processed_data"

os.makedirs(output_folder, exist_ok=True)

# Alle CSV-Dateien im Ordner laden
input_files = glob.glob(os.path.join(input_folder, "*.csv"))

for inp in input_files:
    out = os.path.join(output_folder, os.path.basename(inp).replace(".csv", "_processed.csv"))

    print(f"Loading {inp} ...")
    df = pd.read_csv(inp)

    TIME_COL  = "time"
    SPEED_COL = "TRACKS.MUNIC.SPEED_MERGED (km/h)"
    FUEL_COL  = "TRACKS.MUNIC.MDI_OBD_FUEL (ml)"

    # Zeit umwandeln
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    # Fuel-Differenzen berechnen
    fuel_pts = df.dropna(subset=[FUEL_COL]).copy()
    fuel_pts["fuel_diff"] = fuel_pts[FUEL_COL].diff()
    fuel_pts["time_diff_s"] = fuel_pts[TIME_COL].diff().dt.total_seconds()
    fuel_pts["fuel_rate_ml_s"] = fuel_pts["fuel_diff"] / fuel_pts["time_diff_s"]
    fuel_pts = fuel_pts.dropna(subset=["fuel_rate_ml_s"])

    # Fuel Rate auf alle Zeilen zwischen zwei Messpunkten übertragen
    df["fuel_rate"] = np.nan
    pairs = fuel_pts[[TIME_COL, "fuel_rate_ml_s"]].values

    for i in range(1, len(pairs)):
        start_time_prev = pairs[i-1][0]
        end_time_curr   = pairs[i][0]
        rate = pairs[i][1]
        mask = (df[TIME_COL] > start_time_prev) & (df[TIME_COL] <= end_time_curr)
        df.loc[mask, "fuel_rate"] = rate

    df = df.dropna(subset=["fuel_rate"])
    df.to_csv(out, index=False)
    print(f"✅ Done. Saved preprocessed dataset to {out}\n")
