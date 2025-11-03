import pandas as pd
import numpy as np
import os

input_file  = "fuel_data_speed_merged/863609060548678_merged.csv"
output_folder = "processed_data"

os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(input_file)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").set_index("time")

df["fuel"] = df["TRACKS.MUNIC.MDI_OBD_FUEL (ml)"].ffill().fillna(0)
df["speed"] = df["TRACKS.MUNIC.SPEED_MERGED (km/h)"].interpolate()

window = "30s"
df_window = df.resample(window).agg({
    "fuel": "last",
    "speed": "mean"
})

df_window["fuel_used_ml"] = df_window["fuel"].diff().fillna(0).clip(lower=0)
df_window["speed_smoothed"] = df_window["speed"].rolling(3, min_periods=1).mean()

# Nur Dateiname verwenden
base_name = os.path.basename(input_file)
new_name = base_name.replace(".csv", "_processed.csv")
output_file = os.path.join(output_folder, new_name)

df_window.to_csv(output_file)
print(f"Datei erfolgreich gespeichert: {output_file}")
