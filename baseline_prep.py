import pandas as pd
import os
import glob

input_folder = "fuel_data_speed_merged"
output_folder = "processed_data"

os.makedirs(output_folder, exist_ok=True)

input_files = glob.glob(os.path.join(input_folder, "*.csv"))

for input_file in input_files:
    df = pd.read_csv(input_file)

    # Zeitspalte finden
    time_col = "time" if "time" in df.columns else "received_at"
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Treibstoffspalte finden
    fuel_cols = [c for c in df.columns if "OBD_FUEL" in c]
    if not fuel_cols:
        print(f"⚠️ no fuel column in {input_file}.")
        continue
    fuel_col = fuel_cols[0]
    df["fuel"] = df[fuel_col].ffill().fillna(0)

    # Geschwindigkeitsspalte finden
    speed_cols = [c for c in df.columns if "SPEED" in c]
    if not speed_cols:
        print(f"⚠️ no speed column in {input_file}.")
        continue
    speed_col = speed_cols[0]

    results = []
    prev_time = None
    prev_fuel = None
    first_valid_fuel = None  # markiert den ersten gültigen Wert

    for idx, row in df.iterrows():
        current_time = row[time_col]
        current_fuel = row["fuel"]
        speed = row[speed_col]

        # ersten gültigen Fuel-Wert überspringen
        if first_valid_fuel is None and current_fuel > 0:
            first_valid_fuel = current_fuel
            prev_time = current_time
            prev_fuel = current_fuel
            continue

        # skip, bis wir einen gültigen vorherigen Zeitwert haben
        if prev_time is None:
            prev_time = current_time
            prev_fuel = current_fuel
            continue

        # Zeitdifferenz berechnen
        dt_seconds = (current_time - prev_time).total_seconds()
        if dt_seconds < 1:
            continue

        # Verbrauch berechnen
        fuel_used = max(0, current_fuel - prev_fuel)
        fuel_per_second = fuel_used / dt_seconds if dt_seconds > 0 else 0

        # nur bei Bewegung > 5 km/h den Verbrauch berechnen
        if speed > 5:
            fuel_liters = fuel_used / 1000
            distance_km = speed * (dt_seconds / 3600)
            l_per_100km = (fuel_liters / distance_km) * 100 if distance_km > 0 else None
        else:
            l_per_100km = None

        # unrealistische Ausreißer filtern
        if fuel_per_second > 50:
            fuel_per_second = None
            l_per_100km = None

        results.append({
            "time": current_time,
            "speed_kmh": speed,
            "fuel_used_ml": fuel_used,
            "duration_s": dt_seconds,
            "fuel_ml_per_s": fuel_per_second,
            "fuel_L_per_100km": l_per_100km
        })

        prev_time = current_time
        prev_fuel = current_fuel

    # Ergebnis speichern
    output_file = os.path.join(
        output_folder,
        os.path.basename(input_file).replace(".csv", "_processed.csv")
    )
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"✅ {output_file}")
