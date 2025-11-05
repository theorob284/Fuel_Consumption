import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

# === CONFIG ===
input_folder = "processed_data"
files = glob.glob(os.path.join(input_folder, "*.csv"))

if not files:
    raise FileNotFoundError("⚠️ Keine verarbeiteten Dateien im 'processed_data' Ordner gefunden!")

# === LOAD & COMBINE ===
dfs = [pd.read_csv(f) for f in files]
data = pd.concat(dfs, ignore_index=True)

# === FLEXIBLE SPALTENNAMEN ===
if "speed_kmh" in data.columns:
    speed_col = "speed_kmh"
elif "TRACKS.MUNIC.SPEED_MERGED (km/h)" in data.columns:
    speed_col = "TRACKS.MUNIC.SPEED_MERGED (km/h)"
else:
    raise KeyError("❌ Keine Spalte für Geschwindigkeit gefunden!")

if "fuel_ml_per_s" in data.columns:
    fuel_col = "fuel_ml_per_s"
elif "fuel_rate" in data.columns:
    fuel_col = "fuel_rate"
else:
    raise KeyError("❌ Keine Spalte für Kraftstoffrate gefunden!")

# === FILTER ===
data = data.dropna(subset=[speed_col, fuel_col])
data = data[(data[speed_col] > 5) & (data[fuel_col] > 0) & (data[fuel_col] < 50)]

print(f"✅ Datensätze nach Filterung: {len(data)}")

# === GLÄTTUNG ===
data[speed_col] = data[speed_col].rolling(3, min_periods=1).mean()
data[fuel_col] = data[fuel_col].rolling(3, min_periods=1).mean()

X = data[[speed_col]].values
y = data[fuel_col].values

# === MODEL 1: LINEARE REGRESSION ===
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# === MODEL 2: POLYNOMIAL REGRESSION (Grad 3) ===
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# === MODEL 3: SUPPORT VECTOR REGRESSION (RBF-Kernel) auf Stichprobe ===
sample_for_svr = data.sample(n=min(50000, len(data)), random_state=42)
print(f"🎯 SVR wird auf einer Stichprobe von {len(sample_for_svr)} Punkten trainiert.")

X_svr = sample_for_svr[[speed_col]].values
y_svr = sample_for_svr[fuel_col].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X_svr)
y_scaled = scaler_y.fit_transform(y_svr.reshape(-1, 1)).ravel()

svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
svr.fit(X_scaled, y_scaled)

y_pred_svr = scaler_y.inverse_transform(svr.predict(X_scaled).reshape(-1, 1)).ravel()

# === METRIKEN ===
def metrics(name, y_true, y_pred):
    print(f"\n📊 {name}")
    print(f"R²: {r2_score(y_true, y_pred):.3f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.3f}")

metrics("Lineare Regression", y, y_pred_lin)
metrics("Polynomiale Regression (Grad 3)", y, y_pred_poly)
metrics("Support Vector Regression (RBF, Stichprobe)", y_svr, y_pred_svr)

# === VISUALISIERUNG ===
plt.figure(figsize=(10,6))
plt.scatter(data[speed_col], y, alpha=0.3, label="Tatsächlicher Verbrauch", s=10)

speeds = np.linspace(data[speed_col].min(), data[speed_col].max(), 200).reshape(-1, 1)
plt.plot(speeds, lin_reg.predict(speeds), color="red", label="Linear")
plt.plot(speeds, poly_reg.predict(poly.transform(speeds)), color="orange", label="Polynomial (3. Grad)")
plt.plot(
    speeds,
    scaler_y.inverse_transform(svr.predict(scaler_X.transform(speeds)).reshape(-1,1)),
    color="green", label="SVR (RBF, Stichprobe)"
)

plt.xlabel("Geschwindigkeit [km/h]")
plt.ylabel("Verbrauch [ml/s]")
plt.title("Vergleich verschiedener Modelle für den Verbrauch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
