"""
Visualizes and models the relationship between speed (km/h)
and fuel rate (ml/s) using simple and polynomial regression.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# ========= SETTINGS =========
INPUT_FILE = Path("863609060548678_fuelrate.csv")
SPEED_COL = "TRACKS.MUNIC.SPEED_MERGED (km/h)"
FUEL_COL = "TRACKS.MUNIC.FUEL_RATE (ml/s)"
POLY_DEGREE = 2  # degree for polynomial regression
# ============================


def main():
    # --- Load and clean data ---
    df = pd.read_csv(INPUT_FILE)
    df = df[[SPEED_COL, FUEL_COL]].dropna()
    df = df[df[SPEED_COL] > 0]  # remove any stationary rows

    X = df[[SPEED_COL]]
    y = df[FUEL_COL]

    # --- Linear regression ---
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    y_pred_lin = lin_model.predict(X)
    lin_r2 = r2_score(y, y_pred_lin)

    # --- Polynomial regression ---
    poly = PolynomialFeatures(degree=POLY_DEGREE)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    y_pred_poly = poly_model.predict(X_poly)
    poly_r2 = r2_score(y, y_pred_poly)

    # --- Sort for smoother plotting ---
    sorted_idx = X[SPEED_COL].argsort()
    X_sorted = X.iloc[sorted_idx]
    y_lin_sorted = y_pred_lin[sorted_idx]
    y_poly_sorted = y_pred_poly[sorted_idx]

    # --- Plot ---
    plt.figure(figsize=(9, 6))
    plt.scatter(X, y, s=8, color="gray", alpha=0.5, label="Actual data")
    plt.plot(X_sorted, y_lin_sorted, color="blue", linewidth=2, label=f"Linear fit (R²={lin_r2:.3f})")
    plt.plot(X_sorted, y_poly_sorted, color="red", linewidth=2, label=f"Polynomial fit (R²={poly_r2:.3f})")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Fuel rate (ml/s)")
    plt.title(f"Fuel Consumption vs. Speed\n({INPUT_FILE.name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # --- Summary ---
    print(f"✅ Linear Regression R²: {lin_r2:.3f}")
    print(f"✅ Polynomial Regression (degree={POLY_DEGREE}) R²: {poly_r2:.3f}")
    print("Use the plotted curve to identify efficient speed ranges.")


if __name__ == "__main__":
    main()
