from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import shutil
import numpy as np
import math
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
from typing import Optional, Dict, Any
import os, glob
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

RAW_DIR = Path("fuel_data")                 # original raw CSVs (per trip)
SPEED_MERGED_DIR = Path("fuel_data_speed_merged")    # output of merge_speed
FUELRATE_DIR = Path("fuel_data_with_fuelrate")   # output of fuel_rate_calculator
MODELS_DIR = Path("models")
COMBINED_CSV_PATH = FUELRATE_DIR / "_combined_all_fuelrate.csv"

SPEED_COL = "TRACKS.MUNIC.SPEED_MERGED (km/h)"
TARGET_COL = "TRACKS.MUNIC.FUEL_RATE (ml/s)"
RPM_COL_CANDIDATES = ["TRACKS.MUNIC.MDI_OBD_RPM (rpm)", "RPM", "TRACKS.MUNIC.RPM", "MDI_RPM"]
FUEL_CUM_COL = "TRACKS.MUNIC.MDI_OBD_FUEL (ml)"

GPS_COL = "TRACKS.MUNIC.GPS_SPEED (km/h)"
OBD_COL = "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"
MERGED_COL = "TRACKS.MUNIC.SPEED_MERGED (km/h)"