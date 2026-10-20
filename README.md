# Fuel Consumption Prediction

A machine learning project for analysing vehicle telemetry data and predicting fuel consumption from driving behaviour.

The project processes raw vehicle CSV data, combines multiple speed sources, derives fuel-rate measurements from cumulative fuel readings, engineers driving-related features, and trains a Random Forest regression model to predict fuel consumption.

The project also includes simple linear and polynomial regression models for comparison.

## Project Overview

The raw dataset contains vehicle telemetry collected over time, including:

- GPS speed
- OBD speed
- Cumulative fuel consumption
- Engine RPM, where available
- Timestamps

The processing pipeline converts this raw telemetry into a dataset suitable for machine learning and predicts:

`Fuel rate (ml/s)`

The primary model is a `RandomForestRegressor`.

## Pipeline

```text
Raw CSV files
      ↓
Merge GPS + OBD speed
      ↓
Calculate fuel rate
      ↓
Clean invalid/outlier values
      ↓
Combine processed trips
      ↓
Engineer driving features
      ↓
Temporal train/test split
      ↓
Train Random Forest
      ↓
Evaluate model + generate visualisations
