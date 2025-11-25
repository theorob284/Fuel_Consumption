# Fuel_Consumption User Manual

How to Use the Project
----------------------------------------
1. Put your raw CSV files into the fuel_data folder

These are the original files you want to process.
----------------------------------------
To run:

main.py
----------------------------------------
Menu Options

Option 1 – Prepare the data

1. Merges speed columns
2. Calculates fuel rate

Creates clean files

IMPORTANT
Run this first every time you add new data.
This data is used in training and so only needs to be run once. After that you can use the other models whenever you want.
----------------------------------------
Option 2 – Train Random Forest model

Uses all prepared data
Trains the main machine-learning model
Saves model + graph

Only works after Option 1 is completed.
----------------------------------------
Option 3 – Train baseline models

Builds simple comparison models
Shows quick performance differences
Saves a comparison graph

Also requires Option 1.
----------------------------------------
Typical Workflow 

Add raw files → fuel_data/
Run Option 1
Run Option 2 (main model)
Optionally, Run Option 3 (baseline models)
----------------------------------------
All graphical models saved to the model/ folder
----------------------------------------

To change rf and baseline vales, the respective fields can be found in main.py, in their functions.
More options can be found in their respective files.

