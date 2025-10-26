<!-- # Rakesh_Pedapudi -->
# Milestone 1: Data Preparation & Exploration

**Project:** PrognosAI – AI-Driven Predictive Maintenance System Using Time-Series Sensor Data  
**Cohort:** Infosys Springboard 6.0 – Rakesh Pedapudi

## Overview

This milestone focuses on collecting, exploring, and preprocessing the sensor data required for predictive maintenance modeling.

## Files Included

- `milestone.py` – Python script to load, explore, and clean the sensor data.
- `fd1_train_preprocessed.csv` – Raw/preprocessed training data containing time-series sensor readings and target values.
- `fd1_test_preprocessed.csv` – Raw/preprocessed testing data.
- `fd1_train_cleaned.csv` – Output: Cleaned training data (missing values removed).
- `fd1_test_cleaned.csv` – Output: Cleaned testing data.

## Data Columns

- `timestamp` – Date and time of measurement
- `sensor_1`, `sensor_2`, `sensor_3` – Example sensor readings (numeric)
- `target` – Equipment health status (0: healthy, 1: failure or needs maintenance)

## Steps Performed

1. Loaded CSV sensor datasets using pandas.
2. Explored data for structure, columns, and missing values.
3. Cleaned the data by removing rows with null/missing values.
4. Saved cleaned datasets for use in future milestones.

## How to Run

1. Place all CSV files and milestone.py in the same folder.
2. In terminal, navigate to the folder and run:
3. Review printed outputs and inspect cleaned CSV files.

## Next Steps

- Feature engineering, data visualization, and building predictive models in upcoming milestones.
# Milestone 2: Model Training & Performance Evaluation

**Project:** PrognosAI – AI-Driven Predictive Maintenance System Using Time-Series Sensor Data

## Overview

This milestone covers training the predictive model and evaluating its performance on sensor time-series data.

## Folder Structure

- `milestone2.py`: Trains the neural network model using cleaned sensor data. Saves the best model and training loss plot.
- `evaluate.py`: Loads the trained model and scaler, evaluates predictions on test data, and generates performance plots.
- `models/`: Saved trained models (e.g., `optimized_fd1.h5`).
- `graphs_m2/`: Contains output plots such as training loss curve and predicted vs actual trends.

## Steps Performed

1. Loaded cleaned training and test datasets from Milestone 1.
2. Built and trained a regression model to predict equipment health/remaining useful life.
3. Saved the best-performing model.
4. Evaluated model predictions and calculated performance metrics (e.g., MAE).
5. Generated and saved evaluation plots.

## How to Run

1. Ensure you have cleaned CSV files from Milestone 1.
2. Run:
3. After training completes, run:
4. Output files will be in `models/` and `graphs_m2/`.

## Outputs

- Trained model file (`optimized_fd1.h5`)
- Loss curve (`loss_curve_fd1.png`)
- Predicted vs Actual plot (`pred_vs_actual.png`)
- MAE and other metrics displayed in the terminal

## Next Steps

Proceed to advanced analysis and reporting in Milestone 3 & 4.

---
