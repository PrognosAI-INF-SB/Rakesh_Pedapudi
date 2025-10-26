<!-- # Rakesh_Pedapudi -->
PrognosAI – AI-Driven Predictive Maintenance System Using Time-Series Sensor Data
Cohort: Infosys Springboard 6.0 – Rakesh Pedapudi

Milestone 1: Data Preparation & Exploration
Overview
This milestone focuses on collecting, exploring, and preprocessing the sensor data required for predictive maintenance modeling.

Files Included
milestone.py – Script to load, explore, and clean the sensor data

fd1_train_preprocessed.csv – Raw/preprocessed training data

fd1_test_preprocessed.csv – Raw/preprocessed testing data

fd1_train_cleaned.csv – Cleaned training data (output)

fd1_test_cleaned.csv – Cleaned testing data (output)

Data Columns
timestamp – Date and time of measurement

sensor_1, sensor_2, sensor_3 – Sensor readings (numeric)

target – Equipment health status (0: healthy, 1: failure/maintenance)

Steps Performed
Loaded CSV sensor datasets using pandas

Explored data for structure, columns, and missing values

Cleaned the data by removing rows with null/missing values

Saved cleaned datasets for use in future milestones

How to Run
Place all CSV files and milestone.py in the same folder

Run: python milestone.py

Milestone 2: Model Training & Performance Evaluation
Overview
This milestone covers training the predictive model and evaluating performance on sensor time-series data.

Files & Folders
milestone2.py: Trains the neural network using cleaned sensor data

evaluate.py: Loads the trained model and evaluates on test data

models/: Trained model files (e.g., optimized_fd1.h5)

graphs_m2/: Training loss curve and predicted vs actual plots

Steps Performed
Loaded cleaned train/test from Milestone 1

Built and trained a regression model to predict equipment health/remaining useful life

Saved the best-performing model

Evaluated predictions, calculated MAE and metrics

Generated and saved evaluation plots

How to Run
Ensure cleaned CSV files from Milestone 1 are available

Run:

python milestone2.py

python evaluate.py

Milestone 3 & 4: Advanced Analysis & Alert System
Overview
These milestones focus on advanced model evaluation, creating performance/bias reports, and implementing an alerting system for early warning.

Files & Folders
milestone3.py: Advanced model/evaluation & bias analysis

milestone4.py: Alert system and threshold notification

graphs/: Residual plots, prediction trends, etc.

models_m2/: Updated/optimized model files

result/: Analysis reports, alert logs:

alerts_fd1_milestone4.csv

model_bias_analysis.csv

model_performance.csv

Steps Performed
Residual analysis and error trend plotting

Bias analysis across sensor ranges/conditions

Threshold-based alert logic for predictions

Generation of performance reports and logs

How to Run
Run:

python milestone3.py

python milestone4.py

Milestone 5: Interactive Dashboard & Deployment
Overview
Developed a real-time interactive dashboard using Streamlit to visualize predictions, historical data, and manage alerts.

Files/Folders
dashboard.py – Runs the Streamlit dashboard app

milestone5.py – Prediction engine/data processor for dashboard

future_test_data.csv – Example/test input for predictions

Milestone5_predictions.csv – Output results file

requirements.txt – All dependencies listed for easy setup

Screenshot1.png, Screenshot2.png – Dashboard images

Features
Real-time data upload, prediction, and alerting

Multiple visualizations for sensor trends, prediction vs actual, and error analysis

Download and view detailed predictions and performance logs

How to Run
text
pip install -r requirements.txt
streamlit run dashboard.py
Project Structure Summary
text
Milestone - 01/
    milestone.py
    fd1_train_preprocessed.csv
    fd1_test_preprocessed.csv
    fd1_train_cleaned.csv
    fd1_test_cleaned.csv

Milestone - 02/
    milestone2.py
    evaluate.py
    models/
        optimized_fd1.h5
    graphs_m2/
        loss_curve_fd1.png
        pred_vs_actual.png

Milestone - 03 & 04/
    milestone3.py
    milestone4.py
    graphs/
        fd1_predicted_vs_actual.png
        fd1_residual_distribution.png
        fd1_residual_trend.png
    models_m2/
        optimized_fd1.h5
    result/
        alerts_fd1_milestone4.csv
        model_bias_analysis.csv
        model_performance.csv

Milestone - 05/
    dashboard.py
    milestone5.py
    future_test_data.csv
    Milestone5_predictions.csv
    requirements.txt
    Screenshot1.png
    Screenshot2.png
Requirements
All milestones use Python 3.8+ and these main libraries:

pandas

numpy

tensorflow / keras

scikit-learn

matplotlib

streamlit

See Milestone - 05/requirements.txt for exact pip versions.

Contact
Project by Rakesh Pedapudi
Infosys Springboard 6.0
