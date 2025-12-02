
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

os.makedirs('graphs', exist_ok=True)
os.makedirs('result', exist_ok=True)
os.makedirs('models_m2', exist_ok=True)

# Load cleaned test data
test_df = pd.read_csv('../Milestone - 01/fd1_test_cleaned.csv')
X_test = test_df[['sensor_1', 'sensor_2', 'sensor_3']].values

try:
    scaler = joblib.load('models_m2/scaler_fd1_milestone4.save')
    X_test_scaled = scaler.transform(X_test)
except Exception:
    X_test_scaled = X_test

model = load_model('models_m2/optimized_fd1.h5', compile=False)

# Predict
y_pred = model.predict(X_test_scaled).flatten()

# Alert condition
alert_threshold = 0.8
alerts = (y_pred > alert_threshold).astype(int)

alerts_df = pd.DataFrame({
    'timestamp': test_df['timestamp'],
    'predicted_target': y_pred,
    'alert': alerts
})

alerts_df.to_csv('result/alerts_fd1_milestone4.csv', index=False)

print("Milestone 4 alerts generated.")
