import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
import joblib
import os
os.makedirs('graphs', exist_ok=True)
os.makedirs('result', exist_ok=True)
os.makedirs('models_m2', exist_ok=True)


test_df = pd.read_csv('../Milestone - 01/fd1_test_cleaned.csv')
X_test = test_df[['sensor_1', 'sensor_2', 'sensor_3']].values
y_test = test_df['target'].values

try:
    scaler = joblib.load('models_m2/scaler_fd1_milestone4.save')
    X_test = scaler.transform(X_test)
except Exception:
    pass


model = load_model('models_m2/optimized_fd1.h5', compile=False)



y_pred = model.predict(X_test).flatten()

plt.figure()
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('RUL / Target')
plt.title('Predicted vs Actual')
plt.legend()
os.makedirs('graphs', exist_ok=True)
plt.savefig('graphs/fd1_predicted_vs_actual.png')

residuals = y_test - y_pred
plt.figure()
plt.hist(residuals, bins=30)
plt.xlabel('Residual (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.savefig('graphs/fd1_residual_distribution.png')

plt.figure()
plt.plot(residuals)
plt.xlabel('Sample Index')
plt.ylabel('Residual')
plt.title('Residual Trend')
plt.savefig('graphs/fd1_residual_trend.png')

mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals**2))
results_df = pd.DataFrame({'MAE': [mae], 'RMSE': [rmse]})
os.makedirs('result', exist_ok=True)
results_df.to_csv('result/model_performance.csv', index=False)

bins = pd.cut(test_df['sensor_1'], 5)
bias_df = test_df.copy()
bias_df['residual'] = residuals
bias_analysis = bias_df.groupby(bins)['residual'].mean().reset_index()
bias_analysis.to_csv('result/model_bias_analysis.csv', index=False)
