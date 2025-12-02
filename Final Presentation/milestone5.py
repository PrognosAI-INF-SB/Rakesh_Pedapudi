
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load final model
model = load_model('../Milestone - 03 & 04/models_m2/optimized_fd1.h5', compile=False)

# Load future test data
data = pd.read_csv('future_test_data.csv')
X = data[['sensor_1', 'sensor_2', 'sensor_3']].values

# No scaler used here
X_scaled = X

# Predict
pred = model.predict(X_scaled).flatten()

result = data.copy()
result['predicted_target'] = pred
result.to_csv('Milestone5_predictions.csv', index=False)

print('Final predictions saved to Milestone5_predictions.csv')
