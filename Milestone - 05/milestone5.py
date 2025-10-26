import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

model = load_model('../Milestone - 03 & 04/models_m2/optimized_fd1.h5', compile=False)

data = pd.read_csv('future_test_data.csv')
X = data[['sensor_1', 'sensor_2', 'sensor_3']].values
X_scaled = X 


pred = model.predict(X_scaled).flatten()


result = data.copy()
result['predicted_target'] = pred
result.to_csv('Milestone5_predictions.csv', index=False)
print('Final predictions saved to Milestone5_predictions.csv')
