import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import numpy as np


test_df = pd.read_csv('../Milestone - 01/fd1_test_cleaned.csv')
X_test = test_df[['sensor_1', 'sensor_2', 'sensor_3']].values
y_test = test_df['target'].values


model = load_model('models/optimized_fd1.h5', compile=False)


y_pred = model.predict(X_test).flatten()

mae = np.mean(np.abs(y_pred - y_test))
print(f"Test MAE: {mae:.4f}")

plt.figure()
plt.plot(y_test[:200], label='Actual RUL')
plt.plot(y_pred[:200], label='Predicted RUL')
plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.title('Predicted vs Actual RUL (first 200 samples)')
plt.legend()
plt.savefig('graphs_m2/pred_vs_actual.png')
plt.close()
