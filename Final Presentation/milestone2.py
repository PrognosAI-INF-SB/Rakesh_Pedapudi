
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os

# Load cleaned training data
train_df = pd.read_csv('../Milestone - 01/fd1_train_cleaned.csv')

X = train_df[['sensor_1', 'sensor_2', 'sensor_3']].values
y = train_df['target'].values

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Save best model
os.makedirs('models', exist_ok=True)
checkpoint = ModelCheckpoint(
    'models/optimized_fd1.h5', save_best_only=True,
    monitor='val_loss', mode='min'
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=[checkpoint]
)

# Plot loss curve
os.makedirs('graphs_m2', exist_ok=True)
plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig('graphs_m2/loss_curve_fd1.png')
plt.close()

print("Milestone 2 completed successfully.")
