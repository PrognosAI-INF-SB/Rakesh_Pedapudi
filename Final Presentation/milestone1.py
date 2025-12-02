
import pandas as pd

# Load datasets
train_df = pd.read_csv('fd1_train_preprocessed.csv')
test_df = pd.read_csv('fd1_test_preprocessed.csv')

print("Train data sample:")
print(train_df.head())

print("\nTest data sample:")
print(test_df.head())

print("\nTrain columns:", train_df.columns)
print("\nTest columns:", test_df.columns)

print("\nTrain missing values:\n", train_df.isnull().sum())
print("\nTest missing values:\n", test_df.isnull().sum())

# Clean data
train_clean = train_df.dropna()
test_clean = test_df.dropna()

# Save cleaned files
train_clean.to_csv('fd1_train_cleaned.csv', index=False)
test_clean.to_csv('fd1_test_cleaned.csv', index=False)

print("\nCleaned data saved successfully.")
