import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
from sklearn.preprocessing import MinMaxScaler

file_path = 'https://docs.google.com/spreadsheets/d/1V6cBNRohsvWOEcPUMWyZ6tMAvc25Ji2U/export?format=csv'
data = pd.read_csv(file_path)


# Ensure all values are numeric (convert non-numeric values to NaN)
data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
# Remove rows where all values are zero
data = data.loc[~(data.eq(0).all(axis=1))]

data['resi_value'] = data.apply(lambda row : row['resi_value'] if row['trend_type']==-1 else 100-row['resi_value'], axis = 1)



# # replace all negative values in max_price_move with 0 because out stoploss has been hitted anyway
# data["max_price_move"] = data["max_price_move"].apply(lambda x: max(0, x))
# data
reserve = data
data = data.drop(['time','price','trend_type','max_seen', 'min_seen'], axis=1)
# data = data.drop(['rsi_filter','resistance_filter'], axis=1)
data.info()
data.describe()
data.shape
data.isnull().sum()
# normalized_df = keras.utils.normalize(data)
# type(normalized_df)
# data = pd.DataFrame(normalized_df, columns=["leg_3_to_leg_1_time","leg_2_to_leg_1_time","leg_2_correction_percent","resistance","rsi_filter_passed","normalized_rsi","max_price_move"])
# data
#normalize data
columns_to_normalize = data.columns.difference(['rsi_filter'])
scaler = MinMaxScaler()
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

#print(data)



# Separate features and target variable
X = data.drop('max_move', axis=1)
y = data['max_move']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a StandardScaler for scaling the features
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='sigmoid'),
    Dense(16, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model with mean squared error loss for regression
model.compile(optimizer='sgd', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Test MAE: {test_mae}")

# Predict on new data (example)
predictions = model.predict(X_test_scaled)

import matplotlib.pyplot as plt

# Plot training & validation loss values
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

# Mean Absolute Error
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.show()

test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")

import numpy as np

# Make predictions
predictions = model.predict(X_test_scaled).flatten()

# Compare predictions with actual values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(comparison_df.head(10))  # Display first 10 rows for a quick check

