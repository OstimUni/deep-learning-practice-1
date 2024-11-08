import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import sys
print(sys.executable)


# Get the current working directory
current_folder = os.getcwd()
relative_path = '../RawData/2000_to_2024.xlsx'
full_path = os.path.join(current_folder, relative_path)
# get raw data from excel file
df = pd.read_excel(full_path)
## Preprocessing
# 1. Drop signals with is_closed = False
df = df[df['is_closed'] == True]
# 2. Drop signals with same signal_time
df = df.drop_duplicates(subset=['signal_time'])
# 3. Normalize rsi_value due to trend_type : if trend_type = -1 then rsi_value = 100 - rsi_value
df['rsi_value'] = df.apply(lambda x: 100 - x['rsi_value'] if x['trend_type'] == -1 else x['rsi_value'], axis=1)
# 4. Drop columns that are not needed including 'is_closed', 'trend_type', 'signal_time', 'signal_price',
# 'tp_1_hit', 'tp_2_hit', 'tp_3_hit', 'tp_4_hit', 'tp_5_hit', 'tp_6_hit', 'tp_7_hit', 'tp_8_hit', 
# last_stop_tp_1_number, last_stop_tp_2_number, last_stop_tp_3_number, last_stop_tp_4_number, last_stop_tp_5_number,
# last_stop_tp_6_number, last_stop_tp_7_number, last_stop_tp_8_number, rd_filter_passed , hd_filter_passed
df = df.drop(columns=['max_price_move','pair_name','max_seen_value','min_seen_value','sl_2_hit','sl_1_hit', 'is_closed', 'trend_type', 'signal_time', 'signal_price', 'tp_1_hit', 'tp_2_hit', 'tp_3_hit', 'tp_4_hit', 'tp_5_hit', 'tp_6_hit', 'tp_7_hit', 'tp_8_hit', 'last_stop_tp_1_number', 'last_stop_tp_2_number', 'last_stop_tp_3_number', 'last_stop_tp_4_number', 'last_stop_tp_5_number', 'last_stop_tp_6_number', 'last_stop_tp_7_number', 'last_stop_tp_8_number', 'rd_filter_passed', 'hd_filter_passed'])


# create init data file 
# Save DataFrame as CSV
df.to_csv('processed_data_init.csv', index=False)

# Load dataset
data = pd.read_csv('processed_data_init.csv')

# Convert boolean columns to int for consistency
data['is_big_fractal_resistance'] = data['is_big_fractal_resistance'].astype(int)
data['rsi_filter_passed'] = data['rsi_filter_passed'].astype(int)

# Visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Features")
plt.show()

# Define features and targets
X = data.drop(['last_tp_sl_1_number', 'last_tp_sl_2_number'], axis=1)
y1 = data['last_tp_sl_1_number']
y2 = data['last_tp_sl_2_number']

# Convert targets to categorical
y1 = to_categorical(y1 + 1)  # shift by 1 for range -1 to 8
y2 = to_categorical(y2 + 1)

# Split data into training and testing sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define model architecture
input_layer = Input(shape=(X_train.shape[1],))
hidden_layer_1 = Dense(64, activation='relu')(input_layer)
dropout_1 = Dropout(0.3)(hidden_layer_1)
hidden_layer_2 = Dense(32, activation='relu')(dropout_1)
dropout_2 = Dropout(0.3)(hidden_layer_2)

# Separate outputs for two targets
output1 = Dense(y1.shape[1], activation='softmax', name="output_1")(dropout_2)
output2 = Dense(y2.shape[1], activation='softmax', name="output_2")(dropout_2)

# Compile model
model = Model(inputs=input_layer, outputs=[output1, output2])
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Display model summary
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, [y1_train, y2_train],
                    validation_data=(X_test, [y1_test, y2_test]),
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping])

# Plot training history
plt.figure(figsize=(12, 4))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['output_1_accuracy'], label='Output 1 Training Accuracy')
plt.plot(history.history['val_output_1_accuracy'], label='Output 1 Validation Accuracy')
plt.plot(history.history['output_2_accuracy'], label='Output 2 Training Accuracy')
plt.plot(history.history['val_output_2_accuracy'], label='Output 2 Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Predictions
y1_pred, y2_pred = model.predict(X_test)
y1_pred_classes = np.argmax(y1_pred, axis=1) - 1
y2_pred_classes = np.argmax(y2_pred, axis=1) - 1

# Convert y1_test and y2_test to original classes for comparison
y1_test_classes = np.argmax(y1_test, axis=1) - 1
y2_test_classes = np.argmax(y2_test, axis=1) - 1

# Classification report and confusion matrix for both outputs
print("Classification Report for last_tp_sl_1_number:")
print(classification_report(y1_test_classes, y1_pred_classes))
print("Confusion Matrix for last_tp_sl_1_number:")
print(confusion_matrix(y1_test_classes, y1_pred_classes))

print("\nClassification Report for last_tp_sl_2_number:")
print(classification_report(y2_test_classes, y2_pred_classes))
print("Confusion Matrix for last_tp_sl_2_number:")
print(confusion_matrix(y2_test_classes, y2_pred_classes))
