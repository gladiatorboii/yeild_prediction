import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense, Bidirectional, Conv1D, MaxPooling1D, Flatten, Input, BatchNormalization # Import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Set dataset directory
data_dir = "C:/Users/yuga3/Desktop/new/crop_recommendation/dataset"

# Load all datasets
dfs = []
for file in os.listdir(data_dir):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(data_dir, file))
        df["Dataset"] = file.split(".")[0]
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Encode categorical variables
le_crop = LabelEncoder()
le_soil = LabelEncoder()
df['Crop_Type'] = le_crop.fit_transform(df['Crop_Type'])
df['Soil_Type'] = le_soil.fit_transform(df['Soil_Type'])

# Feature Engineering (Polynomial Features)
poly = PolynomialFeatures(degree=2, interaction_only=True)
X = df.drop(columns=['Yield', 'Dataset'])
y = np.log1p(df['Yield'])

# Split data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply polynomial features transformation
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

# Scale features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_poly)
X_val_scaled = scaler_X.transform(X_val_poly)

# Scale target
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

# Reshape input for CNN and LSTM
X_train_cnn_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_val_cnn_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))



# Define a more complex model
def create_model():
    model = Sequential([
        # Input Layer
        Input(shape=(X_train_cnn_lstm.shape[1], X_train_cnn_lstm.shape[2])),
        # Combined CNN and LSTM
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'), # Increased filters
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(256, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001))), # Increased units
        Dropout(0.3),
        Bidirectional(LSTM(256, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001))), # Increased units, added another LSTM
        Dropout(0.3),
        GRU(128, activation='relu', kernel_regularizer=l2(0.001)), # Increased units, Added GRU
        Dropout(0.3),
        Flatten(),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)), # Increased units
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model

model = create_model()

# Add callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True) # Increased patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=12, min_lr=0.000001) # Increased patience, reduced lr

# Train model
history = model.fit(X_train_cnn_lstm, y_train_scaled,
                    validation_data=(X_val_cnn_lstm, y_val_scaled),
                    epochs=250, # Increased epochs
                    batch_size=128,
                    verbose=1,
                    callbacks=[early_stopping, reduce_lr])

# Evaluate model
y_pred_val = model.predict(X_val_cnn_lstm)
y_pred_val = scaler_y.inverse_transform(y_pred_val).flatten()
y_val_actual = scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()

# Calculate metrics
mae = mean_absolute_error(y_val_actual, y_pred_val)
mse = mean_squared_error(y_val_actual, y_pred_val)
rmse = np.sqrt(mse)
r2 = r2_score(y_val_actual, y_pred_val)

print("\nFinal Model Performance:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Calculate overall accuracy
error_margin = 0.10
overall_accuracy = np.mean(np.abs(y_pred_val - y_val_actual) / y_val_actual <= error_margin) * 100
print(f"Overall Accuracy: {overall_accuracy:.2f}%")

# Save final model and preprocessors
model.save(os.path.join(data_dir, "crop_yield_lstm_model_final.keras"))
joblib.dump(le_crop, os.path.join(data_dir, "le_crop.pkl"))
joblib.dump(le_soil, os.path.join(data_dir, "le_soil.pkl"))
joblib.dump(poly, os.path.join(data_dir, "poly_features.pkl"))
joblib.dump(scaler_X, os.path.join(data_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(data_dir, "scaler_y.pkl"))

print("\nModel trained and saved successfully!")
