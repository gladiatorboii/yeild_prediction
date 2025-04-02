from flask import Flask, request, jsonify
import os
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)

# Set dataset directory
data_dir = "C:/Users/yuga3/Desktop/new/crop_recommendation/dataset"

# Load the trained model
model_path = os.path.join(data_dir, "crop_yield_lstm_model_final.keras")
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    print("Error: Model file not found.")
    model = None

# Load preprocessors
preprocessors = {}
preprocessor_files = ["le_crop.pkl", "le_soil.pkl", "poly_features.pkl", "scaler_X.pkl", "scaler_y.pkl"]

for file in preprocessor_files:
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        preprocessors[file.split('.')[0]] = joblib.load(path)
    else:
        print(f"Error: {file} not found.")

# Function to predict yield
def predict_yield(crop, soil, n, p, k, moisture, temp, land_area):
    if model is None or any(key not in preprocessors for key in ["le_crop", "le_soil", "poly_features", "scaler_X", "scaler_y"]):
        print("Error: Model or preprocessors not loaded correctly.")
        return None

    try:
        # Encode categorical inputs
        crop_encoded = preprocessors["le_crop"].transform([crop])[0]
        soil_encoded = preprocessors["le_soil"].transform([soil])[0]
        
        # Prepare input data
        input_data = np.array([[crop_encoded, soil_encoded, n, p, k, moisture, temp, land_area]])
        
        # Apply polynomial features
        input_poly = preprocessors["poly_features"].transform(input_data)
        
        # Scale the input features
        input_scaled = preprocessors["scaler_X"].transform(input_poly)
        
        # Reshape input for LSTM model
        input_scaled = input_scaled.reshape((1, input_scaled.shape[1], 1))
        
        # Predict
        predicted_yield_scaled = model.predict(input_scaled)
        
        # Inverse transform the prediction
        predicted_yield = preprocessors["scaler_y"].inverse_transform(predicted_yield_scaled.reshape(-1, 1))[0][0]
        
        # Inverse log transform
        return np.expm1(predicted_yield)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    crop = data['crop']
    soil = data['soil']
    n = data['n']
    p = data['p']
    k = data['k']
    moisture = data['moisture']
    temp = data['temp']
    land_area = data['land_area']
    
    # Get the predicted yield
    yield_prediction = predict_yield(crop, soil, n, p, k, moisture, temp, land_area)
    
    if yield_prediction is not None:
        return jsonify({'yield': yield_prediction})
    else:
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
