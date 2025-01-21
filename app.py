import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import os

app = Flask(__name__)

# Initialize model and dataset variables
model = None
dataset = None

# Endpoint to upload data
@app.route('/upload', methods=['POST'])
def upload_data():
    global dataset
    file = request.files.get('file')
    if file:
        # Read CSV file
        dataset = pd.read_csv(file)
        return jsonify({"message": "Data uploaded successfully"}), 200
    else:
        return jsonify({"error": "No file provided"}), 400

# Endpoint to train the model
@app.route('/train', methods=['POST'])
def train_model():
    global model, dataset
    if dataset is None:
        return jsonify({"error": "No dataset uploaded"}), 400

    # Preprocessing the data
    X = dataset[['Temperature', 'Run_Time']]  # Features
    y = dataset['Downtime_Flag']  # Target variable

    # Splitting dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model (Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return jsonify({
        "message": "Model trained successfully",
        "accuracy": accuracy,
        "f1_score": f1
    }), 200

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        return jsonify({"error": "Model is not trained yet"}), 400

    # Get data from the request (Temperature and Run_Time)
    data = request.get_json()
    temperature = data.get('Temperature')
    run_time = data.get('Run_Time')

    if temperature is None or run_time is None:
        return jsonify({"error": "Temperature and Run_Time are required"}), 400

    # Make prediction
    prediction = model.predict([[temperature, run_time]])
    confidence = model.predict_proba([[temperature, run_time]])[0][prediction[0]]

    # Return prediction result in JSON format
    return jsonify({
        "Downtime": "Yes" if prediction[0] == 1 else "No",
        "Confidence": confidence
    }), 200

# Generate a synthetic dataset
def generate_synthetic_data():
    np.random.seed(42)
    num_samples = 1000

    
    machine_ids = np.random.choice([f'M{str(i)}' for i in range(1, 11)], num_samples)
    temperatures = np.random.uniform(70, 120, num_samples)
    run_times = np.random.uniform(30, 300, num_samples)
    
    # Downtime depends on temperature and run_time 
    downtime_flags = np.array([1 if temp > 95 and run_time > 150 else 0 
                               for temp, run_time in zip(temperatures, run_times)])

    # Adding some noise to data for variability (10-15% noise)
    noise = np.random.choice([0, 1], num_samples, p=[0.9, 0.1])
    downtime_flags = np.clip(downtime_flags + noise, 0, 1)  # Ensure flags are binary

    # Create a DataFrame
    data = pd.DataFrame({
        'Machine_ID': machine_ids,
        'Temperature': temperatures,
        'Run_Time': run_times,
        'Downtime_Flag': downtime_flags
    })

    # Saves it to a CSV file
    data.to_csv('synthetic_manufacturing_data.csv', index=False)
    return data

if __name__ == '__main__':
    # Generates synthetic data and saves it
    generate_synthetic_data()

    # Run Flask app
    app.run(debug=True)
