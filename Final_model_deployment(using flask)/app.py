from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the models and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model_logistic = pickle.load(f)

# Define feature columns used during model training
feature_columns = ['year', 'month', 'average_temperature_median', 'maximum_temperature_median',
                   'minimum_temperature_median', 'precipitation_lag_median', 'snow_depth_lag_median',
                   'wind_speed_lag_median', 'maximum_sustained_wind_speed_lag_median',
                   'wind_gust_lag_median', 'dew_point_lag_median', 'fog_lag_mean',
                   'thunder_lag_mean', 'lat_lag_median', 'lon_lag_median']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/benefits.html')
def benefits():
    return render_template('benefits.html')

@app.route('/prediction.html')
def prediction_page():
    return render_template('prediction.html')

@app.route('/about.html')
def about_page():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    year = int(request.form['year'])
    month = int(request.form['month'])

    # Prepare data for logistic regression
    input_features = {
        'year': [year],
        'month': [month]
    }

    # Create DataFrame with input features
    input_df = pd.DataFrame(input_features)

    # Fill missing columns with zeros
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0.0

    # Scale the input data using the loaded scaler
    input_scaled = scaler.transform(input_df[feature_columns])

    # Predict using logistic regression
    prediction = model_logistic.predict(input_scaled)[0]

    if prediction == 1:
        prediction_message = "Fire is present"
    else:
        prediction_message = "Fire is not present"

    # Return prediction result as JSON
    return jsonify({'prediction': prediction_message})

if __name__ == "__main__":
    app.run(debug=True)
