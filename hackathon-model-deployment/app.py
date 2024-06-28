from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

with open('logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    form_data = request.form
    features = [
        form_data['average_temperature_median'],
        form_data['maximum_temperature_median'],
        form_data['minimum_temperature_median'],
        form_data['precipitation_lag_median'],
        form_data['snow_depth_lag_median'],
        form_data['wind_speed_lag_median'],
        form_data['maximum_sustained_wind_speed_lag_median'],
        form_data['wind_gust_lag_median'],
        form_data['dew_point_lag_median'],
        form_data['fog_lag_mean'],
        form_data['thunder_lag_mean'],
        form_data['lat_lag_median'],
        form_data['lon_lag_median'],
        form_data['year'],
        form_data['month']
    ]

    # Convert features to floats
    try:
        features = np.array(features, dtype=float).reshape(1, -1)
    except ValueError as e:
        return f"Input conversion error: {e}"

    # Debug: Print the features to verify the input data
    print("Features for prediction:", features)

    # Make prediction
    prediction = model.predict(features)

    # Debug: Print the prediction to verify the model output
    print("Prediction:", prediction)

    # Return prediction as 0 or 1
    return str(int(prediction[0]))



if __name__ == '__main__':
    app.run(debug=True)