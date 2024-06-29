from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the models
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('logistic_model.pkl', 'rb') as f:
    logistic_model = pickle.load(f)


feature_columns = ['year', 'month', 'average_temperature_median', 'maximum_temperature_median',
                   'minimum_temperature_median', 'precipitation_lag_median', 'snow_depth_lag_median',
                   'wind_speed_lag_median', 'maximum_sustained_wind_speed_lag_median',
                   'wind_gust_lag_median', 'dew_point_lag_median', 'fog_lag_mean',
                   'thunder_lag_mean', 'lat_lag_median', 'lon_lag_median']

@app.route('/')
def home():
    return render_template('index.html')

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


    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0.0

    
    input_scaled = scaler.transform(input_df[feature_columns])

    # Predict using logistic regression
    prediction = logistic_model.predict(input_scaled)[0]

    return f'Prediction: {int(prediction)}'

if __name__ == "__main__":
    app.run(debug=True)
