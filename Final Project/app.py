import flask
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and encoders
with open('accident_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

@app.route('/')
def index():
    # Pass categorical options to the frontend for dropdowns
    options = {col: list(le.classes_) for col, le in encoders.items()}
    return render_template('index.html', options=options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form
        speed_limit = float(request.form['speed_limit'])
        alcohol_level = float(request.form['alcohol_level'])
        weather = request.form['weather']
        road_type = request.form['road_type']
        vehicle_condition = request.form['vehicle_condition']

        # Encode categorical inputs
        weather_enc = encoders['Weather Conditions'].transform([weather])[0]
        road_enc = encoders['Road Type'].transform([road_type])[0]
        vehicle_enc = encoders['Vehicle Condition'].transform([vehicle_condition])[0]

        # Prepare feature vector
        features = np.array([[speed_limit, alcohol_level, weather_enc, road_enc, vehicle_enc]])

        # Predict
        prediction = model.predict(features)[0]

        # Render result
        options = {col: list(le.classes_) for col, le in encoders.items()}
        return render_template('index.html', prediction=prediction, options=options)
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)