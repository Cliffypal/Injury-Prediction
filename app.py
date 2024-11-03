from flask import Flask, request, jsonify, render_template
import pickle
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__, template_folder='template')

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    raise Exception("Model file not found. Please ensure it exists at the specified location.")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

def validate_inputs(data):
    try:
        age = float(data['player-age'])
        weight = float(data['player-weight'])
        height = float(data['player-height'])
        previous_injuries = int(data['previous-injuries'])
        training_intensity = float(data['training-intensity'])
        recovery_time = float(data['recovery-time'])
        return age, weight, height, previous_injuries, training_intensity, recovery_time
    except ValueError:
        raise ValueError("Invalid input type. Ensure all inputs are numbers.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form and validate inputs
        age, weight, height, previous_injuries, training_intensity, recovery_time = validate_inputs(request.form)
        
        # Prepare features for prediction
        input_features = np.array([[age, weight, height, previous_injuries, training_intensity, recovery_time]])
        
        # Make prediction using the loaded model
        prediction = model.predict(input_features)
        
        # Convert prediction to readable result
        result = "Injury Risk: High" if prediction[0] == 1 else "Injury Risk: Low"
        
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return render_template('index.html', prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=False)  # Set to False in production
