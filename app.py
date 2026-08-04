import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global dictionary for models and encoders
models = {}

def load_assets():
    """Load model and label encoders with robust error handling."""
    files_to_load = {
        'model': 'collegename_model.pkl',
        'gender_encoder': 'gender_encoder.pkl',
        'category_encoder': 'category_encoder.pkl',
        'seat_encoder': 'seat_encoder.pkl',
        'target_encoder': 'target_encoder.pkl'
    }
    
    for key, filename in files_to_load.items():
        file_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(file_path):
            print(f"CRITICAL ERROR: File '{filename}' not found at {file_path}")
            sys.exit(1)
        try:
            models[key] = joblib.load(file_path)
            print(f"Successfully loaded: {filename}")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load '{filename}'. Details: {str(e)}")
            sys.exit(1)

# Initialize models upon app starting
load_assets()

@app.route('/', methods=['GET'])
def index():
    """Render main interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to process user input, encode values, predict, and split output."""
    try:
        # Extract inputs from request form
        merit_no_raw = request.form.get('merit_number')
        percentile_raw = request.form.get('percentile')
        gender_raw = request.form.get('gender')
        category_raw = request.form.get('category')
        seat_raw = request.form.get('seat_allotted')

        # 1. Input Validation
        if not all([merit_no_raw, percentile_raw, gender_raw, category_raw, seat_raw]):
            return jsonify({
                'success': False, 
                'error': 'Missing required fields. Please complete all form options.'
            }), 400

        # 2. Numeric Type Conversions
        try:
            merit_number = int(merit_no_raw)
            percentile = float(percentile_raw)
        except ValueError:
            return jsonify({
                'success': False, 
                'error': 'Merit Number must be an integer and Percentile must be a valid number.'
            }), 400

        if not (0 <= percentile <= 100):
            return jsonify({
                'success': False, 
                'error': 'MHTCET Percentile must be strictly between 0 and 100.'
            }), 400

        # 3. Categorical Encodings (with LabelEncoder validation)
        try:
            encoded_gender = int(models['gender_encoder'].transform([str(gender_raw)])[0])
            encoded_category = int(models['category_encoder'].transform([str(category_raw)])[0])
            encoded_seat = int(models['seat_encoder'].transform([str(seat_raw)])[0])
        except ValueError as ve:
            return jsonify({
                'success': False, 
                'error': f'Invalid categorical selection value. Details: {str(ve)}'
            }), 400

        # 4. Feature Construction
        # Feature order strictly matching training data: 
        # ['Merit Number', 'MHTCET Percentile', 'Gender', 'Category', 'Seat Alloted']
        features = np.array([[
            merit_number,
            percentile,
            encoded_gender,
            encoded_category,
            encoded_seat
        ]], dtype=object)

        # 5. Inference Execution
        raw_pred = models['model'].predict(features)
        
        # 6. Target Decoding & Extraction
        decoded_prediction = models['target_encoder'].inverse_transform(raw_pred)[0]

        if " | " in str(decoded_prediction):
            institute, course = str(decoded_prediction).split(" | ", 1)
        else:
            institute = str(decoded_prediction)
            course = "Not Specified"

        # Successful Return payload
        return jsonify({
            'success': True,
            'institute': institute.strip(),
            'course': course.strip()
        }), 200

    except Exception as e:
        # Prevent 500 crashes by catching unexpected internal runtime errors safely
        print(f"UNHANDLED PREDICTION EXCEPTION: {str(e)}")
        return jsonify({
            'success': False, 
            'error': 'An unexpected calculation error occurred. Please verify your inputs and try again.'
        }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
