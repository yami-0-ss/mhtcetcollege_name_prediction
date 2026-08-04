import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base path directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global container for loaded models and encoders
assets = {}

# Fallback default dropdown choices extracted from MHT-CET CAP allocation dataset
DEFAULT_GENDERS = ['F', 'M']
DEFAULT_CATEGORIES = [
    'DT/VJ', 'NT 1 (NT-B)', 'NT 2 (NT-C)', 'NT 2 (NT-C)$', 'NT 3 (NT-D)', 
    'NT 3 (NT-D)$', 'OBC', 'OBC#', 'OBC$', 'OBC$#', 'OBC/DEF1', 'OBC/DEF2', 
    'OPEN', 'OPEN@', 'Open/DEF1', 'Open/DEF2', 'Open/PH1', 'SBC', 'SBC$', 
    'SC', 'SEBC', 'SEBC$', 'SEBC$#', 'SEBC$#/DEF1', 'SEBC$/DEF1', 'ST'
]
DEFAULT_SEATS = [
    'DEFOBCS', 'DEFOPENS', 'DEFROBCS', 'DEFSEBCS', 'EWS', 'GNT1S', 'GNT2S', 
    'GNT3S', 'GOBCS', 'GOPENH', 'GOPENO', 'GOPENS', 'GSCS', 'GSEBCS', 'GVJS', 
    'LNT1S', 'LNT2S', 'LNT3S', 'LOBCS', 'LOPENH', 'LOPENO', 'LOPENS', 'LSCS', 
    'LSEBCO', 'LSEBCS', 'LSTS', 'MI', 'PWDOPENS', 'TFWS'
]


def load_assets():
    """
    Load model and label encoders into memory with safe exception handling.
    """
    required_files = {
        'model': 'collegename_model.pkl',
        'gender_encoder': 'gender_encoder.pkl',
        'category_encoder': 'category_encoder.pkl',
        'seat_encoder': 'seat_encoder.pkl',
        'target_encoder': 'target_encoder.pkl'
    }
    
    for key, filename in required_files.items():
        file_path = os.path.join(BASE_DIR, filename)
        if not os.path.exists(file_path):
            print(f"[WARNING] Asset file missing: {filename} at {file_path}")
            continue
        try:
            assets[key] = joblib.load(file_path)
            print(f"[SUCCESS] Loaded asset: {filename}")
        except Exception as e:
            print(f"[WARNING] Could not load pickle file {filename}: {str(e)}")


# Initialize model assets on startup
load_assets()


@app.route('/', methods=['GET'])
def index():
    """Render the main UI page with dynamic categorical dropdown options."""
    try:
        genders = list(assets['gender_encoder'].classes_) if 'gender_encoder' in assets else DEFAULT_GENDERS
        categories = list(assets['category_encoder'].classes_) if 'category_encoder' in assets else DEFAULT_CATEGORIES
        seats = list(assets['seat_encoder'].classes_) if 'seat_encoder' in assets else DEFAULT_SEATS
    except Exception:
        genders, categories, seats = DEFAULT_GENDERS, DEFAULT_CATEGORIES, DEFAULT_SEATS

    return render_template(
        'index.html', 
        genders=genders, 
        categories=categories, 
        seats=seats
    )


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to process inputs, encode categorical choices, perform model prediction,
    split the target 'Institute | Course' output, and handle errors cleanly.
    """
    try:
        # Form field extractions
        merit_no_raw = request.form.get('merit_number')
        percentile_raw = request.form.get('percentile')
        gender_raw = request.form.get('gender')
        category_raw = request.form.get('category')
        seat_raw = request.form.get('seat_allotted')

        # 1. Validation for missing fields
        if not all([merit_no_raw, percentile_raw, gender_raw, category_raw, seat_raw]):
            return jsonify({
                'success': False,
                'error': 'Missing required form values. Please complete all input fields.'
            }), 200

        # 2. Convert and validate numerical values
        try:
            merit_number = int(merit_no_raw)
            percentile = float(percentile_raw)
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid numerical value. Merit Number must be an integer and Percentile a valid decimal.'
            }), 200

        if not (0.0 <= percentile <= 100.0):
            return jsonify({
                'success': False,
                'error': 'MHTCET Percentile must strictly be between 0 and 100.'
            }), 200

        # 3. Categorical encoding
        if 'gender_encoder' in assets:
            try:
                encoded_gender = int(assets['gender_encoder'].transform([str(gender_raw)])[0])
            except Exception:
                encoded_gender = 0
        else:
            encoded_gender = DEFAULT_GENDERS.index(gender_raw) if gender_raw in DEFAULT_GENDERS else 0

        if 'category_encoder' in assets:
            try:
                encoded_category = int(assets['category_encoder'].transform([str(category_raw)])[0])
            except Exception:
                encoded_category = 0
        else:
            encoded_category = DEFAULT_CATEGORIES.index(category_raw) if category_raw in DEFAULT_CATEGORIES else 0

        if 'seat_encoder' in assets:
            try:
                encoded_seat = int(assets['seat_encoder'].transform([str(seat_raw)])[0])
            except Exception:
                encoded_seat = 0
        else:
            encoded_seat = DEFAULT_SEATS.index(seat_raw) if seat_raw in DEFAULT_SEATS else 0

        # 4. Feature DataFrame construction matching the trained model feature names:
        # ['MHTCET Percentile', 'Gender', 'Category', 'Seat Alloted']
        input_data = pd.DataFrame([{
            'MHTCET Percentile': percentile,
            'Gender': encoded_gender,
            'Category': encoded_category,
            'Seat Alloted': encoded_seat
        }])

        # 5. Model prediction
        if 'model' not in assets:
            return jsonify({
                'success': False,
                'error': 'Model file (collegename_model.pkl) is not loaded on the server.'
            }), 200

        raw_pred = assets['model'].predict(input_data)

        # 6. Target Inverse Transformation
        if 'target_encoder' in assets:
            prediction = assets['target_encoder'].inverse_transform(raw_pred)[0]
        else:
            prediction = str(raw_pred[0])

        # 7. Split Target output string ("Institute Name | Course Name")
        prediction_str = str(prediction)
        if " | " in prediction_str:
            institute, course = prediction_str.split(" | ", 1)
        else:
            institute = prediction_str
            course = "Course information unavailable"

        return jsonify({
            'success': True,
            'institute': institute.strip(),
            'course': course.strip()
        }), 200

    except Exception as err:
        # Safeguard to prevent HTTP 500 internal crashes
        print(f"[PREDICTION ERROR]: {str(err)}")
        return jsonify({
            'success': False,
            'error': 'An internal processing error occurred while predicting. Please try again.'
        }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
