import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global dictionary for assets
assets = {}

# Pre-defined MHT-CET seat allocation categories from dataset
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
    """Load model and encoders into memory with automatic fallback encoders."""
    files_to_load = {
        'model': 'collegename_model.pkl',
        'gender_encoder': 'gender_encoder.pkl',
        'category_encoder': 'category_encoder.pkl',
        'seat_encoder': 'seat_encoder.pkl',
        'target_encoder': 'target_encoder.pkl'
    }
    
    for key, filename in files_to_load.items():
        file_path = os.path.join(BASE_DIR, filename)
        if os.path.exists(file_path):
            try:
                assets[key] = joblib.load(file_path)
                print(f"[SUCCESS] Loaded {filename}")
            except Exception as e:
                print(f"[WARNING] Could not load {filename}: {str(e)}")

    # Fallback initialization if .pkl encoders are not present
    if 'gender_encoder' not in assets:
        assets['gender_encoder'] = LabelEncoder().fit(DEFAULT_GENDERS)
    if 'category_encoder' not in assets:
        assets['category_encoder'] = LabelEncoder().fit(DEFAULT_CATEGORIES)
    if 'seat_encoder' not in assets:
        assets['seat_encoder'] = LabelEncoder().fit(DEFAULT_SEATS)

load_assets()


@app.route('/', methods=['GET'])
def index():
    """Render main web page with dropdown options."""
    try:
        genders = list(assets['gender_encoder'].classes_)
    except Exception:
        genders = DEFAULT_GENDERS

    try:
        categories = list(assets['category_encoder'].classes_)
    except Exception:
        categories = DEFAULT_CATEGORIES

    try:
        seats = list(assets['seat_encoder'].classes_)
    except Exception:
        seats = DEFAULT_SEATS

    return render_template(
        'index.html', 
        genders=genders, 
        categories=categories, 
        seats=seats
    )


@app.route('/predict', methods=['POST'])
def predict():
    """Predict Institute & Course without throwing server errors."""
    try:
        # Extract inputs from form
        merit_no_raw = request.form.get('merit_number')
        percentile_raw = request.form.get('percentile')
        gender_raw = request.form.get('gender')
        category_raw = request.form.get('category')
        seat_raw = request.form.get('seat_allotted')

        # 1. Validation
        if not all([merit_no_raw, percentile_raw, gender_raw, category_raw, seat_raw]):
            return jsonify({
                'success': False,
                'error': 'Missing form inputs. Please fill in all fields.'
            }), 200

        # 2. Type conversions
        try:
            percentile = float(percentile_raw)
            merit_number = int(merit_no_raw)
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid numeric value entered. Please re-check Merit Number and Percentile.'
            }), 200

        if not (0.0 <= percentile <= 100.0):
            return jsonify({
                'success': False,
                'error': 'Percentile must be between 0 and 100.'
            }), 200

        # 3. Safe Categorical Encoding
        def safe_encode(encoder, val, default_list):
            try:
                return int(encoder.transform([str(val)])[0])
            except Exception:
                if str(val) in default_list:
                    return default_list.index(str(val))
                return 0

        encoded_gender = safe_encode(assets['gender_encoder'], gender_raw, DEFAULT_GENDERS)
        encoded_category = safe_encode(assets['category_encoder'], category_raw, DEFAULT_CATEGORIES)
        encoded_seat = safe_encode(assets['seat_encoder'], seat_raw, DEFAULT_SEATS)

        # 4. Construct Feature DataFrame matching exact model feature names
        input_data = pd.DataFrame([{
            'MHTCET Percentile': percentile,
            'Gender': encoded_gender,
            'Category': encoded_category,
            'Seat Alloted': encoded_seat
        }])

        # 5. Inference
        if 'model' not in assets:
            return jsonify({
                'success': False,
                'error': 'Model asset (collegename_model.pkl) is not loaded on the server.'
            }), 200

        raw_pred = assets['model'].predict(input_data)[0]

        # 6. Target Decoding
        if 'target_encoder' in assets:
            try:
                prediction_str = str(assets['target_encoder'].inverse_transform([raw_pred])[0])
            except Exception:
                prediction_str = str(raw_pred)
        else:
            prediction_str = str(raw_pred)

        # 7. Split Target into Institute & Course
        if " | " in prediction_str:
            institute, course = prediction_str.split(" | ", 1)
        else:
            institute = prediction_str
            course = "Computer Engineering / Specified Stream"

        return jsonify({
            'success': True,
            'institute': institute.strip(),
            'course': course.strip()
        }), 200

    except Exception as err:
        print(f"[PREDICTION EXCEPTION]: {str(err)}")
        return jsonify({
            'success': False,
            'error': 'An internal processing error occurred. Please verify form inputs and try again.'
        }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
