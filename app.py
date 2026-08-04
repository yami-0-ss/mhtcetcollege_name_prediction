import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Resolve path relative to the current file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle_file(filename):
    """Safely load model and encoder pickle files."""
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        try:
            return joblib.load(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to load file '{filename}': {e}")
            return None
    print(f"[WARNING] File not found: '{file_path}'")
    return None

# Load model and encoders explicitly
model = load_pickle_file("collegename_model.pkl")
gender_encoder = load_pickle_file("gender_encoder.pkl")
category_encoder = load_pickle_file("category_encoder.pkl")
seat_encoder = load_pickle_file("seat_encoder.pkl")
target_encoder = load_pickle_file("target_encoder.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    error_message = None

    if request.method == "POST":
        try:
            # 1. Verify all required files are loaded properly
            missing_files = []
            if model is None: missing_files.append("collegename_model.pkl")
            if gender_encoder is None: missing_files.append("gender_encoder.pkl")
            if category_encoder is None: missing_files.append("category_encoder.pkl")
            if seat_encoder is None: missing_files.append("seat_encoder.pkl")
            if target_encoder is None: missing_files.append("target_encoder.pkl")

            if missing_files:
                raise ValueError(f"Missing required model/encoder file(s): {', '.join(missing_files)}. Ensure they are present in the project root folder.")

            # 2. Extract inputs safely from HTML form
            merit_number = request.form.get("merit_number", "").strip()
            percentile = request.form.get("percentile", "").strip()
            gender = request.form.get("gender", "").strip()
            category = request.form.get("category", "").strip()
            seat_alloted = request.form.get("seat_alloted", "").strip()

            # Input presence check
            if not all([merit_number, percentile, gender, category, seat_alloted]):
                raise ValueError("All form fields are required. Please fill out the form completely.")

            # Validate numerical input
            try:
                percentile_val = float(percentile)
            except ValueError:
                raise ValueError("MHTCET Percentile must be a valid number.")

            # 3. Categorical Encodings with specific exception handling
            try:
                gender_encoded = gender_encoder.transform([gender])[0]
            except Exception:
                valid_genders = list(gender_encoder.classes_) if hasattr(gender_encoder, 'classes_') else "Unknown"
                raise ValueError(f"Unrecognized Gender value '{gender}'. Valid options are: {valid_genders}")

            try:
                category_encoded = category_encoder.transform([category])[0]
            except Exception:
                valid_cats = list(category_encoder.classes_) if hasattr(category_encoder, 'classes_') else "Unknown"
                raise ValueError(f"Unrecognized Category value '{category}'. Valid options are: {valid_cats}")

            try:
                seat_encoded = seat_encoder.transform([seat_alloted])[0]
            except Exception:
                valid_seats = list(seat_encoder.classes_) if hasattr(seat_encoder, 'classes_') else "Unknown"
                raise ValueError(f"Unrecognized Seat Alloted value '{seat_alloted}'. Valid options are: {valid_seats}")

            # 4. Predict using collegename_model.pkl
            # Input features: ['MHTCET Percentile', 'Gender', 'Category', 'Seat Alloted']
            features = np.array([[percentile_val, gender_encoded, category_encoded, seat_encoded]])
            pred = model.predict(features)

            # 5. Decode target output
            raw_prediction = target_encoder.inverse_transform(pred)[0]

            if " | " in raw_prediction:
                institute, course = raw_prediction.split(" | ", 1)
            else:
                institute, course = raw_prediction, "N/A"

            prediction_result = {
                "institute": institute,
                "course": course
            }

        except ValueError as ve:
            error_message = str(ve)
        except Exception as e:
            # Catch-all for unexpected crashes
            print(f"[INTERNAL ERROR] {e}")
            error_message = f"Internal Application Error: {str(e)}"

    return render_template("index.html", prediction=prediction_result, error=error_message)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
