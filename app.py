import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Define base directory to ensure proper path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle_file(filename):
    """Utility function to safely load pickle/joblib files."""
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None

# Load the trained ML model and encoders
try:
    model = load_pickle_file("collegename_model.pkl")
    gender_encoder = load_pickle_file("gender_encoder.pkl")
    category_encoder = load_pickle_file("category_encoder.pkl")
    seat_encoder = load_pickle_file("seat_encoder.pkl")
    target_encoder = load_pickle_file("target_encoder.pkl")
    print("Model and Encoders loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    error_message = None

    if request.method == "POST":
        try:
            # Check if all required files are loaded
            if not all([model, gender_encoder, category_encoder, seat_encoder, target_encoder]):
                raise ValueError("Model files are missing on the server. Please contact support.")

            # Form Input Extraction
            merit_number = request.form.get("merit_number", "").strip()
            percentile = request.form.get("percentile", "").strip()
            gender = request.form.get("gender", "").strip()
            category = request.form.get("category", "").strip()
            seat_alloted = request.form.get("seat_alloted", "").strip()

            # Input Validations
            if not merit_number or not percentile or not gender or not category or not seat_alloted:
                raise ValueError("All fields are required. Please fill out the form completely.")

            try:
                merit_number = float(merit_number)
            except ValueError:
                raise ValueError("Merit Number must be a valid number.")

            try:
                percentile = float(percentile)
                if not (0 <= percentile <= 100):
                    raise ValueError("MHTCET Percentile must be between 0 and 100.")
            except ValueError:
                raise ValueError("MHTCET Percentile must be a valid numerical percentage.")

            # Categorical Feature Encoding
            try:
                gender_encoded = gender_encoder.transform([gender])[0]
            except Exception:
                raise ValueError(f"Invalid option selected for Gender: {gender}")

            try:
                category_encoded = category_encoder.transform([category])[0]
            except Exception:
                raise ValueError(f"Invalid option selected for Category: {category}")

            try:
                seat_encoded = seat_encoder.transform([seat_alloted])[0]
            except Exception:
                raise ValueError(f"Invalid option selected for Seat Alloted: {seat_alloted}")

            # Construct input vector matching model features:
            # ['MHTCET Percentile', 'Gender', 'Category', 'Seat Alloted']
            features = np.array([[percentile, gender_encoded, category_encoded, seat_encoded]])

            # Prediction
            pred = model.predict(features)

            # Decode Target
            raw_prediction = target_encoder.inverse_transform(pred)[0]

            # Split Prediction into Institute and Course
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
            error_message = f"An unexpected error occurred during prediction: {str(e)}"

    return render_template("index.html", prediction=prediction_result, error=error_message)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
