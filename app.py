import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle_file(filename):
    """Utility to safely load pickle/joblib files from the project directory."""
    file_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None

# Load your specific model and label encoders
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
            # 1. Verify all .pkl files loaded successfully
            missing_files = []
            if model is None: missing_files.append("collegename_model.pkl")
            if gender_encoder is None: missing_files.append("gender_encoder.pkl")
            if category_encoder is None: missing_files.append("category_encoder.pkl")
            if seat_encoder is None: missing_files.append("seat_encoder.pkl")
            if target_encoder is None: missing_files.append("target_encoder.pkl")

            if missing_files:
                raise ValueError(f"Missing model/encoder file(s): {', '.join(missing_files)}. Please make sure they exist in the root folder.")

            # 2. Extract inputs from the HTML form
            merit_number = request.form.get("merit_number", "").strip()
            percentile = request.form.get("percentile", "").strip()
            gender = request.form.get("gender", "").strip()
            category = request.form.get("category", "").strip()
            seat_alloted = request.form.get("seat_alloted", "").strip()

            if not all([merit_number, percentile, gender, category, seat_alloted]):
                raise ValueError("All form fields are required.")

            percentile_val = float(percentile)

            # 3. Transform categorical inputs using saved LabelEncoders
            try:
                gender_encoded = gender_encoder.transform([gender])[0]
            except Exception as e:
                raise ValueError(f"Invalid option selected for Gender: '{gender}'. {str(e)}")

            try:
                category_encoded = category_encoder.transform([category])[0]
            except Exception as e:
                raise ValueError(f"Invalid option selected for Category: '{category}'. {str(e)}")

            try:
                seat_encoded = seat_encoder.transform([seat_alloted])[0]
            except Exception as e:
                raise ValueError(f"Invalid option selected for Seat Alloted: '{seat_alloted}'. {str(e)}")

            # 4. Predict using collegename_model.pkl
            # Input features: ['MHTCET Percentile', 'Gender', 'Category', 'Seat Alloted']
            features = np.array([[percentile_val, gender_encoded, category_encoded, seat_encoded]])
            pred = model.predict(features)

            # 5. Decode target output
            raw_prediction = target_encoder.inverse_transform(pred)[0]

            # Split target into Institute and Course
            if " | " in raw_prediction:
                institute, course = raw_prediction.split(" | ", 1)
            else:
                institute, course = raw_prediction, "N/A"

            prediction_result = {
                "institute": institute,
                "course": course
            }

        except Exception as e:
            error_message = str(e)

    return render_template("index.html", prediction=prediction_result, error=error_message)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
