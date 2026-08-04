import joblib
from sklearn.preprocessing import LabelEncoder

# 1. Load your existing trained model
model = joblib.load("collegename_model.pkl")

# 2. Extract feature names and target classes directly from the model
features = model.feature_names_in_  # ['MHTCET Percentile', 'Gender', 'Category', 'Seat Alloted']
target_classes = model.classes_

print("Detected Features:", features)
print("Target Classes Count:", len(target_classes))

# 3. Define the categorical values used during your model's training
# (Adjust these lists if your training dataset used slightly different names)
genders = ["Female", "Male"]
categories = ["GOPEN", "GSC", "GST", "VJ", "NT1", "NT2", "NT3", "OBC", "SEBC", "EWS"]
seats = ["Home University", "Other than Home University", "State Level"]

# 4. Fit and save LabelEncoders
gender_encoder = LabelEncoder().fit(genders)
category_encoder = LabelEncoder().fit(categories)
seat_encoder = LabelEncoder().fit(seats)
target_encoder = LabelEncoder().fit(target_classes)

joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(category_encoder, "category_encoder.pkl")
joblib.dump(seat_encoder, "seat_encoder.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("Successfully generated all encoder .pkl files!")
