from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model and encoders
# Update these paths as needed
MODEL_PATH = 'model.pkl'  # Path to your trained model
ENCODER_PATH = 'encoders.pkl'  # Path to your encoders (LabelEncoders, MultiLabelBinarizer, etc.)

# Load model and encoders
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)
label_encoders = encoders['label_encoders']  # Dict of LabelEncoders for categorical features
mlb = encoders['mlb']  # MultiLabelBinarizer for diseases

# List of features in the order expected by the model
FEATURES = [
    'Ages', 'Gender', 'Height', 'Weight', 'Activity Level', 'Dietary Preference',
    'Daily Calorie Target', 'Protein', 'Sugar', 'Sodium', 'Calories',
    'Carbohydrates', 'Fiber', 'Fat'
]

@app.route('/')
def index():
    return render_template('health.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Prepare input in the correct order
    input_data = []
    for feature in FEATURES:
        value = data.get(feature)
        # Encode categorical features
        if feature in label_encoders:
            value = label_encoders[feature].transform([value])[0]
        else:
            value = float(value)
        input_data.append(value)
    X = np.array([input_data])
    # Predict
    y_pred = model.predict(X)
    diseases = mlb.inverse_transform(y_pred)
    return jsonify({'diseases': diseases[0] if diseases else []})

if __name__ == '__main__':
    app.run(debug=True) 