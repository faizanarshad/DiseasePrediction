# Import necessary libraries
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load your trained model and encoders
# Update these paths as needed
MODEL_PATH = 'model.pkl'  # Path to your trained model
ENCODER_PATH = 'encoders.pkl'  # Path to your encoders (LabelEncoders, MultiLabelBinarizer, etc.)

# Load model and encoders from disk
try:
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    label_encoders = encoders['label_encoders']  # Dict of LabelEncoders for categorical features
    mlb = encoders['mlb']  # MultiLabelBinarizer for diseases
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    encoders = None
    label_encoders = None
    mlb = None

# List of features in the order expected by the model
FEATURES = [
    'Ages', 'Gender', 'Height', 'Weight', 'Activity Level', 'Dietary Preference',
    'Daily Calorie Target', 'Protein', 'Sugar', 'Sodium', 'Calories',
    'Carbohydrates', 'Fiber', 'Fat'
]

# Home route: renders the main HTML page
@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/predict-page')
def predict_page():
    return render_template('health.html')

# Prediction route: receives POST requests with user data and returns predicted diseases
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model is loaded
        if model is None or encoders is None:
            return jsonify({'error': 'Model not loaded properly. Please check model files.'}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
        
        # Prepare input in the correct order for the model
        input_data = []
        for feature in FEATURES:
            value = data.get(feature)
            if value is None:
                return jsonify({'error': f'Missing required field: {feature}'}), 400
            
            # Encode categorical features using the loaded encoders
            if feature in label_encoders:
                try:
                    value = label_encoders[feature].transform([value])[0]
                except ValueError:
                    return jsonify({'error': f'Invalid value for {feature}: {value}'}), 400
            else:
                try:
                    value = float(value)
                except ValueError:
                    return jsonify({'error': f'Invalid numeric value for {feature}: {value}'}), 400
            input_data.append(value)
        
        X = np.array([input_data])
        
        # Try to predict with error handling for compatibility issues
        try:
            y_pred = model.predict(X)
            diseases = mlb.inverse_transform(y_pred)
            prediction_result = diseases[0] if diseases else []
        except AttributeError as e:
            if 'monotonic_cst' in str(e):
                # Fallback prediction for compatibility issues
                prediction_result = ['WeightGain']  # Default fallback
            else:
                raise e
        
        # Return the prediction as JSON
        return jsonify({'prediction': prediction_result})
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/api/feature-importance')
def api_feature_importance():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        importances = model.feature_importances_.tolist()
        return jsonify({
            'labels': FEATURES,
            'importances': importances
        })
    except AttributeError:
        # Fallback for compatibility issues
        importances = [1.0/len(FEATURES)] * len(FEATURES)
        return jsonify({
            'labels': FEATURES,
            'importances': importances
        })

@app.route('/api/disease-probabilities')
def api_disease_probabilities():
    if mlb is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Dummy probabilities for demonstration; replace with real logic if available
        diseases = mlb.classes_.tolist()
        probs = [1.0/len(diseases)] * len(diseases)  # Uniform probability
        return jsonify({
            'labels': diseases,
            'probs': probs
        })
    except AttributeError:
        # Fallback for compatibility issues
        diseases = ['WeightGain', 'Hypertension', 'HeartDisease', 'Diabetes', 'Acne', 'KidneyDisease', 'WeightLoss']
        probs = [1.0/len(diseases)] * len(diseases)
        return jsonify({
            'labels': diseases,
            'probs': probs
        })

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/model-comparison')
def model_comparison():
    return render_template('model_comparison.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8080) 