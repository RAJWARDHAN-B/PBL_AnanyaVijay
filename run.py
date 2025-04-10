from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and label encoder
model = joblib.load("career_prediction_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Route to test server
@app.route('/')
def home():
    return "Career Prediction API is running."

# Route to predict career
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')  # Should be a list of 10 numbers

        if not features or len(features) != 10:
            return jsonify({'error': 'Please send exactly 10 feature values.'}), 400

        # Preprocess input
        scaled_features = scaler.transform([features])
        probabilities = model.predict_proba(scaled_features)[0] * 100  # Convert to %
        predictions = dict(zip(label_encoder.inverse_transform(range(len(probabilities))), probabilities))

        # Get top 3 predictions
        top_3 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]

        return jsonify({
            "top_3_careers": [
                {"career": career, "confidence": f"{confidence:.2f}%"}
                for career, confidence in top_3
            ]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
