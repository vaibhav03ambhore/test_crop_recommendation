from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model and scaler
with open('crop_recom_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# Define the crop dictionary
crop_dict = {
    'rice': 1,
    'maize': 2,
    'chickpea': 3,
    'kidneybeans': 4,
    'pigeonpeas': 5,
    'mothbeans': 6,
    'mungbean': 7,
    'blackgram': 8,
    'lentil': 9,
    'pomegranate': 10,
    'banana': 11,
    'mango': 12,
    'grapes': 13,
    'watermelon': 14,
    'muskmelon': 15,
    'apple': 16,
    'orange': 17,
    'papaya': 18,
    'coconut': 19,
    'cotton': 20,
    'jute': 21,
    'coffee': 22
}

# Create a reverse mapping
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Parse the input JSON
        data = request.json

        # Extract features
        features = np.array([[data['N'], data['P'], data['K'], data['temperature'], 
                              data['humidity'], data['ph'], data['rainfall']]])
        
        # Scale the features
        scaled_features = scaler.transform(features)
        
        # Get probabilities for all crops
        probabilities = model.predict_proba(scaled_features)[0]

        # Make the prediction (single crop)
        # prediction = model.predict(scaled_features)[0]

        # Get the single crop name using reverse mapping
        # recommended_crop = reverse_crop_dict.get(prediction, "Unknown")

        # return jsonify({'recommended_crops': recommended_crop}), 200 
        
        # Find the crop with the highest probability
        top_idx = np.argmax(probabilities)
        top_probability = probabilities[top_idx]
        recommended_crop = reverse_crop_dict.get(top_idx + 1, "Unknown")  # Add 1 since classes are 1-indexed
        
        return jsonify({
            'recommended_crop': recommended_crop,
            'probability': round(float(top_probability) * 100, 2)
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
