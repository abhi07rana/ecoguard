from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask_cors import CORS
import io

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the model and pre-processing tools
model = load_model('EcoGuard_Air_Quality.h5')  # Load your trained model here
scaler = joblib.load('Health_Incidents_Scaler.pkl')  # Load the scaler
label_encoder = joblib.load('Health_Incidents_LabelEncoder.pkl')  # Load the label encoder

# Define a route to upload the CSV file and make predictions
@app.route('/upload', methods=['POST'])
def upload_file():
    # Ensure a file is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    # Check if the file is CSV
    if file.filename == '' or not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

    try:
        # Read the uploaded CSV file into a DataFrame
        data = pd.read_csv(io.StringIO(file.read().decode('utf-8')))

        # Ensure necessary columns are present
        if 'Health Incidents' not in data.columns:
            return jsonify({"error": "'Health Incidents' column is missing in the CSV."}), 400

        # Preprocess the data
        X = data.drop(columns=['Health Incidents', 'Location'], errors='ignore')  # Drop unnecessary columns
        X_scaled = scaler.transform(X)  # Apply the same scaling as during training

        # Make predictions using the model
        predictions = model.predict(X_scaled)

        # Decode the predictions to the actual labels
        predicted_class = np.argmax(predictions, axis=1)
        predicted_labels = label_encoder.inverse_transform(predicted_class)

        # Add predicted labels to the DataFrame
        data['Predicted Health Incidents'] = predicted_labels

        # Return the results as JSON
        result = data.to_dict(orient='records')  # Convert the DataFrame to a list of dictionaries
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the file: {str(e)}"}), 500

# Default route
@app.route('/')
def home():
    return "EcoGuard Model API is running!"

# Run the Flask app (only for local testing, Render will handle the deployment)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use environment variable for the port
    app.run(debug=True, host='0.0.0.0', port=port)
