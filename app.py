from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import logging
import os


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler
try:
    model = joblib.load('models/model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model/scaler: {str(e)}")
    raise

# Feature order for consistent processing
FEATURE_ORDER = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity', 'TDS_to_Hardness',
    'Chloramine_to_Trihalomethanes'
]

SCALE_COLUMNS = [
    'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 
    'Turbidity', 'TDS_to_Hardness', 'Chloramine_to_Trihalomethanes'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
            'ph': float(request.form['ph']),
            'Hardness': float(request.form['hardness']),
            'Solids': float(request.form['solids']),
            'Chloramines': float(request.form['chloramines']),
            'Sulfate': float(request.form['sulfate']),
            'Conductivity': float(request.form['conductivity']),
            'Organic_carbon': float(request.form['organic_carbon']),
            'Trihalomethanes': float(request.form['trihalomethanes']),
            'Turbidity': float(request.form['turbidity'])
        }
        
        # Calculate engineered features
        data['TDS_to_Hardness'] = data['Solids'] / data['Hardness']
        data['Chloramine_to_Trihalomethanes'] = data['Chloramines'] / data['Trihalomethanes']
        
        # Create and scale DataFrame
        input_df = pd.DataFrame([data], columns=FEATURE_ORDER)
        input_df[SCALE_COLUMNS] = scaler.transform(input_df[SCALE_COLUMNS])
        
        # Predict
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1] * 100
        
        result = {
            'prediction': int(prediction[0]),
            'probability': round(probability, 2),
            'message': 'Potable' if prediction[0] == 1 else 'Not Potable',
            'features': {
                'ph': data['ph'],
                'Hardness': data['Hardness'],
                'Solids': data['Solids'],
                'Chloramines': data['Chloramines'],
                'Sulfate': data['Sulfate'],
                'Conductivity': data['Conductivity'],
                'Organic_carbon': data['Organic_carbon'],
                'Trihalomethanes': data['Trihalomethanes'],
                'Turbidity': data['Turbidity']
            }
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('error.html', error_message="An error occurred during prediction.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Validate JSON input
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 
                          'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400
        
        # Prepare data
        input_data = {
            'ph': float(data['ph']),
            'Hardness': float(data['hardness']),
            'Solids': float(data['solids']),
            'Chloramines': float(data['chloramines']),
            'Sulfate': float(data['sulfate']),
            'Conductivity': float(data['conductivity']),
            'Organic_carbon': float(data['organic_carbon']),
            'Trihalomethanes': float(data['trihalomethanes']),
            'Turbidity': float(data['turbidity'])
        }
        
        # Calculate engineered features
        input_data['TDS_to_Hardness'] = input_data['Solids'] / input_data['Hardness']
        input_data['Chloramine_to_Trihalomethanes'] = input_data['Chloramines'] / input_data['Trihalomethanes']
        
        # Create and scale DataFrame
        input_df = pd.DataFrame([input_data], columns=FEATURE_ORDER)
        input_df[SCALE_COLUMNS] = scaler.transform(input_df[SCALE_COLUMNS])
        
        # Predict
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1] * 100
        
        response = {
            'prediction': int(prediction[0]),
            'probability': round(probability, 2),
            'status': 'Potable' if prediction[0] == 1 else 'Not Potable',
            'features': input_data
        }
        
        return jsonify(response), 200
    
    except ValueError as e:
        return jsonify({'error': 'Invalid input values'}), 400
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500
    
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or default to 5000
    app.run(host='0.0.0.0', port=port)