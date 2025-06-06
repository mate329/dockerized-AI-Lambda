import json
import joblib
import pandas as pd

from aws_lambda_powertools import Logger

# Initialize logger
logger = Logger(service="injury_prediction")

# Load artifacts once when Lambda container starts
model = joblib.load('injury_model.pkl')
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    event_body = json.loads(event.get('body')) if 'body' in event else event
    request_id = context.aws_request_id
    logger.append_keys(request_id=request_id)

    # Validate event structure
    if 'features' not in event_body:
        logger.error(f"Invalid event structure: {event_body}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid event structure'})
        }

    # Extract features from event
    features = event_body['features']
    
    # Validate features match training
    expected_features = metadata['feature_names']
    if list(features.keys()) != expected_features:
        logger.error(f"Feature mismatch: expected {expected_features}, got {list(features.keys())}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Feature mismatch'})
        }

    return perform_prediction(features, expected_features)

def perform_prediction(features, expected_features):
    try:
        # Create DataFrame with correct feature order
        X = pd.DataFrame([features])[expected_features]
        
        # Make prediction
        prediction = model.predict(X)[0]

        logger.info(f"Prediction made successfully: {prediction} days")
        
        # Return response
        return {
            'statusCode': 200,
            'body': json.dumps({
                'injury_duration_days': float(prediction),
                'model_version': metadata['training_date']
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }