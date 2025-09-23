from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List

# Load model and feature info
model_path = os.path.join('models', 'random_forest_model.pkl')
feature_info_path = os.path.join('models', 'feature_info.pkl')

try:
    model = joblib.load(model_path)
    feature_info = joblib.load(feature_info_path)
    feature_names = feature_info['feature_names']
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create dummy model for demonstration
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']

app = FastAPI(title="ML Model API", version="1.0.0")

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    feature_names: List[str]

@app.get("/")
async def root():
    return {"message": "ML Model API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert features to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            feature_names=feature_names
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    return {
        "feature_names": feature_names,
        "model_type": "RandomForestClassifier",
        "input_dimension": len(feature_names)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)