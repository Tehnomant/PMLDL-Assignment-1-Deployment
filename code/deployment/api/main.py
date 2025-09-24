from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List
import uvicorn

app = FastAPI(title="ML Model API", version="1.0.0")

# Initialize as None, load on first use
model = None
feature_names = None
feature_info = None

def load_model():
    global model, feature_names, feature_info
    if model is None:
        try:
            model_path = os.path.join('models', 'random_forest_model.pkl')
            feature_info_path = os.path.join('models', 'feature_info.pkl')
            
            model = joblib.load(model_path)
            feature_info = joblib.load(feature_info_path)
            feature_names = feature_info['feature_names']
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

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
    try:
        load_model()  # Try to load model on health check
        return {"status": "healthy", "model_loaded": model is not None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Load model if not already loaded
        load_model()
        
        # Convert features to numpy array
        features = np.array(request.features).reshape(1, -1)
        
        # Validate input length
        if len(request.features) != len(feature_names):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {len(feature_names)} features, got {len(request.features)}"
            )
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            feature_names=feature_names
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    try:
        load_model()
        return {
            "feature_names": feature_names,
            "model_type": "RandomForestClassifier",
            "input_dimension": len(feature_names),
            "model_loaded": model is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)