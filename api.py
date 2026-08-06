
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from typing import List, Optional

app = FastAPI(
    title="Diabetes Prediction API",
    description="Local API for diabetes classification model",
    version="1.0.0"
)

# Load artifacts at startup
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Request schema — adjust fields to match your actual CSV columns
class PredictionRequest(BaseModel):
    features: List[float]  # Ordered list matching feature_columns

    class Config:
        json_schema_extra = {
            "example": {
                "features": [45.0, 4.7, 75.0, 6.5, 4.2, 1.8, 1.2, 2.5, 0.8, 28.5, 1.0]
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: Optional[List[float]] = None
    input_features: List[float]
    feature_names: List[str]

@app.get("/")
def root():
    return {
        "message": "Diabetes Prediction API is running",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "feature_info": "/feature-info"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/feature-info")
def feature_info():
    return {
        "expected_features": feature_columns,
        "count": len(feature_columns),
        "description": "Send values in this exact order to /predict"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_columns)} features, got {len(request.features)}. "
                   f"Required order: {feature_columns}"
        )

    # Convert to numpy array and scale
    input_array = np.array(request.features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Predict
    pred = model.predict(input_scaled)[0]

    # Get probabilities if available
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0].tolist()

    # Map back to original label if possible
    try:
        label = label_encoder.inverse_transform([pred])[0]
    except:
        label = str(pred)

    return PredictionResponse(
        prediction=int(pred),
        prediction_label=str(label),
        probability=proba,
        input_features=request.features,
        feature_names=feature_columns
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
