import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="YouTube Category Prediction API", version="1.0")

# Points to your successful model artifact from Lab 4
MODEL_PATH = "models/youtube_category_assignable_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Trained model artifact not found at {MODEL_PATH}. Please run Lab 4 tracking first.")

# Load the trained K-Neighbors / TF-IDF Pipeline
model = joblib.load(MODEL_PATH)

class PredictionInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    category_id: str

@app.get("/")
def home():
    return {"status": "online", "message": "YouTube Category Model API is active"}

@app.post("/predict", response_model=PredictionOutput)
def predict_category(payload: PredictionInput):
    try:
        # Generate prediction using your trained pipeline
        pred = model.predict([payload.text])
        return PredictionOutput(category_id=str(pred[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
