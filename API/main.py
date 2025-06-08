# backend/main.py
import io
from datetime import datetime
from typing import Optional

import uvicorn
import httpx
import numpy as np
import motor.motor_asyncio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from . import settings

# --- Charger le modèle au démarrage ---
import tensorflow as tf
MODEL_PATH = settings.MODEL_FILE_PATH 
model = tf.keras.models.load_model(MODEL_PATH)

# Initialiser FastAPI et Mongo
app = FastAPI(title="Prediction API")
client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URI)
db = client.prediction_service

class PredictionRequest(BaseModel):
    prediction: str
    confidence: float
    actual_class: Optional[str]

async def resize_image(image_data: bytes) -> np.ndarray:
    """Resize image to 32×32 and retourne un array normalisé."""
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((32, 32))
    arr = np.array(image).astype("float32") / 255.0
    # (32,32,3) → (1,32,32,3)
    return np.expand_dims(arr, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Lire et prétraiter l'image
    contents = await file.read()
    img_array = await resize_image(contents)

    # 2. Prédiction
    preds = model.predict(img_array)             # shape (1, n_classes)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(preds[0, class_idx])

    # 3. Construction de la réponse
    labels = settings.CLASS_LABELS              # liste de chaînes, ex ["airplane",...]
    predicted_class = labels[class_idx]
    result = {
        "prediction": predicted_class,
        "confidence": confidence
    }

    # 4. Enregistrement dans MongoDB
    await db.predictions.insert_one({
        "prediction": predicted_class,
        "confidence": confidence,
        "timestamp": datetime.utcnow(),
        "actual_class": None
    })

    return JSONResponse(result= {"r":"merci"})  # Remplacer par le résultat de la prédiction

@app.post("/report-error")
async def report_error(data: PredictionRequest):
    doc = await db.predictions.find_one(
        {"prediction": data.prediction, "confidence": data.confidence, "actual_class": None},
        sort=[("_id", -1)]
    )
    if not doc:
        return JSONResponse({"error": "Prediction not found"}, status_code=404)

    # Met à jour la classe réelle
    await db.predictions.update_one({"_id": doc["_id"]}, {"$set": {"actual_class": data.actual_class}})

    # Optionnel : déclencher un retraining
    async with httpx.AsyncClient() as client:
        await client.post(f"{settings.TRAINING_SERVICE_URL}/trigger-retrain")

    return JSONResponse({"message": "Error reported successfully"})

if __name__ == "__main__":
    uvicorn.run("API.main:app", host=settings.HOST, port=settings.PORT, reload=True)
