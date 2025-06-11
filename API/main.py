# backend/main.py
import io
from datetime import datetime
from typing import Optional

import uvicorn
import numpy as np
import motor.motor_asyncio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import settings

# --- Charger le modèle au démarrage ---
import tensorflow as tf

MODEL_PATH = settings.MODEL_FILE_PATH 
print(f"Loading model from {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# Initialiser FastAPI et Mongo
app = FastAPI(title="Prediction API")
client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URI)
db = client.prediction_service

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    prediction: str
    confidence: float
    actual_class: Optional[str]

async def resize_image(image_data: bytes) -> np.ndarray:
    """Resize image to 224x224 and retourne un array normalisé."""
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image).astype("float32")
    # (224,224,3) → (1,224,224,3)
    return np.expand_dims(arr, axis=0)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Lire et prétraiter l'image
    contents = await file.read()
    img_array = await resize_image(contents)

    # 2. Prédiction
    preds = model.predict(img_array)             # shape (1, n_classes)
    print(f"Prediction shape: {preds.shape}, values: {preds}")
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(preds[0, class_idx])

    # 3. Construction de la réponse
    labels = settings.CLASS_LABELS              # liste de chaînes, ex ["airplane",...]
    predicted_class = labels[class_idx]
    result = {
        "prediction": predicted_class,
        "confidence": confidence
    }
    print(f"Prediction result: {result}")

    # 4. Enregistrement dans MongoDB
    """ await db.predictions.insert_one({
        "prediction": predicted_class,
        "confidence": confidence,
        "timestamp": datetime.utcnow(),
        "actual_class": None
    }) """

    return JSONResponse(result)  # Remplacer par le résultat de la prédiction

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

    return JSONResponse({"message": "Error reported successfully"})

if __name__ == "__main__":
    uvicorn.run("API.main:app", host=settings.HOST, port=settings.PORT, reload=True)
