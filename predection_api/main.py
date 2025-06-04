import io
from datetime import datetime

import torch
import torchvision.transforms as T
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image

from utils import (
    authenticate_user,
    create_access_token,
    get_current_user,
)
import settings


app = FastAPI()


# --- 1) Connexion à MongoDB (via motor) ---
@app.on_event("startup")
async def startup_db_client():
    app.mongodb_client = AsyncIOMotorClient(settings.MONGODB_URI)
    app.mongodb = app.mongodb_client[settings.MONGODB_DB_NAME]
    # on peut créer un index par exemple sur "timestamp" si besoin
    await app.mongodb[settings.PREDICTIONS_COLLECTION].create_index("timestamp")


@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb_client.close()


# --- 2) Chargement du modèle PyTorch pré-entraîné ---
class ToyClassifier(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_model(path_weights: str, device: torch.device):
    model = ToyClassifier(num_classes=5)  # adapter au nombre de classes réelles
    state_dict = torch.load(path_weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = load_model("model_weights.pth", DEVICE)

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# --- 3) Endpoint pour obtenir un token OAuth2 (flux “password”) ---
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nom d’utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(user["username"])
    return {"access_token": access_token, "token_type": "bearer"}


# --- 4) Endpoint protégé pour la classification d’image ---
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    '''
    - Reçoit une image (multipart/form-data).
    - Retourne la classe prédite.
    - Enregistre chaque prédiction dans MongoDB.
    - Requiert le header Authorization: Bearer <token>.
    '''
    # Vérifier le type de fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Le fichier envoyé n’est pas une image.",
        )

    # Charger l’image en mémoire
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de traiter l’image.",
        )

    # Prétraitement
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Inférence
    with torch.no_grad():
        outputs = MODEL(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_label = torch.max(probs, dim=1)

    predicted_idx = int(top_label.cpu().item())
    confidence = float(top_prob.cpu().item())

    # Enregistrer la prédiction dans MongoDB
    document = {
        "username": current_user["username"],
        "filename": file.filename,
        "predicted_class_idx": predicted_idx,
        "confidence": confidence,
        "timestamp": datetime.utcnow(),
    }
    await app.mongodb[settings.PREDICTIONS_COLLECTION].insert_one(document)

    # Réponse JSON
    return {
        "filename": file.filename,
        "predicted_class_idx": predicted_idx,
        "confidence": confidence,
    }
