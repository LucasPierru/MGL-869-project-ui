MONGODB_URI = "mongodb://localhost:27017"
MONGODB_DB_NAME = "toy_classifier_db"
PREDICTIONS_COLLECTION = "predictions"
MODEL_FILE_PATH: str = "../ML/models/cnn.keras"
CLASS_LABELS: list[str] = ["airplane","automobile","bird","cat","deer",
                            "dog","frog","horse","ship","truck"]
TRAINING_SERVICE_URL: str = "http://localhost:8001"
HOST: str = "0.0.0.0"
PORT: int = 8000