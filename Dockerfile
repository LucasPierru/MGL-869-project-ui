# Use an official Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Copy project files
COPY API /app/API
COPY ML/models /app/ML/models

# Set environment for FastAPI
ENV MODEL_FILE_PATH="/app/ML/models/efficientnet_model.keras"
ENV HOST="0.0.0.0"
ENV PORT=8000
ENV CLASS_LABELS='["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]'

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port
EXPOSE 8000

# Run the FastAPI app (reload=False for Docker)
CMD ["uvicorn",  "API.main:app", "--host", "0.0.0.0", "--port", "8000"]


