# 📦 Image Classification App (FastAPI + React)

This project allows users to **drag and drop an image** on a web UI, sends it to a **FastAPI backend**, performs **image classification** using a **TensorFlow `.keras` model**, and returns the predicted label. The frontend also updates a selector to reflect the prediction.

---

## 🖥️ Tech Stack

- **Frontend**: React (with drag-and-drop image upload)
- **Backend**: FastAPI + TensorFlow
- **Model Format**: `.keras` (TensorFlow SavedModel format)

---

## 🚀 Getting Started

### 🔧 Backend Setup

#### 1. 🐍 Create and activate a virtual environment

```bash
python -m venv venv
On Windows: cd venv\Scripts
. activate
```

#### 2. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. ▶️ Start the FastAPI server

```bash
uvicorn main:app --reload --port 8000
```

### 🌐 Frontend Setup

#### 1. 📁 Navigate to the frontend folder

```bash
cd frontend
```

#### 2. 📦 Install dependencies

```bash
npm install
```

#### 3. ▶️ Start the React development server

```bash
npm run dev
```

- Frontend will run on port 5173
- Backend will run on port 8000

### 📸 Prediction API

Endpoint: POST /predict

Payload: multipart/form-data with a single file field containing the image.

Response example:

```json
{
  "predicted_class": "cat",
  "confidence": 0.987
}
```

### 🙌 Credits

Built using:

- FastAPI
- TensorFlow
- React
