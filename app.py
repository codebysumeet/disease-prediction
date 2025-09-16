import os
import gdown
import tensorflow as tf
import json
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

app = FastAPI(title="🌿 Plant Disease Prediction API")

# --- Paths ---
MODEL_PATH = "plant_disease_prediction_model.h5"
CLASS_INDICES_PATH = "class_indices.json"

# 🔹 Google Drive file ID (from your shared link)
GDRIVE_FILE_ID = "10W0hrse5qknxel_mg_dxgtz7pqgdSrs2"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# --- Download model if not present ---
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# --- Load model & classes (load once) ---
print("🔄 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)


# --- Helper functions ---
def load_and_preprocess_image(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array


def predict_image_class(image_bytes, top_k=3):
    preprocessed_img = load_and_preprocess_image(image_bytes)
    predictions = model.predict(preprocessed_img)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        class_name = class_indices[str(idx)]
        confidence = float(predictions[idx]) * 100
        results.append({
            "class": class_name,
            "confidence": round(confidence, 2)
        })
    return results


# --- Routes ---
@app.get("/")
def home():
    return {"message": "🌿 Welcome to Plant Disease Prediction API. Use /predict endpoint to upload an image."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        results = predict_image_class(image_bytes, top_k=3)
        return JSONResponse(content={"predictions": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
