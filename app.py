from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model

import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI(title="DeepFake Detection API")

# =========================
# Configuration
# =========================
MODEL_PATH = "model\\model.h5"

FILES_DIR = r"C:\\Users\\aak" # Change this to your actual files directory
# =========================
# Load model ONCE (inference mode)
# =========================
model = load_model(MODEL_PATH, compile=False)


# =========================
# ELA Processing Function
# =========================
def convert_to_ela_image(path: str, quality: int = 85) -> Image.Image:
    """
    Perform Error Level Analysis (ELA) on an image.
    """
    original = Image.open(path).convert("RGB")

    # Temporary compressed image
    temp_path = "temp_ela.jpg"
    original.save(temp_path, "JPEG", quality=quality)
    compressed = Image.open(temp_path)

    # Compute difference
    ela_image = ImageChops.difference(original, compressed)

    # Scale ELA image
    extrema = ela_image.getextrema()
    max_diff = max(ex[1] for ex in extrema) or 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return ela_image


# =========================
# Prediction Endpoint
# =========================
@app.post("/predict/{folder}/{filename}")
def predict_image(folder,filename):
    image_path = os.path.join(FILES_DIR, folder,filename)
    # image_path="E:\\Joget-DX8\\wflow\\app_formuploads\\images\\f1be566a-9683-4ce1-b533-c8a28fb97891\\" + filename
    if not os.path.isfile(image_path):
        print("File not found:", image_path)
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # ELA preprocessing
        ela = convert_to_ela_image(image_path, quality=95)
        ela = ela.resize((128, 128))
        ela_array = np.asarray(ela, dtype=np.float32) / 255.0
        ela_array = np.expand_dims(ela_array, axis=0)

        # Model prediction
        pred = model.predict(ela_array)[0]
        confidence_real = float(pred[0] * 100)
        confidence_fake = float(pred[1] * 100)

        result = "Fake" if confidence_fake > confidence_real else "Real"

        return {
            "file": filename,
            "prediction": result,
            "confidence": max(confidence_real, confidence_fake),
            "probabilities": {
                "real": confidence_real,
                "fake": confidence_fake
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
