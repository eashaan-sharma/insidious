from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.inference import load_model, predict

# ------------------
# App init
# ------------------
app = FastAPI(title="Pneumonia Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------
# Model + transforms (LOAD ONCE)
# ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "model", "cnn_weights.pth")

model = load_model(WEIGHTS_PATH)
model.to(device)

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------
# Health check
# ------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------
# Prediction endpoint
# ------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_tensor = inference_transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        result = predict(image_tensor, model)

        return result

    except Exception as e:
        return {"error": str(e)}

