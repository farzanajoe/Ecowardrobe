from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="EcoWardrobe API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    category: str
    material: str
    brand: str
    price: float
    expected_wear_frequency: float

# ---------------------------------------------------
# Corrected model paths
# ---------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "ml")

MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
LE_ITEM_PATH = os.path.join(BASE_DIR, "le_item.joblib")
LE_MATERIAL_PATH = os.path.join(BASE_DIR, "le_material.joblib")
LE_BRAND_PATH = os.path.join(BASE_DIR, "le_brand.joblib")

def safe_load(path):
    try:
        return load(path)
    except Exception:
        return None

model = safe_load(MODEL_PATH)
le_item = safe_load(LE_ITEM_PATH)
le_material = safe_load(LE_MATERIAL_PATH)
le_brand = safe_load(LE_BRAND_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

def normalize_label(s: str) -> str:
    return ''.join(c for c in s.lower() if c.isalnum())

def encode_label(encoder, value, name):
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    norm = normalize_label(value)
    for cls in classes:
        if normalize_label(str(cls)) == norm:
            return int(encoder.transform([cls])[0])
    raise HTTPException(status_code=400, detail=f"Unknown {name} label: {value}")

@app.post("/predict")
def predict(req: PredictRequest):
    global model

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    item_enc = encode_label(le_item, req.category, "category")
    material_enc = encode_label(le_material, req.material, "material")
    brand_enc = encode_label(le_brand, req.brand, "brand")

    X = [[item_enc, material_enc, brand_enc, req.price, req.expected_wear_frequency]]

    pred = model.predict(X).tolist()
    proba = model.predict_proba(X).tolist() if hasattr(model, "predict_proba") else None

    return {"prediction": pred, "probability": proba}
