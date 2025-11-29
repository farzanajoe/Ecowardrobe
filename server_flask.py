from flask import Flask, request, jsonify, send_from_directory, render_template
from joblib import load
import os
import csv
import statistics
from flask_cors import CORS

app = Flask(__name__, template_folder='template')
CORS(app)

# ---------------------------------------------------------
# Corrected file paths
# ---------------------------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "ml")

MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
LE_ITEM_PATH = os.path.join(BASE_DIR, "le_item.joblib")
LE_MATERIAL_PATH = os.path.join(BASE_DIR, "le_material.joblib")
LE_BRAND_PATH = os.path.join(BASE_DIR, "le_brand.joblib")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", 
                         "Hybrid Sustainable Fashion Class Labels.csv")

# ---------------------------------------------------------
# Safe loader
# ---------------------------------------------------------
def safe_load(path):
    try:
        return load(path)
    except Exception:
        return None

model = safe_load(MODEL_PATH)
le_item = safe_load(LE_ITEM_PATH)
le_material = safe_load(LE_MATERIAL_PATH)
le_brand = safe_load(LE_BRAND_PATH)

# ---------------------------------------------------------
# Brand type bucket mapping (updated with new CSV path)
# ---------------------------------------------------------
BRAND_TYPE_BUCKETS = {}
def _build_brand_type_buckets():
    if not os.path.exists(DATA_PATH):
        return {}

    brands = {}
    try:
        with open(DATA_PATH, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                bid = r.get('Brand_ID')
                try:
                    price = float(r.get('Item_Price_USD') or 0)
                except Exception:
                    continue
                if not bid:
                    continue
                entry = brands.setdefault(bid, {"prices": [], "count": 0})
                entry["prices"].append(price)
                entry["count"] += 1

        brand_list = []
        for bid, info in brands.items():
            if info["prices"]:
                med = statistics.median(info["prices"])
                brand_list.append((bid, med, info["count"]))

        if not brand_list:
            return {}

        brand_list.sort(key=lambda x: x[1])

        n = len(brand_list)
        qsize = max(1, n // 4)
        quartile_map = {1: [], 2: [], 3: [], 4: []}

        for i, (bid, med, cnt) in enumerate(brand_list):
            q = min(4, (i // qsize) + 1)
            quartile_map[q].append((bid, cnt, med))

        for q in quartile_map:
            items = quartile_map[q]
            items.sort(key=lambda x: (-x[1], x[2]))
            quartile_map[q] = [bid for (bid, cnt, med) in items]

        mapping = {
            "fast_fashion": 1,
            "mid_range": 2,
            "sustainable": 3,
            "luxury": 4
        }

        result = {}
        for bt, q in mapping.items():
            candidates = quartile_map.get(q, [])
            if not candidates:
                for offset in (1, -1, 2, -2):
                    alt = q + offset
                    if alt in quartile_map and quartile_map[alt]:
                        candidates = quartile_map[alt]
                        break
            result[bt] = candidates

        return result

    except Exception:
        return {}

BRAND_TYPE_BUCKETS = _build_brand_type_buckets()


def choose_brand_for_type(brand_type: str):
    if not brand_type:
        return None
    bt = brand_type.lower().replace("-", "_")
    candidates = BRAND_TYPE_BUCKETS.get(bt) or []
    return candidates[0] if candidates else None


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/<path:filename>", methods=["GET"])
def serve_static(filename):
    return send_from_directory(os.path.dirname(__file__), filename)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


def normalize_label(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def encode_label(encoder, value, name):
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    norm = normalize_label(value)
    for cls in classes:
        if normalize_label(str(cls)) == norm:
            return int(encoder.transform([cls])[0])
    raise ValueError(f"Unknown {name} label: {value}")


@app.route("/predict", methods=["POST"])
def predict():
    global model, le_item, le_material, le_brand

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json() or {}

    try:
        category = data.get("category")
        material = data.get("material")
        brand = data.get("brand")
        brand_type = data.get("brand_type")
        price = float(data.get("price", 0))

        item_enc = encode_label(le_item, category, "category")
        material_enc = encode_label(le_material, material, "material")

        if brand:
            chosen_brand = brand
        else:
            chosen_brand = choose_brand_for_type(brand_type)

        brand_enc = encode_label(le_brand, chosen_brand, "brand")

        X = [[item_enc, material_enc, brand_enc, price]]

        pred = model.predict(X).tolist()
        proba = model.predict_proba(X).tolist() if hasattr(model, "predict_proba") else None

        return {
            "prediction": pred,
            "probability": proba,
            "chosen_brand": chosen_brand
        }

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))