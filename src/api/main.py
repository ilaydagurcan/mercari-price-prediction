"""
Mercari Price Prediction — FastAPI (Ensemble v2)
Ridge v2 + LightGBM SVD ensemble ile fiyat tahmini.

Çalıştırmak için:
    cd mercari-price-prediction
    uvicorn src.api.main:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.sparse import csr_matrix, hstack

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.api.schemas import HealthResponse, PredictionRequest, PredictionResponse

# Global değişkenler
ridge_model = None
lgb_model = None
tfidf_name_word = None
tfidf_name_char = None
tfidf_desc_word = None
tfidf_combined = None
svd_model = None
le_cat_main = None
le_cat_sub1 = None
le_cat_sub2 = None
le_brand = None
top_brands = None

# Ensemble ağırlıkları (06_score_improvement notebook'undan)
RIDGE_WEIGHT = 0.75
LGB_WEIGHT = 0.25

NUMERIC_FEATURES = [
    'item_condition_id', 'shipping', 'cat_main_enc',
    'cat_sub1_enc', 'cat_sub2_enc', 'brand_enc',
    'desc_word_count', 'name_len', 'has_description'
]


def load_models():
    global ridge_model, lgb_model
    global tfidf_name_word, tfidf_name_char, tfidf_desc_word, tfidf_combined
    global svd_model
    global le_cat_main, le_cat_sub1, le_cat_sub2, le_brand, top_brands

    models_dir = os.path.join(PROJECT_ROOT, "models")

    # Ridge v2
    ridge_model = joblib.load(os.path.join(models_dir, "ridge_v2.joblib"))

    # LightGBM
    lgb_model = lgb.Booster(model_file=os.path.join(models_dir, "lgbm_v2.txt"))

    # TF-IDF v2
    tfidf_name_word = joblib.load(os.path.join(models_dir, "tfidf_name_word_v2.joblib"))
    tfidf_name_char = joblib.load(os.path.join(models_dir, "tfidf_name_char_v2.joblib"))
    tfidf_desc_word = joblib.load(os.path.join(models_dir, "tfidf_desc_word_v2.joblib"))
    tfidf_combined = joblib.load(os.path.join(models_dir, "tfidf_combined_v2.joblib"))

    # SVD
    svd_model = joblib.load(os.path.join(models_dir, "svd_v2.joblib"))

    # Encoders (v1 — aynı encoding kullanılıyor)
    le_cat_main = joblib.load(os.path.join(models_dir, "le_cat_main.joblib"))
    le_cat_sub1 = joblib.load(os.path.join(models_dir, "le_cat_sub1.joblib"))
    le_cat_sub2 = joblib.load(os.path.join(models_dir, "le_cat_sub2.joblib"))
    le_brand = joblib.load(os.path.join(models_dir, "le_brand.joblib"))
    top_brands = joblib.load(os.path.join(models_dir, "top_brands.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Ensemble model yükleniyor...")
    load_models()
    print(f"Yüklendi! Ridge + LightGBM (w={RIDGE_WEIGHT}/{LGB_WEIGHT})")
    yield
    print("Kapatılıyor...")


app = FastAPI(
    title="Mercari Price Prediction API",
    description="Ridge + LightGBM ensemble ile fiyat tahmini (RMSLE: 0.4396)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_label_encode(encoder, value):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        return 0


def preprocess_and_predict(req: PredictionRequest) -> dict:
    # Kategori split
    parts = str(req.category_name).split("/")
    while len(parts) < 3:
        parts.append("missing")
    cat_main, cat_sub1, cat_sub2 = parts[0], parts[1], parts[2]

    # Brand
    brand = req.brand_name
    if not brand or brand.strip() == "":
        brand = "missing"
    elif brand not in top_brands and brand != "missing":
        brand = "other"

    # Description
    desc = req.item_description
    if not desc or desc.strip() == "":
        desc = "No description yet"

    # Label encoding
    cat_main_enc = safe_label_encode(le_cat_main, cat_main)
    cat_sub1_enc = safe_label_encode(le_cat_sub1, cat_sub1)
    cat_sub2_enc = safe_label_encode(le_cat_sub2, cat_sub2)
    brand_enc = safe_label_encode(le_brand, brand)

    # Sayısal features
    desc_word_count = len(str(desc).split())
    name_len = len(str(req.name))
    has_description = 1 if desc != "No description yet" else 0

    numeric = np.array(
        [[req.item_condition_id, req.shipping, cat_main_enc,
          cat_sub1_enc, cat_sub2_enc, brand_enc,
          desc_word_count, name_len, has_description]],
        dtype=np.float32,
    )

    # Combined text
    combined = f"{req.name} {brand} {cat_main} {cat_sub2}"

    # === RIDGE TAHMİNİ ===
    X_nw = tfidf_name_word.transform([req.name])
    X_nc = tfidf_name_char.transform([req.name])
    X_dw = tfidf_desc_word.transform([desc])
    X_cb = tfidf_combined.transform([combined])

    X_ridge = hstack([csr_matrix(numeric), X_nw, X_nc, X_dw, X_cb])
    ridge_pred = float(ridge_model.predict(X_ridge)[0])
    ridge_pred = max(ridge_pred, 0)

    # === LIGHTGBM TAHMİNİ ===
    X_text_sparse = hstack([X_nw, X_nc, X_dw, X_cb])
    X_svd = svd_model.transform(X_text_sparse).astype(np.float32)
    X_lgb = np.hstack([numeric, X_svd])
    lgb_pred = float(lgb_model.predict(X_lgb)[0])
    lgb_pred = max(lgb_pred, 0)

    # === ENSEMBLE ===
    ensemble_pred = RIDGE_WEIGHT * ridge_pred + LGB_WEIGHT * lgb_pred

    # log1p → gerçek fiyat
    ridge_price = max(float(np.expm1(ridge_pred)), 3.0)
    lgb_price = max(float(np.expm1(lgb_pred)), 3.0)
    ensemble_price = max(float(np.expm1(ensemble_pred)), 3.0)

    return {
        "ensemble": round(ensemble_price, 2),
        "ridge": round(ridge_price, 2),
        "lgb": round(lgb_price, 2),
    }


# === ENDPOINTS ===

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=ridge_model is not None and lgb_model is not None,
        version="2.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_price(req: PredictionRequest):
    """
    Ensemble (Ridge + LightGBM) ile fiyat tahmini.

    Örnek:
    ```json
    {
        "name": "Nike Air Max 90 Size 10",
        "category_name": "Men/Shoes/Athletic",
        "brand_name": "Nike",
        "item_condition_id": 2,
        "shipping": 1,
        "item_description": "Brand new with box."
    }
    ```
    """
    if ridge_model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")

    try:
        prices = preprocess_and_predict(req)

        return PredictionResponse(
            predicted_price=prices["ensemble"],
            price_range={
                "low": round(prices["ensemble"] * 0.75, 2),
                "high": round(prices["ensemble"] * 1.25, 2),
                "ridge_estimate": prices["ridge"],
                "lgb_estimate": prices["lgb"],
            },
            model_info={
                "model_type": "Ensemble (Ridge + LightGBM)",
                "val_rmsle": 0.4396,
                "ridge_weight": RIDGE_WEIGHT,
                "lgb_weight": LGB_WEIGHT,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")


@app.post("/advice", tags=["RAG Advisor"])
async def price_advice(req: PredictionRequest):
    """
    Fiyat tahmini + LLM ile doğal dilde açıklama.
    Benzer ürünleri dataset'ten bulur (RAG) ve tahmin gerekçesi üretir.
    """
    if ridge_model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi")

    try:
        prices = preprocess_and_predict(req)

        # RAG advisor
        from src.rag.advisor import PriceAdvisor
        advisor = PriceAdvisor()

        product_info = {
            "name": req.name,
            "brand": req.brand_name or "Unknown",
            "category": req.category_name,
            "condition": req.item_condition_id,
            "shipping": req.shipping,
        }

        prediction_info = {
            "predicted_price": prices["ensemble"],
            "ridge_estimate": prices["ridge"],
            "lgb_estimate": prices["lgb"],
        }

        advice = advisor.explain(product_info, prediction_info)

        return {
            "predicted_price": prices["ensemble"],
            "price_range": {
                "low": round(prices["ensemble"] * 0.75, 2),
                "high": round(prices["ensemble"] * 1.25, 2),
                "ridge_estimate": prices["ridge"],
                "lgb_estimate": prices["lgb"],
            },
            "explanation": advice["explanation"],
            "similar_products": advice["similar_products"],
            "llm_used": advice["llm_used"],
            "model_info": {
                "model_type": "Ensemble (Ridge + LightGBM) + RAG Advisor",
                "val_rmsle": 0.4396,
                "ridge_weight": RIDGE_WEIGHT,
                "lgb_weight": LGB_WEIGHT,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advice hatası: {str(e)}")


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Mercari Price Prediction API v3 (Ensemble + RAG)",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "advice": "POST /advice",
        "model": "Ridge (0.75) + LightGBM (0.25) = RMSLE 0.4396",
    }
