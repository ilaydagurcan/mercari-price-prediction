"""
Mercari Price Prediction — API Şemaları
Pydantic modelleri ile request/response validasyonu.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Fiyat tahmini için gerekli ürün bilgileri."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Ürün adı",
        examples=["Nike Air Max 90 Size 10"],
    )
    category_name: str = Field(
        default="missing/missing/missing",
        description="Kategori (Ana/Alt1/Alt2 formatında)",
        examples=["Women/Tops & Blouses/Blouse"],
    )
    brand_name: str = Field(
        default="missing",
        description="Marka adı",
        examples=["Nike"],
    )
    item_condition_id: int = Field(
        ...,
        ge=1,
        le=5,
        description="Ürün durumu (1=New, 2=Like New, 3=Good, 4=Fair, 5=Poor)",
    )
    shipping: int = Field(
        ...,
        ge=0,
        le=1,
        description="Kargo (0=alıcı öder, 1=satıcı öder)",
    )
    item_description: str = Field(
        default="No description yet",
        max_length=5000,
        description="Ürün açıklaması",
        examples=["Brand new with box. Never worn."],
    )


class PredictionResponse(BaseModel):
    """Fiyat tahmini sonucu."""

    predicted_price: float = Field(
        description="Tahmin edilen fiyat (USD)"
    )
    price_range: dict = Field(
        description="Güven aralığı (alt ve üst sınır)"
    )
    model_info: dict = Field(
        description="Model bilgileri"
    )


class HealthResponse(BaseModel):
    """Sağlık kontrolü yanıtı."""

    status: str
    model_loaded: bool
    version: str
