"""
Mercari Price Advisor — LangChain + RAG
Tahmin sonucunu doğal dilde açıklayan LLM modülü.

Kullanım:
    from src.rag.advisor import PriceAdvisor
    advisor = PriceAdvisor()
    explanation = advisor.explain(product_info, prediction_result)
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Optional

# LLM backend seçimi
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")  # "ollama" veya "openai"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")  # En hafif model
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw/train.tsv")


class SimilarProductRetriever:
    """Dataset'ten benzer ürünleri bulan RAG retriever."""
    
    def __init__(self, max_results=5):
        self.max_results = max_results
        self.df = None
    
    def load(self):
        if self.df is not None:
            return
        if not os.path.exists(DATA_PATH):
            print("⚠️ Dataset bulunamadı, RAG devre dışı")
            return
        
        # Sadece gerekli sütunları yükle
        self.df = pd.read_csv(
            DATA_PATH, sep='\t',
            usecols=['name', 'category_name', 'brand_name', 'price', 'item_condition_id', 'shipping']
        )
        self.df = self.df[self.df['price'] > 0]
        self.df['name_lower'] = self.df['name'].str.lower().str.strip()
        self.df['brand_lower'] = self.df['brand_name'].fillna('').str.lower().str.strip()
        print(f"RAG: {len(self.df):,} ürün yüklendi")
    
    def find_similar(self, name: str, brand: str = "", category: str = "") -> list:
        """Benzer ürünleri bul — marka + kategori + isim eşleşmesi."""
        if self.df is None:
            self.load()
        if self.df is None:
            return []
        
        name_lower = name.lower().strip()
        brand_lower = brand.lower().strip() if brand else ""
        
        # Arama stratejisi: marka + isim kelimesi eşleşmesi
        name_words = set(name_lower.split())
        
        scores = pd.Series(0, index=self.df.index)
        
        # Marka eşleşmesi (+3 puan)
        if brand_lower and brand_lower not in ('missing', 'other', ''):
            scores += (self.df['brand_lower'] == brand_lower).astype(int) * 3
        
        # İsim kelimesi eşleşmesi (her kelime +1 puan)
        for word in name_words:
            if len(word) > 2:  # Kısa kelimeleri atla
                scores += self.df['name_lower'].str.contains(word, regex=False, na=False).astype(int)
        
        # Kategori eşleşmesi (+2 puan)
        if category:
            cat_main = category.split('/')[0].lower()
            scores += self.df['category_name'].fillna('').str.lower().str.startswith(cat_main).astype(int) * 2
        
        # En yüksek skorlu ürünleri al
        top_idx = scores.nlargest(self.max_results * 2).index
        candidates = self.df.loc[top_idx]
        candidates = candidates[scores.loc[top_idx] > 0]  # Skor 0 olanları çıkar
        
        if len(candidates) == 0:
            return []
        
        results = []
        for _, row in candidates.head(self.max_results).iterrows():
            results.append({
                'name': row['name'],
                'brand': row.get('brand_name', 'unknown'),
                'price': float(row['price']),
                'condition': int(row.get('item_condition_id', 0)),
                'shipping': int(row.get('shipping', 0)),
            })
        
        return results


class PriceAdvisor:
    """LLM ile fiyat açıklaması yapan danışman."""
    
    def __init__(self):
        self.retriever = SimilarProductRetriever()
        self.llm_available = False
        self._check_llm()
    
    def _check_llm(self):
        """LLM bağlantısını kontrol et."""
        if LLM_BACKEND == "ollama":
            try:
                import requests
                resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
                if resp.status_code == 200:
                    models = [m['name'] for m in resp.json().get('models', [])]
                    if any(OLLAMA_MODEL.split(':')[0] in m for m in models):
                        self.llm_available = True
                        print(f"✅ Ollama bağlı: {OLLAMA_MODEL}")
                    else:
                        print(f"⚠️ Ollama çalışıyor ama {OLLAMA_MODEL} yüklü değil")
                        print(f"   Yükle: ollama pull {OLLAMA_MODEL}")
            except Exception:
                print("⚠️ Ollama bağlantısı yok (localhost:11434)")
        
        elif LLM_BACKEND == "openai":
            if OPENAI_KEY:
                self.llm_available = True
                print("✅ OpenAI API key mevcut")
            else:
                print("⚠️ OPENAI_API_KEY ortam değişkeni ayarlanmamış")
    
    def _call_ollama(self, prompt: str) -> str:
        """Ollama API'ye istek at."""
        import requests
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 300}
            },
            timeout=60
        )
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return ""
    
    def _call_openai(self, prompt: str) -> str:
        """OpenAI API'ye istek at."""
        import requests
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 300
            },
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return ""
    
    def _build_prompt(self, product: dict, prediction: dict, similar: list) -> str:
        """LLM için prompt oluştur."""
        similar_text = ""
        if similar:
            prices = [s['price'] for s in similar]
            similar_text = f"""
Similar products found in marketplace data:
- Price range: ${min(prices):.0f} - ${max(prices):.0f}
- Average price: ${np.mean(prices):.0f}
- Examples: {', '.join([f"{s['name']} (${s['price']:.0f})" for s in similar[:3]])}
"""
        
        prompt = f"""You are a pricing expert for the Mercari marketplace. Give a brief, helpful explanation of a price prediction.

Product: {product.get('name', 'Unknown')}
Brand: {product.get('brand', 'Unknown')}
Category: {product.get('category', 'Unknown')}
Condition: {product.get('condition', 'Unknown')}
Shipping: {"Seller pays" if product.get('shipping') == 1 else "Buyer pays"}

Model prediction: ${prediction.get('predicted_price', 0):.2f}
- Ridge estimate: ${prediction.get('ridge_estimate', 0):.2f}
- LightGBM estimate: ${prediction.get('lgb_estimate', 0):.2f}
{similar_text}
Write 3-4 sentences explaining why this price makes sense. Mention the brand, condition, and category effects. If similar products were found, reference the price range. Be concise and practical. Do not use bullet points."""
        
        return prompt
    
    def _fallback_explanation(self, product: dict, prediction: dict, similar: list) -> str:
        """LLM yoksa basit template açıklama."""
        pred = prediction.get('predicted_price', 0)
        brand = product.get('brand', 'Unknown')
        category = product.get('category', 'Unknown').split('/')[0]
        condition_map = {1: "New", 2: "Like New", 3: "Good", 4: "Fair", 5: "Poor"}
        cond = condition_map.get(product.get('condition', 0), "Unknown")
        shipping = "seller pays" if product.get('shipping') == 1 else "buyer pays"
        
        explanation = f"The predicted price for this {brand} product is ${pred:.2f}. "
        
        if cond in ("New", "Like New"):
            explanation += f"Being in {cond.lower()} condition increases its value. "
        elif cond in ("Fair", "Poor"):
            explanation += f"The {cond.lower()} condition reduces its market value. "
        
        if shipping == "seller pays":
            explanation += "Seller-paid shipping typically lowers the listing price by $5-6. "
        
        if similar:
            prices = [s['price'] for s in similar]
            explanation += f"Similar items in the {category} category sell for ${min(prices):.0f}-${max(prices):.0f} on average."
        
        return explanation
    
    def explain(self, product: dict, prediction: dict) -> dict:
        """
        Ürün ve tahmin bilgisiyle açıklama üret.
        
        Returns:
            {
                'explanation': str,  # Doğal dilde açıklama
                'similar_products': list,  # Benzer ürünler
                'llm_used': bool,  # LLM kullanıldı mı
            }
        """
        # RAG: Benzer ürünleri bul
        similar = self.retriever.find_similar(
            name=product.get('name', ''),
            brand=product.get('brand', ''),
            category=product.get('category', '')
        )
        
        # LLM ile açıklama
        if self.llm_available:
            prompt = self._build_prompt(product, prediction, similar)
            try:
                if LLM_BACKEND == "ollama":
                    explanation = self._call_ollama(prompt)
                elif LLM_BACKEND == "openai":
                    explanation = self._call_openai(prompt)
                else:
                    explanation = ""
                
                if explanation.strip():
                    return {
                        'explanation': explanation.strip(),
                        'similar_products': similar,
                        'llm_used': True,
                    }
            except Exception as e:
                print(f"LLM hatası: {e}")
        
        # Fallback: Template açıklama
        return {
            'explanation': self._fallback_explanation(product, prediction, similar),
            'similar_products': similar,
            'llm_used': False,
        }
