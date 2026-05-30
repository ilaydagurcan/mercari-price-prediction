"""
Mercari Price Prediction — Final UI
API: uvicorn src.api.main:app --reload --port 8000
UI:  streamlit run src/ui/streamlit_app.py --server.port 8501
"""

import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Mercari", layout="wide")

API_URL = "http://localhost:8000"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data/raw/train.tsv")


@st.cache_data(show_spinner="Loading dataset...")
def load_lookup_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, sep='\t', usecols=['name', 'price'])
    df = df[df['price'] > 0]
    df['name_lower'] = df['name'].str.lower().str.strip()
    return df


def find_actual_price(df, product_name):
    if df is None or not product_name:
        return None
    name_lower = product_name.lower().strip()
    matches = df[df['name_lower'] == name_lower]
    if len(matches) > 0:
        return float(matches['price'].median())
    return None


REAL_SAMPLES = [
    {'key': 'Nike Shoes', 'name': "Nike Rosche",
     'cat': 'Women/Shoes/Athletic', 'brand': 'Nike', 'cond': 2, 'ship': 1,
     'desc': 'Size 8 womens, great condition'},
    {'key': 'Victoria Secret', 'name': "Victoria's Secret Bra 34B",
     'cat': "Women/Women's Accessories/Bras", 'brand': "Victoria's Secret", 'cond': 1, 'ship': 1,
     'desc': 'Brand new with tags.'},
    {'key': 'iPhone Case', 'name': "iPhone 6 case",
     'cat': "Electronics/Cell Phones & Accessories/Cases, Covers & Skins", 'brand': 'Apple', 'cond': 3, 'ship': 1,
     'desc': 'Silicone case. Used but good condition.'},
    {'key': 'Lululemon', 'name': "Lululemon leggings",
     'cat': 'Women/Athletic Apparel/Pants, Tights, Leggings', 'brand': 'Lululemon', 'cond': 2, 'ship': 0,
     'desc': 'Size 4. Black. Great condition.'},
    {'key': 'PINK Hoodie', 'name': "PINK Hoodie",
     'cat': 'Women/Sweats & Hoodies/Hoodie', 'brand': 'PINK', 'cond': 2, 'ship': 1,
     'desc': 'Victoria Secret hoodie. Size small.'},
    {'key': 'MK Bag', 'name': "MK purse",
     'cat': 'Women/Accessories/Handbags', 'brand': 'Michael Kors', 'cond': 2, 'ship': 0,
     'desc': 'Authentic Michael Kors purse.'},
]

MAIN_CATS = ["Women", "Men", "Electronics", "Kids", "Beauty", "Home",
             "Vintage & Collectibles", "Other", "Handmade", "Sports & Outdoors"]
CONDITIONS_EN = {"New (1)": 1, "Like New (2)": 2, "Good (3)": 3, "Fair (4)": 4, "Poor (5)": 5}

NAV_CATS = [
    ("👗", "Women"), ("👕", "Men"), ("📱", "Electronics"), ("🧸", "Toys"),
    ("🎮", "Gaming"), ("👜", "Handbags"), ("🏠", "Home"), ("🕯️", "Vintage"),
    ("💄", "Beauty"), ("🍼", "Kids"), ("🏀", "Sports"), ("💎", "Handmade"),
    ("📋", "Office"), ("🐾", "Pet"), ("⚽", "Outdoor"), ("🔧", "Tools"),
]

st.markdown("""
<style>
    .main .block-container { padding-top: 0; max-width: 100%; }
    #MainMenu, footer { visibility: hidden; }
    
    /* NAV BAR — tam genişlik, kenardan kenara */
    .top-nav {
        background: #002875;
        padding: 14px 24px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin: -1rem -4rem 12px -4rem;
        padding-left: calc(4rem + 24px);
        padding-right: calc(4rem + 24px);
    }
    .nav-logo { color: white; font-size: 28px; font-weight: 700; }
    .nav-search { flex: 1; background: white; border-radius: 4px; padding: 10px 16px; color: #999; font-size: 15px; }
    .nav-links { display: flex; gap: 20px; color: white; font-size: 15px; }
    .sell-btn { background: white; color: #002875; padding: 6px 18px; border-radius: 4px; font-weight: 700; }
    
    /* Kategori nav — ikonlar ve yazılar BÜYÜK */
    .cat-nav { display: flex; justify-content: center; padding: 18px 0; border-bottom: 1px solid #eee; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
    .cat-nav-item { text-align: center; padding: 6px 14px; }
    .cat-nav-icon { font-size: 32px; display: block; margin-bottom: 5px; }
    .cat-nav-label { font-size: 14px; color: #333; font-weight: 500; }
    
    .hero { background: linear-gradient(90deg, #6d28d9 0%, #8b5cf6 100%); border-radius: 16px; padding: 36px; color: white; margin-bottom: 16px; min-height: 240px; }
    .hero-brand { font-size: 24px; letter-spacing: 3px; font-weight: 700; margin-bottom: 14px; }
    .hero-title { background: #d4ff32; color: #1a1a1a; display: inline-block; padding: 8px 20px; font-size: 38px; font-weight: 900; border-radius: 6px; margin-bottom: 12px; }
    .hero-sub { font-size: 22px; font-weight: 600; }
    
    .cat-card { background: #dbeafe; border-radius: 12px; padding: 20px; min-height: 140px; }
    .cat-card-title { font-size: 24px; font-weight: 700; color: #1a1a1a; margin-top: 4px; }
    .cat-card-sub { font-size: 14px; color: #555; }
    
    /* TAHMIN SAYFASI — tüm yazılar büyük */
    .price-hero { background: linear-gradient(135deg, #6d28d9, #9333ea); border-radius: 16px; padding: 36px; color: white; text-align: center; margin-bottom: 14px; }
    .price-big { font-size: 58px; font-weight: 900; line-height: 1.1; margin: 8px 0; }
    .price-label { font-size: 16px; opacity: 0.9; letter-spacing: 1.5px; font-weight: 700; }
    
    .model-card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; margin-top: 10px; }
    .model-item { border-radius: 8px; padding: 16px; text-align: center; }
    
    .acc-box { background: #f9fafb; border-radius: 8px; padding: 16px; }
    .acc-bar-bg { height: 12px; background: #e5e7eb; border-radius: 6px; overflow: hidden; margin-top: 8px; }
    .acc-bar { height: 100%; border-radius: 6px; }
    
    /* Form label'ları — KALIN ve BÜYÜK */
    .stTextInput > label, .stSelectbox > label, .stTextArea > label, .stRadio > label {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #1a1a1a !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Input alanları büyük */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        font-size: 15px !important;
    }
    
    /* Shipping radio yazıları siyah ve büyük */
    .stRadio > div { flex-direction: row !important; gap: 20px !important; }
    .stRadio div[role="radiogroup"] label { color: #1a1a1a !important; }
    .stRadio div[role="radiogroup"] label p { color: #1a1a1a !important; font-size: 15px !important; }
    .stRadio div[role="radiogroup"] label span { color: #1a1a1a !important; }
</style>
""", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "home"
if "sample_data" not in st.session_state:
    st.session_state.sample_data = None


# ============================================================
# HOME PAGE
# ============================================================
if st.session_state.page == "home":
    st.markdown("""
    <div class="top-nav">
        <div class="nav-logo">mercari</div>
        <div class="nav-search">🔍 Search for anything</div>
        <div class="nav-links">
            <span>Sign up</span><span>Log in</span>
            <span>🔔</span><span>🛒</span>
            <span class="sell-btn">Sell</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    cat_html = '<div class="cat-nav">'
    for icon, label in NAV_CATS:
        cat_html += f'<div class="cat-nav-item"><span class="cat-nav-icon">{icon}</span><span class="cat-nav-label">{label}</span></div>'
    cat_html += '</div>'
    st.markdown(cat_html, unsafe_allow_html=True)
    
    col_h, col_b = st.columns([4, 1])
    with col_h:
        st.markdown("""
        <div class="hero">
            <div class="hero-brand">MERCARI</div>
            <div class="hero-title">Daily Deals Await</div>
            <div class="hero-sub">Fresh finds with built-in savings*</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        if st.button("💰 Price Prediction", type="primary", use_container_width=True):
            st.session_state.page = "predict"
            st.rerun()
        st.caption("🤖 ML-powered")
    
    c1, c2, c3 = st.columns(3)
    cards_top = [("Games, consoles", "Gaming", "🎮"),
                 ("Action figures, minis", "Figurines", "🦸"),
                 ("Booster packs, cards", "Trading Cards", "🃏")]
    for col, (sub, title, emoji) in zip([c1, c2, c3], cards_top):
        with col:
            st.markdown(f"""
            <div class="cat-card">
                <div class="cat-card-sub">{sub}</div>
                <div class="cat-card-title">{title}</div>
                <div style="font-size:50px;text-align:right;opacity:0.3;">{emoji}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    cards_bot = [("Phones, tablets", "Electronics", "💻"),
                 ("Plushies, dolls", "Toys", "🧸"),
                 ("Makeup, fragrance", "Beauty", "💄"),
                 ("Designer bags", "Handbags", "👜")]
    for col, (sub, title, emoji) in zip([c1, c2, c3, c4], cards_bot):
        with col:
            st.markdown(f"""
            <div class="cat-card">
                <div class="cat-card-sub">{sub}</div>
                <div class="cat-card-title">{title}</div>
                <div style="font-size:40px;text-align:right;opacity:0.3;">{emoji}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# PREDICTION PAGE
# ============================================================
elif st.session_state.page == "predict":
    st.markdown("""
    <div class="top-nav">
        <div class="nav-logo">mercari</div>
        <div style="flex:1;text-align:center;color:white;">
            <span style="font-weight:700;font-size:20px;">Price Prediction Tool</span>
            <span style="opacity:0.7;font-size:15px;margin-left:10px;">RMSLE 0.4396</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    _, col_b = st.columns([5, 1])
    with col_b:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.sample_data = None
            st.rerun()
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if not health.get("model_loaded"):
            st.warning("⚠️ API çalışıyor ama model yüklenmedi.")
    except Exception:
        st.error("❌ API bağlantısı yok. Çalıştır: `uvicorn src.api.main:app --reload --port 8000`")
        st.stop()
    
    col_form, col_result = st.columns([1, 1], gap="large")
    
    with col_form:
        st.markdown("### 📝 Item Details")
        
        st.markdown("**⚡ Quick Fill — Products from Dataset**")
        sc = st.columns(3)
        for i, sd in enumerate(REAL_SAMPLES):
            with sc[i % 3]:
                if st.button(sd['key'], key=f"s_{i}", use_container_width=True):
                    st.session_state.sample_data = sd
                    st.rerun()
        
        st.markdown("---")
        
        sample = st.session_state.sample_data
        d_name = sample["name"] if sample else ""
        d_brand = sample["brand"] if sample else ""
        d_desc = sample["desc"] if sample else ""
        d_cat_idx = 0
        d_sub = ""
        if sample:
            main_cat = sample["cat"].split("/")[0]
            if main_cat in MAIN_CATS:
                d_cat_idx = MAIN_CATS.index(main_cat)
            d_sub = "/".join(sample["cat"].split("/")[1:])
        d_cond_idx = (sample["cond"] - 1) if sample else 0
        d_ship = sample["ship"] if sample else 1
        
        name = st.text_input("Product Name *", value=d_name, placeholder="e.g. Nike Air Max 90")
        
        c1, c2 = st.columns(2)
        with c1:
            main_cat_sel = st.selectbox("Category", MAIN_CATS, index=d_cat_idx)
        with c2:
            sub_cat = st.text_input("Subcategory", value=d_sub)
        
        brand = st.text_input("Brand", value=d_brand)
        
        c5, c6 = st.columns(2)
        with c5:
            condition = st.selectbox("Condition", list(CONDITIONS_EN.keys()), index=d_cond_idx)
        with c6:
            shipping = st.radio("Shipping", ["Buyer pays", "Seller pays"], index=d_ship, horizontal=True)
        
        description = st.text_area("Description", value=d_desc, height=100)
        
        predict_btn = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    with col_result:
        if predict_btn:
            if not name.strip():
                st.error("Product name is required")
            else:
                lookup_df = load_lookup_data()
                actual_price = find_actual_price(lookup_df, name)
                
                cat_full = f"{main_cat_sel}/{sub_cat}" if sub_cat else main_cat_sel
                payload = {
                    "name": name.strip(),
                    "category_name": cat_full,
                    "brand_name": brand.strip() if brand.strip() else "missing",
                    "item_condition_id": CONDITIONS_EN[condition],
                    "shipping": 0 if shipping == "Buyer pays" else 1,
                    "item_description": description.strip() if description.strip() else "No description yet",
                }
                
                with st.spinner("🧠 Calculating..."):
                    try:
                        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                        resp.raise_for_status()
                        result = resp.json()
                        
                        pred = result["predicted_price"]
                        ridge_est = result["price_range"].get("ridge_estimate", pred)
                        lgb_est = result["price_range"].get("lgb_estimate", pred)
                        rw = result["model_info"].get("ridge_weight", 0.75)
                        lw = result["model_info"].get("lgb_weight", 0.25)
                        
                        if actual_price is not None:
                            diff = pred - actual_price
                            pct = abs(diff) / actual_price * 100 if actual_price > 0 else 0
                            accuracy = max(0, 100 - pct)
                            diff_color = "#d4ff32" if pct < 20 else "#fbbf24"
                            
                            st.markdown(f"""
                            <div class="price-hero">
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                                    <div>
                                        <div class="price-label">🔮 PREDICTED</div>
                                        <div class="price-big">${pred:.2f}</div>
                                    </div>
                                    <div style="border-left:2px solid rgba(255,255,255,0.25);padding-left:20px;">
                                        <div class="price-label">💵 ACTUAL</div>
                                        <div class="price-big">${actual_price:.2f}</div>
                                    </div>
                                </div>
                                <div style="margin-top:14px;font-size:20px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.2);">
                                    Difference: <span style="color:{diff_color};font-weight:700;">${diff:+.2f}</span> ({pct:.1f}%)
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            bar_color = "#10b981" if pct < 20 else "#eab308" if pct < 40 else "#ef4444"
                            rating = "Excellent" if pct < 10 else "Good" if pct < 20 else "Fair" if pct < 40 else "Poor"
                            st.markdown(f"""
                            <div class="acc-box">
                                <div style="display:flex;justify-content:space-between;font-size:18px;">
                                    <span style="font-weight:700;">🎯 {rating}</span>
                                    <span style="font-weight:700;">{accuracy:.1f}%</span>
                                </div>
                                <div class="acc-bar-bg">
                                    <div class="acc-bar" style="width:{accuracy}%;background:{bar_color};"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="price-hero">
                                <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">
                                    <div>
                                        <div class="price-label">🔮 PREDICTED</div>
                                        <div class="price-big">${pred:.2f}</div>
                                    </div>
                                    <div style="border-left:2px solid rgba(255,255,255,0.25);padding-left:20px;">
                                        <div class="price-label">💵 ACTUAL</div>
                                        <div class="price-big" style="opacity:0.4;">—</div>
                                        <div style="font-size:14px;opacity:0.6;">Not in dataset</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="model-card">
                            <div style="font-size:16px;font-weight:700;color:#374151;margin-bottom:12px;">🔬 MODEL BREAKDOWN</div>
                            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
                                <div class="model-item" style="background:#f5f3ff;">
                                    <div style="font-size:14px;color:#7c3aed;font-weight:700;">RIDGE · {rw*100:.0f}%</div>
                                    <div style="font-size:30px;font-weight:800;">${ridge_est:.2f}</div>
                                </div>
                                <div class="model-item" style="background:#ecfdf5;">
                                    <div style="font-size:14px;color:#10b981;font-weight:700;">LIGHTGBM · {lw*100:.0f}%</div>
                                    <div style="font-size:30px;font-weight:800;">${lgb_est:.2f}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # RAG AI Advisor
                        try:
                            advice_resp = requests.post(f"{API_URL}/advice", json=payload, timeout=60)
                            if advice_resp.status_code == 200:
                                advice = advice_resp.json()
                                explanation = advice.get("explanation", "")
                                similar = advice.get("similar_products", [])
                                llm_used = advice.get("llm_used", False)
                                
                                if explanation:
                                    llm_badge = "🤖 AI" if llm_used else "📋 Template"
                                    st.markdown(f"""
                                    <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:12px;padding:20px;margin-top:12px;">
                                        <div style="font-size:16px;font-weight:700;color:#92400e;margin-bottom:10px;">
                                            🧠 AI PRICE ADVISOR <span style="font-size:13px;opacity:0.7;margin-left:8px;">{llm_badge}</span>
                                        </div>
                                        <div style="font-size:16px;color:#78350f;line-height:1.7;">
                                            {explanation}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                if similar:
                                    sim_html = '<div style="margin-top:12px;"><div style="font-size:16px;font-weight:700;color:#374151;margin-bottom:10px;">📦 SIMILAR PRODUCTS IN DATASET</div>'
                                    for sp in similar[:4]:
                                        sim_html += f'<div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #f3f4f6;font-size:16px;"><span style="color:#374151;">{sp["name"][:40]}</span><span style="font-weight:700;color:#1a1a1a;">${sp["price"]:.0f}</span></div>'
                                    sim_html += '</div>'
                                    st.markdown(sim_html, unsafe_allow_html=True)
                        except Exception:
                            pass
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("💡 Fill in the form or click a sample product, then press Predict Price")
