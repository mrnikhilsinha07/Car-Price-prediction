import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CarDekho Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

.hero-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.2rem;
}
.hero-sub { text-align: center; color: rgba(255,255,255,0.55); font-size: 1rem; margin-bottom: 2rem; }

.card {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px; padding: 1.8rem; margin-bottom: 1.2rem; backdrop-filter: blur(10px);
}
.card-title {
    color: #ffd200; font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 1.2rem;
}

.result-box {
    background: linear-gradient(135deg, #f7971e22, #ffd20022);
    border: 2px solid #ffd200; border-radius: 20px; padding: 2rem; text-align: center;
}
.result-label { color: rgba(255,255,255,0.6); font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; }
.result-price {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #f7971e, #ffd200);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin: 0.4rem 0;
}
.result-range { color: rgba(255,255,255,0.45); font-size: 0.82rem; }

.metric-tile {
    background: rgba(255,255,255,0.07); border-radius: 14px; padding: 1rem;
    text-align: center; border: 1px solid rgba(255,255,255,0.08);
}
.metric-value { font-size: 1.4rem; font-weight: 800; color: #ffd200; }
.metric-label { font-size: 0.72rem; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.08em; }

.deal-good    { background: linear-gradient(135deg,#05471c,#0a7a34); border:2px solid #22c55e; border-radius:16px; padding:1.2rem; text-align:center; }
.deal-fair    { background: linear-gradient(135deg,#1a3a5c,#1e5f8c); border:2px solid #3b82f6; border-radius:16px; padding:1.2rem; text-align:center; }
.deal-over    { background: linear-gradient(135deg,#4a0d0d,#8c1a1a); border:2px solid #ef4444; border-radius:16px; padding:1.2rem; text-align:center; }
.deal-emoji   { font-size:2.2rem; }
.deal-label   { font-size:1.1rem; font-weight:800; margin:0.3rem 0 0.2rem; }
.deal-desc    { font-size:0.8rem; color:rgba(255,255,255,0.65); }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important; color: white !important;
}
label { color: rgba(255,255,255,0.75) !important; font-size: 0.85rem !important; font-weight: 500 !important; }

.stButton > button {
    background: linear-gradient(135deg, #f7971e, #ffd200) !important;
    color: #1a1a2e !important; font-weight: 800 !important; font-size: 1.05rem !important;
    border: none !important; border-radius: 14px !important;
    padding: 0.8rem 2.5rem !important; width: 100% !important; transition: all 0.3s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(255,210,0,0.35) !important; }

hr { border-color: rgba(255,255,255,0.08) !important; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BRAND → MODEL MAPPING
# ─────────────────────────────────────────────
BRAND_MODELS = {
    "Audi": ["A4","A6","A8","Q7"],
    "BMW": ["3","5","6","7","X1","X3","X4","X5","Z4"],
    "Bentley": ["Continental"],
    "Datsun": ["GO","RediGO","redi-GO"],
    "Ferrari": ["GTC4Lusso"],
    "Force": ["Gurkha"],
    "Ford": ["Aspire","Ecosport","Endeavour","Figo","Freestyle"],
    "Honda": ["Amaze","CR","CR-V","City","Civic","Jazz","WR-V"],
    "Hyundai": ["Aura","Creta","Elantra","Grand","Santro","Tucson","Venue","Verna","i10","i20"],
    "ISUZU": ["MUX"], "Isuzu": ["D-Max","MUX"],
    "Jaguar": ["F-PACE","XE","XF"],
    "Jeep": ["Compass","Wrangler"],
    "Kia": ["Carnival","Seltos"],
    "Land Rover": ["Rover"],
    "Lexus": ["ES","NX","RX"],
    "MG": ["Hector"],
    "Mahindra": ["Alturas","Bolero","KUV","KUV100","Marazzo","Scorpio","Thar","XUV300","XUV500"],
    "Maruti": ["Alto","Baleno","Celerio","Ciaz","Dzire LXI","Dzire VXI","Dzire ZXI","Eeco","Ertiga","Ignis","S-Presso","Swift","Swift Dzire","Vitara","Wagon R","XL6"],
    "Maserati": ["Ghibli","Quattroporte"],
    "Mercedes-AMG": ["C"],
    "Mercedes-Benz": ["C-Class","CLS","E-Class","GL-Class","GLS","S-Class"],
    "Mini": ["Cooper"],
    "Nissan": ["Kicks","X-Trail"],
    "Porsche": ["Cayenne","Macan","Panamera"],
    "Renault": ["Duster","KWID","Triber"],
    "Rolls-Royce": ["Ghost"],
    "Skoda": ["Octavia","Rapid","Superb"],
    "Tata": ["Altroz","Harrier","Hexa","Nexon","Safari","Tiago","Tigor"],
    "Toyota": ["Camry","Fortuner","Glanza","Innova","Yaris"],
    "Volkswagen": ["Polo","Vento"],
    "Volvo": ["S90","XC","XC60","XC90"],
}

MODEL_PKL        = "car_model.pkl"
PREPROCESSOR_PKL = "car_preprocessor.pkl"
LE_PKL           = "car_le.pkl"
CSV_PATH         = "cardekho_imputated.csv"

# ─────────────────────────────────────────────
# TRAIN & CACHE MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    if os.path.exists(MODEL_PKL) and os.path.exists(PREPROCESSOR_PKL) and os.path.exists(LE_PKL):
        model        = pickle.load(open(MODEL_PKL, "rb"))
        preprocessor = pickle.load(open(PREPROCESSOR_PKL, "rb"))
        le           = pickle.load(open(LE_PKL, "rb"))
        return model, preprocessor, le

    if not os.path.exists(CSV_PATH):
        st.error(f"❌ '{CSV_PATH}' not found! Place it in the same folder as app.py")
        st.stop()

    df = pd.read_csv(CSV_PATH)
    df.drop(columns=["car_name","brand","Unnamed: 0"], inplace=True, errors="ignore")

    X = df.drop(["selling_price"], axis=1)
    y = df["selling_price"]

    le = LabelEncoder()
    X["model"] = le.fit_transform(X["model"])

    num_features   = X.select_dtypes(exclude="object").columns
    onehot_columns = ["seller_type","fuel_type","transmission_type"]

    preprocessor = ColumnTransformer([
        ("OneHotEncoder",  OneHotEncoder(drop="first", handle_unknown="ignore"), onehot_columns),
        ("standardScaler", StandardScaler(),                                     num_features),
    ], remainder="passthrough")

    X_t = preprocessor.fit_transform(X)
    X_train, X_test, y_train, _ = train_test_split(X_t, y, random_state=42, test_size=0.3)

    model = RandomForestRegressor(n_estimators=500, min_samples_split=2,
                                  max_features=8, max_depth=None, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    pickle.dump(model,        open(MODEL_PKL,        "wb"))
    pickle.dump(preprocessor, open(PREPROCESSOR_PKL, "wb"))
    pickle.dump(le,           open(LE_PKL,           "wb"))
    return model, preprocessor, le

@st.cache_data(show_spinner=False)
def load_raw_data():
    df = pd.read_csv(CSV_PATH)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    return df

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict_price(model, preprocessor, le, inputs):
    model_enc = le.transform([inputs["model"]])[0] if inputs["model"] in le.classes_ else 0
    row = pd.DataFrame([{
        "model": model_enc, "vehicle_age": inputs["vehicle_age"],
        "km_driven": inputs["km_driven"], "seller_type": inputs["seller_type"],
        "fuel_type": inputs["fuel_type"], "transmission_type": inputs["transmission_type"],
        "mileage": inputs["mileage"], "engine": inputs["engine"],
        "max_power": inputs["max_power"], "seats": inputs["seats"],
    }])
    return float(model.predict(preprocessor.transform(row))[0])

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────────
def feature_importance_chart(model, preprocessor):
    try:
        ohe        = preprocessor.named_transformers_["OneHotEncoder"]
        ohe_names  = ohe.get_feature_names_out(["seller_type","fuel_type","transmission_type"]).tolist()
        num_names  = list(preprocessor.named_transformers_["standardScaler"].feature_names_in_)
        all_names  = ohe_names + num_names

        imp = model.feature_importances_[:len(all_names)]
        df_imp = pd.DataFrame({"feature": all_names, "importance": imp})

        # Group OHE back to parent
        def parent(f):
            for p in ["seller_type","fuel_type","transmission_type"]:
                if f.startswith(p): return p
            return f
        df_imp["group"] = df_imp["feature"].apply(parent)
        df_imp = df_imp.groupby("group")["importance"].sum().reset_index()
        df_imp.columns = ["Feature","Importance"]

        readable = {
            "vehicle_age":"Vehicle Age","km_driven":"KM Driven",
            "mileage":"Mileage","engine":"Engine CC","max_power":"Max Power",
            "seats":"Seats","seller_type":"Seller Type",
            "fuel_type":"Fuel Type","transmission_type":"Transmission","model":"Car Model"
        }
        df_imp["Feature"] = df_imp["Feature"].map(lambda x: readable.get(x, x))
        df_imp = df_imp.sort_values("Importance", ascending=True)

        fig = go.Figure(go.Bar(
            x=df_imp["Importance"], y=df_imp["Feature"],
            orientation="h",
            marker=dict(
                color=df_imp["Importance"],
                colorscale=[[0,"#302b63"],[0.5,"#f7971e"],[1,"#ffd200"]],
                showscale=False,
            ),
            text=[f"{v:.1%}" for v in df_imp["Importance"]],
            textposition="outside", textfont=dict(color="white", size=11),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"),
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=12)),
            height=340,
        )
        return fig
    except Exception:
        return None

# ─────────────────────────────────────────────
# PRICE TREND CHART (age vs price for same model)
# ─────────────────────────────────────────────
def price_trend_chart(df_raw, selected_brand, selected_model, current_age, predicted_price):
    subset = df_raw[(df_raw["brand"] == selected_brand) & (df_raw["model"] == selected_model)]
    if len(subset) < 3:
        subset = df_raw[df_raw["brand"] == selected_brand]

    if len(subset) < 3:
        return None

    trend = subset.groupby("vehicle_age")["selling_price"].median().reset_index()
    trend.columns = ["Age (Years)", "Median Price (₹)"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["Age (Years)"], y=trend["Median Price (₹)"],
        mode="lines+markers",
        line=dict(color="#f7971e", width=3),
        marker=dict(size=7, color="#ffd200"),
        name="Market Trend",
        hovertemplate="Age: %{x} yrs<br>Price: ₹%{y:,.0f}<extra></extra>",
    ))
    # Highlight current car
    fig.add_trace(go.Scatter(
        x=[current_age], y=[predicted_price],
        mode="markers",
        marker=dict(size=14, color="#22c55e", symbol="star", line=dict(color="white", width=2)),
        name="Your Car",
        hovertemplate="Your Car<br>Age: %{x} yrs<br>Predicted: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Vehicle Age (Years)", gridcolor="rgba(255,255,255,0.07)", color="white"),
        yaxis=dict(title="Price (₹)", gridcolor="rgba(255,255,255,0.07)", color="white",
                   tickformat=",.0f"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        height=300,
    )
    return fig

# ─────────────────────────────────────────────
# SIMILAR CARS TABLE
# ─────────────────────────────────────────────
def similar_cars(df_raw, selected_brand, selected_model, predicted_price, fuel, transmission):
    subset = df_raw[
        (df_raw["brand"] == selected_brand) &
        (df_raw["model"] == selected_model) &
        (df_raw["fuel_type"] == fuel) &
        (df_raw["transmission_type"] == transmission)
    ].copy()

    if len(subset) < 3:
        subset = df_raw[
            (df_raw["brand"] == selected_brand) &
            (df_raw["model"] == selected_model)
        ].copy()

    if len(subset) < 2:
        return None

    subset = subset.sort_values("selling_price").drop_duplicates("selling_price").head(5)
    subset["vs_predicted"] = ((subset["selling_price"] - predicted_price) / predicted_price * 100).round(1)
    subset["Deal"] = subset["vs_predicted"].apply(
        lambda x: "🟢 Cheaper" if x < -5 else ("🔴 Expensive" if x > 5 else "🟡 Similar")
    )
    display = subset[["car_name","vehicle_age","km_driven","fuel_type","selling_price","vs_predicted","Deal"]].copy()
    display.columns = ["Car Name","Age (Yrs)","KM Driven","Fuel","Price (₹)","vs Predicted (%)","Deal"]
    display["Price (₹)"] = display["Price (₹)"].apply(lambda x: f"₹{x:,.0f}")
    display["KM Driven"] = display["KM Driven"].apply(lambda x: f"{x:,}")
    return display

# ─────────────────────────────────────────────
# DEAL RATING
# ─────────────────────────────────────────────
def get_deal_rating(df_raw, selected_brand, selected_model, km_driven, vehicle_age, predicted_price):
    subset = df_raw[
        (df_raw["brand"] == selected_brand) &
        (df_raw["model"] == selected_model) &
        (df_raw["vehicle_age"].between(vehicle_age - 1, vehicle_age + 1)) &
        (df_raw["km_driven"].between(km_driven * 0.7, km_driven * 1.3))
    ]
    if len(subset) < 3:
        subset = df_raw[(df_raw["brand"] == selected_brand) & (df_raw["model"] == selected_model)]

    if len(subset) < 2:
        return "fair", 0

    market_median = subset["selling_price"].median()
    diff_pct = (predicted_price - market_median) / market_median * 100

    if diff_pct < -8:
        return "good", diff_pct
    elif diff_pct > 8:
        return "over", diff_pct
    else:
        return "fair", diff_pct

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
with st.spinner("🔧 Training model on CarDekho data... (first run only, ~30 sec)"):
    model, preprocessor, le = load_or_train_model()

df_raw = load_raw_data()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🚗 CarDekho Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered used car valuation · Real market data · Instant insights</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FORM + RESULT (top row)
# ─────────────────────────────────────────────
col_form, col_result = st.columns([1.1, 1], gap="large")

with col_form:
    st.markdown('<div class="card"><div class="card-title">🏷️ Car Identity</div>', unsafe_allow_html=True)
    brands = sorted(BRAND_MODELS.keys())
    selected_brand = st.selectbox("Brand", brands, index=brands.index("Maruti"))
    models_list    = BRAND_MODELS[selected_brand]
    selected_model = st.selectbox("Model", models_list)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">📋 Condition</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        vehicle_age = st.number_input("Vehicle Age (years)", 0, 30, 3, 1)
    with c2:
        km_driven = st.number_input("KM Driven", 100, 500000, 30000, 1000)
    seller_type = st.selectbox("Seller Type", ["Individual","Dealer","Trustmark Dealer"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">⚙️ Specs</div>', unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        fuel_type = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG","LPG","Electric"])
        engine    = st.number_input("Engine (cc)", 600, 7000, 1200, 50)
        seats     = st.selectbox("Seats", [2,4,5,6,7,8,9], index=2)
    with s2:
        transmission_type = st.selectbox("Transmission", ["Manual","Automatic"])
        max_power = st.number_input("Max Power (bhp)", 30.0, 650.0, 85.0, 0.5)
        mileage   = st.number_input("Mileage (km/l)", 4.0, 35.0, 18.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

    predict_btn = st.button("🔮 Predict Selling Price", use_container_width=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "predicted_price" not in st.session_state:
    st.session_state.predicted_price = None
    st.session_state.inputs          = None

if predict_btn:
    inputs = dict(
        model=selected_model, vehicle_age=vehicle_age, km_driven=km_driven,
        seller_type=seller_type, fuel_type=fuel_type,
        transmission_type=transmission_type, mileage=mileage,
        engine=engine, max_power=max_power, seats=seats,
    )
    with st.spinner("Calculating..."):
        price = predict_price(model, preprocessor, le, inputs)
    st.session_state.predicted_price = price
    st.session_state.inputs          = inputs
    st.session_state.brand           = selected_brand
    st.session_state.model_name      = selected_model

# ─────────────────────────────────────────────
# RESULT COLUMN
# ─────────────────────────────────────────────
with col_result:
    if st.session_state.predicted_price:
        price  = st.session_state.predicted_price
        inp    = st.session_state.inputs
        s_brand = st.session_state.brand
        s_model = st.session_state.model_name
        low, high = price * 0.90, price * 1.10

        # Price box
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Selling Price</div>
            <div class="result-price">₹{price:,.0f}</div>
            <div class="result-range">Range: ₹{low:,.0f} – ₹{high:,.0f}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric tiles
        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown(f'<div class="metric-tile"><div class="metric-value">₹{price/1e5:.1f}L</div><div class="metric-label">In Lakhs</div></div>', unsafe_allow_html=True)
        with t2:
            dep = min(vehicle_age * 8, 70)
            st.markdown(f'<div class="metric-tile"><div class="metric-value">{dep:.0f}%</div><div class="metric-label">~Depreciation</div></div>', unsafe_allow_html=True)
        with t3:
            ppk = price / max(km_driven, 1)
            st.markdown(f'<div class="metric-tile"><div class="metric-value">₹{ppk:.1f}</div><div class="metric-label">Per KM Value</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Deal Rating
        deal, diff = get_deal_rating(df_raw, s_brand, s_model, inp["km_driven"], inp["vehicle_age"], price)
        if deal == "good":
            cls, emoji, label, desc = "deal-good","🤑","Great Deal!", f"{abs(diff):.1f}% below market average — worth buying"
        elif deal == "over":
            cls, emoji, label, desc = "deal-over","⚠️","Overpriced", f"{abs(diff):.1f}% above market average — negotiate first"
        else:
            cls, emoji, label, desc = "deal-fair","✅","Fair Price", "In line with market average"

        st.markdown(f"""
        <div class="{cls}">
            <div class="deal-emoji">{emoji}</div>
            <div class="deal-label" style="color:white">{label}</div>
            <div class="deal-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Summary
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📝 Summary</div>
            <div style="color:rgba(255,255,255,0.8);font-size:0.88rem;line-height:2;">
                🚘 <b style="color:#ffd200">{s_brand} {s_model}</b><br>
                📅 {vehicle_age} yr{"s" if vehicle_age!=1 else ""} old &nbsp;|&nbsp; {km_driven:,} km<br>
                ⛽ {inp['fuel_type']} &nbsp;|&nbsp; {inp['transmission_type']}<br>
                🔧 {inp['engine']}cc &nbsp;|&nbsp; {inp['max_power']} bhp &nbsp;|&nbsp; {inp['mileage']} km/l<br>
                👥 {inp['seats']} Seats &nbsp;|&nbsp; {inp['seller_type']}
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 1.5rem;">
            <div style="font-size:4rem;margin-bottom:1rem;">🚘</div>
            <div style="color:rgba(255,255,255,0.5);font-size:0.95rem;line-height:1.7;">
                Fill in the car details on the left<br>and click <b style="color:#ffd200">Predict</b> to get<br>an instant price estimate.
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div class="card-title">💡 What you'll get</div>
            <div style="color:rgba(255,255,255,0.55);font-size:0.85rem;line-height:2.1;">
                💰 <b style="color:#ffd200">Predicted Price</b> + safe range<br>
                🤑 <b style="color:#ffd200">Deal Rating</b> — Good / Fair / Overpriced<br>
                📊 <b style="color:#ffd200">Feature Importance</b> — what drives price<br>
                📈 <b style="color:#ffd200">Price Trend</b> — how value changes with age<br>
                🔍 <b style="color:#ffd200">Similar Cars</b> — real market comparison
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INSIGHTS SECTION (below, full width)
# ─────────────────────────────────────────────
if st.session_state.predicted_price:
    price   = st.session_state.predicted_price
    inp     = st.session_state.inputs
    s_brand = st.session_state.brand
    s_model = st.session_state.model_name

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="color:#ffd200;font-weight:700;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:1.2rem;">📊 Market Insights</div>', unsafe_allow_html=True)

    ins1, ins2 = st.columns(2, gap="large")

    # Feature Importance
    with ins1:
        st.markdown('<div class="card"><div class="card-title">🧠 What Drives Car Price?</div>', unsafe_allow_html=True)
        fig_imp = feature_importance_chart(model, preprocessor)
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Price Trend
    with ins2:
        st.markdown('<div class="card"><div class="card-title">📈 Price vs Vehicle Age</div>', unsafe_allow_html=True)
        fig_trend = price_trend_chart(df_raw, s_brand, s_model, inp["vehicle_age"], price)
        if fig_trend:
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<p style="color:rgba(255,255,255,0.4);font-size:0.85rem;">Not enough data for this model</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Similar Cars
    st.markdown('<div class="card"><div class="card-title">🔍 Similar Cars in Market</div>', unsafe_allow_html=True)
    df_similar = similar_cars(df_raw, s_brand, s_model, price, inp["fuel_type"], inp["transmission_type"])
    if df_similar is not None:
        st.dataframe(
            df_similar,
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(f'<p style="color:rgba(255,255,255,0.35);font-size:0.75rem;margin-top:0.5rem;">Showing real listings from CarDekho dataset · Your predicted price: ₹{price:,.0f}</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:rgba(255,255,255,0.4);font-size:0.85rem;">Not enough similar listings found</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:rgba(255,255,255,0.2);font-size:0.78rem;padding:0.5rem 0 1rem;">Random Forest · 15,000+ CarDekho Listings · Streamlit 🚗</div>', unsafe_allow_html=True)