# 🚗 CarDekho Used Car Price Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**An end-to-end ML web app that predicts used car prices, rates deals, and delivers real market insights — all from 15,000+ real CarDekho listings.**

&nbsp;·&nbsp; [📊 Dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) nbsp;·&nbsp; [🐛 Report Bug](../../issues)

</div>

---

## 📸 App Preview

```
┌─────────────────────────────────────────────────────┐
│  🚗 CarDekho Price Predictor                        │
│  ─────────────────────────────────────────────────  │
│  [Brand ▼]  [Model ▼]                               │
│  Age: 3 yrs    KM: 30,000                           │
│  Fuel: Petrol  Transmission: Manual                 │
│                                                     │
│  [ 🔮 Predict Selling Price ]                       │
│                         ┌──────────────────────┐    │
│                         │  ₹4,85,000           │    │
│                         │  🤑 Great Deal!      │    │
│                         │  8.2% below market   │    │
│                         └──────────────────────┘    │
│  ┌──────────────────┐  ┌──────────────────────┐     │
│  │ 🧠 Feature       │  │ 📈 Price vs Age      │     │
│  │    Importance    │  │    Trend Chart       │     │
│  └──────────────────┘  └──────────────────────┘     │
│  🔍 Similar Cars in Market (real listings)          │
└─────────────────────────────────────────────────────┘
```

---

## ✨ What Makes This Different

Most car price projects stop at "input features → predict price." This one goes further:

| Feature | What it does |
|---|---|
| 🤑 **Deal Rating Engine** | Compares predicted price vs real market median — tells you if it's a Good Deal, Fair, or Overpriced |
| 🧠 **Feature Importance Chart** | Shows *why* the model predicted that price — which factors matter most |
| 📈 **Price Depreciation Trend** | Real data chart: how this exact model loses value year over year, with your car starred |
| 🔍 **Similar Cars Comparison** | Pulls actual listings from the dataset matching your car's specs |
| ⚡ **Auto-caching** | Model trains once, saves as `.pkl` — instant load on every run after |

---

## 🧠 ML Pipeline

```
Raw CSV (15,411 rows)
        │
        ▼
  Data Cleaning
  • Drop car_name, brand (high cardinality)
  • Remove duplicates
        │
        ▼
  Feature Engineering
  • LabelEncoder → model column
  • OneHotEncoder → seller_type, fuel_type, transmission_type
  • StandardScaler → all numerical features
        │
        ▼
  ColumnTransformer (sklearn Pipeline)
        │
        ▼
  RandomForestRegressor
  • n_estimators = 500
  • max_features = 8
  • Tuned via RandomizedSearchCV (100 iterations, 3-fold CV)
        │
        ▼
  Evaluation
  • R² Score, MAE, RMSE on 30% test set
```

---

## 📊 Model Performance

| Metric | Train | Test |
|--------|-------|------|
| R² Score | ~0.98 | ~0.92 |


---

## 🗂️ Project Structure

```
car-price-predictor/
│
├── app.py                    # Main Streamlit app
├── cardekho_imputated.csv    # Dataset (15,411 listings)
├── requirements.txt          # Dependencies
├── README.md                 # This file
│
└── (auto-generated on first run)
    ├── car_model.pkl         # Trained RF model
    ├── car_preprocessor.pkl  # Fitted ColumnTransformer
    └── car_le.pkl            # Fitted LabelEncoder
```

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/car-price-predictor.git
cd car-price-predictor
```

**2. Install dependencies**
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

**3. Run the app**
```bash
streamlit run app.py
```

> First run trains the model (~30 seconds). After that it loads instantly from `.pkl` files.

---

## 🌐 Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → `app.py`
4. Click **Deploy** — live in 2 minutes ✅

---

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
plotly
```

---

## 📁 Dataset

**CarDekho Used Cars Dataset**
- 15,411 listings across 32 brands and 120+ models
- Features: brand, model, vehicle age, KM driven, fuel type, transmission, engine, max power, mileage, seats, seller type
- Target: `selling_price` (₹40,000 – ₹3.95 Crore)

---

## 👤 Author

**Nikhil Sinha**
- GitHub: https://github.com/mrnikhilsinha07
- LinkedIn: [Nikhil Sinha](https://www.linkedin.com/in/nikhil-sinha-b9a7a3330/)

---

<div align="center">
Made with ❤️ and scikit-learn &nbsp;·&nbsp; Star ⭐ this repo if you found it useful!
</div>
