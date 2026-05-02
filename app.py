import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

st.set_page_config(page_title="House Price AI", layout="wide")

# -----------------------------
# 🔥 UI STYLE
# -----------------------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f0c29, #1a0f3c, #000000);
    color: white;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
}
.neon {
    color: #c77dff;
    text-shadow: 0 0 12px #9d4edd;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
xgb = joblib.load("models/xgboost.pkl")
rf = joblib.load("models/random_forest.pkl")
lr = joblib.load("models/linear_regression.pkl")

scaler = joblib.load("models/scaler.pkl")
columns = joblib.load("models/columns.pkl")

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h1 class='neon'>🏠 House Price AI</h1>", unsafe_allow_html=True)

# -----------------------------
# MODEL SELECT
# -----------------------------
model_choice = st.selectbox("Select Model", ["XGBoost", "Random Forest", "Linear Regression"])
model = {"XGBoost": xgb, "Random Forest": rf, "Linear Regression": lr}[model_choice]

# -----------------------------
# MULTI COLOR SIDEBAR
# -----------------------------
def slider(label, min_val, max_val, default, color, key):
    st.sidebar.markdown(f"<span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
    return st.sidebar.slider(label, min_val, max_val, default, key=key)

st.sidebar.header("Property Details")

input_data = {
    'OverallQual': slider("Quality", 1, 10, 7, "#ff4ecd", "oq"),
    'GrLivArea': slider("Area", 500, 4000, 2000, "#4cc9f0", "area"),
    'GarageCars': slider("Garage", 0, 4, 2, "#80ed99", "garage"),
    'GarageArea': slider("Garage Area", 0, 1000, 500, "#ffd166", "garea"),
    'TotalBsmtSF': slider("Basement", 0, 2000, 1000, "#c77dff", "bsmt"),
    '1stFlrSF': slider("1st Floor", 500, 3000, 1500, "#ff6b6b", "floor"),
    'FullBath': slider("Bath", 0, 4, 2, "#f72585", "bath"),
    'YearBuilt': slider("Year", 1900, 2025, 2005, "#06d6a0", "year")
}

df_input = pd.DataFrame([input_data])

# ALIGN
for col in columns:
    if col not in df_input.columns:
        df_input[col] = 0

df_input = df_input[columns]
scaled = scaler.transform(df_input)

# -----------------------------
# PREDICTION
# -----------------------------
prediction = model.predict(scaled)[0]

st.markdown(f"## 💰 Predicted Price: ₹ {prediction:,.0f}")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/cleaned_train.csv")
X = pd.read_csv("data/X_processed.csv")
y = pd.read_csv("data/y_processed.csv")

# -----------------------------
# 📊 BUSINESS INSIGHTS (CARDS)
# -----------------------------
st.markdown("## 📊 Business Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Price", f"₹ {df['SalePrice'].mean():,.0f}")
col2.metric("Max Price", f"₹ {df['SalePrice'].max():,.0f}")
col3.metric("Min Price", f"₹ {df['SalePrice'].min():,.0f}")

# -----------------------------
# 📈 TIME TREND
# -----------------------------
st.markdown("## 📈 Price Trend")

trend = df.groupby("YearBuilt")["SalePrice"].mean().reset_index()

fig_trend = px.line(trend, x="YearBuilt", y="SalePrice", color_discrete_sequence=['#c77dff'])
fig_trend.update_layout(template="plotly_dark")

st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------
# 📊 ACTUAL VS PREDICTED
# -----------------------------
st.markdown("## 📊 Model Performance")

preds = model.predict(X.sample(300))

fig = px.scatter(
    x=y.sample(300).values.flatten(),
    y=preds,
    color=preds,
    color_continuous_scale='plasma'
)

fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 🧠 SHAP (FAST FIX)
# -----------------------------
st.markdown("## 🧠 SHAP Explainability")

try:
    sample_X = X.sample(100)

    if model_choice in ["XGBoost", "Random Forest"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_X)

        shap.summary_plot(shap_values, sample_X, show=False)
        st.pyplot(bbox_inches='tight')

    else:
        st.info("SHAP for Linear Regression is slow and skipped")

except Exception as e:
    st.warning(f"SHAP skipped: {e}")

# -----------------------------
# 📄 PDF REPORT
# -----------------------------
def generate_pdf():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("House Price Prediction Report", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"Predicted Price: ₹ {prediction:,.0f}", styles['Normal']),
        Spacer(1, 12),
        Paragraph(f"Model Used: {model_choice}", styles['Normal'])
    ]

    doc.build(content)
    return tmp.name

pdf_file = generate_pdf()

with open(pdf_file, "rb") as f:
    st.download_button("📄 Download Report", f, file_name="report.pdf")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("💜 FAANG-Level ML Project")