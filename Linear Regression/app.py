import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
path = r"E:\Downloads\california1.joblib"
loaded_model = joblib.load(path)

model = loaded_model["model"]
columns = loaded_model["columns"].tolist()

# -------------------- CUSTOM CSS --------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #27ae60;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- HEADER --------------------
st.markdown('<div class="main-title">🏠 House Price Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict house prices using a trained Linear Regression model</div>',
    unsafe_allow_html=True
)

# -------------------- SIDEBAR --------------------
st.sidebar.header("📊 Input Features")
st.sidebar.write("Enter the house details below:")

inputs = []

for col in columns:
    value = st.sidebar.number_input(
        label=col,
        min_value=0.0,
        step=0.1,
        format="%.2f"
    )
    inputs.append(value)

# -------------------- MAIN CONTENT --------------------
st.markdown("### 🧮 Model Inputs Preview")

input_df = pd.DataFrame([inputs], columns=columns)
st.dataframe(input_df, use_container_width=True)

# -------------------- PREDICTION --------------------
if st.button("🚀 Predict House Price", use_container_width=True):
    prediction = model.predict([inputs])[0]

    st.markdown(
        f"""
        <div class="prediction-box">
            💰 Estimated House Price <br>
            ${prediction:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit & Scikit-Learn</center>",
    unsafe_allow_html=True
)