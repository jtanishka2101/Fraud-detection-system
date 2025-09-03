# dashboard1.py

import streamlit as st
import pandas as pd
import joblib   # ✅ use joblib instead of pickle
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")   # ✅ fixed to joblib

model = load_model()

# -----------------------------
# Title & Instructions
# -----------------------------
st.title("🚨 Fraud Detection Dashboard")
st.markdown(
    """
    Upload a CSV file of transactions to analyze.  
    The system will predict **fraudulent transactions** and highlight them.  
    """
)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload Transactions CSV", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data (Preview)")
    st.dataframe(data.head(20), use_container_width=True)

    # -----------------------------
    # Run Predictions
    # -----------------------------
    st.subheader("🤖 Running Fraud Detection...")
    if "Class" in data.columns:
      data = data.drop(columns=["Class"])
    predictions = model.predict(data)
    data["Fraud_Prediction"] = predictions

    # -----------------------------
    # Fraud Summary
    # -----------------------------
    fraud_count = int((data["Fraud_Prediction"] == 1).sum())
    total_count = len(data)

    st.success(f"✅ Analysis Complete: {fraud_count} fraudulent transactions out of {total_count}")

    # -----------------------------
    # Fraud Transactions Highlighted
    # -----------------------------
    def highlight_fraud(row):
        return ['background-color: red; color: white;' if row["Fraud_Prediction"] == 1 else '' for _ in row]

    st.subheader("📑 Transactions with Fraud Highlighted")
    st.dataframe(
        data.head(200).style.apply(highlight_fraud, axis=1),
        use_container_width=True
    )

    # -----------------------------
    # Graphs
    # -----------------------------
    st.subheader("📈 Fraud Analysis Charts")

    col1, col2 = st.columns(2)

    # Pie chart
    with col1:
        fraud_pie = data["Fraud_Prediction"].value_counts()
        plt.figure(figsize=(4,4))
        plt.pie(
            fraud_pie,
            labels=["Not Fraud", "Fraud"],
            autopct="%1.1f%%",
            colors=["#66bb6a", "#ef5350"]
        )
        plt.title("Fraud vs Non-Fraud Distribution")
        st.pyplot(plt)

    # Bar chart
    with col2:
        plt.figure(figsize=(6,4))
        fraud_pie.plot(kind="bar", color=["#66bb6a", "#ef5350"])
        plt.xticks(rotation=0)
        plt.title("Fraud vs Non-Fraud Counts")
        plt.ylabel("Number of Transactions")
        st.pyplot(plt)

else:
    st.info("⬆️ Please upload a CSV file to get started.")
