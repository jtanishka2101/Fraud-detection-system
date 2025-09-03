# dashboard.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# -------------------------------
# 1. Load Trained Model
# -------------------------------
model = joblib.load("fraud_detection_model.pkl")

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Financial Fraud Detection Dashboard")
st.markdown("Upload a CSV file to detect fraudulent transactions.")

# -------------------------------
# 2. File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load uploaded data
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Ensure dataset has correct features
    if "Class" in data.columns:
        X = data.drop("Class", axis=1)
        y_true = data["Class"]
    else:
        X = data
        y_true = None

    # -------------------------------
    # 3. Make Predictions
    # -------------------------------
    predictions = model.predict(X)
    prediction_probs = model.predict_proba(X)[:, 1]  # Fraud probability

    results = data.copy()
    results["Fraud_Prediction"] = predictions
    results["Fraud_Probability"] = prediction_probs

    # ✅ Show only first 500 rows to avoid Streamlit cell limit
    st.subheader("Prediction Results (First 500 Rows)")
    sample_results = results.head(500)

    # Highlight fraud cases in red
    def highlight_fraud(row):
        return ['background-color: red' if row["Fraud_Prediction"] == 1 else '' for _ in row]

    st.dataframe(sample_results.style.apply(highlight_fraud, axis=1))

    # -------------------------------
    # 4. Visualizations
    # -------------------------------
    st.subheader("📊 Data Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        # Fraud vs Non-Fraud
        st.markdown("**Fraud vs Non-Fraud Count**")
        fig, ax = plt.subplots()
        sns.countplot(x="Fraud_Prediction", data=results, ax=ax)
        ax.set_xticklabels(["Non-Fraud", "Fraud"])
        st.pyplot(fig)

    with col2:
        # Transaction Amount Distribution
        if "Amount_Scaled" in results.columns:
            st.markdown("**Transaction Amount Distribution (Scaled)**")
            fig, ax = plt.subplots()
            sns.histplot(results["Amount_Scaled"], bins=50, kde=True, ax=ax)
            st.pyplot(fig)

    # Time-series Fraud Cases (if timestamp available)
    if "Time_Scaled" in results.columns:
        st.subheader("⏱️ Time-Series of Fraud Cases")
        fraud_over_time = results.groupby("Time_Scaled")["Fraud_Prediction"].sum().reset_index()
        fig, ax = plt.subplots()
        sns.lineplot(x="Time_Scaled", y="Fraud_Prediction", data=fraud_over_time, ax=ax)
        st.pyplot(fig)

    # -------------------------------
    # 5. Alerts
    # -------------------------------
    if results["Fraud_Prediction"].sum() > 0:
        st.error(f"⚠️ ALERT: {results['Fraud_Prediction'].sum()} suspicious transactions detected!")
    else:
        st.success("✅ No fraudulent transactions detected.")
