import streamlit as st
import numpy as np
import joblib
import pandas as pd

# --- Load Model, Scaler, and Feature Order ---
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = joblib.load("feature_order.pkl")

st.title("Customer Segmentation App using KMeans")
st.write("Predict which customer cluster a person belongs to based on demographic and purchase behavior.")

# --- Sidebar Inputs (match feature_order exactly) ---
st.sidebar.header("Enter Customer Details")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
income = st.sidebar.number_input("Income (per year)", min_value=1000, max_value=200000, value=90000)
totalspend = st.sidebar.number_input("Total Spend (last 2 years)", min_value=0, max_value=20000, value=1800)
numstorepurchases = st.sidebar.number_input("Number of Store Purchases", min_value=0, max_value=50, value=15)
numwebpurchases = st.sidebar.number_input("Number of Web Purchases", min_value=0, max_value=50, value=8)
numwebvisitsmonth = st.sidebar.slider("Number of Web Visits per Month", 0, 20, 4)
recency = st.sidebar.slider("Recency (days since last purchase)", 0, 100, 10)

# --- Build feature vector in the exact order ---
input_data = np.array([[age, income, totalspend, numstorepurchases, numwebpurchases, numwebvisitsmonth, recency]])

# --- Scale Input ---
input_df = pd.DataFrame(input_data, columns=feature_order)
scaled_features = scaler.transform(input_df)


# --- Predict Cluster ---
if st.button("Predict Cluster"):
    cluster = model.predict(scaled_features)[0]
    st.success(f"🧭 Predicted Customer Cluster: **Cluster {cluster}**")

    # Optional interpretation
    if cluster == 0:
        st.info("Cluster 0: High income, frequent shoppers.")
    elif cluster == 1:
        st.info("Cluster 1: Moderate spenders, occasional online buyers.")
    else:

        st.info("Cluster 2: Low spend, high recency customers.")
