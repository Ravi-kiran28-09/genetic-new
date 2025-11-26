import streamlit as st
import pandas as pd
import joblib

# Load trained model and feature columns
model = joblib.load("genetic_disorder_subclass_model.pkl")
x_columns = joblib.load("x2_columns.pkl")

st.title("🧬 Genetic Disorder Subclass Prediction")

st.write("Provide the details below to predict the disorder subclass.")

# --- Identify categorical features from one-hot encoded columns ---
categorical_features = {}
numeric_features = []

for col in x_columns:
    if "_" in col:  # one-hot encoded col like Gender_Male
        base, value = col.split("_", 1)
        categorical_features.setdefault(base, []).append(value)
    else:
        numeric_features.append(col)

# --- Streamlit inputs ---
user_inputs = {}

# Numeric fields
for num_feat in numeric_features:
    user_inputs[num_feat] = st.number_input(f"{num_feat}", value=0)

# Categorical fields
for cat_feat, options in categorical_features.items():
    choice = st.selectbox(f"{cat_feat}", options)
    user_inputs[cat_feat] = choice

# --- Build DataFrame ---
raw_input_df = pd.DataFrame([user_inputs])

# One-hot encode and align with training columns
encoded_input = pd.get_dummies(raw_input_df)
encoded_input = encoded_input.reindex(columns=x_columns, fill_value=0)

# --- Prediction ---
if st.button("🔮 Predict Disorder Subclass"):
    prediction = model.predict(encoded_input)[0]
    st.success(f"Predicted Disorder Subclass: **{prediction}**")
