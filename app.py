import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model, scaler, and explainer
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Optional SHAP explainer
try:
    with open("xgb_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    has_shap = True
except:
    has_shap = False

st.title("Ola Driver Churn Prediction")

# Sample input form
st.header("Enter Driver Info:")
driver_input = {
    'Age': st.number_input("Age", 18, 65, 30),
    'Gender': st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1]),
    'Income': st.number_input("Income", 10000, 500000, 50000),
    'Joining Designation': st.selectbox("Joining Designation (1 being lowest)", [1, 2, 3, 4, 5]),
    'Total Business Value': st.number_input("Total Business Value", 0, 10000000, 10000),
    'Education_Level': st.selectbox("Education Level (2 being highest)", [0, 1, 2]),
    'Last_grade': st.slider("Last Grade (5 being highest)", 1, 5, 3),
    'Income_increased': st.selectbox("Income Increased (No=0, Yes=1)", [0, 1]),
    'Last_rat': st.slider("Last Rating", 1, 5, 3),
    'Grade_improved': st.selectbox("Grade Improved (-1 = Decrease, 0 = No change, 1 = Increase)", [-1, 0, 1]),
    'Rating_change': st.slider("Rating Change", -4, 4, 0)
}

if st.button("Predict Churn"):
    input_df = pd.DataFrame([driver_input])
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    st.success(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
    st.info(f"Churn Probability: {prob:.2%}")

    # SHAP explanation
    if has_shap:
        import shap

        shap_vals = explainer.shap_values(pd.DataFrame(scaled_input, columns=input_df.columns))
        shap_array = shap_vals[0] if isinstance(shap_vals, list) else shap_vals

        shap_df = pd.DataFrame({
            "Feature Name": input_df.columns,
            "SHAP Value": shap_array[0]
        })
        shap_df["Impact Direction"] = shap_df["SHAP Value"].apply(
            lambda x: "↑ Towards Churn" if x > 0 else "↓ Away from Churn"
        )
        shap_df["SHAP Value"] = shap_df["SHAP Value"].round(2)
        shap_df["|SHAP|"] = shap_df["SHAP Value"].abs()

        top_shap = shap_df.sort_values(by="|SHAP|", ascending=False).head(5).drop(columns="|SHAP|")

        st.subheader("Top SHAP Features")
        st.dataframe(top_shap)
