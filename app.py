import streamlit as st
import joblib
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb

st.set_page_config(page_title="Risk Calculator", layout="wide")

@st.cache_resource
def load_models():
    logit = joblib.load('lr_baseline_model.pkl')
    xgboost = joblib.load('xgb_zero_harm_model.pkl')
    return logit, xgboost

def main():
    st.title("In-patient Death Risk Calculator")
    logit_model, xgb_model = load_models()
    model_choice = st.sidebar.selectbox("Select Model", ["Logit Regression", "XGBoost"])

    col1, col2 = st.columns(2)
    with col1:
        st.header("Patient Basics")
        weight = st.number_input("Weight (kg)", 0.0, 200.0, 70.0)
        episode = st.number_input("Episode Number", 1, 100, 1)
        emergency = st.checkbox("Emergency Admission")
        cardiac_surg = st.checkbox("Cardiac Surgery")

    with col2:
        st.header("Risk Factors")
        risk_factors = [
            "Parental smoking history", "URTI / Chest infection", "Congenital Heart Disease",
            "Developmental delay", "Autistic Spectrum Disorder", "Epilepsy", "Cancer",
            "Renal Impairment", "Liver impairment", "Preterm",
            "Impaired conscious state (GCS < 13)", "Congenital anomalies / handicap"
        ]
        selections = {rf: st.checkbox(rf) for rf in risk_factors}

    input_data = {
        'const': 1.0,
        'Weight': weight,
        'Emergency': 1 if emergency else 0,
        'Cardiac surgery': 1 if cardiac_surg else 0,
        'RF.Parental smoking history': 1 if selections["Parental smoking history"] else 0,
        'RF.URTI / Chest infection': 1 if selections["URTI / Chest infection"] else 0,
        'RF.Congenital Heart Disease': 1 if selections["Congenital Heart Disease"] else 0,
        'RF.Developmental delay': 1 if selections["Developmental delay"] else 0,
        'RF.Autistic Spectrum Disorder': 1 if selections["Autistic Spectrum Disorder"] else 0,
        'RF.Epilepsy': 1 if selections["Epilepsy"] else 0,
        'RF.Neuromuscular disorder': 0,
        'RF.Cancer': 1 if selections["Cancer"] else 0,
        'RF.Renal Impairment': 1 if selections["Renal Impairment"] else 0,
        'RF.Liver impairment': 1 if selections["Liver impairment"] else 0,
        'RF.Preterm': 1 if selections["Preterm"] else 0,
        'RF.Impaired conscious state (GCS_13)': 1 if selections["Impaired conscious state (GCS < 13)"] else 0,
        'RF.Congenital anomalies / handicap': 1 if selections["Congenital anomalies / handicap"] else 0,
        'Episode': episode
    }
    
    features_df = pd.DataFrame([input_data])

    if st.button("Calculate Risk"):
        if model_choice == "Logit Regression":
            prob = logit_model.predict(features_df)[0]
        else:
            xgb_feats = features_df.drop(columns=['const'])
            prob = xgb_model.predict_proba(xgb_feats)[0][1]
            
        st.metric("Estimated Risk", f"{prob:.2%}")
        if prob > 0.5: st.error("High Risk")
        else: st.success("Standard Risk")

if __name__ == '__main__':
    main()
