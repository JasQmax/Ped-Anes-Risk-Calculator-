import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Risk Calculator", layout="wide")

@st.cache_resource
def load_models():
    logit = joblib.load('lr_baseline_model.pkl')
    xgboost = joblib.load('xgb_zero_harm_model.pkl')
    rf_critical = joblib.load('rf_critical_model.pkl')
    shap_explainer = joblib.load('shap_explainer.pkl')
    return logit, xgboost, rf_critical, shap_explainer

def main():
    st.title("In-patient Death Risk Calculator")
    logit_model, xgb_model, rf_model, shap_explainer = load_models()
    model_choice = st.sidebar.selectbox("Select Model", ["Logit Regression", "XGBoost", "Random Forest Critical"])

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
        # Remove 'const' for models that don't need it
        features_for_prediction = features_df.drop(columns=['const'])
        
        if model_choice == "Logit Regression":
            prob = logit_model.predict(features_df)[0]
        elif model_choice == "XGBoost":
            prob = xgb_model.predict_proba(features_for_prediction)[0][1]
        else:  # Random Forest Critical
            prob = rf_model.predict_proba(features_for_prediction)[0][1]
            
        # Display risk prediction
        st.metric("Estimated Risk", f"{prob:.2%}")
        if prob > 0.5:
            st.error("High Risk")
        else:
            st.success("Standard Risk")
        
        # SHAP Explanation
        st.subheader("Prediction Breakdown (SHAP Analysis)")
        try:
            # Generate SHAP values
            shap_values = shap_explainer.shap_values(features_for_prediction)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]  # For binary classification, take class 1
            else:
                shap_vals = shap_values
            
            # Force plot
            st.write("**Feature Contributions to Prediction:**")
            force_plot = shap.force_plot(
                shap_explainer.expected_value if not isinstance(shap_explainer.expected_value, list) else shap_explainer.expected_value[1],
                shap_vals[0],
                features_for_prediction.iloc[0],
                feature_names=features_for_prediction.columns.tolist(),
                matplotlib=True,
                show=False
            )
            st.pyplot(force_plot)
            
            # Summary plot (feature importance)
            st.write("**Top Risk Factors:**")
            fig, ax = plt.subplots()
            shap.summary_plot(
                shap_vals,
                features_for_prediction,
                plot_type="bar",
                matplotlib=True,
                show=False,
                max_display=10
            )
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {str(e)}")

if __name__ == '__main__':
    main()
