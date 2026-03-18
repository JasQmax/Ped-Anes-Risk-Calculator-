import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Risk Calculator", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .header-gradient {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            color: white;
        }
        .header-gradient h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header-gradient p {
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }
        .risk-high {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .risk-high h3 {
            color: #c62828;
            margin: 0;
            font-size: 1.5em;
        }
        .risk-high p {
            color: #d32f2f;
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .risk-low {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .risk-low h3 {
            color: #2e7d32;
            margin: 0;
            font-size: 1.5em;
        }
        .risk-low p {
            color: #388e3c;
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    logit = joblib.load('lr_baseline_model.pkl')
    xgboost = joblib.load('xgb_zero_harm_model.pkl')
    rf_critical = joblib.load('rf_critical_model.pkl')
    shap_explainer = joblib.load('shap_explainer.pkl')
    return logit, xgboost, rf_critical, shap_explainer

def main():
    # Header with gradient background
    st.markdown("""
        <div class="header-gradient">
            <h1>🏥 In-patient Death Risk Calculator</h1>
            <p>Evidence-based pediatric anesthesia risk assessment tool</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Model selection in sidebar
    st.sidebar.markdown("## ⚙️ Settings")
    logit_model, xgb_model, rf_model, shap_explainer = load_models()
    model_choice = st.sidebar.selectbox(
        "Select Risk Model",
        ["Logit Regression", "XGBoost", "Random Forest Critical"],
        help="Choose which ML model to use for risk prediction"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### ℹ️ About This Tool
    This calculator uses machine learning models trained on pediatric anesthesia data to estimate patient risk.
    
    **Always combine with clinical judgment.**
    """)
    
    # Main content
    with st.expander("👤 Patient Basics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            weight = st.number_input("⚖️ Weight (kg)", 0.0, 200.0, 70.0)
        with col2:
            episode = st.number_input("🔢 Episode Number", 1, 100, 1)
        with col3:
            emergency = st.checkbox("🚨 Emergency Admission")
        with col4:
            cardiac_surg = st.checkbox("❤️ Cardiac Surgery")
    
    with st.expander("⚠️ Risk Factors", expanded=True):
        risk_factors = [
            "Parental smoking history", 
            "URTI / Chest infection", 
            "Congenital Heart Disease",
            "Developmental delay", 
            "Autistic Spectrum Disorder", 
            "Epilepsy", 
            "Cancer",
            "Renal Impairment", 
            "Liver impairment", 
            "Preterm",
            "Impaired conscious state (GCS < 13)", 
            "Congenital anomalies / handicap"
        ]
        
        cols = st.columns(2)
        selections = {}
        for i, rf in enumerate(risk_factors):
            with cols[i % 2]:
                selections[rf] = st.checkbox(rf, key=f"rf_{i}")
    
    # Information box
    st.info("💡 **Clinical Tip:** This calculator should be used as a supplementary tool alongside comprehensive clinical assessment and patient history.")
    
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
    
    # Calculate button with custom styling
    col1, col2, col3 = st.columns([1, 1, 2])
    with col2:
        calculate_button = st.button("📊 Calculate Risk", use_container_width=True, key="calc_button")
    
    if calculate_button:
        # Remove 'const' for models that don't need it
        features_for_prediction = features_df.drop(columns=['const'])
        
        if model_choice == "Logit Regression":
            prob = logit_model.predict(features_df)[0]
        elif model_choice == "XGBoost":
            prob = xgb_model.predict_proba(features_for_prediction)[0][1]
        else:  # Random Forest Critical
            prob = rf_model.predict_proba(features_for_prediction)[0][1]
        
        # Display risk prediction with improved styling
        if prob > 0.5:
            st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ HIGH RISK</h3>
                    <p>{{prob:.1%}}</p>
                    <p style="font-size: 0.9em; color: #d32f2f;">Risk of in-patient death</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ STANDARD RISK</h3>
                    <p>{{prob:.1%}}</p>
                    <p style="font-size: 0.9em; color: #388e3c;">Risk of in-patient death</p>
                </div>
            """, unsafe_allow_html=True)
        
        # SHAP Explanation
        st.subheader("📈 Prediction Breakdown (SHAP Analysis)")
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
            st.warning(f"⚠️ SHAP explanation could not be generated: {str(e)}")

if __name__ == '__main__':
    main()