import shap

# Other imports and code here...  

# Example of force_plot
shap.force_plot(explainer.expected_value, shap_values, features)

# Example of summary_plot
shap.summary_plot(shap_values, features, ax=ax)