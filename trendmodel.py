import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import aic, bic
from io import BytesIO
from fpdf import FPDF

# Streamlit settings
st.set_page_config(page_title="Trend models for time series [by Suman_econ UAS(B)]", layout="wide")
st.title("📈 Trend models for time series data [by Suman_econ UAS(B)]")

# Introduction
st.markdown("""
### 📘 Introduction

Trend models help analyze how an economic variable behaves over time. They are vital for:
- Forecasting long-term changes
- Understanding growth or instability patterns
- Supporting policy and investment decisions

**Instructions:**
- Ensure the first column is a Date or Year
- The remaining columns should be numeric variables (e.g., GDP, Production)
""")

# Upload
uploaded_file = st.file_uploader("📤 Upload CSV, XLSX, or XLS file", type=["csv", "xlsx", "xls"])

def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

if uploaded_file:
    df = load_data(uploaded_file)
    df.columns = df.columns.astype(str)
    st.write("### 📄 Data Preview")
    st.dataframe(df.head())

    time_col = df.columns[0]
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected_columns = st.multiselect("📌 Select variable(s) for trend analysis", options=numeric_cols, default=numeric_cols)

    if selected_columns:
        results = []
        plot_buffer = BytesIO()
        fig, ax = plt.subplots(figsize=(14, 6))

        for col in selected_columns:
            y = df[col].dropna().values
            x = np.arange(1, len(y) + 1)
            data = pd.DataFrame({'x': x, 'y': y})

            # Fit models
            models = {
                'Linear': sm.OLS(data['y'], sm.add_constant(data['x'])).fit(),
                'Quadratic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2)))).fit(),
                'Cubic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2, data['x']**3)))).fit(),
                'Quartic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2, data['x']**3, data['x']**4)))).fit(),
                'Exponential': sm.OLS(np.log(data['y']), sm.add_constant(data['x'])).fit()
            }

            model_styles = {
                'Linear': {'color': 'red', 'linestyle': '-', 'linewidth': 1.5},
                'Quadratic': {'color': 'green', 'linestyle': '--'},
                'Cubic': {'color': 'purple', 'linestyle': ':', 'linewidth': 1},
                'Quartic': {'color': 'black', 'linestyle': '--'},
                'Exponential': {'color': 'brown', 'linestyle': '-.', 'linewidth': 1.5}
            }

            ax.scatter(x, y, color='dodgerblue', label=f'{col} Actual', s=25)

            for name, model in models.items():
                y_pred = model.fittedvalues if name != 'Exponential' else np.exp(model.fittedvalues)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                results.append({
                    'Variable': col,
                    'Model': name,
                    'R2': model.rsquared,
                    'Adj R2': model.rsquared_adj,
                    'RMSE': rmse,
                    'AIC': aic(model.llf, len(y), model.df_model+1),
                    'BIC': bic(model.llf, len(y), model.df_model+1),
                    'Interpretation': f"R2={model.rsquared:.3f}, AdjR2={model.rsquared_adj:.3f}, RMSE={rmse:.2f}, AIC={aic(model.llf, len(y), model.df_model+1):.1f}, BIC={bic(model.llf, len(y), model.df_model+1):.1f}"
                })

                ax.plot(x, y_pred, label=name, **model_styles[name])

        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title("Actual Data vs Fitted Values for Various Models")
        ax.legend(title="Model")
        plt.tight_layout()
        plt.savefig(plot_buffer, format='png')
        st.image(plot_buffer)

        # Summary Table
        result_df = pd.DataFrame(results)
        st.write("### 📊 Model Comparison Summary")
        st.dataframe(result_df)

        # Download Buttons
        st.markdown("### 💾 Download Options")
        def convert_df(df): return df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Table as CSV", data=convert_df(result_df), file_name="model_summary.csv", mime="text/csv")
        st.download_button("🖼 Download Plot as PNG", data=plot_buffer.getvalue(), file_name="trend_plot.png", mime="image/png")

        # PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "Trend Model Analysis Report\n\n")
        for idx, row in result_df.iterrows():
            pdf.multi_cell(0, 10, f"{row['Variable']} - {row['Model']}: {row['Interpretation']}")
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_output = BytesIO(pdf_bytes)
        st.download_button("📄 Download Report as PDF", data=pdf_output, file_name="trend_report.pdf", mime="application/pdf")

        # Policy Brief
        st.markdown("""
        ---
        ### 🧩 Policy Brief

        Based on the best-fitting models (lowest AIC/BIC):
        - Forecast future economic indicators with confidence
        - Identify structural trends, volatility, or seasonal shifts
        - Plan interventions or investments aligned with projected trends
        - Create transparent data-driven governance strategies
        """)

# Footer
st.markdown("""
---
App developed by **Suman_econ UAS(B)**  
For support, contact: sumanecon.uas@outlook.in
""")
