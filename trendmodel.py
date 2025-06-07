import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import aic, bic
from io import BytesIO
from fpdf import FPDF

# App Title
st.set_page_config(page_title="Trend models for time series data [by Suman_econ UAS(B)]", layout="wide")
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
- You may select one, multiple, or all columns for analysis
""")

# Upload File
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
        best_models = []
        plot_buffer = BytesIO()
        plot_buffer_actual = BytesIO()

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        df = df.sort_values(by=time_col).reset_index(drop=True)

        # Line styles for different models
        line_styles = {
            'Linear': 'solid',
            'Quadratic': 'dashed',
            'Cubic': 'dashdot',
            'Quartic': 'dotted',
            'Exponential': (0, (3, 5, 1, 5))
        }

        # Plot 1: Index X-axis
        plt.figure(figsize=(14, 6))
        for col in selected_columns:
            y = df[col].dropna().values
            x = np.arange(1, len(y) + 1)
            data = pd.DataFrame({'x': x, 'y': y})

            models = {
                'Linear': sm.OLS(data['y'], sm.add_constant(data['x'])).fit(),
                'Quadratic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2)))).fit(),
                'Cubic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2, data['x']**3)))).fit(),
                'Quartic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2, data['x']**3, data['x']**4)))).fit(),
                'Exponential': sm.OLS(np.log(data['y']), sm.add_constant(data['x'])).fit()
            }

            plt.plot(x, y, label=f"{col} Actual", linewidth=2)

            model_metrics = []
            for name, model in models.items():
                y_pred = model.fittedvalues if name != 'Exponential' else np.exp(model.fittedvalues)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                model_aic = aic(model.llf, len(y), model.df_model+1)
                model_bic = bic(model.llf, len(y), model.df_model+1)
                results.append({
                    'Variable': col,
                    'Model': name,
                    'R2': model.rsquared,
                    'Adj R2': model.rsquared_adj,
                    'RMSE': rmse,
                    'AIC': model_aic,
                    'BIC': model_bic,
                    'Interpretation': f"R2={model.rsquared:.3f}, AdjR2={model.rsquared_adj:.3f}, RMSE={rmse:.2f}, AIC={model_aic:.1f}, BIC={model_bic:.1f}"
                })
                model_metrics.append((name, model, model_aic + model_bic))
                plt.plot(x, y_pred, label=f"{col} - {name}", linestyle=line_styles[name])

            best_model_name, best_model_obj, _ = sorted(model_metrics, key=lambda x: x[2])[0]
            best_models.append((col, best_model_name, best_model_obj, x, y))

        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Actual vs Fitted Trends (Index X-axis)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_buffer, format='png')
        st.image(plot_buffer, caption="📉 Plot with Index on X-axis")

        # Plot 2: Time Column X-axis
        plt.figure(figsize=(14, 6))
        for col in selected_columns:
            y = df[col].dropna().values
            x_time = df[time_col].iloc[:len(y)]
            x_num = np.arange(1, len(y) + 1)
            data = pd.DataFrame({'x': x_num, 'y': y})

            models = {
                'Linear': sm.OLS(data['y'], sm.add_constant(data['x'])).fit(),
                'Quadratic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2)))).fit(),
                'Cubic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2, data['x']**3)))).fit(),
                'Quartic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2, data['x']**3, data['x']**4)))).fit(),
                'Exponential': sm.OLS(np.log(data['y']), sm.add_constant(data['x'])).fit()
            }

            plt.plot(x_time, y, label=f"{col} Actual", linewidth=2)
            for name, model in models.items():
                y_pred = model.fittedvalues if name != 'Exponential' else np.exp(model.fittedvalues)
                plt.plot(x_time, y_pred, label=f"{col} - {name}", linestyle=line_styles[name])

        plt.xlabel("Date/Year")
        plt.ylabel("Value")
        plt.title("Actual vs Fitted Trends (Time X-axis)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_buffer_actual, format='png')
        st.image(plot_buffer_actual, caption="📉 Plot with Date/Year on X-axis")

        # Results Table
        result_df = pd.DataFrame(results)
        st.write("### 📊 Model Comparison Summary")
        st.dataframe(result_df)

        # Best model and forecast
        st.markdown("### ⭐ Best Models and Forecasts")
        for var, model_name, model_obj, x, y in best_models:
            st.markdown(f"**{var}: Best model → {model_name}**")
            X_fore = np.arange(len(x)+1, len(x)+7)
            if model_name == 'Linear':
                X_pred = sm.add_constant(X_fore)
            elif model_name == 'Quadratic':
                X_pred = sm.add_constant(np.column_stack((X_fore, X_fore**2)))
            elif model_name == 'Cubic':
                X_pred = sm.add_constant(np.column_stack((X_fore, X_fore**2, X_fore**3)))
            elif model_name == 'Quartic':
                X_pred = sm.add_constant(np.column_stack((X_fore, X_fore**2, X_fore**3, X_fore**4)))
            elif model_name == 'Exponential':
                X_pred = sm.add_constant(X_fore)

            forecast = model_obj.predict(X_pred)
            if model_name == 'Exponential':
                forecast = np.exp(forecast)

            forecast_years = pd.date_range(df[time_col].iloc[-1], periods=6, freq='Y')
            forecast_df = pd.DataFrame({'Year': forecast_years.year, 'Forecast': forecast})
            st.write(forecast_df)

        # Downloads
        st.markdown("### 💾 Download Options")
        def convert_df(df): return df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Table as CSV", data=convert_df(result_df), file_name="model_summary.csv", mime="text/csv")
        st.download_button("🖼 Download Plot (Index) as PNG", data=plot_buffer.getvalue(), file_name="trend_plot_index.png", mime="image/png")
        st.download_button("🖼 Download Plot (Time) as PNG", data=plot_buffer_actual.getvalue(), file_name="trend_plot_time.png", mime="image/png")

        # PDF generation
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "Trend Model Analysis Report\n\n")
        for idx, row in result_df.iterrows():
            pdf.multi_cell(0, 10, f"{row['Variable']} - {row['Model']}: {row['Interpretation']}")
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        st.download_button("📄 Download Report as PDF", data=BytesIO(pdf_bytes), file_name="trend_report.pdf", mime="application/pdf")

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
For support, reach out via university research forums or contact the developer.
""")
