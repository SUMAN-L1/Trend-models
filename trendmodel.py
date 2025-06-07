import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import aic, bic
from io import BytesIO
from fpdf import FPDF

st.set_page_config(page_title="Trend models for time series data [by Suman_econ UAS(B)]", layout="wide")
st.title("📈 Trend models for time series data [by Suman_econ UAS(B)]")

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
        try:
            x_full = pd.to_datetime(df[time_col], errors='coerce')
            if x_full.isnull().all():
                x_full = df[time_col]
        except:
            x_full = df[time_col]

        line_styles = {
            'Linear': '-',
            'Quadratic': '--',
            'Cubic': '-.',
            'Quartic': ':',
            'Exponential': (0, (3, 1, 1, 1))
        }

        results = []
        best_models = {}

        plt.figure(figsize=(14, 6))
        plot_buffer = BytesIO()

        for col in selected_columns:
            y = df[col].dropna().values
            valid_index = df[col].dropna().index
            x_values = x_full.loc[valid_index].values
            x_index = np.arange(1, len(y) + 1)
            data = pd.DataFrame({'x': x_index, 'y': y})

            models = {
                'Linear': sm.OLS(data['y'], sm.add_constant(data['x'])).fit(),
                'Quadratic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2)))).fit(),
                'Cubic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2, data['x']**3)))).fit(),
                'Quartic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2, data['x']**3, data['x']**4)))).fit(),
                'Exponential': sm.OLS(np.log(data['y']), sm.add_constant(data['x'])).fit()
            }

            plt.plot(x_values, y, label=f"{col} Actual", linewidth=2, color='black')

            model_scores = []
            for name, model in models.items():
                y_pred = model.fittedvalues if name != 'Exponential' else np.exp(model.fittedvalues)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                score = aic(model.llf, len(y), model.df_model+1) + bic(model.llf, len(y), model.df_model+1)

                model_scores.append((name, score, model, y_pred))

                plt.plot(x_values, y_pred, label=f"{col} - {name}", linestyle=line_styles[name])

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

            best_model = min(model_scores, key=lambda x: x[1])
            best_models[col] = best_model  # (name, score, model, y_pred)

        plt.xlabel("Year / Date")
        plt.ylabel("Value")
        plt.title("Actual vs Fitted Trends")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_buffer, format='png')
        st.image(plot_buffer)

        # Display summary table with highlight
        result_df = pd.DataFrame(results)

        def highlight_best(s):
            is_best = (
                (result_df['Variable'] == s['Variable']) &
                (result_df['Model'] == best_models[s['Variable']][0])
            )
            return ['background-color: orange; font-weight: bold' if b else '' for b in is_best]

        st.write("### 📊 Model Comparison Summary")
        st.dataframe(result_df.style.apply(highlight_best, axis=1))

        # Forecasting
        forecast_buffer = BytesIO()
        plt.figure(figsize=(12, 5))

        st.write("### 🔮 Forecast Plot (Best Models Only)")
        for col in selected_columns:
            y = df[col].dropna().values
            x_index = np.arange(1, len(y) + 1)
            future_periods = 6
            future_x = np.arange(len(y) + 1, len(y) + future_periods + 1)

            best_name, _, best_model, _ = best_models[col]

            if best_name == 'Linear':
                X_pred = sm.add_constant(np.concatenate([x_index, future_x]))
            elif best_name == 'Quadratic':
                X_pred = sm.add_constant(np.column_stack([np.concatenate([x_index, future_x])**i for i in range(1, 3)]))
            elif best_name == 'Cubic':
                X_pred = sm.add_constant(np.column_stack([np.concatenate([x_index, future_x])**i for i in range(1, 4)]))
            elif best_name == 'Quartic':
                X_pred = sm.add_constant(np.column_stack([np.concatenate([x_index, future_x])**i for i in range(1, 5)]))
            elif best_name == 'Exponential':
                X_pred = sm.add_constant(np.concatenate([x_index, future_x]))
            else:
                continue

            pred_values = best_model.predict(X_pred)
            if best_name == 'Exponential':
                pred_values = np.exp(pred_values)

            all_x = list(x_full.loc[df[col].dropna().index].values) + [f"F{i}" for i in range(1, future_periods+1)]
            plt.plot(all_x, pred_values, label=f"{col} Forecast ({best_name})", linestyle='--')

        plt.title("Forecast using Best Model")
        plt.xlabel("Year / Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(forecast_buffer, format='png')
        st.image(forecast_buffer)

        # Downloads
        st.markdown("### 💾 Download Options")

        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        st.download_button("⬇️ Download Table as CSV", data=convert_df(result_df), file_name="model_summary.csv", mime="text/csv")
        st.download_button("🖼 Download Plot as PNG", data=plot_buffer.getvalue(), file_name="trend_plot.png", mime="image/png")
        st.download_button("🖼 Download Forecast Plot as PNG", data=forecast_buffer.getvalue(), file_name="forecast_plot.png", mime="image/png")

        # PDF generation
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
For support, reach out via university research forums or contact the developer.
""")
