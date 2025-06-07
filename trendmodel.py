import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import aic, bic
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
import seaborn as sns

# Set page config
st.set_page_config(page_title="Trend models for time series data [by Suman_econ UAS(B)]", layout="wide")
st.title("📈 Trend models for time series data [by Suman_econ UAS(B)]")

# Introduction
st.markdown("""
### 📘 Introduction
Trend models help understand time-based movements in economic variables.
They are critical for forecasting, investment decisions, and policy formulation.

**Instructions:**
- Ensure your dataset has the first column as Date or Year.
- The rest of the columns should contain numeric data.
- You can analyze one or more columns at a time.
""")

# Upload file
uploaded_file = st.file_uploader("📤 Upload CSV, XLSX, or XLS file", type=["csv", "xlsx", "xls"])

# Context for dynamic policy briefs
user_context = st.text_area("🧠 Optional: Add real-time context (e.g., market disruptions, export bans, price volatility)")

def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

if uploaded_file:
    df = load_data(uploaded_file)
    df.columns = df.columns.astype(str)
    df.dropna(how="all", axis=1, inplace=True)
    time_col = df.columns[0]

    # Convert to datetime
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    if df[time_col].isnull().any():
        st.warning("Some rows have invalid or missing date/time values. These rows will be removed.")
        df = df.dropna(subset=[time_col])

    df = df.sort_values(by=time_col)
    df = df.reset_index(drop=True)

    # Interpolate missing values in numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    interpolated = df[numeric_cols].interpolate(method='linear', limit_direction='both')
    if interpolated.isnull().sum().sum() > 0:
        st.warning("Linear interpolation could not fill all missing values. Remaining NAs will be excluded.")
    else:
        st.info("Missing values handled using linear interpolation.")
    df[numeric_cols] = interpolated

    st.write("### 📄 Data Preview")
    st.dataframe(df.head())

    selected_columns = st.multiselect("📌 Select variable(s) for trend analysis", options=numeric_cols, default=numeric_cols)

    if selected_columns:
        results = []
        tab1, tab2 = st.tabs(["📊 Trend Plot", "📋 Dashboard"])

        with tab1:
            fig = go.Figure()
            line_styles = ['solid', 'dash', 'dot', 'dashdot', 'longdash']

            for col in selected_columns:
                y = df[col].dropna().values
                x = np.arange(1, len(y)+1)
                data = pd.DataFrame({'x': x, 'y': y})

                models = {
                    'Linear': sm.OLS(data['y'], sm.add_constant(data['x'])).fit(),
                    'Quadratic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2)))).fit(),
                    'Cubic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2, data['x']**3)))).fit(),
                    'Quartic': sm.OLS(data['y'], sm.add_constant(np.column_stack((data['x'], data['x']**2, data['x']**3, data['x']**4)))).fit(),
                    'Exponential': sm.OLS(np.log(data['y']), sm.add_constant(data['x'])).fit()
                }

                best_model = None
                best_aic = float('inf')

                for i, (name, model) in enumerate(models.items()):
                    y_pred = model.fittedvalues if name != 'Exponential' else np.exp(model.fittedvalues)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    current_aic = aic(model.llf, len(y), model.df_model+1)
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_model = name
                    results.append({
                        'Variable': col,
                        'Model': name,
                        'R2': model.rsquared,
                        'Adj R2': model.rsquared_adj,
                        'RMSE': rmse,
                        'AIC': current_aic,
                        'BIC': bic(model.llf, len(y), model.df_model+1),
                        'Interpretation': f"R2={model.rsquared:.3f}, AdjR2={model.rsquared_adj:.3f}, RMSE={rmse:.2f}, AIC={current_aic:.1f}"
                    })

                    fig.add_trace(go.Scatter(
                        x=df[time_col],
                        y=y_pred,
                        mode='lines',
                        name=f"{col} - {name}",
                        line=dict(dash=line_styles[i % len(line_styles)])
                    ))
                fig.add_trace(go.Scatter(x=df[time_col], y=y, mode='markers', name=f"{col} Actual", marker=dict(size=6)))
                st.success(f"📌 Best model for **{col}** is: {best_model}")

            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            result_df = pd.DataFrame(results)
            st.write("### 📋 Model Summary Table")
            st.dataframe(result_df)

            # KPI Dashboard
            st.markdown("### 📌 Model KPI Cards")
            for col in selected_columns:
                best_row = result_df[(result_df['Variable'] == col)].sort_values("AIC").iloc[0]
                st.metric(label=f"{col} - Best Model", value=best_row['Model'], delta=f"R2: {best_row['R2']:.2f}")

            def convert_df(df): return df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Table as CSV", data=convert_df(result_df), file_name="model_summary.csv", mime="text/csv")

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, "Trend Model Analysis Report", ln=1, align='C')
            pdf.set_font("Arial", size=12)
            pdf.ln(10)
            for idx, row in result_df.iterrows():
                pdf.multi_cell(0, 10, f"{row['Variable']} - {row['Model']}: {row['Interpretation']}")

            if user_context.strip():
                pdf.ln(10)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, "User Context:", ln=True)
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 10, user_context)

            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            pdf_output = BytesIO(pdf_bytes)
            st.download_button("📄 Download Report as PDF", data=pdf_output, file_name="trend_report.pdf", mime="application/pdf")

            st.markdown("""
            ---
            ### 🧩 Policy Brief
            Based on AIC/BIC and RMSE, select the model with the best fit. Forecasts can help in:
            - Allocating budget or subsidies
            - Planning logistics and storage
            - Targeting seasonal interventions

            **Auto-detected trend insights**:
            """)
            for col in selected_columns:
                best = result_df[result_df['Variable'] == col].sort_values("AIC").iloc[0]
                if best['Model'] == "Exponential":
                    st.info(f"🔹 {col}: Shows exponential growth. Policy can focus on infrastructure scalability and risk mitigation.")
                elif best['Model'] == "Linear":
                    st.info(f"🔹 {col}: Displays linear growth. Consider steady policy interventions or capacity planning.")
                elif best['Model'] in ["Quadratic", "Cubic", "Quartic"]:
                    st.info(f"🔹 {col}: Shows non-linear trend. Adaptive and responsive policies may be needed.")
            if user_context:
                st.markdown(f"**User-stated context considered:** {user_context}")

# Footer
st.markdown("""
---
App developed by **Suman_econ UAS(B)**  
For improvements or academic collaboration, contact through university channels.
""")
