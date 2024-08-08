import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Function to fit and plot regression models
def fit_and_plot_regression(x, y, degree):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    
    model = LinearRegression()
    model.fit(x_poly, y)
    
    y_pred = model.predict(x_poly)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    x_poly_const = sm.add_constant(x_poly)
    ols_model = sm.OLS(y, x_poly_const).fit()
    p_values = np.round(ols_model.pvalues[1:], 4)  # Skip intercept p-value
    coefficients = np.round(model.coef_[1:], 4)  # Skip intercept coefficient
    intercept = np.round(model.intercept_, 4)
    
    equation = f"Y = {intercept} + " + " + ".join([f"{coef}*X^{i}" for i, coef in enumerate(coefficients, start=1)])
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label=f'Estimated (degree {degree})')
    plt.title(f'Degree {degree} Regression\nMSE: {mse:.2f}, R^2: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    return mse, r2, intercept, coefficients, p_values, equation, model, x_poly

# Function to fit and plot Cobb-Douglas model
def fit_and_plot_cobb_douglas(x, y):
    x_log = np.log(x)
    y_log = np.log(y)
    
    x_log_const = sm.add_constant(x_log)
    model = sm.OLS(y_log, x_log_const).fit()
    
    y_pred_log = model.predict(x_log_const)
    y_pred = np.exp(y_pred_log)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    intercept = np.round(model.params[0], 4)
    coefficients = np.round(model.params[1:], 4)  # Skip intercept coefficient
    p_values = np.round(model.pvalues[1:], 4)  # Skip intercept p-value
    equation = f"ln(Y) = {intercept} + {coefficients[0]}*ln(X)"
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label='Estimated (Cobb-Douglas)')
    plt.title(f'Cobb-Douglas Regression\nMSE: {mse:.2f}, R^2: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    return mse, r2, intercept, coefficients, p_values, equation, model, x_log, y_log

# Function to forecast using the best model
def forecast_best_model(best_model, x, y, model_type, additional_params=None):
    if model_type == 'Linear':
        model, x_poly = additional_params
        last_x = x[-1]
        future_x = np.array([last_x + i for i in range(1, 4)])
        future_x_poly = PolynomialFeatures(degree=1).fit_transform(future_x.reshape(-1, 1))
        future_y_pred = model.predict(future_x_poly)
        
    elif model_type == 'Quadratic':
        model, x_poly = additional_params
        last_x = x[-1]
        future_x = np.array([last_x + i for i in range(1, 4)])
        future_x_poly = PolynomialFeatures(degree=2).fit_transform(future_x.reshape(-1, 1))
        future_y_pred = model.predict(future_x_poly)
        
    elif model_type == 'Quartic':
        model, x_poly = additional_params
        last_x = x[-1]
        future_x = np.array([last_x + i for i in range(1, 4)])
        future_x_poly = PolynomialFeatures(degree=4).fit_transform(future_x.reshape(-1, 1))
        future_y_pred = model.predict(future_x_poly)
        
    elif model_type == 'Cobb-Douglas':
        model, x_log, y_log = additional_params
        last_x = x[-1]
        future_x = np.array([last_x + i for i in range(1, 4)])
        future_x_log = np.log(future_x)
        future_y_log_pred = model.predict(sm.add_constant(future_x_log))
        future_y_pred = np.exp(future_y_log_pred)
        
    return future_x, future_y_pred

st.title('Time Series Trend Analysis')

uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Data Preview:", df.head())

        # Model Equations Table
        model_info = {
            'Model': [
                'Linear Regression',
                'Quadratic Regression',
                'Quartic Regression',
                'Cobb-Douglas Regression'
            ],
            'Equation': [
                'Y = a + bX',
                'Y = a + bX + cX^2',
                'Y = a + bX + cX^2 + dX^3 + eX^4',
                'ln(Y) = a + b*ln(X)'
            ]
        }
        model_df = pd.DataFrame(model_info)
        st.write("### Model Equations")
        st.table(model_df)

        time_column = st.selectbox("Select Time Column", df.columns)
        value_column = st.selectbox("Select Value Column", df.columns)

        x = df[[time_column]].values
        y = df[value_column].values
        
        # Linear Regression
        st.subheader('Linear Regression (Degree 1)')
        mse1, r2_1, intercept1, coef1, pval1, equation1, model1, x_poly1 = fit_and_plot_regression(x, y, degree=1)
        st.write(f"**Model:** {equation1}")
        st.write(f"**Coefficients:** Intercept = {intercept1}, b = {coef1[0]}")
        st.write(f"**P-Values:** b = {pval1[0]}")
        st.write(f"MSE: {mse1:.2f}, R²: {r2_1:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {coef1[0]} means that for each unit increase in X, Y increases by {coef1[0]} units.")
        
        # Quadratic Regression
        st.subheader('Quadratic Regression (Degree 2)')
        mse2, r2_2, intercept2, coef2, pval2, equation2, model2, x_poly2 = fit_and_plot_regression(x, y, degree=2)
        st.write(f"**Model:** {equation2}")
        st.write(f"**Coefficients:** Intercept = {intercept2}, b = {coef2[0]}, c = {coef2[1]}")
        st.write(f"**P-Values:** b = {pval2[0]}, c = {pval2[1]}")
        st.write(f"MSE: {mse2:.2f}, R²: {r2_2:.2f}")
        st.write(f"**Interpretation:** The coefficient c = {coef2[1]} indicates the curvature of the quadratic relationship.")
        
        # Quartic Regression
        st.subheader('Quartic Regression (Degree 4)')
        mse4, r2_4, intercept4, coef4, pval4, equation4, model4, x_poly4 = fit_and_plot_regression(x, y, degree=4)
        st.write(f"**Model:** {equation4}")
        st.write(f"**Coefficients:** Intercept = {intercept4}, b = {coef4[0]}, c = {coef4[1]}, d = {coef4[2]}, e = {coef4[3]}")
        st.write(f"**P-Values:** b = {pval4[0]}, c = {pval4[1]}, d = {pval4[2]}, e = {pval4[3]}")
        st.write(f"MSE: {mse4:.2f}, R²: {r2_4:.2f}")
        st.write(f"**Interpretation:** The coefficient e = {coef4[3]} captures the highest order polynomial effect on Y.")
        
        # Cobb-Douglas Regression
        st.subheader('Cobb-Douglas Regression')
        mse_cd, r2_cd, intercept_cd, coef_cd, pval_cd, equation_cd, model_cd, x_log, y_log = fit_and_plot_cobb_douglas(x, y)
        st.write(f"**Model:** {equation_cd}")
        st.write(f"**Coefficients:** Intercept = {intercept_cd}, b = {coef_cd[0]}")
        st.write(f"**P-Values:** b = {pval_cd[0]}")
        st.write(f"MSE: {mse_cd:.2f}, R²: {r2_cd:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {coef_cd[0]} represents the elasticity of Y with respect to X.")
        
        # Final Results Table
        st.subheader('Model Comparison')
        comparison_data = {
            'Model': ['Linear', 'Quadratic', 'Quartic', 'Cobb-Douglas'],
            'MSE': [mse1, mse2, mse4, mse_cd],
            'R²': [r2_1, r2_2, r2_4, r2_cd],
            'Intercept': [intercept1, intercept2, intercept4, intercept_cd],
            'Coefficients': [coef1, coef2, coef4, coef_cd],
            'P-Values': [pval1, pval2, pval4, pval_cd]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # Identify the best model
        best_model_idx = comparison_df['R²'].idxmax()
        best_model = comparison_df.iloc[best_model_idx]
        best_model_name = comparison_df['Model'][best_model_idx]
        
        # Forecasting
        if best_model_name == 'Linear':
            future_x, future_y_pred = forecast_best_model(best_model, x, y, 'Linear', additional_params=(model1, x_poly1))
        elif best_model_name == 'Quadratic':
            future_x, future_y_pred = forecast_best_model(best_model, x, y, 'Quadratic', additional_params=(model2, x_poly2))
        elif best_model_name == 'Quartic':
            future_x, future_y_pred = forecast_best_model(best_model, x, y, 'Quartic', additional_params=(model4, x_poly4))
        elif best_model_name == 'Cobb-Douglas':
            future_x, future_y_pred = forecast_best_model(best_model, x, y, 'Cobb-Douglas', additional_params=(model_cd, x_log, y_log))
        
        st.subheader('Forecast for Next 3 Periods')
        forecast_df = pd.DataFrame({
            'Future Time': future_x,
            'Forecasted Value': future_y_pred
        })
        st.write(forecast_df)

        st.write(f"**Best Model:** {best_model_name} Regression")
        st.markdown(f"<span style='color: darkorange; font-weight: bold;'>Best Model based on R²: {best_model_name} Regression</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
