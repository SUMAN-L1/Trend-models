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

        x = df[[time_column]].values.flatten()
        y = df[value_column].values

        # Linear Regression
        st.subheader('Linear Regression (Degree 1)')
        mse1, r2_1, intercept1, coef1, pval1, equation1, model1, x_poly1 = fit_and_plot_regression(x, y, degree=1)
        st.write(f"**Model:** {equation1}")
        st.write(f"**Coefficients:** Intercept = {intercept1}, b = {coef1[0]}")
        st.write(f"**P-Values:** b = {pval1[0]}")
        st.write(f"MSE: {mse1:.2f}, R²: {r2_1:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {coef1[0]} means that for each unit increase in X, Y increases by {coef1[0]} units.")
        
        # Quadratic Regr
