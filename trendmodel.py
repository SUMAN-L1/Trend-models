import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy.optimize import curve_fit
import io

# Function to fit and plot exponential model
def fit_and_plot_exponential(x, y):
    def exponential_model(x, a, b):
        return a * np.exp(b * x)
    
    # Fit model
    popt, _ = curve_fit(exponential_model, x, y, p0=(1, 0.1))
    a, b = popt
    
    y_pred = exponential_model(x, a, b)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    equation = f"Y = {a:.4f} * e^({b:.4f} * X)"
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label='Estimated (Exponential)')
    plt.title(f'Exponential Regression\nMSE: {mse:.2f}, R^2: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    return mse, r2, a, b, equation

# Function to fit and plot modified exponential model
def fit_and_plot_modified_exponential(x, y):
    def modified_exponential_model(x, a, b, c):
        return a * np.exp(b * x) + c
    
    # Fit model
    popt, _ = curve_fit(modified_exponential_model, x, y, p0=(1, 0.1, 1))
    a, b, c = popt
    
    y_pred = modified_exponential_model(x, a, b, c)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    equation = f"Y = {a:.4f} * e^({b:.4f} * X) + {c:.4f}"
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label='Estimated (Modified Exponential)')
    plt.title(f'Modified Exponential Regression\nMSE: {mse:.2f}, R^2: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    return mse, r2, a, b, c, equation

# Function to fit and plot regression models
def fit_and_plot_regression(x, y, degree):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x.reshape(-1, 1))
    
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

# Function to fit and plot quadratic B model
def fit_and_plot_quadratic_b(x, y):
    def quadratic_b_model(x, a, b, c):
        return a + b * x - c * x**2
    
    # Fit model
    popt, _ = curve_fit(quadratic_b_model, x, y, p0=(1, 1, 1))
    a, b, c = popt
    
    y_pred = quadratic_b_model(x, a, b, c)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    equation = f"Y = {a:.4f} + {b:.4f} * X - {c:.4f} * X^2"
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label='Estimated (Quadratic B)')
    plt.title(f'Quadratic B Regression\nMSE: {mse:.2f}, R^2: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    return mse, r2, a, b, c, equation

# Function to fit and plot Cobb-Douglas model
def fit_and_plot_cobb_douglas(x, y):
    x_log = np.log(x)
    y_log = np.log(y)
    
    x_log_const = sm.add_constant(x_log.reshape(-1, 1))  # Ensure x_log is 2D
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
def forecast_best_model(model_type, x, y, additional_params):
    last_x = x[-1]
    future_x = np.array([last_x + i for i in range(1, 4)])
    
    if model_type == 'Linear':
        def linear_model(x, a, b):
            return a + b * x
        future_y_pred = linear_model(future_x, *additional_params)
        
    elif model_type == 'Quadratic':
        def quadratic_model(x, a, b, c):
            return a + b * x + c * x**2
        future_y_pred = quadratic_model(future_x, *additional_params)
        
    elif model_type == 'Quartic':
        def quartic_model(x, a, b, c, d, e):
            return a + b * x + c * x**2 + d * x**3 + e * x**4
        future_y_pred = quartic_model(future_x, *additional_params)
        
    elif model_type == 'Exponential':
        def exponential_model(x, a, b):
            return a * np.exp(b * x)
        future_y_pred = exponential_model(future_x, *additional_params)
        
    elif model_type == 'Modified Exponential':
        def modified_exponential_model(x, a, b, c):
            return a * np.exp(b * x) + c
        future_y_pred = modified_exponential_model(future_x, *additional_params)
        
    elif model_type == 'Quadratic B':
        def quadratic_b_model(x, a, b, c):
            return a + b * x - c * x**2
        future_y_pred = quadratic_b_model(future_x, *additional_params)
        
    elif model_type == 'Cobb-Douglas':
        def cobb_douglas_model(x, a, b):
            return np.exp(a + b * np.log(x))
        future_y_pred = cobb_douglas_model(future_x, *additional_params)
        
    return future_x, future_y_pred

# Streamlit app layout
st.title('Regression Models Analysis')

st.write("### Model Equations")
st.write(pd.DataFrame({
    'Model': ['Linear', 'Quadratic', 'Quartic', 'Exponential', 'Modified Exponential', 'Quadratic B', 'Cobb-Douglas'],
    'Equation': [
        'Y = a + b*X',
        'Y = a + b*X + c*X^2',
        'Y = a + b*X + c*X^2 + d*X^3 + e*X^4',
        'Y = a * e^(b * X)',
        'Y = a * e^(b * X) + c',
        'Y = a + b*X - c*X^2',
        'ln(Y) = a + b*ln(X)'
    ]
}))

uploaded_file = st.file_uploader("Choose an Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')

    st.write("### Data Overview")
    st.write(df.head())
    
    st.write("### Select Columns")
    year_col = st.selectbox("Select Year Column", df.columns)
    y_col = st.selectbox("Select Dependent Variable Column", df.columns)
    
    x = df[year_col].values
    y = df[y_col].values
    
    # Linear Regression
    st.write("### Linear Regression")
    mse_linear, r2_linear, intercept_linear, coef_linear, p_value_linear, equation_linear, model_linear, x_poly_linear = fit_and_plot_regression(x, y, 1)
    
    # Quadratic Regression
    st.write("### Quadratic Regression")
    mse_quadratic, r2_quadratic, intercept_quadratic, coef_quadratic, p_value_quadratic, equation_quadratic, model_quadratic, x_poly_quadratic = fit_and_plot_regression(x, y, 2)
    
    # Quartic Regression
    st.write("### Quartic Regression")
    mse_quartic, r2_quartic, intercept_quartic, coef_quartic, p_value_quartic, equation_quartic, model_quartic, x_poly_quartic = fit_and_plot_regression(x, y, 4)
    
    # Exponential Regression
    st.write("### Exponential Regression")
    mse_exponential, r2_exponential, a_exp, b_exp, equation_exp = fit_and_plot_exponential(x, y)
    
    # Modified Exponential Regression
    st.write("### Modified Exponential Regression")
    mse_mod_exp, r2_mod_exp, a_mod_exp, b_mod_exp, c_mod_exp, equation_mod_exp = fit_and_plot_modified_exponential(x, y)
    
    # Quadratic B Regression
    st.write("### Quadratic B Regression")
    mse_quad_b, r2_quad_b, a_quad_b, b_quad_b, c_quad_b, equation_quad_b = fit_and_plot_quadratic_b(x, y)
    
    # Cobb-Douglas Regression
    st.write("### Cobb-Douglas Regression")
    mse_cobb_douglas, r2_cobb_douglas, intercept_cobb_douglas, coef_cobb_douglas, p_value_cobb_douglas, equation_cobb_douglas, model_cobb_douglas, x_log, y_log = fit_and_plot_cobb_douglas(x, y)

    # Determine best model based on MSE and R^2
    models = {
        'Linear': (mse_linear, r2_linear),
        'Quadratic': (mse_quadratic, r2_quadratic),
        'Quartic': (mse_quartic, r2_quartic),
        'Exponential': (mse_exponential, r2_exponential),
        'Modified Exponential': (mse_mod_exp, r2_mod_exp),
        'Quadratic B': (mse_quad_b, r2_quad_b),
        'Cobb-Douglas': (mse_cobb_douglas, r2_cobb_douglas)
    }
    
    best_model = min(models, key=lambda k: models[k][0])  # Minimum MSE
    best_r2_model = max(models, key=lambda k: models[k][1])  # Maximum R^2

    # Highlight best model based on both criteria
    st.markdown(f"<h1 style='color: orange; font-size: 30px;'>Best Model Based on MSE and R²: {best_model}</h1>", unsafe_allow_html=True)

    # Forecasting future values using the best model
    if best_model == 'Linear':
        future_x, future_y_pred = forecast_best_model('Linear', x, y, 'Linear', additional_params=(model_linear, x_poly_linear))
    elif best_model == 'Quadratic':
        future_x, future_y_pred = forecast_best_model('Quadratic', x, y, 'Quadratic', additional_params=(model_quadratic, x_poly_quadratic))
    elif best_model == 'Quartic':
        future_x, future_y_pred = forecast_best_model('Quartic', x, y, 'Quartic', additional_params=(model_quartic, x_poly_quartic))
    elif best_model == 'Exponential':
        future_x, future_y_pred = forecast_best_model('Exponential', x, y, 'Exponential', additional_params=(a_exp, b_exp))
    elif best_model == 'Modified Exponential':
        future_x, future_y_pred = forecast_best_model('Modified Exponential', x, y, 'Modified Exponential', additional_params=(a_mod_exp, b_mod_exp, c_mod_exp))
    elif best_model == 'Quadratic B':
        future_x, future_y_pred = forecast_best_model('Quadratic B', x, y, 'Quadratic B', additional_params=(a_quad_b, b_quad_b, c_quad_b))
    elif best_model == 'Cobb-Douglas':
        future_x, future_y_pred = forecast_best_model('Cobb-Douglas', x, y, 'Cobb-Douglas', additional_params=(model_cobb_douglas, x_log, y_log))

    st.write(f"### Forecasting using the Best Model: {best_model}")
    st.write(pd.DataFrame({
        'Future Time': future_x,
        'Predicted Value': future_y_pred
    }))

