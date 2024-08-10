import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy.optimize import curve_fit

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
        future_y_log_pred = model.predict(sm.add_constant(future_x_log.reshape(-1, 1)))  # Ensure future_x_log is 2D
        future_y_pred = np.exp(future_y_log_pred)
        
    elif model_type == 'Exponential':
        a, b = additional_params
        last_x = x[-1]
        future_x = np.array([last_x + i for i in range(1, 4)])
        future_y_pred = a * np.exp(b * future_x)
        
    elif model_type == 'Modified Exponential':
        a, b, c = additional_params
        last_x = x[-1]
        future_x = np.array([last_x + i for i in range(1, 4)])
        future_y_pred = a * np.exp(b * future_x) + c
        
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
        
        # Select columns for independent and dependent variables
        independent_var = st.selectbox('Select the Time/Year column:', df.columns)
        dependent_var = st.selectbox('Select the dependent variable column:', df.columns)

        # Extract columns
        x = df[independent_var].values
        y = df[dependent_var].values

        # Model Equations Table
        model_info = {
            'Model': ['Linear', 'Quadratic', 'Quartic', 'Cobb-Douglas', 'Exponential', 'Modified Exponential'],
            'Equation': [],
            'MSE': [],
            'R2': []
        }

        # Linear Regression
        linear_mse, linear_r2, linear_intercept, linear_coefficients, linear_p_values, linear_equation, linear_model, linear_x_poly = fit_and_plot_regression(x, y, degree=1)
        model_info['Equation'].append(linear_equation)
        model_info['MSE'].append(linear_mse)
        model_info['R2'].append(linear_r2)
        
        # Quadratic Regression
        quadratic_mse, quadratic_r2, quadratic_intercept, quadratic_coefficients, quadratic_p_values, quadratic_equation, quadratic_model, quadratic_x_poly = fit_and_plot_regression(x, y, degree=2)
        model_info['Equation'].append(quadratic_equation)
        model_info['MSE'].append(quadratic_mse)
        model_info['R2'].append(quadratic_r2)

        # Quartic Regression
        quartic_mse, quartic_r2, quartic_intercept, quartic_coefficients, quartic_p_values, quartic_equation, quartic_model, quartic_x_poly = fit_and_plot_regression(x, y, degree=4)
        model_info['Equation'].append(quartic_equation)
        model_info['MSE'].append(quartic_mse)
        model_info['R2'].append(quartic_r2)

        # Cobb-Douglas Regression
        cobb_douglas_mse, cobb_douglas_r2, cobb_douglas_intercept, cobb_douglas_coefficients, cobb_douglas_p_values, cobb_douglas_equation, cobb_douglas_model, cobb_douglas_x_log, cobb_douglas_y_log = fit_and_plot_cobb_douglas(x, y)
        model_info['Equation'].append(cobb_douglas_equation)
        model_info['MSE'].append(cobb_douglas_mse)
        model_info['R2'].append(cobb_douglas_r2)
        
        # Exponential Regression
        exponential_mse, exponential_r2, exponential_a, exponential_b, exponential_equation = fit_and_plot_exponential(x, y)
        model_info['Equation'].append(exponential_equation)
        model_info['MSE'].append(exponential_mse)
        model_info['R2'].append(exponential_r2)

        # Modified Exponential Regression
        mod_exp_mse, mod_exp_r2, mod_exp_a, mod_exp_b, mod_exp_c, mod_exp_equation = fit_and_plot_modified_exponential(x, y)
        model_info['Equation'].append(mod_exp_equation)
        model_info['MSE'].append(mod_exp_mse)
        model_info['R2'].append(mod_exp_r2)

        # Display model info
        model_df = pd.DataFrame(model_info)
        st.write("Model Equations and Metrics:")
        st.write(model_df)

        # Determine the best model
        best_model_idx = model_df['R2'].idxmax()
        best_model_name = model_df.iloc[best_model_idx]['Model']
        st.write(f"**Best Model:** {best_model_name}")

        # Forecast with the best model
        additional_params = None
        if best_model_name == 'Linear':
            additional_params = (linear_model, linear_x_poly)
        elif best_model_name == 'Quadratic':
            additional_params = (quadratic_model, quadratic_x_poly)
        elif best_model_name == 'Quartic':
            additional_params = (quartic_model, quartic_x_poly)
        elif best_model_name == 'Cobb-Douglas':
            additional_params = (cobb_douglas_model, cobb_douglas_x_log, cobb_douglas_y_log)
        elif best_model_name == 'Exponential':
            additional_params = (exponential_a, exponential_b)
        elif best_model_name == 'Modified Exponential':
            additional_params = (mod_exp_a, mod_exp_b, mod_exp_c)

        future_x, future_y_pred = forecast_best_model(best_model_name, x, y, best_model_name, additional_params)

        # Display forecast
        forecast_df = pd.DataFrame({independent_var: future_x, dependent_var: future_y_pred})
        st.write("Forecast for the Next 3 Time Points:")
        st.write(forecast_df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
