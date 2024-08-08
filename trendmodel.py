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

        # Model Equations Table
        model_info = {
            'Model': ['Linear', 'Quadratic', 'Quartic', 'Cobb-Douglas', 'Exponential', 'Modified Exponential'],
            'Equation': [
                'Y = a + bX',
                'Y = a + bX + cX^2',
                'Y = a + bX + cX^2 + dX^3 + eX^4',
                'ln(Y) = a + b*ln(X)',
                'Y = a * e^(b * X)',
                'Y = a * e^(b * X) + c'
            ]
        }
        model_info_df = pd.DataFrame(model_info)
        st.write("Model Equations:")
        st.table(model_info_df)
        
        # Extract columns
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values

        # Linear Regression
        st.subheader('Linear Regression')
        mse1, r2_1, intercept1, coef1, pval1, equation1, model1, x_poly1 = fit_and_plot_regression(x, y, 1)
        st.write(f"**Model:** {equation1}")
        st.write(f"**Coefficients:** Intercept = {intercept1}, b = {coef1[0]}")
        st.write(f"**P-Values:** b = {pval1[0]}")
        st.write(f"MSE: {mse1:.2f}, R²: {r2_1:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {coef1[0]} indicates the rate of change in Y for each unit change in X.")
        
        # Quadratic Regression
        st.subheader('Quadratic Regression')
        mse2, r2_2, intercept2, coef2, pval2, equation2, model2, x_poly2 = fit_and_plot_regression(x, y, 2)
        st.write(f"**Model:** {equation2}")
        st.write(f"**Coefficients:** Intercept = {intercept2}, b = {coef2[0]}, c = {coef2[1]}")
        st.write(f"**P-Values:** b = {pval2[0]}, c = {pval2[1]}")
        st.write(f"MSE: {mse2:.2f}, R²: {r2_2:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {coef2[0]} captures the linear effect, and c = {coef2[1]} represents the curvature effect.")
        
        # Quartic Regression
        st.subheader('Quartic Regression')
        mse4, r2_4, intercept4, coef4, pval4, equation4, model4, x_poly4 = fit_and_plot_regression(x, y, 4)
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
        
        # Exponential Regression
        st.subheader('Exponential Regression')
        mse_exp, r2_exp, a_exp, b_exp, equation_exp = fit_and_plot_exponential(x, y)
        st.write(f"**Model:** {equation_exp}")
        st.write(f"**Coefficients:** a = {a_exp}, b = {b_exp}")
        st.write(f"MSE: {mse_exp:.2f}, R²: {r2_exp:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {b_exp} represents the growth rate of Y with respect to X.")
        
        # Modified Exponential Regression
        st.subheader('Modified Exponential Regression')
        mse_mod_exp, r2_mod_exp, a_mod_exp, b_mod_exp, c_mod_exp, equation_mod_exp = fit_and_plot_modified_exponential(x, y)
        st.write(f"**Model:** {equation_mod_exp}")
        st.write(f"**Coefficients:** a = {a_mod_exp}, b = {b_mod_exp}, c = {c_mod_exp}")
        st.write(f"MSE: {mse_mod_exp:.2f}, R²: {r2_mod_exp:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {b_mod_exp} represents the growth rate, and c = {c_mod_exp} is a constant offset.")
        
        # Final Results Table
        st.subheader('Model Comparison')
        comparison_data = {
            'Model': ['Linear', 'Quadratic', 'Quartic', 'Cobb-Douglas', 'Exponential', 'Modified Exponential'],
            'MSE': [mse1, mse2, mse4, mse_cd, mse_exp, mse_mod_exp],
            'R²': [r2_1, r2_2, r2_4, r2_cd, r2_exp, r2_mod_exp],
            'Intercept': [intercept1, intercept2, intercept4, intercept_cd, np.nan, np.nan],
            'Coefficients': [coef1, coef2, coef4, coef_cd, f'a = {a_exp}, b = {b_exp}', f'a = {a_mod_exp}, b = {b_mod_exp}, c = {c_mod_exp}'],
            'P-Values': [pval1, pval2, pval4, pval_cd, np.nan, np.nan]
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
        elif best_model_name == 'Exponential':
            future_x, future_y_pred = forecast_best_model(best_model, x, y, 'Exponential', additional_params=(a_exp, b_exp))
        elif best_model_name == 'Modified Exponential':
            future_x, future_y_pred = forecast_best_model(best_model, x, y, 'Modified Exponential', additional_params=(a_mod_exp, b_mod_exp, c_mod_exp))
        
        st.subheader('Forecast for Next 3 Periods')
        forecast_df = pd.DataFrame({
            'Future Time': future_x.flatten(),  # Ensure 1D array
            'Forecasted Value': future_y_pred.flatten()  # Ensure 1D array
        })
        st.write(forecast_df)

        st.write(f"**Best Model:** {best_model_name} Regression")
        st.markdown(f"<span style='color: darkorange; font-weight: bold;'>Best Model based on R²: {best_model_name} Regression</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
