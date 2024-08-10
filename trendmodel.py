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
    rmse = np.sqrt(mse)
    
    equation = f"Y = {a:.4f} * e^({b:.4f} * X)"
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label='Estimated (Exponential)')
    plt.title(f'Exponential Regression\nMSE: {mse:.2f}, R²: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Interpretations and details
    st.write(f"**Exponential Model Equation:** {equation}")
    st.write(f"**Coefficients:** a = {a:.4f}, b = {b:.4f}")
    st.write(f"**R²:** {r2:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**Interpretation:**")
    st.write(f"1. **Coefficient a ({a:.4f})**: This represents the initial value when X (Time) is 0.")
    st.write(f"2. **Coefficient b ({b:.4f})**: This represents the growth rate. If b > 0, the value is increasing exponentially over time; if b < 0, the value is decreasing exponentially over time.")
    st.write(f"3. **Decision Insight**: If the goal is to assess growth, a positive b indicates consistent growth, while a negative b suggests a decline. The magnitude of b reflects the speed of this change.")
    
    return mse, r2, rmse, a, b, equation

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
    rmse = np.sqrt(mse)
    
    equation = f"Y = {a:.4f} * e^({b:.4f} * X) + {c:.4f}"
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label='Estimated (Modified Exponential)')
    plt.title(f'Modified Exponential Regression\nMSE: {mse:.2f}, R²: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Interpretations and details
    st.write(f"**Modified Exponential Model Equation:** {equation}")
    st.write(f"**Coefficients:** a = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
    st.write(f"**R²:** {r2:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**Interpretation:**")
    st.write(f"1. **Coefficient a ({a:.4f})**: This is the initial scale factor when X (Time) is 0.")
    st.write(f"2. **Coefficient b ({b:.4f})**: This indicates the growth rate, similar to the basic exponential model.")
    st.write(f"3. **Coefficient c ({c:.4f})**: This is the adjustment factor that shifts the entire curve up or down.")
    st.write(f"4. **Decision Insight**: The model adjusts for baseline shifts with c. A positive b still indicates growth, but c allows for a baseline that isn't at zero, reflecting underlying trends in the data.")
    
    return mse, r2, rmse, a, b, c, equation

# Function to fit and plot regression models
def fit_and_plot_regression(x, y, degree):
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(x_poly, y)
    
    y_pred = model.predict(x_poly)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mse)
    
    x_poly_const = sm.add_constant(x_poly)
    ols_model = sm.OLS(y, x_poly_const).fit()
    p_values = np.round(ols_model.pvalues[1:], 4)  # Skip intercept p-value
    coefficients = np.round(model.coef_[1:], 4)  # Skip intercept coefficient
    intercept = np.round(model.intercept_, 4)
    
    equation = f"Y = {intercept} + " + " + ".join([f"{coef}*X^{i}" for i, coef in enumerate(coefficients, start=1)])
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label=f'Estimated (degree {degree})')
    plt.title(f'Degree {degree} Regression\nMSE: {mse:.2f}, R²: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Interpretations and details
    st.write(f"**Degree {degree} Polynomial Model Equation:** {equation}")
    st.write(f"**Intercept:** {intercept}")
    st.write(f"**Coefficients:** {coefficients}")
    st.write(f"**P-Values:** {p_values}")
    st.write(f"**R²:** {r2:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**Interpretation:**")
    st.write(f"1. **Intercept ({intercept})**: This represents the starting value of Y when X (Time) is 0.")
    st.write(f"2. **Coefficients**: Each coefficient represents the contribution of the corresponding polynomial term to the dependent variable Y.")
    st.write(f"   - **Linear Term ({coefficients[0]}):** Indicates the rate of change. A positive value means Y increases as X increases, and a negative value means Y decreases as X increases.")
    st.write(f"   - **Quadratic Term (degree 2):** If present, this term captures the curvature in the relationship between X and Y. A positive coefficient suggests a U-shaped relationship, while a negative coefficient suggests an inverted U-shape.")
    st.write(f"   - **Higher-Order Terms:** If the model includes cubic or quartic terms, these capture more complex shapes in the data.")
    st.write(f"3. **Decision Insight**: The decision-maker should look at the sign and magnitude of each term. For example, a large positive quadratic term might suggest acceleration in the growth, while a negative term suggests deceleration or a peak.")
    
    return mse, r2, rmse, intercept, coefficients, p_values, equation, model, x_poly

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
    rmse = np.sqrt(mse)
    
    intercept = np.round(model.params[0], 4)
    coefficients = np.round(model.params[1:], 4)  # Skip intercept coefficient
    p_values = np.round(model.pvalues[1:], 4)  # Skip intercept p-value
    equation = f"ln(Y) = {intercept} + {coefficients[0]}*ln(X)"
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label='Estimated (Cobb-Douglas)')
    plt.title(f'Cobb-Douglas Regression\nMSE: {mse:.2f}, R²: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    # Interpretations and details
    st.write(f"**Cobb-Douglas Model Equation:** {equation}")
    st.write(f"**Coefficients:** {coefficients}")
    st.write(f"**P-Values:** {p_values}")
    st.write(f"**R²:** {r2:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**Interpretation:**")
    st.write(f"1. **Intercept ({intercept}):** This represents the logarithm of the baseline output when the independent variable is at its baseline level (X=1).")
    st.write(f"2. **Coefficient ({coefficients[0]}):** This indicates the elasticity of Y with respect to X. If this coefficient is greater than 1, Y is highly elastic, meaning a 1% increase in X results in more than a 1% increase in Y. If it's less than 1, Y is inelastic.")
    st.write(f"3. **Decision Insight**: The Cobb-Douglas model is particularly useful for analyzing production functions or growth scenarios. A highly elastic relationship suggests that the dependent variable responds strongly to changes in the independent variable.")
    
    return mse, r2, rmse, intercept, coefficients, p_values, equation, model, x_log, y_log

# Forecast function remains the same as previously provided

# Main Streamlit app code
st.title("Time Series Trend Analysis")

uploaded_file = st.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("Data Preview:")
        st.write(df.head())

        # User selects Time/Year and Dependent Variable columns
        time_column = st.selectbox("Select the Time/Year column:", df.columns)
        dependent_column = st.selectbox("Select the Dependent Variable column:", df.columns)
        
        x = df[time_column].values
        y = df[dependent_column].values
        
        model_info = {
            'Model': ['Linear', 'Quadratic', 'Quartic', 'Cobb-Douglas', 'Exponential', 'Modified Exponential'],
            'MSE': [],
            'R^2': [],
            'Equation': []
        }
        
        # Linear Regression (Degree 1)
        mse, r2, rmse, intercept, coefficients, p_values, equation, linear_model, x_poly = fit_and_plot_regression(x, y, degree=1)
        model_info['MSE'].append(mse)
        model_info['R^2'].append(r2)
        model_info['Equation'].append(equation)

        # Quadratic Regression (Degree 2)
        mse, r2, rmse, intercept, coefficients, p_values, equation, quad_model, x_poly = fit_and_plot_regression(x, y, degree=2)
        model_info['MSE'].append(mse)
        model_info['R^2'].append(r2)
        model_info['Equation'].append(equation)
        
        # Quartic Regression (Degree 4)
        mse, r2, rmse, intercept, coefficients, p_values, equation, quartic_model, x_poly = fit_and_plot_regression(x, y, degree=4)
        model_info['MSE'].append(mse)
        model_info['R^2'].append(r2)
        model_info['Equation'].append(equation)
        
        # Cobb-Douglas Regression
        mse, r2, rmse, intercept, coefficients, p_values, equation, cobb_douglas_model, x_log, y_log = fit_and_plot_cobb_douglas(x, y)
        model_info['MSE'].append(mse)
        model_info['R^2'].append(r2)
        model_info['Equation'].append(equation)
        
        # Exponential Regression
        mse, r2, rmse, a, b, equation = fit_and_plot_exponential(x, y)
        model_info['MSE'].append(mse)
        model_info['R^2'].append(r2)
        model_info['Equation'].append(equation)
        
        # Modified Exponential Regression
        mse, r2, rmse, a, b, c, equation = fit_and_plot_modified_exponential(x, y)
        model_info['MSE'].append(mse)
        model_info['R^2'].append(r2)
        model_info['Equation'].append(equation)
        
        # Display the final model equations and metrics
        df_results = pd.DataFrame(model_info)
        
        # Highlight the best model (Lowest MSE & Highest R^2)
        best_model_idx = df_results[['MSE', 'R^2']].mean(axis=1).idxmin()
        df_results.iloc[best_model_idx, :] = df_results.iloc[best_model_idx, :].apply(lambda x: f"**{x}**", axis=1)
        df_results.style.applymap(lambda x: 'background-color: darkorange' if '**' in str(x) else '')
        
        st.write("Model Comparison:")
        st.table(df_results)
        
        # Forecasting
        best_model_type = df_results['Model'][best_model_idx]
        additional_params = None
        
        if best_model_type == 'Linear':
            additional_params = (linear_model, x_poly)
        elif best_model_type == 'Quadratic':
            additional_params = (quad_model, x_poly)
        elif best_model_type == 'Quartic':
            additional_params = (quartic_model, x_poly)
        elif best_model_type == 'Cobb-Douglas':
            additional_params = (cobb_douglas_model, x_log, y_log)
        elif best_model_type == 'Exponential':
            additional_params = (a, b)
        elif best_model_type == 'Modified Exponential':
            additional_params = (a, b, c)
        
        future_x, future_y_pred = forecast_best_model(best_model_type, x, y, best_model_type, additional_params)
        
        st.write(f"Forecast for the next 3 periods using the best model ({best_model_type}):")
        forecast_df = pd.DataFrame({
            'Time/Year': future_x,
            'Forecasted Value': future_y_pred
        })
        st.write(forecast_df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a file to start analysis.")
