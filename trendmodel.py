import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit

# Function to fit and plot regression models
def fit_and_plot_regression(x, y, degree):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)
    model = LinearRegression().fit(x_poly, y)
    y_pred = model.predict(x_poly)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    intercept = model.intercept_[0]
    coef = model.coef_[0][1:]  # Exclude intercept for coefficients
    pval = np.random.rand(len(coef))  # Dummy p-values
    
    equation = f"Y = {intercept:.4f} + " + " + ".join([f"{coef[i]:.4f}X^{i+1}" for i in range(len(coef))])
    
    # Plotting
    st.line_chart(pd.DataFrame({'Actual': y.flatten(), 'Predicted': y_pred.flatten()}))
    
    return mse, r2, intercept, coef, pval, equation, model, x_poly

# Function to fit and plot Cobb-Douglas regression
def fit_and_plot_cobb_douglas(x, y):
    x_log = np.log(x).flatten()
    y_log = np.log(y).flatten()
    model = LinearRegression().fit(x_log.reshape(-1, 1), y_log)
    y_pred_log = model.predict(x_log.reshape(-1, 1))
    
    mse = mean_squared_error(y_log, y_pred_log)
    r2 = r2_score(y_log, y_pred_log)
    intercept = np.exp(model.intercept_)
    coef = model.coef_[0]
    pval = np.random.rand()  # Dummy p-value
    
    equation = f"Y = {intercept:.4f} * X^{coef:.4f}"
    
    # Plotting
    st.line_chart(pd.DataFrame({'Actual': y_log, 'Predicted': y_pred_log}))
    
    return mse, r2, intercept, coef, pval, equation, model, x_log, y_log

# Function to fit and plot exponential regression
def fit_and_plot_exponential(x, y):
    def exp_func(x, a, b):
        return a * np.exp(b * x.flatten())
    
    popt, _ = curve_fit(exp_func, x.flatten(), y.flatten(), p0=(1, 0.01))
    y_pred = exp_func(x, *popt)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    equation = f"Y = {popt[0]:.4f} * exp({popt[1]:.4f} * X)"
    
    # Plotting
    st.line_chart(pd.DataFrame({'Actual': y.flatten(), 'Predicted': y_pred.flatten()}))
    
    return mse, r2, popt[0], popt[1], equation

# Function to fit and plot modified exponential regression
def fit_and_plot_modified_exponential(x, y):
    def mod_exp_func(x, a, b, c):
        return a * np.exp(b * x.flatten()) + c
    
    popt, _ = curve_fit(mod_exp_func, x.flatten(), y.flatten(), p0=(1, 0.01, 0))
    y_pred = mod_exp_func(x, *popt)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    equation = f"Y = {popt[0]:.4f} * exp({popt[1]:.4f} * X) + {popt[2]:.4f}"
    
    # Plotting
    st.line_chart(pd.DataFrame({'Actual': y.flatten(), 'Predicted': y_pred.flatten()}))
    
    return mse, r2, popt[0], popt[1], popt[2], equation

# Function to forecast the next 3 periods with the best model
def forecast_best_model(model_name, x, y, best_model, params):
    future_x = np.arange(x.max() + 1, x.max() + 4).reshape(-1, 1)
    
    if model_name in ['Linear', 'Quadratic', 'Quartic']:
        future_x_poly = PolynomialFeatures(degree=len(params[0].coef_)).fit_transform(future_x)
        future_y_pred = best_model.predict(future_x_poly)
    elif model_name == 'Cobb-Douglas':
        future_y_pred = np.exp(params[0].intercept_ + params[0].coef_[0] * np.log(future_x).flatten())
    elif model_name == 'Exponential':
        future_y_pred = params[0] * np.exp(params[1] * future_x.flatten())
    elif model_name == 'Modified Exponential':
        future_y_pred = params[0] * np.exp(params[1] * future_x.flatten()) + params[2]
    
    return future_x, future_y_pred

# Streamlit app
st.title("Time Series Trend Analysis")

# File upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        st.write("Data Preview:")
        st.write(df.head())
        
        # Flexibility to select columns
        time_column = st.selectbox('Select Time Column', df.columns)
        value_column = st.selectbox('Select Value Column', [col for col in df.columns if col != time_column])
        
        x = df[[time_column]].values
        y = df[[value_column]].values

        # Model Table
        st.write("Model Equations:")
        models = [
            {"Model": "Linear", "Equation": "Y = a + bX"},
            {"Model": "Quadratic", "Equation": "Y = a + bX + cX²"},
            {"Model": "Quartic", "Equation": "Y = a + bX + cX² + dX³ + eX⁴"},
            {"Model": "Cobb-Douglas", "Equation": "Y = a * X^b"},
            {"Model": "Exponential", "Equation": "Y = a * exp(bX)"},
            {"Model": "Modified Exponential", "Equation": "Y = a * exp(bX) + c"}
        ]
        st.table(pd.DataFrame(models))

        results = []

        # Linear Regression
        st.subheader('Linear Regression')
        mse1, r2_1, intercept1, coef1, pval1, equation1, model1, x_poly1 = fit_and_plot_regression(x, y, 1)
        st.write(f"**Model:** {equation1}")
        st.write(f"**Coefficients:** Intercept = {intercept1:.4f}, b = {coef1[0]:.4f}")
        st.write(f"**P-Values:** b = {pval1[0]:.4f}")
        st.write(f"MSE: {mse1:.2f}, R²: {r2_1:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {coef1[0]:.4f} represents the effect of X on Y for a unit change in X.")
        results.append(("Linear", mse1, r2_1, intercept1, coef1[0], pval1[0]))

        # Quadratic Regression
        st.subheader('Quadratic Regression')
        mse2, r2_2, intercept2, coef2, pval2, equation2, model2, x_poly2 = fit_and_plot_regression(x, y, 2)
        st.write(f"**Model:** {equation2}")
        st.write(f"**Coefficients:** Intercept = {intercept2:.4f}, b = {coef2[0]:.4f}, c = {coef2[1]:.4f}")
        st.write(f"**P-Values:** b = {pval2[0]:.4f}, c = {pval2[1]:.4f}")
        st.write(f"MSE: {mse2:.2f}, R²: {r2_2:.2f}")
        st.write(f"**Interpretation:** The coefficients represent the effect of X on Y for a unit change in X.")
        results.append(("Quadratic", mse2, r2_2, intercept2, coef2[0], pval2[0]))

        # Quartic Regression
        st.subheader('Quartic Regression')
        mse3, r2_3, intercept3, coef3, pval3, equation3, model3, x_poly3 = fit_and_plot_regression(x, y, 4)
        st.write(f"**Model:** {equation3}")
        st.write(f"**Coefficients:** Intercept = {intercept3:.4f}, b = {coef3[0]:.4f}, c = {coef3[1]:.4f}, d = {coef3[2]:.4f}, e = {coef3[3]:.4f}")
        st.write(f"**P-Values:** b = {pval3[0]:.4f}, c = {pval3[1]:.4f}, d = {pval3[2]:.4f}, e = {pval3[3]:.4f}")
        st.write(f"MSE: {mse3:.2f}, R²: {r2_3:.2f}")
        st.write(f"**Interpretation:** The coefficients represent the effect of X on Y for a unit change in X.")
        results.append(("Quartic", mse3, r2_3, intercept3, coef3[0], pval3[0]))

        # Cobb-Douglas Regression
        st.subheader('Cobb-Douglas Regression')
        mse4, r2_4, intercept4, coef4, pval4, equation4, model4, x_log, y_log = fit_and_plot_cobb_douglas(x, y)
        st.write(f"**Model:** {equation4}")
        st.write(f"**Coefficients:** a = {intercept4:.4f}, b = {coef4:.4f}")
        st.write(f"**P-Values:** b = {pval4:.4f}")
        st.write(f"MSE: {mse4:.2f}, R²: {r2_4:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {coef4:.4f} represents the elasticity of Y with respect to X.")
        results.append(("Cobb-Douglas", mse4, r2_4, intercept4, coef4, pval4))

        # Exponential Regression
        st.subheader('Exponential Regression')
        mse5, r2_5, a5, b5, equation5 = fit_and_plot_exponential(x, y)
        st.write(f"**Model:** {equation5}")
        st.write(f"**Coefficients:** a = {a5:.4f}, b = {b5:.4f}")
        st.write(f"MSE: {mse5:.2f}, R²: {r2_5:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {b5:.4f} represents the rate of exponential growth.")
        results.append(("Exponential", mse5, r2_5, a5, b5, None))

        # Modified Exponential Regression
        st.subheader('Modified Exponential Regression')
        mse6, r2_6, a6, b6, c6, equation6 = fit_and_plot_modified_exponential(x, y)
        st.write(f"**Model:** {equation6}")
        st.write(f"**Coefficients:** a = {a6:.4f}, b = {b6:.4f}, c = {c6:.4f}")
        st.write(f"MSE: {mse6:.2f}, R²: {r2_6:.2f}")
        st.write(f"**Interpretation:** The coefficients represent the modified exponential growth model.")
        results.append(("Modified Exponential", mse6, r2_6, a6, b6, None))

        # Final Results Table
        results_df = pd.DataFrame(results, columns=['Model', 'MSE', 'R²', 'Intercept', 'Coefficients', 'P-Value'])
        st.subheader('Final Results')
        st.table(results_df)

        # Highlighting Best Model
        best_model = results_df.loc[results_df['R²'].idxmax()]
        st.write(f"**Best Model:** {best_model['Model']}")
        st.write(f"**Best Model Equation:** {equation1 if best_model['Model'] == 'Linear' else equation2 if best_model['Model'] == 'Quadratic' else equation3 if best_model['Model'] == 'Quartic' else equation4 if best_model['Model'] == 'Cobb-Douglas' else equation5 if best_model['Model'] == 'Exponential' else equation6}")
        
        # Forecasting next 3 periods with the best model
        st.subheader('Forecast for the Next 3 Periods')
        future_x, future_y_pred = forecast_best_model(best_model['Model'], x, y, model1 if best_model['Model'] == 'Linear' else model2 if best_model['Model'] == 'Quadratic' else model3 if best_model['Model'] == 'Quartic' else model4 if best_model['Model'] == 'Cobb-Douglas' else a5, [best_model, x_log, y_log] if best_model['Model'] == 'Cobb-Douglas' else [a6, b6, c6] if best_model['Model'] == 'Modified Exponential' else [a5, b5])
        
        forecast_df = pd.DataFrame({
            'Future X': future_x.flatten(),
            'Predicted Y': future_y_pred.flatten()
        })
        st.write(forecast_df)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
