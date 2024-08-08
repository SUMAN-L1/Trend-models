import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def fit_and_plot_regression(x, y, degree):
    # Fit polynomial regression
    X_poly = np.vander(x, N=degree+1, increasing=True)
    model = sm.OLS(y, sm.add_constant(X_poly)).fit()
    y_pred = model.predict(sm.add_constant(X_poly))
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    intercept = model.params[0]
    coef = model.params[1:]
    pval = model.pvalues[1:]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x, y_pred, color='red', label='Fitted Line')
    plt.title(f'{degree}-Degree Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    st.pyplot(plt)
    
    equation = ' + '.join([f'{coef[i]:.4f}X^{i+1}' for i in range(len(coef))])
    equation = f'Y = {intercept:.4f} + {equation}'
    
    return mse, r2, intercept, coef, pval, equation, model, X_poly

def fit_and_plot_cobb_douglas(x, y):
    # Cobb-Douglas Model
    x_log = np.log(x)
    y_log = np.log(y)
    model = sm.OLS(y_log, sm.add_constant(x_log)).fit()
    y_pred = np.exp(model.predict(sm.add_constant(x_log)))
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    intercept = np.exp(model.params[0])
    coef = model.params[1]
    pval = model.pvalues[1]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x, y_pred, color='red', label='Fitted Line')
    plt.title('Cobb-Douglas Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    st.pyplot(plt)
    
    equation = f'ln(Y) = {model.params[0]:.4f} + {model.params[1]:.4f} * ln(X)'
    
    return mse, r2, intercept, coef, pval, equation, model, x_log, y_log

def fit_and_plot_exponential(x, y):
    # Exponential Model
    def exponential_model(params, x):
        return params[0] * np.exp(params[1] * x)
    
    def residuals(params, x, y):
        return y - exponential_model(params, x)
    
    def exp_fit(params):
        return residuals(params, x, y)
    
    initial_params = [y[0], 0.1]
    result = least_squares(exp_fit, initial_params)
    a, b = result.x
    y_pred = exponential_model(result.x, x)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x, y_pred, color='red', label='Fitted Line')
    plt.title('Exponential Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    st.pyplot(plt)
    
    equation = f'Y = {a:.4f} * e^({b:.4f} * X)'
    
    return mse, r2, a, b, equation

def fit_and_plot_modified_exponential(x, y):
    # Modified Exponential Model
    def modified_exponential_model(params, x):
        return params[0] * np.exp(params[1] * x) + params[2]
    
    def residuals(params, x, y):
        return y - modified_exponential_model(params, x)
    
    def mod_exp_fit(params):
        return residuals(params, x, y)
    
    initial_params = [y[0], 0.1, 0]
    result = least_squares(mod_exp_fit, initial_params)
    a, b, c = result.x
    y_pred = modified_exponential_model(result.x, x)
    
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Actual Data')
    plt.plot(x, y_pred, color='red', label='Fitted Line')
    plt.title('Modified Exponential Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    st.pyplot(plt)
    
    equation = f'Y = {a:.4f} * e^({b:.4f} * X) + {c:.4f}'
    
    return mse, r2, a, b, c, equation

def forecast_best_model(best_model, x, y, model_name, additional_params):
    future_x = np.array(range(len(x) + 1, len(x) + 4)).reshape(-1, 1)
    
    if model_name == 'Linear':
        model, X_poly = additional_params
        future_x_poly = np.vander(future_x.flatten(), N=X_poly.shape[1], increasing=True)
        future_y_pred = model.predict(sm.add_constant(future_x_poly))
    elif model_name == 'Quadratic':
        model, X_poly = additional_params
        future_x_poly = np.vander(future_x.flatten(), N=X_poly.shape[1], increasing=True)
        future_y_pred = model.predict(sm.add_constant(future_x_poly))
    elif model_name == 'Quartic':
        model, X_poly = additional_params
        future_x_poly = np.vander(future_x.flatten(), N=X_poly.shape[1], increasing=True)
        future_y_pred = model.predict(sm.add_constant(future_x_poly))
    elif model_name == 'Cobb-Douglas':
        model, x_log, y_log = additional_params
        future_x_log = np.log(future_x)
        future_y_pred = np.exp(model.predict(sm.add_constant(future_x_log)))
    elif model_name == 'Exponential':
        a, b = additional_params
        future_y_pred = a * np.exp(b * future_x.flatten())
    elif model_name == 'Modified Exponential':
        a, b, c = additional_params
        future_y_pred = a * np.exp(b * future_x.flatten()) + c
    
    return future_x, future_y_pred

# Streamlit app
st.title('Time Series Regression Analysis')

uploaded_file = st.file_uploader("Upload your CSV/XLSX file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        st.write("Data Preview:")
        st.write(df.head())

        # Display Model Equations Table
        model_info = {
            'Model': ['Linear', 'Quadratic', 'Quartic', 'Cobb-Douglas', 'Exponential', 'Modified Exponential'],
            'Equation': [
                'Y = a + bX',
                'Y = a + bX + cX^2',
                'Y = a + bX + cX^2 + dX^3 + eX^4',
                'ln(Y) = a + b * ln(X)',
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
        st.write(f"**Coefficients:** Intercept = {intercept1:.4f}, b = {coef1[0]:.4f}")
        st.write(f"**P-Values:** b = {pval1[0]:.4f}")
        st.write(f"MSE: {mse1:.2f}, R²: {r2_1:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {coef1[0]:.4f} indicates the change in Y for a unit change in X.")

        # Quadratic Regression
        st.subheader('Quadratic Regression')
        mse2, r2_2, intercept2, coef2, pval2, equation2, model2, x_poly2 = fit_and_plot_regression(x, y, 2)
        st.write(f"**Model:** {equation2}")
        st.write(f"**Coefficients:** Intercept = {intercept2:.4f}, b1 = {coef2[0]:.4f}, b2 = {coef2[1]:.4f}")
        st.write(f"**P-Values:** b1 = {pval2[0]:.4f}, b2 = {pval2[1]:.4f}")
        st.write(f"MSE: {mse2:.2f}, R²: {r2_2:.2f}")
        st.write(f"**Interpretation:** The coefficient b2 = {coef2[1]:.4f} represents the curvature of the parabola.")

        # Quartic Regression
        st.subheader('Quartic Regression')
        mse3, r2_3, intercept3, coef3, pval3, equation3, model3, x_poly3 = fit_and_plot_regression(x, y, 4)
        st.write(f"**Model:** {equation3}")
        st.write(f"**Coefficients:** Intercept = {intercept3:.4f}, b1 = {coef3[0]:.4f}, b2 = {coef3[1]:.4f}, b3 = {coef3[2]:.4f}, b4 = {coef3[3]:.4f}")
        st.write(f"**P-Values:** b1 = {pval3[0]:.4f}, b2 = {pval3[1]:.4f}, b3 = {pval3[2]:.4f}, b4 = {pval3[3]:.4f}")
        st.write(f"MSE: {mse3:.2f}, R²: {r2_3:.2f}")
        st.write(f"**Interpretation:** The coefficient b4 = {coef3[3]:.4f} influences the shape of the quartic curve.")

        # Cobb-Douglas Regression
        st.subheader('Cobb-Douglas Regression')
        mse4, r2_4, intercept4, coef4, pval4, equation4, model4, x_log4, y_log4 = fit_and_plot_cobb_douglas(x, y)
        st.write(f"**Model:** {equation4}")
        st.write(f"**Coefficients:** Intercept = {intercept4:.4f}, b = {coef4:.4f}")
        st.write(f"**P-Values:** b = {pval4:.4f}")
        st.write(f"MSE: {mse4:.2f}, R²: {r2_4:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {coef4:.4f} represents the elasticity of Y with respect to X.")

        # Exponential Regression
        st.subheader('Exponential Regression')
        mse5, r2_5, a5, b5, equation5 = fit_and_plot_exponential(x, y)
        st.write(f"**Model:** {equation5}")
        st.write(f"**Coefficients:** a = {a5:.4f}, b = {b5:.4f}")
        st.write(f"MSE: {mse5:.2f}, R²: {r2_5:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {b5:.4f} represents the growth rate in the exponential model.")

        # Modified Exponential Regression
        st.subheader('Modified Exponential Regression')
        mse6, r2_6, a6, b6, c6, equation6 = fit_and_plot_modified_exponential(x, y)
        st.write(f"**Model:** {equation6}")
        st.write(f"**Coefficients:** a = {a6:.4f}, b = {b6:.4f}, c = {c6:.4f}")
        st.write(f"MSE: {mse6:.2f}, R²: {r2_6:.2f}")
        st.write(f"**Interpretation:** The coefficient b = {b6:.4f} influences the growth rate, and c = {c6:.4f} is the offset term.")

        # Final Results Table
        results_df = pd.DataFrame({
            'Model': ['Linear', 'Quadratic', 'Quartic', 'Cobb-Douglas', 'Exponential', 'Modified Exponential'],
            'Equation': [
                equation1,
                equation2,
                equation3,
                equation4,
                equation5,
                equation6
            ],
            'Intercept': [intercept1, intercept2, intercept3, intercept4, None, None],
            'Coefficients': [
                f'{coef1[0]:.4f}',
                f'{coef2[0]:.4f}, {coef2[1]:.4f}',
                f'{coef3[0]:.4f}, {coef3[1]:.4f}, {coef3[2]:.4f}, {coef3[3]:.4f}',
                f'{coef4:.4f}',
                f'{a5:.4f}',
                f'{a6:.4f}, {b6:.4f}'
            ],
            'P-Values': [
                f'{pval1[0]:.4f}',
                f'{pval2[0]:.4f}, {pval2[1]:.4f}',
                f'{pval3[0]:.4f}, {pval3[1]:.4f}, {pval3[2]:.4f}, {pval3[3]:.4f}',
                f'{pval4:.4f}',
                None,
                None
            ],
            'MSE': [mse1, mse2, mse3, mse4, mse5, mse6],
            'R²': [r2_1, r2_2, r2_3, r2_4, r2_5, r2_6]
        })

        st.write("Final Results:")
        st.write(results_df)

        # Highlight best model
        best_model_idx = results_df['R²'].idxmax()
        best_model_name = results_df.loc[best_model_idx, 'Model']
        st.write(f"**Best Model based on R²:** {best_model_name}")

        # Forecast future values with the best model
        if best_model_name == 'Linear':
            future_x, future_y_pred = forecast_best_model('Linear', x, y, best_model_name, (model1, x_poly1))
        elif best_model_name == 'Quadratic':
            future_x, future_y_pred = forecast_best_model('Quadratic', x, y, best_model_name, (model2, x_poly2))
        elif best_model_name == 'Quartic':
            future_x, future_y_pred = forecast_best_model('Quartic', x, y, best_model_name, (model3, x_poly3))
        elif best_model_name == 'Cobb-Douglas':
            future_x, future_y_pred = forecast_best_model('Cobb-Douglas', x, y, best_model_name, (model4, x_log4, y_log4))
        elif best_model_name == 'Exponential':
            future_x, future_y_pred = forecast_best_model('Exponential', x, y, best_model_name, (a5, b5))
        elif best_model_name == 'Modified Exponential':
            future_x, future_y_pred = forecast_best_model('Modified Exponential', x, y, best_model_name, (a6, b6, c6))
        
        st.write("Forecast for the next 3 periods:")
        forecast_df = pd.DataFrame({'Period': future_x.flatten(), 'Forecast': future_y_pred})
        st.write(forecast_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
