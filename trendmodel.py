import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import statsmodels.api as sm

# Function Definitions

def fit_and_plot_regression(x, y, degree):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x.reshape(-1, 1))
    model = LinearRegression().fit(x_poly, y)

    y_pred = model.predict(x_poly)
    mse = mean_squared_error(y, y_pred)
    r2 = model.score(x_poly, y)

    x_range = np.linspace(x.min(), x.max(), 100)
    x_range_poly = poly.transform(x_range.reshape(-1, 1))
    y_range_pred = model.predict(x_range_poly)

    # Plotting
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x_range, y_range_pred, color='red', label=f'Fit Degree {degree}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Polynomial Degree {degree} Regression')
    plt.legend()
    st.pyplot(plt)

    intercept = model.intercept_
    coefficients = model.coef_
    equation = f"y = {intercept:.4f} + " + " + ".join([f"{coeff:.4f}*x^{i}" for i, coeff in enumerate(coefficients[1:], start=1)])
    
    # Fitting the model with statsmodels for p-values
    x_poly_const = sm.add_constant(x_poly)
    sm_model = sm.OLS(y, x_poly_const).fit()
    p_values = sm_model.pvalues

    return mse, r2, intercept, coefficients, p_values, equation, model, x_poly

def fit_and_plot_exponential(x, y):
    def exponential_func(x, a, b):
        return a * np.exp(b * x)

    params, _ = curve_fit(exponential_func, x, y, p0=(1, 0.01))
    y_pred = exponential_func(x, *params)
    mse = mean_squared_error(y, y_pred)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range_pred = exponential_func(x_range, *params)

    # Plotting
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x_range, y_range_pred, color='red', label='Exponential Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Exponential Regression')
    plt.legend()
    st.pyplot(plt)

    a, b = params
    equation = f"y = {a:.4f} * exp({b:.4f} * x)"

    return mse, r2, a, b, equation

def fit_and_plot_modified_exponential(x, y):
    def modified_exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c

    params, _ = curve_fit(modified_exponential_func, x, y, p0=(1, 0.01, 1))
    y_pred = modified_exponential_func(x, *params)
    mse = mean_squared_error(y, y_pred)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    x_range = np.linspace(x.min(), x.max(), 100)
    y_range_pred = modified_exponential_func(x_range, *params)

    # Plotting
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x_range, y_range_pred, color='red', label='Modified Exponential Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Modified Exponential Regression')
    plt.legend()
    st.pyplot(plt)

    a, b, c = params
    equation = f"y = {a:.4f} * exp({b:.4f} * x) + {c:.4f}"

    return mse, r2, a, b, c, equation

def fit_and_plot_cobb_douglas(x, y):
    x_log = np.log(x)
    y_log = np.log(y)

    model = LinearRegression().fit(x_log.reshape(-1, 1), y_log)

    y_log_pred = model.predict(x_log.reshape(-1, 1))
    mse = mean_squared_error(y_log, y_log_pred)
    r2 = model.score(x_log.reshape(-1, 1), y_log)

    x_range = np.linspace(x_log.min(), x_log.max(), 100)
    y_log_range_pred = model.predict(x_range.reshape(-1, 1))
    y_range_pred = np.exp(y_log_range_pred)

    # Plotting
    plt.figure()
    plt.scatter(x_log, y_log, color='blue', label='Log Data')
    plt.plot(x_range, y_log_range_pred, color='red', label='Cobb-Douglas Fit (Log-Log)')
    plt.xlabel('log(X)')
    plt.ylabel('log(Y)')
    plt.title('Cobb-Douglas Regression')
    plt.legend()
    st.pyplot(plt)

    intercept = model.intercept_
    coefficient = model.coef_[0]
    equation = f"log(y) = {intercept:.4f} + {coefficient:.4f}*log(x)"

    # Interpretation as Cobb-Douglas
    a = np.exp(intercept)
    b = coefficient

    return mse, r2, intercept, [b], model.pvalues if hasattr(model, 'pvalues') else None, equation, model, x_log, y_log

def fit_and_plot_quadratic_b(x, y):
    x_squared = x**2
    X_design = np.vstack((x, x_squared)).T
    X_design_const = sm.add_constant(X_design)

    model = sm.OLS(y, X_design_const).fit()
    y_pred = model.predict(X_design_const)

    mse = mean_squared_error(y, y_pred)
    r2 = model.rsquared
    intercept = model.params[0]
    coefficients = model.params[1:]
    p_values = model.pvalues

    x_range = np.linspace(x.min(), x.max(), 100)
    x_range_squared = x_range**2
    x_range_design = np.vstack((x_range, x_range_squared)).T
    x_range_design_const = sm.add_constant(x_range_design)
    y_range_pred = model.predict(x_range_design_const)

    # Plotting
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data')
    plt.plot(x_range, y_range_pred, color='red', label='QuadraticB Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('QuadraticB Regression')
    plt.legend()
    st.pyplot(plt)

    equation = f"y = {intercept:.4f} + {coefficients[0]:.4f}*x - {coefficients[1]:.4f}*x^2"

    return mse, r2, intercept, coefficients, p_values, equation, model

def forecast_best_model(model_name, x, y, model_type, additional_params=None):
    if model_type in ['Linear', 'Quadratic', 'Quartic']:
        model, x_poly = additional_params
        last_x = x_poly[-1, :]
        future_x_poly = np.vstack([last_x + i for i in range(1, 4)])
        future_y_pred = model.predict(future_x_poly)
        
    elif model_type == 'Cobb-Douglas':
        model, x_log, y_log = additional_params
        future_x_log = np.log(x[-1] + np.arange(1, 4))
        future_y_log_pred = model.predict(future_x_log.reshape(-1, 1))
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

    elif model_type == 'QuadraticB':
        model = additional_params
        last_x = x[-1]
        future_x = np.array([last_x + i for i in range(1, 4)])
        future_x_squared = future_x**2
        future_x_reshaped = np.vstack((future_x, future_x_squared)).T
        future_x_reshaped_const = sm.add_constant(future_x_reshaped)
        future_y_pred = model.predict(future_x_reshaped_const)

    return future_x, future_y_pred

# Streamlit App

def main():
    st.title("Regression Analysis")

    uploaded_file = st.file_uploader("Upload CSV, Excel (.xlsx, .xls) file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Read data
            if uploaded_file.name.endswith(('xlsx', 'xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            st.write("Data preview:")
            st.write(df.head())
            
            # Assuming the first two columns are X and Y
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values
            
            # Apply different models
            st.subheader('Linear Regression')
            mse_linear, r2_linear, intercept_linear, coeff_linear, p_values_linear, eq_linear, linear_model, linear_x_poly = fit_and_plot_regression(x, y, 1)
            st.write(f"R2: {r2_linear:.4f}, Intercept: {intercept_linear:.4f}, Coefficients: {coeff_linear[1]:.4f}")
            st.write(f"Equation: {eq_linear}")
            
            st.subheader('Quadratic Regression')
            mse_quadratic, r2_quadratic, intercept_quadratic, coeff_quadratic, p_values_quadratic, eq_quadratic, quadratic_model, quadratic_x_poly = fit_and_plot_regression(x, y, 2)
            st.write(f"R2: {r2_quadratic:.4f}, Intercept: {intercept_quadratic:.4f}, Coefficients: {coeff_quadratic[1]:.4f}, {coeff_quadratic[2]:.4f}")
            st.write(f"Equation: {eq_quadratic}")
            
            st.subheader('Quartic Regression')
            mse_quartic, r2_quartic, intercept_quartic, coeff_quartic, p_values_quartic, eq_quartic, quartic_model, quartic_x_poly = fit_and_plot_regression(x, y, 4)
            st.write(f"R2: {r2_quartic:.4f}, Intercept: {intercept_quartic:.4f}, Coefficients: {', '.join([f'{c:.4f}' for c in coeff_quartic[1:]])}")
            st.write(f"Equation: {eq_quartic}")

            st.subheader('Cobb-Douglas Regression')
            mse_cobb_douglas, r2_cobb_douglas, intercept_cobb_douglas, coeff_cobb_douglas, p_values_cobb_douglas, eq_cobb_douglas, cobb_douglas_model, cobb_douglas_x_log, cobb_douglas_y_log = fit_and_plot_cobb_douglas(x, y)
            st.write(f"R2: {r2_cobb_douglas:.4f}, Intercept: {intercept_cobb_douglas:.4f}, Coefficient: {coeff_cobb_douglas[0]:.4f}")
            st.write(f"Equation: {eq_cobb_douglas}")

            st.subheader('Exponential Regression')
            mse_exponential, r2_exponential, a_exponential, b_exponential, eq_exponential = fit_and_plot_exponential(x, y)
            st.write(f"R2: {r2_exponential:.4f}, Coefficients: a: {a_exponential:.4f}, b: {b_exponential:.4f}")
            st.write(f"Equation: {eq_exponential}")

            st.subheader('Modified Exponential Regression')
            mse_mod_exp, r2_mod_exp, a_mod_exp, b_mod_exp, c_mod_exp, eq_mod_exp = fit_and_plot_modified_exponential(x, y)
            st.write(f"R2: {r2_mod_exp:.4f}, Coefficients: a: {a_mod_exp:.4f}, b: {b_mod_exp:.4f}, c: {c_mod_exp:.4f}")
            st.write(f"Equation: {eq_mod_exp}")

            st.subheader('QuadraticB Regression')
            mse_quadratic_b, r2_quadratic_b, intercept_quadratic_b, coeff_quadratic_b, p_values_quadratic_b, eq_quadratic_b, quadratic_b_model = fit_and_plot_quadratic_b(x, y)
            st.write(f"R2: {r2_quadratic_b:.4f}, Intercept: {intercept_quadratic_b:.4f}, Coefficients: b: {coeff_quadratic_b[0]:.4f}, c: {coeff_quadratic_b[1]:.4f}")
            st.write(f"Equation: {eq_quadratic_b}")

            # Determine Best Model
            model_info = {
                'Model': ['Linear', 'Quadratic', 'Quartic', 'Cobb-Douglas', 'Exponential', 'Modified Exponential', 'QuadraticB'],
                'MSE': [mse_linear, mse_quadratic, mse_quartic, mse_cobb_douglas, mse_exponential, mse_mod_exp, mse_quadratic_b],
                'R2': [r2_linear, r2_quadratic, r2_quartic, r2_cobb_douglas, r2_exponential, r2_mod_exp, r2_quadratic_b],
                'Equation': [eq_linear, eq_quadratic, eq_quartic, eq_cobb_douglas, eq_exponential, eq_mod_exp, eq_quadratic_b],
                'Interpretation': [f'Intercept: {intercept_linear:.4f}, Coeff: {coeff_linear[1]:.4f}', 
                                    f'Intercept: {intercept_quadratic:.4f}, Coeff: {coeff_quadratic[1]:.4f}, {coeff_quadratic[2]:.4f}',
                                    f'Intercept: {intercept_quadratic:.4f}, Coeff: {coeff_quadratic[1]:.4f}, {coeff_quadratic[2]:.4f}, {coeff_quadratic[3]:.4f}',
                                    f'Intercept: {intercept_cobb_douglas:.4f}, Coefficient: {coeff_cobb_douglas[0]:.4f}', 
                                    f'a: {a_exponential:.4f}, b: {b_exponential:.4f}',
                                    f'a: {a_mod_exp:.4f}, b: {b_mod_exp:.4f}, c: {c_mod_exp:.4f}',
                                    f'Intercept: {intercept_quadratic_b:.4f}, b: {coeff_quadratic_b[0]:.4f}, c: {coeff_quadratic_b[1]:.4f}']
            }
            model_info_df = pd.DataFrame(model_info)
            best_model_idx = model_info_df['R2'].idxmax()
            best_model_name = model_info_df.loc[best_model_idx, 'Model']
            best_model_mse = model_info_df.loc[best_model_idx, 'MSE']
            best_model_r2 = model_info_df.loc[best_model_idx, 'R2']
            best_model_equation = model_info_df.loc[best_model_idx, 'Equation']
            best_model_interpretation = model_info_df.loc[best_model_idx, 'Interpretation']

            st.markdown(f"<div style='color: orange; font-size:30px; font-weight:bold;'>"
                        f"Best Model: {best_model_name} "
                        f"(MSE: {best_model_mse:.4f}, R^2: {best_model_r2:.4f})"
                        f"</div>", unsafe_allow_html=True)
            st.write(f"Equation: {best_model_equation}")
            st.write(f"Interpretation: {best_model_interpretation}")

            # Forecast using the best model
            st.subheader('Forecasting with the Best Model')
            if best_model_name == 'Linear':
                future_x, future_y_pred = forecast_best_model(best_model_name, x, y, 'Linear', additional_params=(linear_model, linear_x_poly))
            elif best_model_name == 'Quadratic':
                future_x, future_y_pred = forecast_best_model(best_model_name, x, y, 'Quadratic', additional_params=(quadratic_model, quadratic_x_poly))
            elif best_model_name == 'Quartic':
                future_x, future_y_pred = forecast_best_model(best_model_name, x, y, 'Quartic', additional_params=(quartic_model, quartic_x_poly))
            elif best_model_name == 'Cobb-Douglas':
                future_x, future_y_pred = forecast_best_model(best_model_name, x, y, 'Cobb-Douglas', additional_params=(cobb_douglas_model, cobb_douglas_x_log, cobb_douglas_y_log))
            elif best_model_name == 'Exponential':
                future_x, future_y_pred = forecast_best_model(best_model_name, x, y, 'Exponential', additional_params=(a_exponential, b_exponential))
            elif best_model_name == 'Modified Exponential':
                future_x, future_y_pred = forecast_best_model(best_model_name, x, y, 'Modified Exponential', additional_params=(a_mod_exp, b_mod_exp, c_mod_exp))
            elif best_model_name == 'QuadraticB':
                future_x, future_y_pred = forecast_best_model(best_model_name, x, y, 'QuadraticB', additional_params=quadratic_b_model)

            st.write(f"Next 3 values for {best_model_name} model:")
            st.write(pd.DataFrame({'X': future_x, 'Predicted Y': future_y_pred}))

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
