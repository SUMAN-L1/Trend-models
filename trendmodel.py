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
    p_values = ols_model.pvalues[1:]  # Skip intercept p-value
    coefficients = model.coef_[1:]  # Skip intercept coefficient
    
    equation = f"Y = {model.intercept_:.2f} + " + " + ".join([f"{coef:.2f}*X^{i}" for i, coef in enumerate(coefficients, start=1)])
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label=f'Estimated (degree {degree})')
    plt.title(f'Degree {degree} Regression\nMSE: {mse:.2f}, R^2: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    return mse, r2, coefficients, p_values

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
    
    coefficients = model.params[1:]  # Skip intercept coefficient
    p_values = model.pvalues[1:]  # Skip intercept p-value
    equation = f"ln(Y) = {model.params[0]:.2f} + {model.params[1]:.2f}*ln(X)"
    
    plt.figure()
    plt.scatter(x, y, color='blue', label='Actual')
    plt.plot(x, y_pred, color='red', label='Estimated (Cobb-Douglas)')
    plt.title(f'Cobb-Douglas Regression\nMSE: {mse:.2f}, R^2: {r2:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    
    return mse, r2, coefficients, p_values

st.title('Time Series Trend Analysis')

uploaded_file = st.file_uploader("Upload your CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Data Preview:", df.head())

        time_column = st.selectbox("Select Time Column", df.columns)
        value_column = st.selectbox("Select Value Column", df.columns)

        x = df[[time_column]].values
        y = df[value_column].values
        
        st.subheader('Linear Regression (Degree 1)')
        st.write("**Model:** Y = a + bX")
        mse1, r2_1, coef1, pval1 = fit_and_plot_regression(x, y, degree=1)
        st.write(f"MSE: {mse1:.2f}, R²: {r2_1:.2f}")
        st.write(f"Coefficients: {coef1}")
        st.write(f"P-Values: {pval1}")
        
        st.subheader('Quadratic Regression (Degree 2)')
        st.write("**Model:** Y = a + bX + cX^2")
        mse2, r2_2, coef2, pval2 = fit_and_plot_regression(x, y, degree=2)
        st.write(f"MSE: {mse2:.2f}, R²: {r2_2:.2f}")
        st.write(f"Coefficients: {coef2}")
        st.write(f"P-Values: {pval2}")
        
        st.subheader('Quartic Regression (Degree 4)')
        st.write("**Model:** Y = a + bX + cX^2 + dX^3 + eX^4")
        mse4, r2_4, coef4, pval4 = fit_and_plot_regression(x, y, degree=4)
        st.write(f"MSE: {mse4:.2f}, R²: {r2_4:.2f}")
        st.write(f"Coefficients: {coef4}")
        st.write(f"P-Values: {pval4}")

        st.subheader('Cobb-Douglas Regression')
        st.write("**Model:** ln(Y) = a + b*ln(X)")
        mse_cd, r2_cd, coef_cd, pval_cd = fit_and_plot_cobb_douglas(x, y)
        st.write(f"MSE: {mse_cd:.2f}, R²: {r2_cd:.2f}")
        st.write(f"Coefficients: {coef_cd}")
        st.write(f"P-Values: {pval_cd}")
        
        st.subheader('Model Comparison')
        comparison_data = {
            'Model': ['Linear', 'Quadratic', 'Quartic', 'Cobb-Douglas'],
            'MSE': [mse1, mse2, mse4, mse_cd],
            'R²': [r2_1, r2_2, r2_4, r2_cd],
            'Coefficients': [coef1, coef2, coef4, coef_cd],
            'P-Values': [pval1, pval2, pval4, pval_cd]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        best_model_idx = comparison_df['R²'].idxmax()
        best_model = comparison_df.iloc[best_model_idx]
        
        st.write(f"""
        ### Interpretation of Results
        - **Linear Regression (Degree 1)**: This model fits a straight line to the data. It is useful for identifying overall linear trends but may not capture more complex patterns in the data.
        - **Quadratic Regression (Degree 2)**: This model fits a parabola to the data. It can capture simple curvilinear trends and is more flexible than linear regression but may still miss more complex patterns.
        - **Quartic Regression (Degree 4)**: This model fits a quartic polynomial (degree 4) to the data. It can capture more complex trends and fluctuations. However, it may also overfit the data, especially if the true underlying trend is simpler.
        - **Cobb-Douglas Regression**: This model fits a Cobb-Douglas function to the data, which is useful for modeling relationships where growth rates are proportional. It is often used in economics but can be applied to other fields.
        - **Model Selection**: Compare the MSE, R², coefficients, and p-values of the models. Lower MSE and higher R² indicate a better fit. However, be cautious of overfitting with higher-degree polynomials. Choose the model that balances fit and simplicity.
        """)
        
        st.markdown(f"**Best Model:** {best_model['Model']} Regression", unsafe_allow_html=True)
        st.markdown(f"<span style='color: darkorange; font-weight: bold;'>Best Model based on R²: {best_model['Model']} Regression</span>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
