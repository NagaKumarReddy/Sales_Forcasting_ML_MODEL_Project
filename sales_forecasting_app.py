import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import plotly.express as px
import plotly.graph_objects as go
import io

# --- Sidebar ---
st.sidebar.title('📊 Sales Forecasting App')
st.sidebar.info('Upload your sales data (Excel file) and get forecasts, diagnostics, and evaluation metrics.\n\n**Steps:**\n1. Upload your file\n2. Review EDA\n3. See model results\n4. Download or screenshot outputs')
st.sidebar.markdown('---')
st.sidebar.write('Developed with ❤️ using Streamlit')

# --- Main Banner ---
st.markdown('<h1 style="text-align:center; color:#4F8BF9;">Sales Forecasting with Time Series Analysis 🚀</h1>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:gray;">Upload your sales data and get instant insights!</div>', unsafe_allow_html=True)
st.markdown('---')

uploaded_file = st.file_uploader('⬆️ **Upload Excel file**', type=['xlsx'])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.markdown('### 🗂️ Raw Data Preview')
    st.dataframe(df.head())
    st.markdown('---')

    # Preprocessing
    if 'Order Date' not in df.columns or 'Sales' not in df.columns:
        st.error("❌ Dataset must contain 'Order Date' and 'Sales' columns.")
        st.stop()
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df.set_index('Order Date', inplace=True)
    df['Sales'] = df['Sales'].fillna(0)

    # --- EDA Section ---
    st.markdown('<div style="background-color:#eaf6fb; padding:16px; border-radius:8px;">', unsafe_allow_html=True)
    st.markdown('### 🔍 Exploratory Data Analysis')
    # Histogram
    fig1 = px.histogram(df, x='Sales', nbins=50, color_discrete_sequence=['#4F8BF9'])
    fig1.update_layout(title='Sales Distribution', xaxis_title='Sales', yaxis_title='Frequency')
    st.plotly_chart(fig1, use_container_width=True)
    # Boxplot
    fig2 = px.box(df, x='Sales', color_discrete_sequence=['#b3d8fd'])
    fig2.update_layout(title='Boxplot of Sales', xaxis_title='Sales')
    st.plotly_chart(fig2, use_container_width=True)
    # Category bar chart
    if 'Category' in df.columns:
        cat_sales = df.groupby('Category')['Sales'].sum().reset_index()
        fig3 = px.bar(cat_sales, x='Category', y='Sales', color='Category', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig3.update_layout(title='Total Sales by Category', yaxis_title='Total Sales')
        st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('---')

    # --- Outlier Removal (preprocessing) ---
    sales_99 = df['Sales'].quantile(0.99)
    df = df[df['Sales'] <= sales_99]

    # Monthly sales trend
    st.markdown('### 📈 Monthly Sales Trend')
    monthly_sales = df['Sales'].resample('M').sum().reset_index()
    fig4 = px.line(monthly_sales, x='Order Date', y='Sales', markers=True, line_shape='spline', color_discrete_sequence=['#4F8BF9'])
    fig4.update_layout(title='Monthly Sales Over Time', xaxis_title='Date', yaxis_title='Sales')
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('---')

    # Stationarity check
    st.markdown('### 🧪 Stationarity Check (ADF Test)')
    def adf_test(series):
        adf_result = adfuller(series.dropna(), autolag='AIC')
        adf_stat, p_value, usedlag, nobs = adf_result[:4]
        result = {'ADF Statistic': adf_stat, 'p-value': p_value, '# Lags Used': usedlag, 'N Obs': nobs}
        if len(adf_result) > 4:
            crit_values = adf_result[4]
            for key, val in crit_values.items():
                result[f'Critical Value ({key})'] = val
        return result
    st.info('ADF Test on Monthly Sales:')
    st.write(adf_test(monthly_sales['Sales']))
    log_sales = np.log1p(monthly_sales['Sales'])
    st.info('ADF Test on Log-Transformed Sales:')
    st.write(adf_test(log_sales))
    log_diff_sales = log_sales.diff().dropna()
    st.info('ADF Test on Log-Differenced Sales:')
    st.write(adf_test(log_diff_sales))
    # Log-differenced sales plot
    fig5 = px.line(x=log_diff_sales.index, y=log_diff_sales.values, markers=True, color_discrete_sequence=['#4F8BF9'])
    fig5.update_layout(title='Log-Differenced Monthly Sales', xaxis_title='Date', yaxis_title='Log-Diff Sales')
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown('---')

    # ARIMA and SARIMA grid search
    st.markdown('### 🤖 Model Selection: ARIMA & SARIMA')
    import warnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    warnings.filterwarnings('ignore')
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    best_aic = float('inf')
    best_order = None
    best_seasonal_order = None
    best_model_type = 'ARIMA'
    # ARIMA grid search
    for order in pdq:
        try:
            model = ARIMA(log_diff_sales, order=order)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
                best_seasonal_order = None
                best_model_type = 'ARIMA'
        except:
            continue
    # SARIMA grid search
    for order in pdq:
        for seasonal_order in seasonal_pdq:
            try:
                model = SARIMAX(log_diff_sales, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
                    best_model_type = 'SARIMA'
            except:
                continue
    if best_model_type == 'ARIMA':
        st.success(f'Best Model: ARIMA{best_order} (AIC: {best_aic:.2f})')
    else:
        st.success(f'Best Model: SARIMA{best_order}x{best_seasonal_order} (AIC: {best_aic:.2f})')
    st.markdown('---')

    # Train/test split
    st.markdown('### 🏁 Model Training & Forecasting')
    train = log_diff_sales[:-12]
    test = log_diff_sales[-12:]
    if best_model_type == 'ARIMA':
        model = ARIMA(train, order=best_order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=12)
    else:
        model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=12)
    forecast_cumsum = forecast.cumsum() + log_sales.iloc[-13]
    forecast_original = np.expm1(forecast_cumsum)
    test_cumsum = test.cumsum() + log_sales.iloc[-13]
    test_original = np.expm1(test_cumsum)

    # Forecast plot
    st.markdown('#### 📊 Forecast vs Actual')
    forecast_df = pd.DataFrame({
        'Date': list(train.index) + list(test.index),
        'Type': ['Train']*len(train) + ['Test']*len(test),
        'Actual': list(np.expm1(log_sales.loc[train.index])) + list(test_original),
        'Forecast': [None]*len(train) + list(forecast_original)
    })
    fig6 = px.line(forecast_df, x='Date', y='Actual', color='Type', color_discrete_map={'Train':'#b3d8fd','Test':'#4F8BF9'})
    fig6.add_scatter(x=test.index, y=forecast_original, mode='lines+markers', name='Forecast', line=dict(dash='dash', color='#f9a04f'))
    fig6.update_layout(title='Sales Forecast vs Actual', xaxis_title='Date', yaxis_title='Sales')
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown('---')

    # --- Metrics Section ---
    st.markdown('<div style="background-color:#f9f9f9; padding:16px; border-radius:8px;">', unsafe_allow_html=True)
    st.markdown('### 🏆 Evaluation Metrics')
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    mae = mean_absolute_error(test_original, forecast_original)
    rmse = np.sqrt(mean_squared_error(test_original, forecast_original))
    mape = mean_absolute_percentage_error(test_original, forecast_original)
    accuracy = 100 - mape

    # Classification-based precision/recall (sales increase/decrease)
    from sklearn.metrics import precision_score, recall_score
    test_direction = np.sign(np.diff(test_original))
    forecast_direction = np.sign(np.diff(forecast_original))
    test_class = (test_direction > 0).astype(int)
    forecast_class = (forecast_direction > 0).astype(int)
    if len(test_class) > 0 and len(forecast_class) > 0:
        precision = precision_score(test_class, forecast_class, zero_division='warn')
        recall = recall_score(test_class, forecast_class, zero_division='warn')
    else:
        precision = recall = 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('MAE', f'{mae:.2f}')
    col2.metric('RMSE', f'{rmse:.2f}')
    col3.metric('MAPE', f'{mape:.2f}%')
    col4.metric('Accuracy', f'{accuracy:.2f}%')
    col5.metric('Precision', f'{precision*100:.2f}%')
    st.metric('Recall', f'{recall*100:.2f}%')

    if accuracy >= 80 and precision*100 >= 80 and recall*100 >= 80:
        st.success('🎉 All key metrics are above 80%! Model performance is strong.')
    else:
        st.warning('⚠️ One or more metrics are below 80%. Consider tuning your model or reviewing your data.')
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('---')

    # Residual diagnostics
    st.markdown('### 🧾 Model Diagnostics')
    residuals = model_fit.resid
    # Residuals plot
    fig7 = px.line(x=residuals.index, y=residuals.values, color_discrete_sequence=['#4F8BF9'])
    fig7.update_layout(title='Residuals of ARIMA Model', xaxis_title='Date', yaxis_title='Residuals')
    st.plotly_chart(fig7, use_container_width=True)
    # ACF plot
    max_lags = min(24, len(residuals) // 2 - 1)
    # ACF plot
    buf = io.BytesIO()
    fig_acf, ax_acf = plt.subplots(figsize=(10,4))
    plot_acf(residuals, lags=max_lags, ax=ax_acf)
    ax_acf.set_title('ACF of Residuals')
    plt.tight_layout()
    fig_acf.savefig(buf, format='png')
    st.image(buf.getvalue(), caption='ACF of Residuals', use_column_width=True)
    buf.close()
    # PACF plot
    buf = io.BytesIO()
    fig_pacf, ax_pacf = plt.subplots(figsize=(10,4))
    plot_pacf(residuals, lags=max_lags, ax=ax_pacf)
    ax_pacf.set_title('PACF of Residuals')
    plt.tight_layout()
    fig_pacf.savefig(buf, format='png')
    st.image(buf.getvalue(), caption='PACF of Residuals', use_column_width=True)
    buf.close()
    st.info('If residuals are white noise (no autocorrelation), the model is well specified.')
    st.success('Analysis complete! Scroll up to review all results.')
else:
    st.info('Please upload an Excel file to begin.') 