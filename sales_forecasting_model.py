import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
import warnings
from statsmodels.tsa.stattools import adfuller
import numpy as np
import itertools
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Ignore harmless warnings
warnings.filterwarnings('ignore')
register_matplotlib_converters()

# Load the dataset
df = pd.read_excel('global_superstore_2016.xlsx')

# Preview the data
print(df.head())

# --- Data Preprocessing ---
# Assume 'Order Date' and 'Sales' columns exist
# Convert 'Order Date' to datetime
if 'Order Date' in df.columns:
    df['Order Date'] = pd.to_datetime(df['Order Date'])
else:
    raise ValueError("'Order Date' column not found in dataset.")

# Set 'Order Date' as index
df.set_index('Order Date', inplace=True)

# Resample sales by month (sum)
monthly_sales = df['Sales'].resample('M').sum()

# --- Visualization ---
plt.figure(figsize=(12,6))
plt.plot(monthly_sales, label='Monthly Sales')
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# --- Model Training (ARIMA) ---
# Simple ARIMA model as baseline
# Split data into train and test
train = monthly_sales[:-12]
test = monthly_sales[-12:]

# Fit ARIMA model (order can be tuned)
model = ARIMA(train, order=(1,1,1))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=12)

# --- Visualization of Forecast ---
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, forecast, label='Forecast', linestyle='--')
plt.title('Sales Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# --- Evaluation ---
mse = mean_squared_error(test, forecast)
print(f'Mean Squared Error: {mse:.2f}')

# --- Data Cleaning & Exploration ---
# Check for missing values
missing_summary = df.isnull().sum()
print('Missing values per column:\n', missing_summary)

# Fill missing sales with 0 (or could use interpolation)
df['Sales'] = df['Sales'].fillna(0)

# Outlier detection (simple: sales > 99th percentile)
sales_99 = df['Sales'].quantile(0.99)
outliers = df[df['Sales'] > sales_99]
print(f'Number of outliers (Sales > 99th percentile): {len(outliers)}')

# EDA: Histogram of Sales
plt.figure(figsize=(8,4))
plt.hist(df['Sales'], bins=50, color='skyblue', edgecolor='black')
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# EDA: Boxplot of Sales
plt.figure(figsize=(6,3))
plt.boxplot(df['Sales'], vert=False)
plt.title('Boxplot of Sales')
plt.xlabel('Sales')
plt.show()

# EDA: Sales by Category (if available)
if 'Category' in df.columns:
    plt.figure(figsize=(8,4))
    df.groupby('Category')['Sales'].sum().plot(kind='bar')
    plt.title('Total Sales by Category')
    plt.ylabel('Total Sales')
    plt.show()

# --- Stationarity Check ---
def adf_test(series, title=''):
    print(f'\nADF Test: {title}')
    adf_result = adfuller(series.dropna(), autolag='AIC')
    adf_stat, p_value, usedlag, nobs = adf_result[:4]
    labels = ['ADF Statistic', 'p-value', '# Lags Used', 'Number of Observations Used']
    out = pd.Series([adf_stat, p_value, usedlag, nobs], index=labels)
    if len(adf_result) > 4:
        crit_values = adf_result[4]
        for key, val in crit_values.items():
            out[f'Critical Value ({key})'] = val
    print(out)
    if p_value <= 0.05:
        print('=> Stationary (reject H0)')
    else:
        print('=> Not stationary (fail to reject H0)')

# Check original series
adf_test(monthly_sales, 'Original Monthly Sales')

# If not stationary, try log transform
log_sales = np.log1p(monthly_sales)
adf_test(log_sales, 'Log-Transformed Sales')

# If still not stationary, try differencing
log_diff_sales = log_sales.diff().dropna()
adf_test(log_diff_sales, 'Log-Differenced Sales')

# Plot transformed series
plt.figure(figsize=(12,4))
plt.plot(log_diff_sales, label='Log-Differenced Sales')
plt.title('Log-Differenced Monthly Sales')
plt.legend()
plt.show()

# --- Model Tuning: Grid Search for ARIMA Parameters ---
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
best_aic = float('inf')
best_order = None
for order in pdq:
    try:
        model = ARIMA(log_diff_sales, order=order)
        model_fit = model.fit()
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_order = order
    except Exception as e:
        continue
print(f'Best ARIMA order: {best_order} with AIC: {best_aic}')

# Use best_order for modeling
train = log_diff_sales[:-12]
test = log_diff_sales[-12:]
model = ARIMA(train, order=best_order)
model_fit = model.fit()
forecast = model_fit.forecast(steps=12)

# --- Model Diagnostics ---
residuals = model_fit.resid
plt.figure(figsize=(12,4))
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.show()

plt.figure(figsize=(12,4))
plot_acf(residuals, lags=24)
plt.title('ACF of Residuals')
plt.show()

plt.figure(figsize=(12,4))
plot_pacf(residuals, lags=24)
plt.title('PACF of Residuals')
plt.show()

print('If residuals are white noise (no autocorrelation), the model is well specified.')

# --- Evaluation Metrics ---
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --- Simple Rolling Forecast Backtesting ---
window = 12
rolling_forecasts = []
rolling_truth = []
for i in range(window, len(log_diff_sales)):
    train_rolling = log_diff_sales[:i]
    test_rolling = log_diff_sales[i:i+1]
    try:
        model_rolling = ARIMA(train_rolling, order=best_order)
        model_rolling_fit = model_rolling.fit()
        forecast_rolling = model_rolling_fit.forecast(steps=1)
        # Convert back to original scale
        forecast_cumsum = forecast_rolling.cumsum() + log_sales.iloc[i-1]
        forecast_original = np.expm1(forecast_cumsum)
        test_cumsum = test_rolling.cumsum() + log_sales.iloc[i-1]
        test_original = np.expm1(test_cumsum)
        rolling_forecasts.append(forecast_original.values[0])
        rolling_truth.append(test_original.values[0])
    except:
        continue
rolling_mae = mean_absolute_error(rolling_truth, rolling_forecasts)
rolling_rmse = np.sqrt(mean_squared_error(rolling_truth, rolling_forecasts))
rolling_mape = mean_absolute_percentage_error(rolling_truth, rolling_forecasts)
print(f'Rolling MAE: {rolling_mae:.2f}')
print(f'Rolling RMSE: {rolling_rmse:.2f}')
print(f'Rolling MAPE: {rolling_mape:.2f}%') 