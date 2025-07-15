import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Example: y_true and y_pred are actual and predicted sales values
# Replace with your actual arrays
# y_true = ...
# y_pred = ...

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Accuracy for regression (using MAPE as a proxy)
def regression_accuracy(y_true, y_pred):
    return 100 - mean_absolute_percentage_error(y_true, y_pred)

# Precision and recall are not standard for regression/time series forecasting.
# They are used for classification tasks. For regression, use MAE, RMSE, MAPE, R^2, etc.

# Example usage:
# mae = mean_absolute_error(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# mape = mean_absolute_percentage_error(y_true, y_pred)
# acc = regression_accuracy(y_true, y_pred)
# print(f'MAE: {mae:.2f}')
# print(f'RMSE: {rmse:.2f}')
# print(f'MAPE: {mape:.2f}%')
# print(f'Accuracy (100-MAPE): {acc:.2f}%')

# If you want to use precision/recall, you must first convert the regression output to a classification problem (e.g., did sales increase/decrease), which is not typical for forecasting. 