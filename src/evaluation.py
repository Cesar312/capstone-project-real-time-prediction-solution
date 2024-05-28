import os
import sys
import numpy as np
import json

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from src.paths import METRICS_DIR

def smape(y_true, y_pred):
    
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def evaluate_metrics(y_true, y_pred):

    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Squared Error
    mse = mean_squared_error(y_true, y_pred)

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # R-squared Score   
    r2 = r2_score(y_true, y_pred)

    # Pearson Correlation Coefficient
    r, _ = pearsonr(y_true, y_pred)

    # Symmetric Mean Absolute Percentage Error
    smape_score = smape(y_true, y_pred)


    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'SMAPE': smape_score,
        'R': r,
        'R-squared': r2
    }
    
    return metrics

def convert_to_serializable(metrics):

    for key, value in metrics.items():
        if isinstance(value, np.generic):
            metrics[key] = value.item()
        elif isinstance(value, np.ndarray):
            metrics[key] = value.tolist()

    return metrics

def save_metrics(metrics, file_name= METRICS_DIR / f'metrics.json'):

    metrics = convert_to_serializable(metrics)

    try: 
        if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
                with open(file_name, 'r') as file:
                     data = json.load(file)
        else:
            data = []
    except (FileNotFoundError, json.decoder.JSONDecodeError):
         data = []
         
    data.append(metrics)

    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)