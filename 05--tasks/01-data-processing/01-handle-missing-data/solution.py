import numpy as np
from scipy import stats

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    for col_i in range(data.shape[1]):
        column = data[:, col_i]
        non_nan = column[~np.isnan(column)]
        if len(non_nan) == 0:
            continue
        if strategy == "mean":
            fill_val = np.mean(non_nan)
        elif strategy == "median":
            fill_val = np.median(non_nan)
        elif strategy == "mode":
            fill_val = stats.mode(non_nan, keepdims=False).mode
        else:
            raise ValueError("Strategy must be 'mean', 'median', or 'mode'")
        data[:, col_i] = np.where(np.isnan(column), fill_val, column)
    return data