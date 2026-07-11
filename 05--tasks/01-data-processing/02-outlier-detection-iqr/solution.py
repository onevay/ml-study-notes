import numpy as np

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
    data = np.array(data)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = float(q1 - k * iqr)
    upper_bound = float(q3 + k * iqr)
    cleaned_data = [float(x) for x in data[(data >= lower_bound) & (data <= upper_bound)]]
    outlier_indices = [int(i) for i in np.where((data < lower_bound) | (data > upper_bound))[0]]
    return {
        "cleaned_data": cleaned_data,
        "outlier_indices": outlier_indices,
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4)
    }