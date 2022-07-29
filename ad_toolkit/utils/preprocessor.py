import pandas as pd
import numpy as np


def window_data(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    if window_size < 2:
        return data
    windowed_data = []
    for i in range(len(data) - window_size + 1):
        windowed_data.append(
            np.array(data[i:i + window_size]).flatten()
        )
    index = data.index[window_size-1:]
    return pd.DataFrame(data=windowed_data, index=index)
