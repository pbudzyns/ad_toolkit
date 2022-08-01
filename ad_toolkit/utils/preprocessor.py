import pandas as pd
import numpy as np


def window_data(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Concatenates consecutive time steps into one vector.

    Parameters
    ----------
    data
        Data to process.
    window_size
        Number of time steps to concatenate together.

    Returns
    -------
    pd.DataFrame
        Windowed data.
    """
    if window_size < 2:
        return data
    windowed_data = []
    for i in range(len(data) - window_size + 1):
        windowed_data.append(
            np.array(data[i:i + window_size]).flatten()
        )
    index = data.index[window_size-1:]
    return pd.DataFrame(data=windowed_data, index=index)
