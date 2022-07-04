import numpy as np


def euc_dist(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.sum((x - y)**2))
