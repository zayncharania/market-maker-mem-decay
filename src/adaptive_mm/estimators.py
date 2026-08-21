import numpy as np

def ewma(values, half_life):
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    if np.ndim(values) != 1:
        raise ValueError("values must be one-dimensional")

    decay = 2 ** (-1/half_life)
    m = 0
    w = 0
    estimates = np.empty(len(values))

    for i, value in enumerate(values):
        m = decay * m + value
        w = decay * w + 1
        estimates[i] = m / w

    return estimates
