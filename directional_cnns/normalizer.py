import numpy as np


def std_normalizer(data):
    """
    Normalizes data to zero mean and unit variance.

    :param data: data
    :return: standardized data
    """
    # normalize as 64 bit, to avoid numpy warnings
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    data = data.copy() - mean
    if std != 0.:
        data = data / std
    return data.astype(np.float16)

