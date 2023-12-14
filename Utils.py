# Import tensorflow
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

import pandas as pd
import seaborn as sns
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt



def normalize_window(windows_batch, return_minmax= False):
    """
    Normalize a batch of time series windows using min-max scaling.

    Parameters:
    - windows_batch (numpy.ndarray):    A batch of time series windows with shape (batch_size, window_length, 1)
                                        or (window_length, 1) for a single sample in the batch.
    - return_minmax (bool, optional):   If True, return both the scaled batch and the min-max array for each sample.
                                        If False, return only the scaled batch. Default is False.

    Returns:
    - numpy.ndarray or tuple: If return_minmax is False, returns the scaled batch.
                              If return_minmax is True, returns a tuple (scaled_batch, minmax_batch_array).

    """

    single_sample = False
    if len(windows_batch.shape) == 2:
        single_sample = True

    # Squeeze last dimension
    windows_batch = np.squeeze(windows_batch)
    batch_size = len(windows_batch)

    # For each sample compute min and max value
    if single_sample:
        minmax_batch_array = np.array([[np.min(windows_batch)], [np.max(windows_batch)]])
    else:
        minmax_batch_array = np.array([[np.min(windows_batch[i]), np.max(windows_batch[i])] for i in range(batch_size)])
   
    # Perform min-max scaling for each sample
    if single_sample:
        scaled_batch = np.array([(windows_batch - minmax_batch_array[0]) / (minmax_batch_array[1] - minmax_batch_array[0])])
    else:
        scaled_batch = np.array([(windows_batch[i] - minmax_batch_array[i, 0]) / (minmax_batch_array[i, 1] - minmax_batch_array[i, 0]) for i in range(batch_size)])

    # Return results
    if return_minmax:
        return scaled_batch, minmax_batch_array
    else:
        return scaled_batch
    


def inverse_normalize_window(windows_batch, minmax_batch_array):
    """
    Inverse normalize a batch of time series windows using the provided min-max array.

    Parameters:
    - windows_batch (numpy.ndarray):    A batch of time series windows with shape (batch_size, window_length, 1)
                                        or (window_length, 1) for a single sample in the batch.
    - minmax_batch_array (numpy.ndarray): An array containing the min and max values for each sample in the batch.
                                        Shape is (batch_size, 2) for multiple samples or (2,) for a single sample.

    Returns:
    - numpy.ndarray: The inverse normalized batch.

    """
    single_sample = False
    if len(windows_batch.shape) == 1:
        single_sample = True

    # Squeeze last dimension
    windows_batch = np.squeeze(windows_batch)
    batch_size = len(windows_batch)

    # Inverse min-max scaling for each sample
    if single_sample:
        inverse_scaled_batch = minmax_batch_array[0] + (windows_batch * (minmax_batch_array[1] - minmax_batch_array[0]))
    else:
        inverse_scaled_batch = np.array([minmax_batch_array[i, 0] + (windows_batch[i] * (minmax_batch_array[i, 1] - minmax_batch_array[i, 0])) for i in range(batch_size)])

    return inverse_scaled_batch