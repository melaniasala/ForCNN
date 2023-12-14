# Import tensorflow
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

import pandas as pd
import seaborn as sns
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
plt.rc('font', size=16)
from sklearn.preprocessing import MinMaxScaler



def normalize_window(windows_batch, return_minmax= False):
    # squeeze last dimension
    windows_batch =np.squeeze(windows_batch)
    batch_size = len(windows_batch)

    # For each sample compute min and max value
    minmax_batch_array = np.array([[np.min(windows_batch[i]), np.max(windows_batch[i])] for i in range(batch_size)])

    # Perform min-max scaling for each sample
    scaled_batch = np.array([(windows_batch[i] - minmax_batch_array[i,0]) / (minmax_batch_array[i,1] - minmax_batch_array[i,0]) for i in range(batch_size)])

    if return_minmax:
        return scaled_batch, minmax_batch_array

    else:
        return scaled_batch