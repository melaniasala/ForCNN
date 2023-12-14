# Import tensorflow
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from io import BytesIO
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



class TimeSeriesToImageLayer(tfkl.Layer):
    def __init__(self):
        super(TimeSeriesToImageLayer, self).__init__()

    def timeseries_to_image(self, timeseries_tensor):
        images = []
        timeseries_array = timeseries_tensor.numpy()

        for window in timeseries_array:
            fig, ax = plt.subplots(facecolor='black')
            ax.plot(window, color='white', linewidth=0.8)
            ax.axis('off')

            # Set figure size explicitly to 224x224 pixels
            fig.set_size_inches(224 / 100, 224 / 100)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove unnecessary margins

            # Save the plot in memory using BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
            buffer.seek(0)  # Reset the buffer position to the start

            # Read the image from the buffer and convert it to grayscale
            img = plt.imread(buffer)
            img_gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale

            # Create an RGB representation with the same grayscale information in each channel
            img_rgb = np.stack((img_gray, img_gray, img_gray), axis=-1)
            images.append(img_rgb)
            plt.close(fig)  # Close the figure
            
        images_array = np.array(images)
        images_tensor = tf.convert_to_tensor(images_array, dtype=tf.float32)
        return images_tensor

    def call(self, inputs):
        images = tf.py_function(self.timeseries_to_image, [inputs], tf.float32)
        return images





class NormalizeWindowLayer(tfkl.Layer):
    def __init__(self, return_minmax=False):
        super(NormalizeWindowLayer, self).__init__()
        self.return_minmax = return_minmax


    def normalize_window(self, windows_batch):
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

        # Return results as Keras tensors
        scaled_batch = tf.convert_to_tensor(scaled_batch, dtype=tf.float32)
        minmax_batch_array = tf.convert_to_tensor(minmax_batch_array, dtype=tf.float32)

        if self.return_minmax:
            return scaled_batch, minmax_batch_array
        else:
            return scaled_batch
        

    def call(self, inputs):
        scaled_batch, minmax_batch_array = tf.py_function(
            self.normalize_window,
            [inputs],
            [tf.float32, tf.float32] if self.return_minmax else tf.float32
        )
        
        return (scaled_batch, minmax_batch_array) if self.return_minmax else scaled_batch

        


class InverseNormalizeWindowLayer(tfkl.Layer):
    def inverse_normalize_window(self, windows_batch, minmax_batch_array):
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

        # Convert result to TensorFlow tensor
        inverse_scaled_batch = tf.convert_to_tensor(inverse_scaled_batch, dtype=tf.float32)

        return inverse_scaled_batch
    

    def call(self, inputs):
        # Assuming inputs is a tuple (windows_batch, minmax_batch_array)
        inverse_scaled_batch = tf.py_function(
            self.inverse_normalize_window,
            inputs,
            tf.float32
        )

        return inverse_scaled_batch
