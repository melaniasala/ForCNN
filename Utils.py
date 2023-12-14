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


class timeseries_to_image_layer(tfkl.Layer):
    def __init__(self):
        super(timeseries_to_image_layer, self).__init__()

    def timeseries_to_image(self, timeseries_tensor):
        """
        Converts an array of windows into a tensor of rgb images
        
        Parameters:

            - timeseries_tensor is a keras tensor of shape(number of time_series, number of values in the timeseries)
        
        returns:

            - tensor shaped (num_images,height,width,num_channels)
        """
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

        return np.array(images)

    def call(self, inputs):
        return tf.convert_to_tensor(self.timeseries_to_image(inputs))