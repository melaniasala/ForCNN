import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

def timeseries_to_image(timeseries_array):
    '''
    Converts an array of windows into a tensor of rgb images
    
    PARAMETERS:

        - timeseries_array is a list or an array of timeseries of shape(number of time_series, number of values in the timeseries)
    
    OUTPUT:

        - tensor shaped (num_images,height,width,num_channels)
    '''
    # Set up plot parameters
    images = []
    for window in timeseries_array:
        fig, ax = plt.subplots(facecolor='black')
        ax.plot(window, color='white', linewidth=0.8)
        ax.axis('off')

        # Set figure size explicitly to 224x224 pixels
        fig.set_size_inches(224/100, 224/100)
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