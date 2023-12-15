import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
# from io import BytesIO
# import pandas as pd
# import seaborn as sns
# import numpy as np


from Utils import NormalizeWindowLayer, InverseNormalizeWindowLayer, TimeSeriesToImageLayer



class SDConvBlock(tfkl.Layer):
    def __init__(self, filters, name):
        super(SDConvBlock, self).__init__(name=name)

        # First convolutional layer
        self.conv1 =tfkl.Conv2D(filters, (3, 3), padding='same', name=f'{name}_conv1')
        self.batch_norm1 = tfkl.BatchNormalization(name=f'{name}_batch_norm1')
        self.relu1 = tfkl.ReLU(name=f'{name}_relu1')

        # Second convolutional layer
        self.conv2 = tfkl.Conv2D(filters, (3, 3), padding='same', name=f'{name}_conv2')
        self.batch_norm2 = tfkl.BatchNormalization(name=f'{name}_batch_norm2')
        self.relu2 = tfkl.ReLU(name=f'{name}_relu2')

        # Third convolutional layer
        self.conv3 = tfkl.Conv2D(filters, (3, 3), padding='same', name=f'{name}_conv3')
        self.batch_norm3 = tfkl.BatchNormalization(name=f'{name}_batch_norm3')

        # Identity shortcut connection
        self.add = tfkl.Add(name=f'{name}_add')

        # Final ReLU activation
        self.relu_out = tfkl.ReLU(name=f'{name}_relu_out')

    def call(self, inputs):
        # Forward pass through the convolutional block
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # Shortcut connection
        shortcut = self.add([inputs, x])

        # Final ReLU activation
        output = self.relu_out(shortcut)

        return output
    



class SDStack(tfkl.Layer):
    def __init__(self, filters, num_blocks, name):
        super(SDStack, self).__init__(name=name)

        # Create a list of SDConvBlock instances to form the stack
        self.blocks = [SDConvBlock(filters, f'{name}_block_{i+1}') for i in range(num_blocks)]

        # Final convolution layer for reducing spatial size
        self.final_conv = tfkl.Conv2D(filters, (2, 2), strides=(2, 2), padding='valid', name=f'{name}_final_conv')


    def call(self, inputs):
        # Apply each SDConvBlock in the stack sequentially
        x = inputs
        for block in self.blocks:
            x = block(x)
        
        x = self.final_conv(x)
        return x
    


class SDNetwork(tfkl.Layer):
    def __init__(self, input_shape, name='SDNetwork'):
        super(SDNetwork, self).__init__(name=name)

        filters = 64  # TODO

        # Stacks of SDConvBlocks (using values from paper)
        num_blocks_per_stack = 3
        num_stacks = 5
        self.stacks = [SDStack(filters * 2**i, num_blocks_per_stack, f'stack_{i+1}') for i in range(num_stacks)]

        # Concatenate feature maps from all stacks
        self.concat = tfkl.Concatenate(axis=-1, name='concatenate')

        # Flatten the concatenated feature maps
        self.flatten = tfkl.Flatten(name='flatten')




class ForCNN(tf.keras.Model):
    def __init__(self, window, telescope, cnn_type='ResNet'):
        super(ForCNN, self).__init__()

        self.telescope = telescope
        self.window = window

        # Choose the CNN architecture based on the input string
        if cnn_type == 'ResNet':
            self.cnn = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif cnn_type == 'VGG':
            self.cnn = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif cnn_type == 'SD':
            self.cnn = SDNetwork(input_shape=(224, 224, 3), name='SDNetwork')
        else:
            raise ValueError(f"Unsupported cnn_type: {cnn_type}. Choose from 'ResNet', 'VGG', or 'SD'.")

        # Define pre-processing layers
        self.inputs = tfk.Input(shape=(window), name="input") # TODO: check input shape. No 1 at the end! fix also layer for normalization
        self.normalize_window = NormalizeWindowLayer(return_minmax= True)
        self.to_images = TimeSeriesToImageLayer()

        # CNN layers already defined

        self.global_average_pooling =  tfkl.GlobalAveragePooling2D(name="global_avg_pool")
        self.batch_normalization =  tfkl.BatchNormalization(name="batch_normalization")

        # Define fully connected layers
        # TODO: fix number of neurons
        self.dense1 = tfkl.Dense(1024, activation=tf.keras.activations.swish, name="dense1") # TODO: swish only for classification...?
        self.dense2 = tfkl.Dense(512, activation=tf.keras.activations.swish, name="dense2") 
        self.dense3 = tfkl.Dense(128, activation=tf.keras.activations.swish, name="dense3") 
        self.dense_output = tfkl.Dense(self.telescope, activation='linear', name="dense_output") 
        # in the paper:
        # self.dense1 = tfkl.Dense(1024, activation=tf.keras.activations.relu, name="dense_fc1")
        # self.dense2 = tfkl.Dense(1024, activation=tf.keras.activations.relu, name="dense_fc2")
        # self.dense_output = tfkl.Dense(self.telescope, activation='linear', name="dense_output")


        # Define post-processing layers
        self.denormalize_window = InverseNormalizeWindowLayer()


    def call(self, inputs):
        # Define the forward pass in the call method
        x = self.inputs(inputs) #??
        x, min_max = self.normalize_window(x)
        x = self.to_images(x)
        x = self.cnn(x)
        x = self.global_average_pooling(x)
        x = self.batch_normalization(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense_output(x)
        output = self.denormalize_window(x,min_max)

        return output   