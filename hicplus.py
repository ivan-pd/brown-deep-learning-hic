import numpy as np
import tensorflow as tf
from keras.layers import ReLU, Conv2D

conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5

class HiCPlus(tf.keras.Model):
    
    def __init__(self):
        super().__init__()

        self.batch_size = 250
        self.opt = tf.keras.optimizers.SGD(learning_rate=0.00001)

        self.conv1 = Conv2D(conv2d1_filters_numbers, conv2d1_filters_size, input_shape=(40,40, 1))
        self.conv2 = Conv2D(conv2d2_filters_numbers, conv2d2_filters_size)
        self.conv3 = Conv2D(conv2d3_filters_numbers, conv2d3_filters_size)
        self.relu = ReLU()

    def call(self, inputs):
        # Convoltion 1
        x = self.conv1(inputs)
        x = self.relu(x)

        # Convolution 2
        x = self.conv2(x)
        x = self.relu(x)

        # Convolution 3
        x = self.conv3(x)
        x = self.relu(x)

        return x

    def loss_function(self, predictions, targets):
        # print(predictions.shape)
        # print(targets.shape)

        mse = tf.keras.losses.MeanSquaredError()
       
        return  mse(targets, predictions)

    def accuracy(self, predictions, targets):
        
        # Predictions and targets are (batch_size, 28, 28, 1)

        predictions_flat = tf.reshape(predictions, [self.batch_size, -1])
        targets_flat = tf.reshape(targets, [self.batch_size, -1])

        coefficents = []
        for i in range(predictions_flat.shape[0]):
            x = predictions_flat[i]
            y = targets_flat[i]
            r = np.corrcoef(x, y)
            coefficents.append(r[0][1])

        return np.average(coefficents)




