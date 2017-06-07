# -*- coding: utf-8 -*-

"""
Based on the tflearn example located here:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""
from __future__ import division, print_function, absolute_import

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

import ParseData as pad
import numpy as np

# The size of the classnames
maxSize = 101

def resizeData(value):
### This function resize the data from 2d to 3d
	vector = np.zeros(maxSize)
	vector[value -1] = 1
	return vector

# Load the data set
X = pad.train_data
Y = pad.train_labels
X_test = pad.test_data
Y_test = pad.test_labels

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:

network = input_data(shape=[None, 28, 28],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# Step 1: Convolution
network = conv_1d(network, 28, 2, activation='relu')

# Step 2: Max pooling
network = max_pool_1d(network, 1)

# Step 3: Convolution again
network = conv_1d(network, 56, 2, activation='relu')

# Step 4: Convolution yet again
network = conv_1d(network, 56, 2, activation='relu')

# Step 5: Max pooling again
network = max_pool_1d(network, 1)

# Step 6: Fully-connected 392 node neural network
network = fully_connected(network, 392, activation='relu')

# Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)

# Step 8: Fully-connected neural network with two outputs (0=isn't a bird, 1=is a bird) to make the final prediction
network = fully_connected(network, 1, activation='softmax')

# Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Changing the size of Y and Y_test
newY = []
tempY = []
for counter in range(len(Y)):
	tempVar = resizeData(Y[counter][0])
	tempY.append(tempVar)

newYT = []
tempYT = []
for counter in range(len(Y)):
	tempVar = resizeData(Y[counter][0])
	tempYT.append(tempVar)

Y = tempY
Y_test = tempYT

# Wrap the network in a model object
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='bird-classifier.tfl.ckpt')

# Train it! We'll do 50 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=84,
          snapshot_epoch=True,
          run_id='bird-classifier')

# Save model when training is complete to a file
model.save("bird-classifier.tfl")
print("Network trained and saved as bird-classifier.tfl!")
