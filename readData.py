
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

maxSize = 101


def resizeData(value):
### This function resize the data from 2d to 3d
  vector = np.zeros(maxSize)
  vector[value -1] = 1
  return vector


X_test = pad.test_data
Y_test = pad.test_labels

# Reshaping the two X data lists
newXT = []

for counter in range(len(X_test)):
  newXT.append(X_test[counter].reshape(28, 28))

X_test = newXT
X_test = np.array(X_test)
X_test = X_test.astype('float32')




# Changing the size of Y and Y_test


tempYT = []
for counter in range(len(Y_test)):
  tempVar = resizeData(Y_test[counter][0])
  tempYT.append(tempVar)

Y_test = tempYT
Y_test = np.array(Y_test)
Y_test = Y_test.astype('float32')

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 28, 28],
           data_preprocessing=img_prep,
           data_augmentation=img_aug)

model = tflearn.DNN(network)


model.load('object-classifier')

result = model.predict(X_test)