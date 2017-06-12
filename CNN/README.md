# NNimageclassification
Neural Network for Image Classification final project for UCLA M156: Machine Learning, Spring 2017

	• Code the neural network 
	• Define a training set and a test set
	• Test several different designs of your network
	• Evaluate performance in various scenarios (different classes to discriminate)
	• Bonus: Pre-process the data with PCA to see how it affects performance and running time

	Caltech 101, 28x28 Silhouettes
	
DUNLAP, LAUREN • HU, CARSON • LIU, JIACHEN


list of website file correspondence:
http://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/  -- kerasNN

https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721 -- mediumNN


Instruction:

$python testCNN.py

Best Parameter: Conolution_1d 28 -> Maxpool -> Convolution_1d 56 ->  Convolution_1d 224 ->  Max_pool -> Fully_connected 1568 -> Fully_connected 101

Best result: 90.54%