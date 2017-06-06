#sample of how to initialize

import network3
import mnist_loader

train,valid,test=mnist_loader.load_data()
net=network3.Network([784,442,101], activationFunc=network3.sigmoid)
#net.SGD(train,30,10,0.5,5,test)
#just using sigmoid, training data with 30 epochs minibatch size 10, learning rate 0.5, regularization param 5, test data:
#really we're supposed to do net.SGD(train,30,10,0.5,5,valid), then feedforward the test data. woops.
#Anyways the result for this was accurancy: 1468/2307. theres some overfitting since we're using that as our evaluation data

net.SGD(train,30,10,0.5,5,valid)
print net.accuracy(test)
#the way you're supposed to do it. 61.64% accuracy on test data. Had 61.97% accuracy on validation data, 82% accuracy on training data