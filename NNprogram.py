"""
    Program loads caltech101_silhouettes_28 data set
    and runs multiclass neural network with SGD/softmax to classify images
"""

import Dataset
import Network


(train, valid, test) = Dataset.load_data()

''' call the pca() function with the dataset label you want to run pca on
        classnames (1, 101)
        train_data (4100, 784)
        val_data (2264, 784)
        test_data (2307, 784)
        train_labels (4100, 1)
        val_labels (2264, 1)
        test_labels (2307, 1)
'''
Dataset.pca("train_data")

trainNN = Network.Network([784,442,101], activationFunc=Network.relu)

'''SGD input: training_data, epochs, mini_batch_size, eta, lmbda = 0.0,
        evaluation_data=None,
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)
output: tuple containing four lists
        the (per-epoch) costs on the evaluation data,
        the accuracies on the evaluation data,
        the costs on the training data,
        and the accuracies on the training data.
    ! list empty if the corresponding flag is not set !
'''
#eCost, accurEval, costTrain, accurTrain = trainNN.SGD(train,30,10,.1,30,valid)
