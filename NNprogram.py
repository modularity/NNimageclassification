"""
    Program loads caltech101_silhouettes_28 data set
    and runs multiclass neural network with SGD/softmax to classify images
"""

import Dataset
from Network import Network


(train, val, test) = Dataset.load_data()

trainNN = Network([2, 3, 1])

'''SGD input: training_data, epochs, mini_batch_size, eta, lmbda = 0.0,
        evaluation_data=None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False)
output: tuple containing four lists
        the (per-epoch) costs on the evaluation data,
        the accuracies on the evaluation data,
        the costs on the training data,
        and the accuracies on the training data.
    ! list empty if the corresponding flag is not set !
'''
(eCost, accurEval, costTrain, accurTrain) = trainNN.SGD(train, 10, 2, .5, None, False, False, True, True)

print(eCost)
