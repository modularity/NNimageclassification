import scipy.io
import numpy as np

def load_data():
    mat = scipy.io.loadmat('caltech101_silhouettes_28.mat')
    mat2 = scipy.io.loadmat('caltech101_silhouettes_28_split1.mat')
    
    classnames = mat['classnames']
    X = mat['X']
    Y = mat['Y']
    
    #note: labels are indexed down by one so as to better fit python arrays
    test_data = mat2['test_data']
    test_labels=mat2['test_labels'] - 1 
    train_data=mat2['train_data']
    train_labels=mat2['train_labels'] - 1
    val_data=mat2['val_data']
    val_labels=mat2['val_labels'] - 1
    
    training_inputs= [np.reshape(x, (784, 1)) for x in train_data]
    training_labels = [vectorized_result(y) for y in train_labels]
    val_inputs = [np.reshape(x, (784, 1)) for x in val_data]
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data]
    
    train = zip(training_inputs, training_labels)
    val = zip(val_inputs, val_labels.flatten())
    test=zip(test_inputs, test_labels.flatten())
    return (train, val, test)

def vectorized_result(j):
    e = np.zeros((101,1))
    e[j] = 1.0
    return e
    

    