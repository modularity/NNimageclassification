"""
    DATASET: CalTech 101 28x28 silhouettes
    Dimensions:
        classnames (1, 101)
        train_data (4100, 784)
        val_data (2264, 784)
        test_data (2307, 784)
        train_labels (4100, 1)
        val_labels (2264, 1)
        test_labels (2307, 1)
"""

import scipy.io

def load_data():

    ''' #might need later
    mat = scipy.io.loadmat('caltech101_silhouettes_28.mat')
    classnames = mat['classnames']
    X = mat['X']
    Y = mat['Y']
    '''

    # load file
    mat2 = scipy.io.loadmat('caltech101_silhouettes_28_split1.mat')

    # parse dataset
    test_data = mat2['test_data'] #(2307, 1)
    test_labels=mat2['test_labels'] #(2307, 784)
    train_data=mat2['train_data'] #(4100, 784)
    train_labels=mat2['train_labels'] #(4100, 1)
    val_data=mat2['val_data'] #(2264, 784)
    val_labels=mat2['val_labels'] #(2264, 1)

    # pre-process dataset for NN
    # Network.SGD() needs a list of tuples ``(x, y)`` representing the inputs and the desired outputs.
    train = train_data + train_labels
    val = val_data + val_labels
    test= test_data + test_labels

    return (train, val, test)
