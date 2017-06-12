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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data(preprocess=False):

    ''' #might need later
    mat = scipy.io.loadmat('caltech101_silhouettes_28.mat')
    classnames = mat['classnames']
    X = mat['X']
    Y = mat['Y']
    '''

    # load file
    mat2 = scipy.io.loadmat('caltech101_silhouettes_28_split1.mat')

    # parse dataset
    test_data = mat2['test_data'] #(2307, 784)
    test_labels=mat2['test_labels'] -1 #(2307, 1)
    train_data=mat2['train_data'] #(4100, 784)
    train_labels=mat2['train_labels'] -1 #(4100, 1)
    val_data=mat2['val_data'] #(2264, 784)
    val_labels=mat2['val_labels'] -1 #(2264, 1)

    if(preprocess):
        pca=PCA(n_components=100).fit(train_data) #100x4100
        train_data=pca.transform(train_data)
        val_data=pca.transform(val_data)  
        test_data=pca.transform(test_data)
 
    n=len(train_data[0])
    training_inputs= [np.reshape(x, (n, 1)) for x in train_data]
    training_labels = [vectorized_result(y) for y in train_labels]
    val_inputs = [np.reshape(x, (n, 1)) for x in val_data]
    test_inputs = [np.reshape(x, (n, 1)) for x in test_data]


    train = zip(training_inputs, training_labels)
    val = zip(val_inputs, val_labels.flatten())
    test=zip(test_inputs, test_labels.flatten())
    return (train, val, test)

def vectorized_result(j):
    e = np.zeros((101,1))
    e[j] = 1.0
    return e

def pca(dataset):

    # load file
    mat2 = scipy.io.loadmat('caltech101_silhouettes_28_split1.mat')
    # parse dataset
    X = mat2[dataset] #(4100L, 784L)

    '''
    results = mlab.PCA(data)

    #this will return an array of variance percentages for each component
    results.fracs

    #this will return an array of the data projected into PCA space
    results.Y
    '''

    # calculate covariance matrix, alt: use np.cov(X.T)
    mu = np.mean(X, axis=0)
    sigma = (X-mu).T.dot((X-mu)) / (X.shape[0]-1)

    # calculate eigendecomposition
    evals, evecs = np.linalg.eig(sigma)

    '''
    # create tuple to sort w decending values
    eigs = [ (np.abs(evals[i]), evecs[:,i] for i in range(len(evals)) ]
    eigs.sort()
    eigs.reverse()
    '''

    # calculate the explained variance for each component
    total = sum(evals)
    var_exp = evals/total*100
    # calculate the cumulative explained variance, for each additional component
    cum_var_exp = np.cumsum(var_exp)
    index = np.arange(len(evals))

    fig1 = plt.figure(1) # plot for ranked var exp
    plt.bar(index, var_exp, color='b')
    # label the axes
    plt.xlabel("Components")
    plt.ylabel("Variance Explained")
    plt.xlim(-25,300)
    plt.ylim(0, 15) # testing found all significance below 10
    plt.title("PCA analysis for each component")

    fig = plt.figure(2) # plot for cum var exp
    plt.plot(index, cum_var_exp, color='b')
    # label the axes
    plt.xlabel("Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.xlim(-25,len(evals))
    plt.title("PCA analysis of cumulative components")

    plt.show() # show the plot
