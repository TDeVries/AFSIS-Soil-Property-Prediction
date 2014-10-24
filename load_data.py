import numpy as np
import cPickle
import gzip
import os
import sys

import theano
import theano.tensor as T

import pandas as pd

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn import cross_validation

import warnings
warnings.filterwarnings('ignore')

def _shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_data(path, fold = 0):

    dataset = pd.read_csv(path)
    dataset = dataset[dataset.P < 6]
    
    Z = pd.read_csv('../sorted_test.csv')
    X = np.hstack([dataset.values[:,1:2655], dataset.values[:,2671:3579]]) #excludes Co2 bands and location data
    
    Y = dataset.values[:,-5:] #train on all at the same time

    Z = np.hstack([Z.values[:,1:2655], Z.values[:,2671:3579]])  #excludes Co2 bands and location data

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    Z = scaler.transform(Z)

    X, Y = shuffle(X, Y, random_state = fold)

    train_set_x = X
    train_set_y = Y
    valid_set_x = train_set_x
    valid_set_y = train_set_y
    
    def _shared_dataset(data_xy):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX))
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'float64')

    train_set_x, train_set_y = _shared_dataset((train_set_x, train_set_y))
    valid_set_x, valid_set_y = _shared_dataset((valid_set_x, valid_set_y))
    test_set_x, test_set_y = valid_set_x, valid_set_y
    #test_set_x, test_set_y = _shared_dataset((test_set_x, test_set_y))
    Z = theano.shared(np.asarray(Z, dtype=theano.config.floatX), borrow=True)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval, Z

