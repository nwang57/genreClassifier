import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from sklearn.cross_validation import train_test_split
from features import read_features
from utils import plot_scores, plot_learning_curve, normalize_features, clean_data, impute_nan

import sys
import os
import numpy as np

def train_model(X, y):
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)
    X_train, X_test = impute_nan(X_train, X_test)
    X_train, X_test = normalize_features(X_train, X_test)

    lays = [('input', layers.InputLayer),
              ('hidden', layers.DenseLayer),
              ('output', layers.DenseLayer),
             ]

    net = NeuralNet(
        layers = lays,
        input_shape=(None, 23),
        hidden_num_units=200,
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=10,

        update = nesterov_momentum,
        update_learning_rate= 0.01,
        update_momentum=0.9,

        max_epochs=10,
        verbose=1,
        )
    net.fit(X_train, y_train)
    test_score = net.score(X_test, y_test)
    train_score = net.score(X_train, y_train)
    return train_score, test_score


if __name__ == "__main__":
    features = ['zcr', 'rms', 'sc', 'sr', 'sf','mfcc']
    X, y = read_features(features)
    X = clean_data(X)
    train_score, test_score = train_model(X, y)
    print("train score: ", train_score)
    print("test score: ", test_score)
