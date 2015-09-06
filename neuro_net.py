import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

from sklearn.cross_validation import train_test_split
from features import read_features
from utils import plot_scores, plot_learning_curve, normalize_features, clean_data, impute_nan

from sknn.mlp import Classifier, Layer
from sklearn.grid_search import GridSearchCV
import sys
import os
import numpy as np

def train_nolearn_model(X, y):
    '''
        NeuralNet with nolearn
    '''
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
        hidden_num_units=10,
        objective_loss_function=lasagne.objectives.categorical_crossentropy,
        output_nonlinearity=lasagne.nonlinearities.sigmoid,
        output_num_units=10,


        update = nesterov_momentum,
        update_learning_rate= 0.001,
        update_momentum=0.9,

        max_epochs=10,
        verbose=1,
        )
    #net.fit(X_train, y_train)
    #predicted = net.predict(X_test)
    test_score = net.predict(X_test, y_test)
    train_score = net.score(X_train, y_train)
    return train_score, test_score

def train_sknn(X, y):
    '''
        NeuralNet with sknn
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 5)
    X_train, X_test = impute_nan(X_train, X_test)
    X_train, X_test = normalize_features(X_train, X_test)
    nn = Classifier(
        layers=[
            Layer("Tanh", units=12),
            Layer("Softmax")],
        learning_rate=0.005,
        n_iter=25)

    # gs = GridSearchCV(nn, param_grid={
    #     'learning_rate': [0.05, 0.01, 0.005, 0.001],
    #     'hidden0__units': [4, 8, 12,100],
    #     'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]})
    # gs.fit(X_train, y_train)
    # print(gs.best_estimator_)
    nn.fit(X_train, y_train)
    predicted = nn.predict(X_test).flatten()
    labels = y_test
    return predicted, labels

if __name__ == "__main__":
    features = ['zcr', 'rms', 'sc', 'sr', 'sf','mfcc']
    X, y = read_features(features)
    X = clean_data(X)
    # train_score, test_score = train_nolearn_model(X, y)
    # print("train score: ", train_score)
    # print("test score: ", test_score)

    predicted, true_value = train_sknn(X,y)
    print("predicted: ", predicted)
    print("true_value: ", true_value)
    print("accuracy: ", np.sum(predicted == true_value) / float(len(predicted)))
    #the test accuracy is only 38%, need more tunning!!!
