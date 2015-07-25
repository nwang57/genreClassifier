import numpy as np

from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from features import read_features
from utils import plot_scores

def train_model(X, y, n, w):
    #initialize the clf
    weights = w #['uniform','distance']
    n_neighbors = n
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights = weights)

    #split the dataset
    crossvalidation = cross_validation.StratifiedKFold(y, n_folds=5)

    #fit the model
    clfs = []
    cms = []
    train_scores = []
    test_scores = []


    for train, test in crossvalidation:
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        knn_clf.fit(X_train, y_train)

        #evaluate the model
        train_score = knn_clf.score(X_train, y_train)
        train_scores.append(train_score)
        test_score = knn_clf.score(X_test, y_test)
        test_scores.append(test_score)

        y_predict = knn_clf.predict(X_test)
        cm = confusion_matrix(y_test, y_predict)
        cms.append(cm)

    return np.mean(test_scores), np.mean(train_scores), np.asarray(cms)

if __name__ == "__main__":
    features = ['zcr', 'rms', 'sc', 'sr', 'sf']
    X, y = read_features(features)

    test_scores = []
    train_scores = []
    cm_norms = []
    ks = [1,2,4,8,16,32]
    for i in ks:
        test_score, train_score, cms = train_model(X, y, i, 'distance')
        test_scores.append(test_score)
        train_scores.append(train_score)
        cm_norms.append(np.sum(cms, axis=0))
    print("test: ", test_scores)
    plot_scores(ks, test_scores, train_scores)


