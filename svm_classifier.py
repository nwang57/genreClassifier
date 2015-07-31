import numpy as np

from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from features import read_features
from utils import plot_scores, plot_learning_curve, normalize_features


def train_model(X, y, c):
    svm_clf = SVC(kernel='linear', C=c)

    crossvalidation = cross_validation.StratifiedKFold(y, n_folds=5)

    #fit the model
    clfs = []
    cms = []
    train_scores = []
    test_scores = []

    for train, test in crossvalidation:
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        X_train, X_test = normalize_features(X_train, X_test)

        svm_clf.fit(X_train, y_train)

        train_score = svm_clf.score(X_train, y_train)
        train_scores.append(train_score)

        test_score = svm_clf.score(X_test, y_test)
        test_scores.append(test_score)

        y_predict = svm_clf.predict(X_test)
        cm = confusion_matrix(y_test, y_predict)
        cms.append(cm)

    return np.mean(test_scores), np.mean(train_scores), np.asarray(cms)

def find_best_params(X, y):
    C_range = [1E6, 1E7, 1E8]
    for c in C_range:
        test_score, train_score, cms = train_model(X, y, c)
        print("For C: %f, train_score=%f, test_score=%f" % (c, train_score, test_score))
        print()

if __name__ == "__main__":
    features = ['zcr', 'rms', 'sc', 'sr', 'sf', 'mfcc']
    X, y = read_features(features)
    X = np.nan_to_num(X)
    find_best_params(X,y)
