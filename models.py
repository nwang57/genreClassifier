import numpy as np

from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from features import read_features
from utils import plot_scores, plot_learning_curve, normalize_features



def train_model(X, y, clf):

    #split the dataset
    crossvalidation = cross_validation.StratifiedKFold(y, n_folds=5)

    #fit the model
    cms = []
    train_scores = []
    test_scores = []


    for train, test in crossvalidation:
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        X_train, X_test = normalize_features(X_train, X_test)
        #print(X_train[0])

        clf.fit(X_train, y_train)

        #evaluate the model
        train_score = clf.score(X_train, y_train)
        train_scores.append(train_score)
        test_score = clf.score(X_test, y_test)
        test_scores.append(test_score)

        y_predict = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_predict)
        cms.append(cm)

    return np.mean(test_scores), np.mean(train_scores), np.asarray(cms)

def find_best_k(X, y):
    test_scores = []
    train_scores = []
    cm_norms = []
    ks = [1,2,4,8,16,32, 48, 64]
    for i in ks:
        knn = KNeighborsClassifier(n_neighbors=i, weights = 'distance')
        test_score, train_score, cms = train_model(X, y, knn)
        test_scores.append(test_score)
        train_scores.append(train_score)
        cm_norms.append(np.sum(cms, axis=0))
    print("test: ", test_scores)
    print("train: ", train_scores)
    #plot_scores(ks, test_scores, train_scores)

def svm_tuning(X, y):
    C_range = [1E6, 1E7, 1E8]
    for c in C_range:
        svm = SVC(kernel='linear', C=c)
        test_score, train_score, cms = train_model(X, y, svm)
        print("For C: %f, train_score=%f, test_score=%f" % (c, train_score, test_score))
        print()

def rf_tuning(X, y):
    n_trees = [100, 500]
    for n in n_trees:
        rf = RandomForestClassifier(n_estimators=n)
        test_score, train_score, cms = train_model(X, y, rf)
        print("For n_tree: %f, train_score=%f, test_score=%f" % (n, train_score, test_score))
        print()


if __name__ == "__main__":
    features = ['zcr', 'rms', 'sc', 'sr', 'sf','mfcc']
    X, y = read_features(features)

    X = np.nan_to_num(X)
    X[X >= 1E308] = 0
    X[X <= -1E308] = 0

    #find_best_k(X,y)
    rf_tuning(X, y)

    #plot learning curve for 8nn
    # clf = KNeighborsClassifier(n_neighbors=8, weights = 'distance')
    # crossvalidation = cross_validation.StratifiedKFold(y, n_folds=5)
    # plot_learning_curve(clf, "knn_learning_curve", X, y, cv= crossvalidation, n_jobs=4)



