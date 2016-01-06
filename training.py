import numpy as np
from sklearn import cross_validation
from sklearn.svm import SVC
from preprocess import parse_xy

__author__ = 'maeglin89273'

if __name__ == "__main__":
    x, y = parse_xy("./dataset/edges_train.csv")

    clf = SVC(kernel="rbf", gamma=1, C=1)
    scores = cross_validation.cross_val_score(clf, x, y, cv=10)
    print(scores)
    print(np.mean(scores))