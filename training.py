import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import preprocess
import preprocess_original

__author__ = 'maeglin89273'

if __name__ == "__main__":
    preprocess_original.RESAMPLE_SIZE = 8
    data = preprocess_original.resample("./dataset/pendigits-orig_formatted.tra", preprocess_original.arc_len_resample)

    x, y = data[:, :-1], data[:, -1]
    x = preprocess.compute_angles_of_edges(x)

    print("train started")

    # clf = SVC(kernel="rbf", gamma=0.6, C=5)
    # scores = cross_validation.cross_val_score(clf, x, y, cv=10)
    #
    # print(scores)
    # print(np.mean(scores))

