import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import preprocess
import preprocess_original

__author__ = 'maeglin89273'

def grid_search_opt(x, y, clf, param_grid, cv):
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid_search.fit(x, y)
    print("best score: %s%%" % (100 * grid_search.best_score_))
    print("parameters: %s" % grid_search.best_params_)

if __name__ == "__main__":

    # data = preprocess_original.resample("./dataset/pendigits-orig_formatted.tra", "arc_len")
    # x, y = data[:, :-1], data[:, -1]
    o_x, y = preprocess.parse_xy("./dataset/pendigits-resampled_train.csv")


    print("train started")
    clf = SVC()

    x = preprocess.compute_angles_of_edges(o_x)
    param_grid = {"kernel":["rbf"], "gamma": np.linspace(0.2, 0.4, 3), "C": np.linspace(4, 7, 5)}
    grid_search_opt(x, y, clf, param_grid, 5)


    #x = preprocess.compute_angles_between_edges(o_x)
    #param_grid = {"kernel":["rbf"], "gamma": np.linspace(0.1, 0.5, 5), "C": np.linspace(10, 30, 10)}
    #grid_search_opt(x, y, clf, param_grid, 3)


    # data = preprocess_original.resample("./dataset/pendigits-orig_formatted.tra", "poly_approx")
    # o_x = data[:, :-1]
    # x = preprocess.compute_angles_of_edges(o_x)
    # param_grid = {"kernel":["rbf"], "gamma": np.linspace(0.4, 0.7, 7), "C": np.linspace(10, 20, 10)}
    # grid_search_opt(x, y, clf, param_grid, 3)

