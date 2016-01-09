import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import preprocess
import preprocess_original

__author__ = 'maeglin89273'


FINAL_CLASSIFIER = KNeighborsClassifier(n_neighbors=3, weights="distance") #98.3
FEATURE_EXTRACION_FUNC = preprocess.compute_len_angle_of_edges

def grid_search_opt(x, y, clf, param_grid, cv):
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid_search.fit(x, y)
    print("best score: %s%%" % (100 * grid_search.best_score_))
    print(grid_search.grid_scores_[0].cv_validation_scores)
    print("parameters: %s" % grid_search.best_params_)


def eval_test_set():
    tr_x, tr_y = preprocess.parse_xy("./dataset/pendigits_train.csv")
    ts_x, ts_y = preprocess.parse_xy("./dataset/pendigits_test.csv")

    print("train started")

    tr_x = FEATURE_EXTRACION_FUNC(tr_x)
    ts_x = FEATURE_EXTRACION_FUNC(ts_x)

    FINAL_CLASSIFIER.fit(tr_x, tr_y)
    pd_y = FINAL_CLASSIFIER.predict(ts_x)
    print(100 * FINAL_CLASSIFIER.score(ts_x, ts_y))
    print(confusion_matrix(ts_y, pd_y, np.arange(0, 10)))

if __name__ == "__main__":
    eval_test_set()

    # x, y = preprocess.parse_xy("./dataset/pendigits_train.csv")
    # x = FEATURE_EXTRACION_FUNC(x)
    # clf = KNeighborsClassifier()
    #
    # param_grid = {"n_neighbors": np.arange(1, 11), "weights":["distance", "uniform"]}
    # grid_search_opt(x, y, clf, param_grid, 5)

