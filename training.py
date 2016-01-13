import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import preprocess
import preprocess_original

__author__ = 'maeglin89273'


FINAL_CLASSIFIER = SVC(kernel="rbf", gamma=0.125, C=7.2) #99.7

FEATURE_EXTRACION_FUNC = preprocess.compute_normalized_edges
RESAMPLED = "-resampled" # or blank string "" for given dataset

def grid_search_opt(x, y, clf, param_grid, cv):
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid_search.fit(x, y)
    print("best score: %s%%" % (100 * grid_search.best_score_))
    print(grid_search.grid_scores_[0].cv_validation_scores)
    print("parameters: %s" % grid_search.best_params_)


def eval_test_set():
    tr_x, tr_y = preprocess.parse_xy("./dataset/pendigits%s_train.csv" % RESAMPLED)
    ts_x, ts_y = preprocess.parse_xy("./dataset/pendigits%s_test.csv" % RESAMPLED)

    print("train started")

    tr_x, ts_x = FEATURE_EXTRACION_FUNC(tr_x, ts_x)

    FINAL_CLASSIFIER.fit(tr_x, tr_y)
    pd_y = FINAL_CLASSIFIER.predict(ts_x)
    print(100 * FINAL_CLASSIFIER.score(ts_x, ts_y))
    print(confusion_matrix(ts_y, pd_y, np.arange(0, 10)))
    print(f1_score(ts_y, pd_y, average="weighted"))

if __name__ == "__main__":
    eval_test_set()

    # x, y = preprocess.parse_xy("./dataset/pendigits%s_train.csv" % RESAMPLED)
    # x = FEATURE_EXTRACION_FUNC(x)

    # clf = SVC()
    # clf = KNeighborsClassifier()

    # param_grid = {"kernel": ["rbf"], "gamma": np.linspace(1, 2, 8), "C": np.linspace(2, 8, 8)}
    # param_grid = {"n_neighbors": np.arange(1, 11), "weights": ["distance", "uniform"]}

    # grid_search_opt(x, y, clf, param_grid, 5)
