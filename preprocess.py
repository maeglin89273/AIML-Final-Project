from sklearn import cross_validation
from sklearn.manifold import SpectralEmbedding, MDS
from sklearn.svm import SVC

__author__ = 'maeglin89273'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from mpl_toolkits.mplot3d import Axes3D

TRIAN_FILE_NAME = "pendigits_train.csv"

def parse_x_y():
    data = np.genfromtxt(TRIAN_FILE_NAME, delimiter=",")
    x, y = data[:, :-1], data[:, -1]
    return x, y

def compute_slope(x):
    points = x.reshape((-1, 8, 2))

    point_diff = points[:, 1:, :] - points[:, :-1, :]

    point_slope = np.arctan2(point_diff[:, :, 1], point_diff[:, :, 0])

    return point_slope


def plot_data(x):
    for i in range(364, 370):
        print(y[i])
        num_vec = x[i]
        plt.plot(num_vec[:, 0], num_vec[:, 1], "o-")
        plt.show()
        plt.close()

def plot_3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=y)
    plt.show()
    plt.close()

if __name__ == "__main__":
    x, y = parse_x_y()
    x_splot = compute_slope(x)

    clf = SVC(kernel="rbf", gamma=1, C=1)
    scores = cross_validation.cross_val_score(clf, x_splot, y, cv=10)
    print(scores)

