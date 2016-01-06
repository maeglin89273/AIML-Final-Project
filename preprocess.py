from sklearn import cross_validation
from sklearn.manifold import SpectralEmbedding, MDS
from sklearn.svm import SVC

__author__ = 'maeglin89273'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from mpl_toolkits.mplot3d import Axes3D

TRIAN_FILE_NAME = "./dataset/pendigits-resample_train.csv"


def parse_xy(fname):
    data = np.genfromtxt(fname, delimiter=",")
    x, y = data[:, :-1], data[:, -1]
    return x, y


def compute_edges(x):
    edges = x[:, 2:] - x[:, :-2]
    return edges


def compute_angles_of_edges(x):
    edges_xy_in_pair = compute_edges(x).reshape((-1, 7, 2))
    angles_of_edges = np.arctan2(edges_xy_in_pair[:, :, 1], edges_xy_in_pair[:, :, 0])

    return angles_of_edges


def compute_angles_between_edges(x):
    angles_of_edges = compute_angles_of_edges(x)
    angles_between_edges = angles_of_edges[:, 1:] - angles_of_edges[:, :-1]
    return angles_between_edges


def plot_points_with_edges(x, y):
    for i in range(6, 15):
        print(y[i])
        num_vec = x[i].reshape(-1, 2)
        plt.plot(num_vec[:, 0], num_vec[:, 1], "o-")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.show()
        plt.close()


def plot_3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=y)
    plt.show()
    plt.close()


def save_data(fname, x, y):
    np.savetxt(fname, np.hstack((x, y.reshape(-1, 1))), delimiter=",")


if __name__ == "__main__":

    x, y = parse_xy(TRIAN_FILE_NAME)
    # plot_points_with_edges(x, y)
    save_data("./dataset/edges_train.csv", compute_edges(x), y)
    save_data("./dataset/angles_of_edges_train.csv", compute_angles_of_edges(x), y)
    save_data("./dataset/angles_between_edges_train.csv", compute_angles_between_edges(x), y)