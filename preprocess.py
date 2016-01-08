__author__ = 'maeglin89273'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from mpl_toolkits.mplot3d import Axes3D

#RUNNING CONFIGS:
DATASET = "./dataset/pendigits-resampled_train.csv"
PURPOSE = "plot_points"

def parse_xy(dataset_fname):
    data = np.genfromtxt(dataset_fname, delimiter=",")
    x, y = data[:, :-1], data[:, -1]
    return x, y

def save_data(fname, x, y):
    np.savetxt(fname, np.hstack((x, y.reshape(-1, 1))), delimiter=",")


def compute_edges(x):
    edges = x[:, 2:] - x[:, :-2]
    return edges


def compute_angles_of_edges(x):
    edges = compute_edges(x)
    edges = edges.reshape((edges.shape[0], -1, 2))
    angles_of_edges = np.arctan2(edges[:, :, 1], edges[:, :, 0])

    return angles_of_edges

def compute_len_angle_of_edges(x):
    edges = compute_edges(x)
    edges = edges.reshape((edges.shape[0], -1, 2))
    angles_of_edges = np.arctan2(edges[:, :, 1], edges[:, :, 0])
    len_of_angles = np.linalg.norm(edges, axis=2)
    len_of_angles = len_of_angles / (np.max(len_of_angles) - np.min(len_of_angles))
    return np.hstack((angles_of_edges, len_of_angles))

def compute_angles_between_edges(x):
    angles_of_edges = compute_angles_of_edges(x)
    angles_between_edges = (angles_of_edges[:, 1:] - angles_of_edges[:, :-1]) % (2 * np.pi)

    return angles_between_edges

def plot_points_with_edges(x, y):
    for i in range(5, 10):
        print(y[i])
        num_vec = x[i].reshape(-1, 2)
        plt.plot(num_vec[:, 0], num_vec[:, 1], "o-")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.show()
        plt.close()


def plot_3d(points, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=y)
    plt.show()
    plt.close()

if __name__ == "__main__":

    x, y = parse_xy(DATASET)
    if PURPOSE == "plot_points":
        plot_points_with_edges(x, y)
    else:
        save_data("./dataset/edges_train.csv", compute_edges(x), y)
        save_data("./dataset/angles_of_edges_train.csv", compute_angles_of_edges(x), y)
        save_data("./dataset/angles_between_edges_train.csv", compute_angles_between_edges(x), y)
