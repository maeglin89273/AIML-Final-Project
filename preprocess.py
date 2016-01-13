__author__ = 'maeglin89273'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from mpl_toolkits.mplot3d import Axes3D

#RUNNING CONFIGS:
DATASET = "./dataset/pendigits_train.csv"
PURPOSE = "plot_points"

def parse_xy(dataset_fname):
    data = np.genfromtxt(dataset_fname, delimiter=",")
    x, y = data[:, :-1], data[:, -1]
    return x, y

def save_data(fname, x, y):
    np.savetxt(fname, np.hstack((x, y.reshape(-1, 1))), delimiter=",")


def compute_edges(x):
    return x[:, 2:] - x[:, :-2]

def compute_normalized_edges(tr_x, ts_x=None):
    edges = compute_edges(tr_x)
    if ts_x != None:
        return min_max_normalize(edges, compute_edges(ts_x))

    return min_max_normalize(edges)


def compute_angle_of_edges(tr_x, ts_x=None):
    tr_edges = compute_edges(tr_x)
    tr_edges = tr_edges.reshape((tr_edges.shape[0], -1, 2))
    tr_angles_of_edges = np.arctan2(tr_edges[:, :, 1], tr_edges[:, :, 0])

    if ts_x != None:
        ts_edges = compute_edges(ts_x)
        ts_edges = ts_edges.reshape((ts_edges.shape[0], -1, 2))
        ts_angles_of_edges = np.arctan2(ts_edges[:, :, 1], ts_edges[:, :, 0])
        return tr_angles_of_edges, ts_angles_of_edges

    return tr_angles_of_edges

def compute_len_of_edges(tr_x, ts_x=None):
    tr_edges = compute_edges(tr_x)
    tr_edges = tr_edges.reshape((tr_edges.shape[0], -1, 2))
    tr_len_of_angles = np.linalg.norm(tr_edges, axis=2)

    if ts_x != None:
        ts_edges = compute_edges(tr_x)
        ts_edges = tr_edges.reshape((ts_edges.shape[0], -1, 2))
        ts_len_of_angles = np.linalg.norm(ts_edges, axis=2)
        return min_max_normalize(tr_len_of_angles, ts_len_of_angles)

    return min_max_normalize(tr_len_of_angles)

def compute_len_angle_of_edges(tr_x, ts_x=None):
    tr_edges = compute_edges(tr_x)
    tr_edges = tr_edges.reshape((tr_edges.shape[0], -1, 2))
    tr_angles_of_edges = np.arctan2(tr_edges[:, :, 1], tr_edges[:, :, 0])
    tr_len_of_edges = np.linalg.norm(tr_edges, axis=2)

    if ts_x != None:
        ts_edges = compute_edges(ts_x)
        ts_edges = ts_edges.reshape((ts_edges.shape[0], -1, 2))
        ts_angles_of_edges = np.arctan2(ts_edges[:, :, 1], ts_edges[:, :, 0])
        ts_len_of_edges = np.linalg.norm(ts_edges, axis=2)
        tr_len_of_edges, ts_len_of_edges = min_max_normalize(tr_len_of_edges, ts_len_of_edges)
        tr_len_of_edges *= 2 * np.pi
        ts_len_of_edges *= 2 * np.pi
        return np.hstack((tr_angles_of_edges, tr_len_of_edges)), np.hstack((ts_angles_of_edges, ts_len_of_edges))

    tr_len_of_edges = min_max_normalize(tr_len_of_edges) * 2 * np.pi
    return np.hstack((tr_angles_of_edges, tr_len_of_edges))

def min_max_normalize(tr_x, ts_x=None):
    x_min = np.min(tr_x)
    dx = np.max(tr_x) - x_min
    if ts_x != None:
        return (tr_x - x_min) / dx, (ts_x - x_min) / dx

    return (tr_x - x_min) / dx


def compute_angle_between_edges(tr_x, ts_x=None):

    if ts_x != None:
        tr_angles_of_edges, ts_angles_of_edges = compute_angle_of_edges(tr_x, ts_x)
        tr_angles_between_edges = (tr_angles_of_edges[:, 1:] - tr_angles_of_edges[:, :-1]) % (2 * np.pi)
        ts_angles_between_edges = (ts_angles_of_edges[:, 1:] - ts_angles_of_edges[:, :-1]) % (2 * np.pi)
        return tr_angles_of_edges, ts_angles_of_edges

    tr_angles_of_edges = compute_angle_of_edges(tr_x)
    tr_angles_between_edges = (tr_angles_of_edges[:, 1:] - tr_angles_of_edges[:, :-1]) % (2 * np.pi)
    return tr_angles_between_edges


def plot_points_with_edges(x, y):
    for i in range(0, 10):
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
        save_data("./dataset/angles_of_edges_train.csv", compute_angle_of_edges(x), y)
        save_data("./dataset/angles_between_edges_train.csv", compute_angle_between_edges(x), y)
        save_data("./dataset/angle_len_of_edges_train.csv", compute_len_angle_of_edges(x, y))