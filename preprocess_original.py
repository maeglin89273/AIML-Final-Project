from collections import deque
from preprocess import plot_points_with_edges

__author__ = 'maeglin89273'

import numpy as np
import matplotlib.pyplot as plt

RESAMPLE_SIZE = 8
def resample(fname, resample_algorithm):
    new_data_queue = deque()
    with open(fname, "r") as fin:
        for line in fin:
            points_data = np.fromstring(line, sep=",")
            label, points = points_data[0], points_data[1:]
            max, min = np.max(points), np.min(points)
            points = 100 * ((points - min) / (max - min))

            points = resample_algorithm(points.reshape((-1, 2)))

            d = np.hstack((np.ravel(points), label))
            new_data_queue.append(d)


    np.savetxt(fname+".csv", np.array(new_data_queue), delimiter=",")



def arc_len_resampling(points):
    new_points = deque()


    edges = points[1:] - points[:-1]
    edges_len = np.linalg.norm(edges, axis=1)
    ARC_LEN = np.sum(edges_len) / (RESAMPLE_SIZE - 1)

    current_point = points[0]
    current_len = 0

    new_points.append(current_point)
    for edge_len, start_point, end_point in zip(edges_len, points[:-1], points[1:]):

        if (current_len + edge_len) > ARC_LEN:
            residual = ARC_LEN - current_len
            current_len = edge_len - residual
            current_point = start_point + (end_point - start_point) * (residual / edge_len)
            new_points.append(current_point)

            while current_len > ARC_LEN:

                current_len -= ARC_LEN
                residual += ARC_LEN
                current_point = start_point + (end_point - start_point) * (residual / edge_len)
                new_points.append(current_point)

        else:
            current_len += edge_len

    if len(new_points) < RESAMPLE_SIZE:
        new_points.append(points[-1])

    return np.array(new_points)

def plot_original_number(num_vec):
    plt.plot(num_vec[:, 0], num_vec[:, 1], "o-")
    plt.ylim(0, 500)
    plt.xlim(0, 500)
    plt.show()
    plt.close()

if __name__ == "__main__":
    filename = "./dataset/pendigits-orig_formatted.tes"
    PERPOSE = "forma"
    if PERPOSE == "format":
        full_accu = deque()
        with open(filename, "r") as in_file:
            unit_accu = deque()
            for line in in_file:

                if line != "\n":
                    unit_accu.append(line.rstrip())
                else:
                    full_accu.append(",".join(unit_accu) + "\n")
                    unit_accu.clear()


        with open(filename, "w") as out_file:
            out_file.writelines(full_accu)

    else:
        resample(filename, arc_len_resampling)
