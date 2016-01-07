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


    return np.array(new_data_queue)



def arc_len_resample(points):
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

def polygonal_approx_resample(points):
    if points.shape[0] <= RESAMPLE_SIZE:
        print("warning: points are too few to be resampled, size: %s" % points.shape[0])
        return points

    result = deque((points[0],))
    result.extend(polygonal_approx_impl(points, 0, points.shape[0] - 1, RESAMPLE_SIZE))
    result.append(points[-1])
    return np.array(result)

def polygonal_approx_impl(points, start_point_idx, end_point_idx, remaining_sample_num):
    start_point = points[start_point_idx]
    closing_edge = points[end_point_idx] - start_point

    max_cross = 0
    best_point_idx = end_point_idx
    for i in range(start_point_idx + 1, end_point_idx + 1):
        edge_of_triangle = points[i] - start_point
        abs_cross = np.absolute(np.cross(edge_of_triangle.cross, closing_edge))
        if abs_cross > max_cross:
            max_cross = abs_cross
            best_point_idx = i

    remaining_sample_num -= 1
    left_remaining_sample_num = int(remaining_sample_num * float(best_point_idx - start_point_idx) / (end_point_idx - start_point_idx))
    right_remaining_sample_num = remaining_sample_num - left_remaining_sample_num

    points_collector = deque((points[best_point_idx],))
    if left_remaining_sample_num > 0:
        points_collector.extendleft(points_collector.epolygonal_approx_impl(points, start_point_idx, best_point_idx, left_remaining_sample_num))

    if right_remaining_sample_num > 0:
        points_collector.extend(polygonal_approx_impl(points, best_point_idx, end_point_idx, right_remaining_sample_num))

    return points_collector

def plot_original_number(num_vec):
    plt.plot(num_vec[:, 0], num_vec[:, 1], "o-")
    plt.ylim(0, 500)
    plt.xlim(0, 500)
    plt.show()
    plt.close()

if __name__ == "__main__":
    filename = "./dataset/pendigits-orig_formatted.tra"
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

        np.savetxt("./dataset/pendigits-resample.csv", resample(filename, arc_len_resample), delimiter=",")