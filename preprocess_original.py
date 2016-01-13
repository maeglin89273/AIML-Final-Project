from collections import deque
from preprocess import plot_points_with_edges

__author__ = 'maeglin89273'

import numpy as np
import matplotlib.pyplot as plt

#RUNNING CONFIGS:
ORIGINAL_FILE = "./dataset/pendigits-orig_formatted.tes"
RESAMPLED_FILE = "./dataset/pendigits-resampled_test.csv"

PREVIEW_FIGURES_START = 100
PREVIEW_FIGURES_END = 102

RESAMPLE_SIZE = 8
RESAMPLING_ALGORITHM = "arc_len"
PURPOSE = "spec_proof"


def resample(fname, resampling_algorithm):
    new_data_queue = deque()
    resampling_func = RESAMPLING_FUNC_TABLE[resampling_algorithm]
    with open(fname, "r") as fin:
        for line in fin:
            points_datum = np.fromstring(line, sep=",")
            label, points = points_datum[0], points_datum[1:]
            points = points.reshape(-1, 2)
            points = resampling_func(min_max_normalize(points))

            d = np.hstack((np.ravel(points), label))
            new_data_queue.append(d)

    return np.array(new_data_queue)

def resample_comparison(fname):
     with open(fname, "r") as fin:
        resample_result_table = {"Original": None, "Arc Length": None, "Polygonal Approximation": None}
        for i, line in enumerate(fin):

            points_datum = np.fromstring(line, sep=",")
            label, points = points_datum[0], points_datum[1:]
            if label == 0 or label == 6 or label == 8:
                points = min_max_normalize(points.reshape((-1, 2)))

                resample_result_table["Original"] = points
                resample_result_table["Arc Length"] = arc_len_resample(points)
                resample_result_table["Polygonal Approximation"] = poly_approx_resample(points)
                multi_resampling_plot(resample_result_table)


def compute_properties_of_edges(points):
    edges = points[1:] - points[:-1]
    len_of_edges = np.linalg.norm(edges, axis=1)
    angle_of_edges = np.arctan2(edges[:, 1], edges[:, 0])

    return np.hstack((edges, len_of_edges[:, np.newaxis], angle_of_edges[:, np.newaxis]))

def resample_proof_of_generalization(fname):
    #proof by variance

    arc_len_collection = [deque() for i in range(0, 10)]
    poly_approx_collection = [deque() for i in range(0, 10)]

    with open(fname, "r") as fin:
        for i, line in enumerate(fin):
            points_datum = np.fromstring(line, sep=",")
            label, points = int(points_datum[0]), points_datum[1:]
            points = min_max_normalize(points.reshape((-1, 2)))

            arc_len_result = arc_len_resample(points)
            poly_approx_result = poly_approx_resample(points)

            arc_len_collection[label].append(compute_properties_of_edges(arc_len_result))
            poly_approx_collection[label].append(compute_properties_of_edges(poly_approx_result))

    arc_len_variance_of_digits = np.empty((10, RESAMPLE_SIZE - 1, 4))
    poly_approx_variance_of_digits = np.empty((10, RESAMPLE_SIZE - 1, 4))
    arc_len_variance_mean_of_digits = np.empty((10, 4))
    poly_approx_variance_mean_of_digits = np.empty((10, 4))

    for i in range(0, 10):
        # 7 * 4 variances of 7 edges and 4 related properties
        arc_len_variance_of_digits[i] = np.array(arc_len_collection[i]).var(axis=0)
        poly_approx_variance_of_digits[i] = np.array(poly_approx_collection[i]).var(axis=0)

    poly_var_greater_arc_var_percentage = np.sum(poly_approx_variance_of_digits >= arc_len_variance_of_digits) / poly_approx_variance_of_digits.size
    print("%s%% of variances of poly_approx are greater than the variances of arc_len" % (poly_var_greater_arc_var_percentage * 100))

    arc_len_variance_mean_of_digits = arc_len_variance_of_digits.mean(axis=1)
    poly_approx_variance_mean_of_digits = poly_approx_variance_of_digits.mean(axis=1)

    print("\"means of variance of 7 edges\" of digits by arc_len:\n%s" % arc_len_variance_mean_of_digits)
    print("\"means of variance of 7 edges\" of digits by poly_approx:\n%s" % poly_approx_variance_mean_of_digits)

    print("variance summary of 4 properties by arc_len: %s" % arc_len_variance_mean_of_digits.mean(axis=0))
    print("variance summary of 4 properties by poly_approx: %s" % poly_approx_variance_mean_of_digits.mean(axis=0))

def is_on_edge(point, edge_start, edge_end):
    a = np.linalg.norm(edge_end - point)
    b = np.linalg.norm(point - edge_start)
    c = np.linalg.norm(edge_end - edge_start)
    return np.abs(a + b - c) <= 1e-12

def point_edge_distance_square(point, edge_start, edge_end):
    b = edge_end - edge_start
    a = point - edge_start
    if np.all(b == 0):
        return np.linalg.norm(a)

    return np.cross(b, a) ** 2 / np.inner(b, b)

def mse_of_resampled_curve(original, resampled):
    resampled_edge_end_idx = 1
    resampled_last_idx = resampled.shape[0] - 1
    sq_errors_of_points = np.empty((original.shape[0],))

    for i, (start_point, end_point) in enumerate(zip(original[: -1], original[1:])):
        sq_errors_of_points[i] = point_edge_distance_square(start_point, resampled[resampled_edge_end_idx - 1], resampled[resampled_edge_end_idx])
        while resampled_edge_end_idx < resampled_last_idx and is_on_edge(resampled[resampled_edge_end_idx], start_point, end_point):
            resampled_edge_end_idx += 1

    sq_errors_of_points[0] = 0
    sq_errors_of_points[-1] = 0
    return sq_errors_of_points.mean()

def resample_proof_of_specialization(fname):
    error_collection = deque()
    with open(fname, "r") as fin:

        for i, line in enumerate(fin):
            points_datum = np.fromstring(line, sep=",")
            label, points = int(points_datum[0]), points_datum[1:]
            points = min_max_normalize(points.reshape((-1, 2)))

            arc_len_result = arc_len_resample(points)
            poly_approx_result = poly_approx_resample(points)

            error_collection.append(np.array([mse_of_resampled_curve(points, arc_len_result), mse_of_resampled_curve(points, poly_approx_result)]))

    error_data = np.array(error_collection)
    print("%s%% of MSEs of arc_len are greater than the MSEs of poly_approx" % (100 * np.sum(error_data[:, 0] > error_data[:, 1]) / error_data.shape[0]))
    print("mean of MSEs: %s" % error_data.mean(axis=0))
    print("var of MSEs: %s" % error_data.var(axis=0))

def multi_resampling_plot(points_with_title):
    f, axarr = plt.subplots(ncols=len(points_with_title), sharex=True)
    for ax, title in zip(axarr, points_with_title):
        points = points_with_title[title]
        ax.plot(points[:, 0], points[:, 1], "o-")
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    plt.show()
    plt.close()

def min_max_normalize(points):
    bottom_left = np.min(points, axis=0)
    points -= bottom_left

    points = 100 * (points / np.max(points))
    return points

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

def poly_approx_resample(points):
    if points.shape[0] <= RESAMPLE_SIZE:
        print("warning: points are too few to be resampled, size: %s" % points.shape[0])
        return points

    result = deque((points[0],))
    result.extend(__poly_approx_inner__(points, 0, points.shape[0] - 1, RESAMPLE_SIZE - 2))
    result.append(points[-1])

    return np.array(result)

def __poly_approx_inner__(points, start_point_idx, end_point_idx, remaining_sample_num):
    start_point = points[start_point_idx]
    closing_edge = points[end_point_idx] - start_point

    max_cross = -1
    best_point_idx = end_point_idx
    #find the highest vertex
    for i in range(start_point_idx + 1, end_point_idx):
        edge_of_triangle = points[i] - start_point
        abs_cross = np.absolute(np.cross(edge_of_triangle, closing_edge))
        if abs_cross > max_cross:
            max_cross = abs_cross
            best_point_idx = i

    remaining_sample_num -= 1

    #calculate partitional ratio
    remaining_candidates_num = end_point_idx - start_point_idx - 2
    left_remaining_sample_num = right_remaining_sample_num = 0
    if remaining_candidates_num > 0:
        left_remaining_sample_num = int(remaining_sample_num * float(best_point_idx - start_point_idx - 1) / remaining_candidates_num)
        right_remaining_sample_num = remaining_sample_num - left_remaining_sample_num

    #divide into two searching subsets
    points_collector = deque()
    if left_remaining_sample_num > 0:
        points_collector.extend(__poly_approx_inner__(points, start_point_idx, best_point_idx, left_remaining_sample_num))

    points_collector.append(points[best_point_idx])

    if right_remaining_sample_num > 0:
        points_collector.extend(__poly_approx_inner__(points, best_point_idx, end_point_idx, right_remaining_sample_num))

    return points_collector

RESAMPLING_FUNC_TABLE = {"arc_len": arc_len_resample, "poly_approx": poly_approx_resample}

def plot_original_number(num_vec):
    plt.plot(num_vec[:, 0], num_vec[:, 1], "o-")
    plt.ylim(0, 500)
    plt.xlim(0, 500)
    plt.show()
    plt.close()

def advanced_format(filename):
    full_data = deque()
    with open(filename, "r") as in_file:
        datum = deque()
        for line in in_file:

            if line != "\n":
                datum.append(line.rstrip())
            else:
                full_data.append(",".join(datum) + "\n")
                datum.clear()


    with open(filename, "w") as out_file:
        out_file.writelines(full_data)


if __name__ == "__main__":
    if PURPOSE == "format":
        advanced_format(ORIGINAL_FILE)
    elif PURPOSE == "compare":
        resample_comparison(ORIGINAL_FILE)

    elif PURPOSE == "gen_proof":
        resample_proof_of_generalization(ORIGINAL_FILE)

    elif PURPOSE == "spec_proof":
        resample_proof_of_specialization(ORIGINAL_FILE)

    elif "resample":
        np.savetxt(RESAMPLED_FILE, resample(ORIGINAL_FILE, RESAMPLING_ALGORITHM), delimiter=",")