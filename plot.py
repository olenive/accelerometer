import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence

import parse
import features


nanoseconds_in_one_second = 1000000000
nanoseconds_in_10_seconds = 10000000000


def overlay_series(x: np.ndarray, series: list):
    fig, ax = plt.subplots(1, figsize=(20, 10))
    for y in series:
        ax.plot(x, y)
    return fig, ax


def series_t_x_y_z(interval, axis):
    t, x, y, z = parse.relative_time_and_accelerations(interval)
    t_min = 0
    t_max = nanoseconds_in_10_seconds
    y_min = -20
    y_max = 20
    for i, v in enumerate([x, y, z]):
        axis[i].plot(t, v)
        axis[i].set_xlim([t_min, t_max])
        axis[i].set_ylim([y_min, y_max])


def intervals_by_activity(data, activities):
    for activity in activities:
        print(activity)
        intervals = []
        for key, value in data.items():
            if activity in key:
                intervals = intervals + list(value)
        fig, axis = plt.subplots(3, figsize=(20, 10))
        for interval in intervals:
            series_t_x_y_z(interval, axis)


def feature_histograms_for_activities(interval_features, activities, bins):
    for activity in activities:
        activity_intervals = parse.collect_dict_values_by_key_content(interval_features, activity)
        feature_vectors = features.extract_vectors_from_dict(activity_intervals)
        for i, vector in enumerate(feature_vectors):
            plt.figure(figsize=(15, 5))
            plt.hist(
                vector,
                bins=bins,
                normed=1, facecolor='blue', alpha=0.5)
            plt.title(activity + " x_" + str(i))
            plt.show()


def feature_scatter_for_activities(interval_features, activities, colours):
    fig, ax = plt.subplots(figsize=(10,10))
    for i, activity in enumerate(activities):
        activity_intervals = parse.collect_dict_values_by_key_content(interval_features, activity)
        x = features.extract_vectors_from_dict(activity_intervals)
        ax.scatter(x[0], x[1], c=colours[i], alpha=0.4, marker='.', label=activity)
    ax.set_xlim((0, 200))
    ax.set_ylim((0, 40))
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax.set_title("")
    plt.xlabel("mean absolute magnitude change per second")
    plt.ylabel("mean angle change per second")
    ax.legend()
    plt.show()


def confusion_matrix(array: np.ndarray, labels: Sequence[str]) -> None:
    sns.heatmap(
        array,
        cmap="jet",
        square=True,
        annot=True,
        xticklabels=labels,
        yticklabels=labels,
        fmt='g'
    )
