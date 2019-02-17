import numpy as np
import matplotlib.pyplot as plt

import parse

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
