import numpy as np
import matplotlib.pyplot as plt


def overlay_series(x: np.ndarray, series: list):
    fig, ax = plt.subplots(1, figsize=(20, 10))
    for y in series:
        ax.plot(x, y)
    return fig, ax
