import numpy as np


nanoseconds_in_one_second = 1000000000

def mean_of_magnitudes(x: np.ndarray) -> np.ndarray:
    return np.mean(np.linalg.norm(x, axis=0))


def variance_of_magnitudes(x: np.ndarray) -> np.ndarray:
    return np.var(np.linalg.norm(x, axis=0))


def difference(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 1:
        return x[1:] - x[: -1]
    else:
        return x[:, 1:] - x[:, : -1]


def difference_per_second(x: np.ndarray, times_in_nanoseconds: np.ndarray) -> np.ndarray:
    return difference(x) / (difference(times_in_nanoseconds) / nanoseconds_in_one_second)


def changes_in_magnitudes_per_second(x: np.ndarray, times_in_nanoseconds: np.ndarray) -> np.ndarray:
    """Calculate changes in magnitude of acceleration between measurements and convert to change per second."""
    magnitudes = np.linalg.norm(x, axis=0)
    return difference_per_second(magnitudes, times_in_nanoseconds)


def unit_vector(v):
    return v / np.linalg.norm(v)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.arccos(np.clip(np.dot(
        unit_vector(v1), unit_vector(v2)
    ), -1.0, 1.0))


def angle_difference(x: np.ndarray) -> np.ndarray:
    out = []
    for i in range(1, x.shape[1]):
        out.append(angle_between_vectors(
            x[:, i - 1], x[:, i]
        ))
    return np.array(out)


def angle_difference_per_second(x: np.ndarray, times_in_nanoseconds: np.ndarray) -> np.ndarray:
    angles = angle_difference(x)
    return angles / (difference(times_in_nanoseconds) / nanoseconds_in_one_second)
