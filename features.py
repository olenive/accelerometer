import numpy as np


def mean_of_magnitudes(x: np.ndarray) -> np.ndarray:
    return np.mean(np.linalg.norm(x, axis=0))


def variance_of_magnitudes(x: np.ndarray) -> np.ndarray:
    return np.var(np.linalg.norm(x, axis=0))


def difference(x: np.ndarray) -> np.ndarray:
    return x[: -1] - x[1:]


def changes_in_magnitudes_per_seconds(x: np.ndarray, times_in_nanoseconds: np.ndarray) -> np.ndarray:
    """Calculate changes in magnitude of acceleration between measurements and convert to change per second."""
    nanoseconds_in_one_second = 1000000000
    magnitudes = np.linalg.norm(x, axis=0)
    return difference(magnitudes) / (difference(times_in_nanoseconds) / nanoseconds_in_one_second)


def unit_vector(v):
    return v / np.linalg.norm(v)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.arccos(np.clip(np.dot(
        unit_vector(v1), unit_vector(v2)
    ), -1.0, 1.0))
