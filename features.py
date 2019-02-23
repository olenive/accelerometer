import numpy as np
from itertools import chain
from typing import Tuple, Iterable, Callable, Any, Dict

import parse


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


def difference_per_second(times_in_nanoseconds: np.ndarray, x: np.ndarray) -> np.ndarray:
    return difference(x) / (difference(times_in_nanoseconds) / nanoseconds_in_one_second)


def magnitude_change_per_second(times_in_nanoseconds: np.ndarray, x: np.ndarray)  -> np.ndarray:
    """Calculate changes in magnitude of acceleration between measurements and convert to change per second."""
    magnitudes = np.linalg.norm(x, axis=0)
    return difference_per_second(times_in_nanoseconds, magnitudes)


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


def angle_change_per_second(times_in_nanoseconds: np.ndarray, x: np.ndarray)  -> np.ndarray:
    angles = angle_difference(x)
    return angles / (difference(times_in_nanoseconds) / nanoseconds_in_one_second)


def calculate_from_measurements(measurements: Iterable[Tuple[int, str, int, float, float, float]],
                                feature_function: Callable[[np.ndarray, np.ndarray], Any],
                                ) -> Any:
    """Apply feature calculating function to a measurement interval."""
    times, x, y, z = parse.relative_time_and_accelerations(measurements)
    accelerations = np.array([x, y, z])
    return feature_function(times, accelerations)


def vectors_for_intervals(
    intervals: Dict[Tuple[int, str], Iterable[Iterable[Tuple[int, str, int, float, float, float]]]],
    feature_functions: Iterable[Callable[[np.ndarray, np.ndarray], Any]],
) -> Dict[Tuple[int, str], Iterable[Iterable[float]]]:
    """Apply feature calculating functions to all measurement intervals in dictionary.

    The resulting dictionary of feature vectors can be passed to methods that evaluate classifier performance.
    """
    out = dict()
    for key, values in intervals.items():
        feature_vectors = []
        for measurements in values:
            feature_vectors.append(
                tuple(calculate_from_measurements(measurements, f) for f in feature_functions)
            )
        out[key] = tuple(feature_vectors)
    return out


def extract_vectors_from_dict(interval_features: Dict[Tuple[int, str], Iterable[Iterable[float]]]
                              ) -> Iterable[np.ndarray]:
    """Produce a vector of values for each feature from a dictionary of features per measurement interval.

    These vectors can then be used for plotting and for fitting distributions to feature values.
    """
    out = []
    all_values = tuple(chain(*interval_features.values()))
    num_features = len(all_values[0])
    for i in range(num_features):
        out.append(
            np.array([x[i] for x in all_values])
        )
    return out


def mean_magnitude_change_per_second(t, x) -> float:
    return np.mean(magnitude_change_per_second(t, x))


def mean_absolute_magnitude_change_per_second(t, x) -> float:
    return np.mean(np.absolute(magnitude_change_per_second(t, x)))


def mean_angle_change_per_second(t, x) -> float:
    return np.mean(angle_change_per_second(t, x))
