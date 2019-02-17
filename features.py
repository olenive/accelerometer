import numpy as np
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


def difference_per_second(x: np.ndarray, times_in_nanoseconds: np.ndarray) -> np.ndarray:
    return difference(x) / (difference(times_in_nanoseconds) / nanoseconds_in_one_second)


def magnitude_change_per_second(x: np.ndarray, times_in_nanoseconds: np.ndarray) -> np.ndarray:
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


def angle_change_per_second(x: np.ndarray, times_in_nanoseconds: np.ndarray) -> np.ndarray:
    angles = angle_difference(x)
    return angles / (difference(times_in_nanoseconds) / nanoseconds_in_one_second)


def calculate_for_measurements(measurements: Iterable[Tuple[int, str, int, float, float, float]],
                               feature_function: Callable[[np.ndarray, np.ndarray], Any],
                               ) -> Any:
    times, x, y, z = parse.relative_time_and_accelerations(measurements)
    accelerations = np.array([x, y, z])
    return feature_function(accelerations, times)


def calculate_for_intervals(data: Iterable[Iterable[Tuple[int, str, int, float, float, float]]],
                            feature_function: Callable[[np.ndarray, np.ndarray], Any]
                            ) -> Tuple[Any]:
    return tuple([calculate_for_measurements(x, feature_function) for x in data])


def calculate_for_dict(data: Dict[Tuple[int, str], Iterable[Iterable[Tuple[int, str, int, float, float, float]]]],
                       feature_function: Callable[[np.ndarray, np.ndarray], Any]
                       ) -> Dict[Tuple[int, str], Tuple[Any]]:
    return {k: calculate_for_intervals(v, feature_function) for k, v in data.items()}


def mean_magnitude_change_per_second(x, t) -> float:
    return np.mean(magnitude_change_per_second(x, t))


def mean_absolute_magnitude_change_per_second(x, t) -> float:
    return np.mean(np.absolute(magnitude_change_per_second(x, t)))


def mean_angle_change_per_second(x, t) -> float:
    return np.mean(angle_change_per_second(x, t))
