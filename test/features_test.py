import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

import features


nanoseconds_in_one_second = 1000000000


def test_mean_of_magnitudes_returns_expected_array_for_2d_case():
    given = np.array([
        [3, 5, 8, 7],
        [4, 12, 15, 24]
    ])
    expected = np.sum(np.array([5, 13, 17, 25])) / 4
    result = features.mean_of_magnitudes(given)
    assert_array_equal(result, expected)


def test_difference_works_for_1d_case():
    given = np.array(
        [10, 20, 40, 100],
    )
    expected = np.array(
        [10, 20, 60],
    )
    result = features.difference(given)
    assert_array_equal(result, expected)


def test_difference_works_for_2d_case():
    given = np.array([
        [10, 20, 40, 100],
        [1, 2, 4, 10]
    ])
    expected = np.array([
        [10, 20, 60],
        [1, 2, 6]
    ])
    result = features.difference(given)
    assert_array_equal(result, expected)


def test_magnitude_change_per_second_returns_expected_values_for_2d_case():
    accelerations = np.array([
        [3, 5, 8, 7],
        [4, 12, 15, 24]
    ])
    times_in_nanoseconds = np.array([
        0,
        1000000,
        3000000,
        3500000,
    ])
    expected = np.array([
        (13 - 5) / (1000000 / nanoseconds_in_one_second),
        (17 - 13) / (2000000 / nanoseconds_in_one_second),
        (25 - 17) / (500000 / nanoseconds_in_one_second),
    ])
    result = features.magnitude_change_per_second(accelerations, times_in_nanoseconds)
    assert_array_equal(result, expected)


def test_angle_between_vectors_returns_60_degrees_in_radians():
    v1 = np.array([0, 1, 0])
    v2 = np.array([0, 1, np.sqrt(3)])
    expected = 60 * np.pi / 180
    result = features.angle_between_vectors(v1, v2)
    assert result == expected


def test_angle_between_vectors_returns_zero_for_vectors_in_the_same_direction():
    v1 = np.array([5, 1.5, 2])
    v2 = np.array([50, 15, 20])
    result = features.angle_between_vectors(v1, v2)
    assert_almost_equal(result, 0.0)


def test_angle_between_vectors_returns_half_pi_for_orthogonal_vectors():
    v1 = np.array([0, 1, 0])
    v2 = np.array([0, 0, np.sqrt(3)])
    expected = np.pi / 2
    result = features.angle_between_vectors(v1, v2)
    assert result == expected


def test_angle_between_vectors_returns_pi_for_vectors_in_opposite_directions():
    v1 = np.array([1, 1, 0])
    v2 = np.array([-1, -1, 0])
    expected = np.pi
    result = features.angle_between_vectors(v1, v2)
    assert_almost_equal(result, expected)


def test_angle_difference_returns_expected_array():
    a = np.sqrt(3)
    vectors_in_columns = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, -1],
        [0, 0, a, 0, 0],
    ])
    angles_between_column_vectors = np.array([
        np.pi / 2,
        60 * np.pi / 180,
        60 * np.pi / 180,
        np.pi
    ])
    result = features.angle_difference(vectors_in_columns)
    assert_array_equal(result, angles_between_column_vectors)


def test_angle_change_per_second_returns_expected_array():
    a = np.sqrt(3)
    vectors_in_columns = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, -1],
        [0, 0, a, 0, 0],
    ])
    times_in_nanoseconds = np.array([
        0,
        1000000,
        3000000,
        3500000,
        4000000,
    ])
    expected = np.array([
        np.pi / 2 / (1000000 / nanoseconds_in_one_second),
        60 * np.pi / 180 / (2000000 / nanoseconds_in_one_second),
        60 * np.pi / 180 / (500000 / nanoseconds_in_one_second),
        np.pi / (500000 / nanoseconds_in_one_second),
    ])
    result = features.angle_change_per_second(vectors_in_columns, times_in_nanoseconds)
    assert_array_equal(result, expected)


def test_calculate_from_measurements_returns_expected_values():
    measurements = (
        (2, 'Walking', 10000000, 10, 14, 18),
        (2, 'Walking', 50000000, 11, 15, 19),
        (2, 'Walking', 100000000, 12, 16, 20),
        (2, 'Walking', 200000000, 13, 17, 21),
    )
    result = features.calculate_from_measurements(measurements, lambda t, x: (t, x))
    expected_x = np.array([
        [10, 11, 12, 13],
        [14, 15, 16, 17],
        [18, 19, 20, 21]
    ])
    expected_t = np.array([10000000, 50000000, 100000000, 200000000]) - 10000000
    assert_array_equal(result[0], expected_t)
    assert_array_equal(result[1], expected_x)


def test_calculate_from_measurements_returns_expected_sum():
    measurements = (
        (2, 'Walking', 10000000, 10, 14, 18),
        (2, 'Walking', 50000000, 11, 15, 19),
        (2, 'Walking', 100000000, 12, 16, 20),
        (2, 'Walking', 200000000, 13, 17, 21),
    )
    expected = np.sum(list(range(10, 22)))  # 186
    result = features.calculate_from_measurements(measurements, lambda t, x: np.sum(x))
    assert result == expected


def test_vectors_for_intervals_returns_expected_values():
    intervals = {
        (2, "Walking"): (
            (
                (2, 'Walking', 10000000, 10, 14, 18),
                (2, 'Walking', 50000000, 11, 15, 19),
                (2, 'Walking', 100000000, 12, 16, 20),
                (2, 'Walking', 200000000, 13, 17, 21),
            ),
            (
                (2, 'Walking', 10000000, 0, 0, 0),
                (2, 'Walking', 50000000, 1, 0, 0),
                (2, 'Walking', 100000000, 2, 0, 0),
                (2, 'Walking', 200000000, 3, 0, 0),
            )
        ),
        (2, "Standing"): (
            (
                (2, 'Standing', 10000000, 0, 0, 0),
                (2, 'Standing', 50000000, 0, 0, 0),
                (2, 'Standing', 100000000, 0, 0, 0),
                (2, 'Standing', 200000000, 0, 0, 0),
            ),
            (
                (2, 'Standing', 10000000, 0, 0, 0),
                (2, 'Standing', 50000000, 1, 0, 0),
                (2, 'Standing', 100000000, 2, 0, 0),
                (2, 'Standing', 200000000, 3, 0, 0),
            )
        )
    }
    expected = {
        (2, "Walking"): ((186, 15.5), (6, 0.5)),
        (2, "Standing"): ((0, 0), (6, 0.5))
    }
    feature_functions = (
        lambda t, x: np.sum(x),
        lambda t, x: np.mean(x),
    )
    result = features.vectors_for_intervals(intervals, feature_functions)
    assert result == expected


# def test_vector_for_measurement_sreturns_expected_values():
#     measurements = (
#         (2, 'Walking', 10000000, 10, 14, 18),
#         (2, 'Walking', 50000000, 11, 15, 19),
#         (2, 'Walking', 100000000, 12, 16, 20),
#         (2, 'Walking', 200000000, 13, 17, 21),
#     )
#     expected = (186, 15.5)
#     feature_functions = (
#         lambda
#     )
#     result = features.vector_for_measurements(intervals, lambda t, x: np.sum(x))
#     assert result == expected


# def test_calculate_for_intervals_returns_expected_values():
#     intervals = (
#         (
#             (2, 'Walking', 10000000, 10, 14, 18),
#             (2, 'Walking', 50000000, 11, 15, 19),
#             (2, 'Walking', 100000000, 12, 16, 20),
#             (2, 'Walking', 200000000, 13, 17, 21),
#         ),
#         (
#             (2, 'Walking', 10000000, 0, 0, 0),
#             (2, 'Walking', 50000000, 1, 0, 0),
#             (2, 'Walking', 100000000, 2, 0, 0),
#             (2, 'Walking', 200000000, 3, 0, 0),
#         )
#     )
#     expected = (186, 6)
#     result = features.calculate_for_intervals(intervals, lambda x, t: np.sum(x))
#     assert result == expected
#
#
# def test_calculate_for_dict_returns_expected_dict():
#     intervals = {
#         (2, "Walking"): (
#             (
#                 (2, 'Walking', 10000000, 10, 14, 18),
#                 (2, 'Walking', 50000000, 11, 15, 19),
#                 (2, 'Walking', 100000000, 12, 16, 20),
#                 (2, 'Walking', 200000000, 13, 17, 21),
#             ),
#             (
#                 (2, 'Walking', 10000000, 0, 0, 0),
#                 (2, 'Walking', 50000000, 1, 0, 0),
#                 (2, 'Walking', 100000000, 2, 0, 0),
#                 (2, 'Walking', 200000000, 3, 0, 0),
#             )
#         ),
#         (2, "Standing"): (
#             (
#                 (2, 'Standing', 10000000, 0, 0, 0),
#                 (2, 'Standing', 50000000, 0, 0, 0),
#                 (2, 'Standing', 100000000, 0, 0, 0),
#                 (2, 'Standing', 200000000, 0, 0, 0),
#             ),
#             (
#                 (2, 'Standing', 10000000, 0, 0, 0),
#                 (2, 'Standing', 50000000, 1, 0, 0),
#                 (2, 'Standing', 100000000, 2, 0, 0),
#                 (2, 'Standing', 200000000, 3, 0, 0),
#             )
#         )
#     }
#     expected = {(2, "Walking"): (186, 6), (2, "Standing"): (0, 6)}
#     result = features.calculate_for_dict(intervals, lambda x, t: np.sum(x))
#     assert result == expected



