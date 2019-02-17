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


def test_changes_in_magnitudes_per_second_returns_expected_values_for_2d_case():
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
    result = features.changes_in_magnitudes_per_second(accelerations, times_in_nanoseconds)
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


def test_changes_in_angle_per_second_returns_expected_array():
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
    result = features.angle_difference_per_second(vectors_in_columns, times_in_nanoseconds)
    assert_array_equal(result, expected)
