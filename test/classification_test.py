import numpy as np
from numpy.testing import assert_array_equal

import classification
from classification import GaussianNaiveBayesClassifier, KNNClassifier


def test_train_test_folds_returns_expected_values():
    ids = (100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110)
    sequence = (4, 5, 1, 0, 2, 3, 7, 6, 10, 9, 8)
    n = 3
    test_1 = {104, 105, 101}
    train_1 = {100, 102, 103, 106, 107, 108, 109, 110}
    test_2 = {100, 102, 103}
    train_2 = {101, 104, 105, 106, 107, 108, 109, 110}
    test_3 = {107, 106, 110}
    train_3 = {100, 101, 102, 103, 104, 105, 108, 109}
    expected = (
        (train_1, test_1),
        (train_2, test_2),
        (train_3, test_3),
    )
    result = classification.train_test_folds(ids, sequence, n)
    assert result == expected


def test_normal_pdf_returns_expected_value():
    expected = 0.10328830949345566
    result = GaussianNaiveBayesClassifier.normal_pdf(3, 5, 10)
    assert result == expected


def test_estimate_activity_probabilities():
    given = {
        (2, "Walking"): (186, 6),
        (2, "Standing"): (0, 6, 7),
        (3, "Walking"): (200,),
        (3, "Standing"): (),
        (4, "Walking"): (),
        (4, "Standing"): (10,),
    }
    activities = {"Standing", "Walking", "Jogging"}
    expected = {
        "Standing": 4 / 7,
        "Walking": 3 / 7,
        "Jogging": 0.0,
    }
    result = GaussianNaiveBayesClassifier.estimate_activity_probabilities(given, activities)
    assert result == expected


def test_confusion_matrix_from_pairs_returns_expected_array():
    given = (
        ("A", "B"),
        ("A", "A"),
        ("A", "A"),
        ("A", "A"),
        ("B", "B"),
        ("B", "B"),
        ("B", "B"),
        ("B", "B"),
        ("B", "A"),
        ("B", "A"),
        ("C", "C"),
        ("C", "A"),
        ("C", "A"),
        ("B", "C"),
    )
    expected = np.array([
       # A  B  C
        [3, 1, 0],  # A
        [2, 4, 1],  # B
        [2, 0, 1],  # C
    ])
    expected_labels = ("A", "B", "C")
    result, result_labels = classification.confusion_matrix_from_pairs(given)
    assert_array_equal(result, expected)
    assert result_labels == expected_labels


def test_data_dict_to_points_and_labels_returns_expected_values_and_labels():
    given = {
        (1, "Jogging"): (
            (70, 12),
        ),
        (1, "Downstairs"): (
            (85, 12),
            (81, 12),
            (70, 12),
        ),
        (2, "Jogging"): (),
        (3, "Jogging"): (
            (150, 30),
        )
    }
    points = (
        (70, 12),
        (85, 12),
        (81, 12),
        (70, 12),
        (150, 30),
    )
    labels = (
        "Jogging",
        "Downstairs",
        "Downstairs",
        "Downstairs",
        "Jogging",
    )
    expected = points, labels
    result = KNNClassifier.data_dict_to_points_and_labels(given)

    # import pdb; pdb.set_trace()
    assert result == expected
