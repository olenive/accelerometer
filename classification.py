import numpy as np
from itertools import chain
from typing import Iterable, Tuple, Dict, List

import parse
import features


def train_test_folds(ids: List, shuffled_index_sequence: Iterable, num_folds: int) -> Iterable[Tuple[set, set]]:
    # noinspection PyTypeChecker
    length_test = len(ids) // num_folds
    out = []
    for i in range(num_folds):
        test_indices = shuffled_index_sequence[i * length_test: i * length_test + length_test]
        train_indices = (shuffled_index_sequence[: i * length_test:] +
                         shuffled_index_sequence[i * length_test + length_test:]
                         )
        test_ids = set([ids[i] for i in test_indices])
        train_ids = set([ids[i] for i in train_indices])
        out.append((train_ids, test_ids))
    return tuple(out)


def confusion_matrix_from_pairs(pairs: Iterable[Tuple[str, str]]) -> np.ndarray:
    labels = sorted(set(chain(*pairs)))
    matrix = np.zeros((len(labels), len(labels)))
    for pair in pairs:
        actual = labels.index(pair[0])
        predicted = labels.index(pair[1])
        matrix[actual, predicted] += 1
    return matrix, tuple(labels)


class GaussianNaiveBayesClassifier:
    """Naive Bayes classifier that assumes an underlying Gaussian distribution for each feature."""
    def __init__(
        self,
        data: Dict[Tuple[int, str], Iterable[Iterable[float]]],
        activities: set
    ) -> None:
        self.activities = activities
        self.activity_feature_means_vars = GaussianNaiveBayesClassifier.feature_means_and_variances(
            data, activities
        )
        self.p_activities = GaussianNaiveBayesClassifier.estimate_activity_probabilities(
            data, activities
        )

    @staticmethod
    def feature_means_and_variances(
        data: Dict[Tuple[int, str], Iterable[Iterable[float]]],
        activities: Iterable[str]
    ) -> Dict[str, Iterable[Tuple[float, float]]]:
        out = dict()
        for activity in activities:
            activity_features = parse.collect_dict_values_by_key_content(data, activity)
            value_vectors = features.extract_vectors_from_dict(activity_features)
            means_and_variances = []
            for vector in value_vectors:
                means_and_variances.append((np.mean(vector), np.var(vector)))
            out[activity] = means_and_variances
        # noinspection PyTypeChecker
        return out

    @staticmethod
    def estimate_activity_probabilities(
        data: Dict[Tuple[int, str], Iterable[Iterable[float]]],
        activities: set
    ) -> Dict[str, float]:
        activity_counts = dict()
        total_count = 0
        # Count the number of feature vectors for each activity.
        for activity in activities:
            count = 0
            activity_features = parse.collect_dict_values_by_key_content(data, activity)
            for v in activity_features.values():
                # noinspection PyTypeChecker
                count += len(v)
            activity_counts[activity] = count
            total_count += count
        # Convert counts to probability estimate.
        estimate = dict()
        for activity in activities:
            estimate[activity] = activity_counts[activity] / total_count
        return estimate

    @staticmethod
    def normal_pdf(x: float, mean: float, variance: float):
        exponent = (- (x - mean) ** 2) / (2 * variance)
        coefficient = 1 / np.sqrt(2 * np.pi * variance)
        return coefficient * np.exp(exponent)

    def product_p_x_given_activity(self, x: Iterable[float], activity: str
                                   ) -> np.ndarray:
        p_x_given_class = []
        class_mean_var = self.activity_feature_means_vars[activity]
        for i, x_i in enumerate(x):
            mean = class_mean_var[i][0]
            variance = class_mean_var[i][1]
            p_x_given_class.append(
                GaussianNaiveBayesClassifier.normal_pdf(x_i, mean, variance)
            )
        return np.prod(p_x_given_class)

    def p_activity_given_x(self, x: Iterable[float]):
        """NB estimate of probability of activities given a feature vector."""
        p_class_given_x = dict()
        for activity in self.activities:
            p_class_given_x[activity] = self.p_activities[activity] * self.product_p_x_given_activity(x, activity)
        return p_class_given_x

    def predict_from_feature_vector(self, x: Iterable[float]) -> str:
        """Predict activity given a feature vector."""
        p_classes = self.p_activity_given_x(x)
        keys = []
        probabilities = []
        for key, value in p_classes.items():
            keys.append(key)
            probabilities.append(value)
        index: int = np.argmax(probabilities)
        return keys[index]

    def predicted_and_labeled_pairs(
            self,
            data: Dict[Tuple[int, str], Iterable[Iterable[float]]],
    ) -> Iterable[Tuple[str, str]]:
        """Given new data, return pairs of predicted and known classes."""
        pairs = []
        for key, values in data.items():
            for x in values:
                predicted_label = self.predict_from_feature_vector(x)
                known_label = key[1]
                pairs.append((known_label, predicted_label))
        return tuple(pairs)
