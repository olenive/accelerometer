import numpy as np
from itertools import chain
from typing import Sequence, Tuple, Dict, List, Set

import parse
import features


def train_test_folds(ids: List, shuffled_index_sequence: Sequence, num_folds: int) -> Sequence[Tuple[set, set]]:
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


def confusion_matrix_from_pairs(pairs: Sequence[Tuple[str, str]]) -> Tuple[np.ndarray, Sequence[str]]:
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
        data: Dict[Tuple[int, str], Sequence[Sequence[float]]],
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
        data: Dict[Tuple[int, str], Sequence[Sequence[float]]],
        activities: Set[str]
    ) -> Dict[str, Sequence[Tuple[float, float]]]:
        out = dict()
        for activity in activities:
            activity_features = parse.collect_dict_values_by_key_content(data, activity)
            value_vectors = features.extract_vectors_from_dict(activity_features)
            means_and_variances = []
            for vector in value_vectors:
                means_and_variances.append((float(np.mean(vector)), float(np.var(vector))))
            out[activity] = means_and_variances
        return out

    @staticmethod
    def estimate_activity_probabilities(
        data: Dict[Tuple[int, str], Sequence[Sequence[float]]],
        activities: set
    ) -> Dict[str, float]:
        activity_counts = dict()
        total_count = 0
        # Count the number of feature vectors for each activity.
        for activity in activities:
            count = 0
            activity_features = parse.collect_dict_values_by_key_content(data, activity)
            for v in activity_features.values():
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

    def product_p_x_given_activity(self, x: Sequence[float], activity: str
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

    def p_activity_given_x(self, x: Sequence[float]):
        """NB estimate of probability of activities given a feature vector."""
        p_class_given_x = dict()
        for activity in self.activities:
            p_class_given_x[activity] = self.p_activities[activity] * self.product_p_x_given_activity(x, activity)
        return p_class_given_x

    def predict_from_feature_vector(self, x: Sequence[float]) -> str:
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
            data: Dict[Tuple[int, str], Sequence[Sequence[float]]],
    ) -> Sequence[Tuple[str, str]]:
        """Given new data, return pairs of predicted and known classes."""
        pairs = []
        for key, values in data.items():
            for x in values:
                predicted_label = self.predict_from_feature_vector(x)
                known_label = key[1]
                pairs.append((known_label, predicted_label))
        return tuple(pairs)


class KNNClassifier:
    def __init__(
        self,
        data: Dict[Tuple[int, str], Sequence[Tuple[float]]],
    ) -> None:
        self.locations, self.labels = KNNClassifier.data_dict_to_points_and_labels(data)

    @staticmethod
    def data_dict_to_points_and_labels(data: Dict[Tuple[int, str], Sequence[Tuple[float]]]
                                       ) -> Tuple[Tuple[Tuple[float]], Tuple[str]]:
        """Take a dictionary mapping labels to feature vectors and return matching sequences of features and labels."""
        feature_vectors = []
        labels = []
        for key, sequence in data.items():
            for vector in sequence:
                feature_vectors.append(vector)
                labels.append(key[1])
        return tuple(feature_vectors), tuple(labels)

    @staticmethod
    def distances_to_points(point: Sequence[float], points: Sequence[Tuple]) -> Sequence[float]:
        """Calculate L2 distance between a given point and a sequence of other points"""
        return tuple(float(np.linalg.norm(np.array(x) - np.array(point))) for x in points)

    @staticmethod
    def sort_distances_and_labels(distances: Sequence[float], labels: Sequence[str]
                                  ) -> Tuple[Sequence[float], Sequence[str]]:
        """Sort distances and also sort labels so that they match the re-ordered distances."""
        sort_index = np.argsort(distances)
        sorted_distances: Tuple[float] = tuple(distances[i] for i in sort_index)
        sorted_labels: Tuple[str] = tuple(labels[i] for i in sort_index)
        return sorted_distances, sorted_labels

    @staticmethod
    def resolve_ties(sorted_labels: Sequence[str], k: int) -> str:
        """Pick the most frequent label in top k neighbours or decrement k and try again if there is a tie."""
        top_k = list(sorted_labels[:k])
        # Get labels and corresponding counts of items.
        counts = []
        labels = []
        for label in set(top_k):
            labels.append(label)
            counts.append(list(top_k).count(label))
        # Check for ties
        if len(set(counts)) == len(counts):
            # No duplicated counts so return label corresponding to highest count.
            return labels[int(np.argmax(counts))]
        else:
            # Decrease k and try again.
            return KNNClassifier.resolve_ties(sorted_labels, k - 1)

    def predict_from_feature_vector(self, x: Sequence[float], k: int) -> str:
        """Predict activity given a feature vector."""
        distances = KNNClassifier.distances_to_points(x, self.locations)
        _, sorted_labels = KNNClassifier.sort_distances_and_labels(distances, self.labels)
        return KNNClassifier.resolve_ties(sorted_labels, k)
