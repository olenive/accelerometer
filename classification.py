import numpy as np
from typing import Iterable, Tuple


def train_test_folds(ids: Iterable, shuffled_index_sequence: Iterable, num_folds: int) -> Iterable[Tuple[set, set]]:
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
