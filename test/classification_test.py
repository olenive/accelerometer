import classification


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
