import pytest

import parse


def test_raw_data_string_to_timepoint_strings_returns_expected_list_of_strings():
    sample = (  # Note that not all readings are separated by a new line.  This is the case in the input data.
        '33,Jogging,49105962326000,-0.6946377,12.680544,0.50395286;\n' +
        '33,Jogging,49106062271000,5.012288,11.264028,0.95342433;\n' +
        '33,Jogging,49106112167000,4.0,10.882658,-0.08172209;33,Jogging,49106222305000,-0.612,18.496431,3.0237172;\n' +
        '33,Jogging,49106332290000,-1.1849703,12.108489,7.205164;\n' +
        '20,Walking,0,0,0,0.0;\n' +
        '19, Sitting, 131623531465000, 8.88, -1.33, 1.61;\n\n'  # Multiple new lines at the end of the file
    )
    expected = (
        '33,Jogging,49105962326000,-0.6946377,12.680544,0.50395286',
        '33,Jogging,49106062271000,5.012288,11.264028,0.95342433',
        '33,Jogging,49106112167000,4.0,10.882658,-0.08172209',
        '33,Jogging,49106222305000,-0.612,18.496431,3.0237172',
        '33,Jogging,49106332290000,-1.1849703,12.108489,7.205164',
        '20,Walking,0,0,0,0.0',
        '19, Sitting, 131623531465000, 8.88, -1.33, 1.61',
    )
    result = parse.raw_data_string_to_timepoint_strings(sample)
    assert(result == expected)


def test_timepoint_strings_to_timepoint_tuples_returns_expected_tuple_of_tuples():
    given = (
        '33,Jogging,49106332290000,-1.1849703,12.108489,7.205164',
        '20,Walking,0,0,0,0.0',
        '19, Sitting, 131623531465000, 8.88, -1.33, 1.61',
    )
    expected = (
        (33, "Jogging", 49106332290000, -1.1849703, 12.108489, 7.205164),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
        (19, "Sitting", 131623531465000, 8.88, -1.33, 1.61),
    )
    result = parse.timepoint_strings_to_timepoint_tuples(given)
    assert(result == expected)


def test_extract_user_set_returns_expected_set_of_integers():
    given = (
        (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
        (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
        (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
        (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
        (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
        (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
        (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
    )
    expected = {33, 20, 19}
    result = parse.extract_user_set(given)
    assert(result == expected)


def test_extract_user_set_returns_expected_set_of_strings():
    given = (
        (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
        (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
        (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
        (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
        (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
        (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
        (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
    )
    expected = {"Jogging", "Walking", "Sitting"}
    result = parse.extract_activity_set(given)
    assert(result == expected)


def test_select_matching_timepoints_returns_expected_user_timepoints():
    given = (
        (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
        (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
        (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
        (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
        (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
        (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
        (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
    )
    result = parse.select_matching_timepoints(given, column=0, value=19)
    expected = (
        (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
        (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
    )
    assert(result == expected)


def test_select_matching_timepoints_returns_expected_activity_timepoints():
    given = (
        (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
        (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
        (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
        (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
        (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
        (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
        (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
    )
    result = parse.select_matching_timepoints(given, column=1, value="Walking")
    expected = (
        (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
        (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
    )
    assert(result == expected)


def test_timepoints_by_user_returns_expected_dictionary():
    given = (
        (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
        (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
        (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
        (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
        (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
        (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
        (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
    )
    expected = {
        33: (
            (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
            (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
            (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
            (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
            (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        ),
        20: (
            (20, "Walking", 0, 0.0, 0.0, 0.0),
        ),
        19: (
            (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
            (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
        )
    }
    result = parse.timepoints_by_user(given)
    assert(result == expected)


def test_timepoints_by_activity_returns_expected_dictionary():
    given = (
        (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
        (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
        (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
        (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
        (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
        (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
        (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
    )
    expected = {
        "Jogging": (
            (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
            (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
            (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
        ),
        "Walking": (
            (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
            (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
            (20, "Walking", 0, 0.0, 0.0, 0.0),
        ),
        "Sitting": (
            (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
            (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
        )
    }
    result = parse.timepoints_by_activity(given)
    assert(result == expected)


# def test_timepoints_by_user_and_activity_returns_expected_dictionary_of_dictionaries():
#     given = (
#         (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
#         (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
#         (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
#         (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
#         (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
#         (20, "Walking", 0, 0.0, 0.0, 0.0),
#         (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
#         (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
#     )
#     expected = {
#         33: {
#             "Jogging": (
#                 (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
#                 (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
#                 (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
#             ),
#             "Walking": (
#                 (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
#                 (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
#             )
#         },
#         20: {
#             "Walking": (
#                 (20, "Walking", 0, 0.0, 0.0, 0.0),
#             )
#         },
#         19: {
#             "Sitting": (
#                 (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
#                 (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
#             )
#         },
#     }
#     result = parse.timepoints_by_user_and_activity(given)
#     assert(result == expected)


def test_timepoints_by_user_and_activity_returns_dictionary_of_tuples_to_tuples():
    given = (
        (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
        (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
        (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
        (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
        (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
        (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
        (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
    )
    expected = {
        (33, "Jogging"): (
            (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413),
            (33, 'Jogging', 49183932357000, -1.0760075, 3.445948, 8.049625),
            (33, 'Jogging', 49184042312000, -1.0760075, 5.2165933, 6.891896),
        ),
        (33, "Walking"): (
            (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
            (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        ),
        (33, "Sitting"): (),
        (20, "Jogging"): (),
        (20, "Walking"): (
            (20, "Walking", 0, 0.0, 0.0, 0.0),
        ),
        (20, "Sitting"): (),
        (19, "Jogging"): (),
        (19, "Walking"): (),
        (19, "Sitting"): (
            (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
            (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
        )
    }
    result = parse.timepoints_by_user_and_activity(given)
    assert(result == expected)


def test_timepoint_is_valid_returns_true_for_normal_looking_data():
    given = (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413)
    assert parse.timepoint_is_valid(given)


def test_timepoint_is_valid_returns_false_for_float_zeros():
    given = (20, "Walking", 0, 0.0, 0.0, 0.0)
    assert not parse.timepoint_is_valid(given)


def test_timepoint_is_valid_returns_false_for_integer_zeros():
    given = (20, "Walking", 0, 0, 0, 0.0)
    assert not parse.timepoint_is_valid(given)


def test_next_valid_timepoint_returns_none_if_starting_at_last_timepoint():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
    )
    result = parse.next_valid_timepoint(given, 2)
    assert(result is None)


def test_next_valid_timepoint_returns_next_valid_timepoint():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
    )
    result = parse.next_valid_timepoint(given, 0)
    assert (result == (1, 'Jogging', 50000000, 3.95, 12.26, -2.68))


def test_next_valid_timepoint_returns_next_timepoint():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (20, "Walking", 0, 0, 0, 0.0),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
    )
    result = parse.next_valid_timepoint(given, 0)
    assert (result == (1, 'Jogging', 50000000, 3.95, 12.26, -2.68))


def test_next_valid_timepoint_returns_none_if_no_valid_timepoints_remain_after_index():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (20, "Walking", 0, 0, 0, 0.0),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
    )
    result = parse.next_valid_timepoint(given, 2)
    assert(result is None)


def test_split_into_intervals_returns_two_expected_intervals():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
    )
    expected = (
        (
            (1, 'Jogging', 0, 4.48, 14.18, -2.11),
            (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        ),
        (
            (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
            (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
            (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
            (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
        )
    )
    # Interval length of 0.2 seconds or 200,000,000 nanoseconds
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    # import pdb; pdb.set_trace()
    assert(result == expected)


def test_split_into_intervals_returns_two_expected_intervals_ignoring_trailing_data():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
        (1, 'Jogging', 1050000000, 6.66, 10.0, 11.73),
        (1, 'Jogging', 1100000000, 1.76, 9.85, 1.99)
    )
    expected = (
        (
            (1, 'Jogging', 0, 4.48, 14.18, -2.11),
            (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        ),
        (
            (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
            (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
            (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
            (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
        )
    )
    # Interval length of 0.2 seconds or 200,000,000 nanoseconds
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    # import pdb; pdb.set_trace()
    assert(result == expected)


def test_split_into_intervals_returns_three_expected_intervals():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
        (1, 'Jogging', 1050000000, 6.66, 10.0, 11.73),
        (1, 'Jogging', 1100000000, 1.76, 9.85, 1.99),
        (1, 'Jogging', 1149000000, -0.0, -3.214402, 1.334794),
        (1, 'Jogging', 1199999999, -2.7513103, 9.615966, 12.4489975),
    )
    expected = (
        (
            (1, 'Jogging', 0, 4.48, 14.18, -2.11),
            (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        ),
        (
            (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
            (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
            (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
            (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
        ),
        (
            (1, 'Jogging', 1050000000, 6.66, 10.0, 11.73),
            (1, 'Jogging', 1100000000, 1.76, 9.85, 1.99),
            (1, 'Jogging', 1149000000, -0.0, -3.214402, 1.334794),
            (1, 'Jogging', 1199999999, -2.7513103, 9.615966, 12.4489975),
        ),
    )
    # Interval length of 0.2 seconds or 200,000,000 nanoseconds
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    assert(result == expected)


# def test_split_into_intervals_does_not_return_last_interval_if_it_is_too_short():
#     given = (
#         (1, 'Jogging', 0, 4.48, 14.18, -2.11),
#         (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
#         (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
#         (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
#         (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
#         (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
#         (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
#         (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
#         (1, 'Jogging', 4500000000, 6.66, 10.0, 11.73),
#         (1, 'Jogging', 5000000000, 1.76, 9.85, 1.99),
#     )
#     expected = (
#         (
#             (1, 'Jogging', 0, 4.48, 14.18, -2.11),
#             (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
#             (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
#             (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
#         ),
#         (
#             (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
#             (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
#             (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
#             (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
#         ),
#     )
#     # Interval length of 0.2 seconds or 200,000,000 nanoseconds
#     result = parse.split_into_intervals(
#         data=given,
#         interval_duration_in_nanoseconds=200000000,
#         maximum_gap_in_nanoseconds=100000000
#     )
#     assert(result == expected)
    
    
# def test_split_into_intervals_does_not_return_middle_interval_if_it_is_too_short():
#     given = (
#         (1, 'Jogging', 0, 4.48, 14.18, -2.11),
#         (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
#         (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
#         (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
#         (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
#         (1, 'Jogging', 299000000, 1.61, 12.07, -2.18),
#         (1, 'Jogging', 4500000000, 6.66, 10.0, 11.73),
#         (1, 'Jogging', 5000000000, 1.76, 9.85, 1.99),
#         (1, 'Jogging', 5490000000, -0.0, -3.214402, 1.334794),
#         (1, 'Jogging', 5999999999, -2.7513103, 9.615966, 12.4489975),
#     )
#     expected = (
#         (
#             (1, 'Jogging', 0, 4.48, 14.18, -2.11),
#             (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
#             (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
#             (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
#         ),
#         (
#             (1, 'Jogging', 4500000000, 6.66, 10.0, 11.73),
#             (1, 'Jogging', 5000000000, 1.76, 9.85, 1.99),
#             (1, 'Jogging', 5490000000, -0.0, -3.214402, 1.334794),
#             (1, 'Jogging', 5999999999, -2.7513103, 9.615966, 12.4489975),
#         ),
#     )
#     # Interval length of 0.2 seconds or 200,000,000 nanoseconds
#     result = parse.split_into_intervals(
#         data=given,
#         interval_duration_in_nanoseconds=200000000,
#         maximum_gap_in_nanoseconds=100000000
#     )
#     assert(result == expected)
    

def test_split_into_intervals_works_for_intervals_of_lengths_5():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 140000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 200000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 1310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 1351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 1399000000, 7.06, 11.35, 0.89),
        (1, 'Jogging', 1450000000, 6.66, 10.0, 11.73),
        (1, 'Jogging', 1500000000, 1.76, 9.85, 1.99),
        (1, 'Jogging', 1549000000, -0.0, -3.214402, 1.334794),
        (1, 'Jogging', 1599999999, -2.7513103, 9.615966, 12.4489975),
    )
    expected = (
        (
            (1, 'Jogging', 0, 4.48, 14.18, -2.11),
            (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (1, 'Jogging', 140000000, 5.24, 7.21, -5.56),
            (1, 'Jogging', 200000000, 7.27, 5.79, -6.51),
        ),
        (
            (1, 'Jogging', 1310000000, 1.61, 12.07, -2.18),
            (1, 'Jogging', 1351000000, 1.5, 17.69, -3.6),
            (1, 'Jogging', 1399000000, 7.06, 11.35, 0.89),
            (1, 'Jogging', 1450000000, 6.66, 10.0, 11.73),
            (1, 'Jogging', 1500000000, 1.76, 9.85, 1.99),
        ),
    )
    # Interval length of 0.2 seconds or 200,000,000 nanoseconds
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    assert(result == expected)


def test_split_into_intervals_ignores_all_zero_rows():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 0, 0, 0, 0.0),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 0, 0, 0, 0.0),
        (1, 'Jogging', 0, 0, 0, 0.0),
        (1, 'Jogging', 450000000, 6.66, 10.0, 11.73),
        (1, 'Jogging', 500000000, 1.76, 9.85, 1.99),
        (1, 'Jogging', 0, 0, 0, 0.0),
        (1, 'Jogging', 0, 0, 0, 0.0),
        (1, 'Jogging', 0, 0, 0, 0.0),
        (1, 'Jogging', 549000000, -0.0, -3.214402, 1.334794),
        (1, 'Jogging', 599999999, -2.7513103, 9.615966, 12.4489975),
    )
    expected = (
        (
            (1, 'Jogging', 0, 4.48, 14.18, -2.11),
            (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 100000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        ),
        (
            (1, 'Jogging', 450000000, 6.66, 10.0, 11.73),
            (1, 'Jogging', 500000000, 1.76, 9.85, 1.99),
            (1, 'Jogging', 549000000, -0.0, -3.214402, 1.334794),
            (1, 'Jogging', 599999999, -2.7513103, 9.615966, 12.4489975),
        ),
    )
    # Interval length of 0.2 seconds or 200,000,000 nanoseconds
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    assert(result == expected)


def test_split_into_intervals_drops_interval_if_gap_too_great_due_to_all_zero_rows():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 140000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 200000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 1310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 0, 0, 0, 0.0),
        (1, 'Jogging', 0, 0, 0, 0.0),
        (1, 'Jogging', 0, 0, 0, 0.0),
        (1, 'Jogging', 1500000000, 1.76, 9.85, 1.99),
        (1, 'Jogging', 1549000000, -0.0, -3.214402, 1.334794),
        (1, 'Jogging', 1599999999, -2.7513103, 9.615966, 12.4489975),
    )
    expected = (
        (
            (1, 'Jogging', 0, 4.48, 14.18, -2.11),
            (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (1, 'Jogging', 140000000, 5.24, 7.21, -5.56),
            (1, 'Jogging', 200000000, 7.27, 5.79, -6.51),
        ),
    )
    # Interval length of 0.2 seconds or 200,000,000 nanoseconds
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    assert(result == expected)


def test_split_into_intervals_raises_if_time_decreases():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 100000, 7.06, 11.35, 0.89),
        (1, 'Jogging', 4500000000, 6.66, 10.0, 11.73),
        (1, 'Jogging', 5000000000, 1.76, 9.85, 1.99),
        (1, 'Jogging', 5490000000, -0.0, -3.214402, 1.334794),
        (1, 'Jogging', 5999999999, -2.7513103, 9.615966, 12.4489975),
    )
    with pytest.raises(ValueError):
        parse.split_into_intervals(
            data=given,
            interval_duration_in_nanoseconds=200000000,
            maximum_gap_in_nanoseconds=100000000
        )


def test_split_into_intervals_raises_if_id_changes():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (2, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 400000000, 7.06, 11.35, 0.89),
        (1, 'Jogging', 4500000000, 6.66, 10.0, 11.73),
        (1, 'Jogging', 5000000000, 1.76, 9.85, 1.99),
        (1, 'Jogging', 5490000000, -0.0, -3.214402, 1.334794),
        (1, 'Jogging', 5999999999, -2.7513103, 9.615966, 12.4489975),
    )
    with pytest.raises(ValueError):
        parse.split_into_intervals(
            data=given,
            interval_duration_in_nanoseconds=200000000,
            maximum_gap_in_nanoseconds=100000000
        )


def test_split_into_intervals_raises_if_activity_changes():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 400000000, 7.06, 11.35, 0.89),
        (1, 'Jogging', 4500000000, 6.66, 10.0, 11.73),
        (1, 'Jogging', 5000000000, 1.76, 9.85, 1.99),
        (1, 'Jogging', 5490000000, -0.0, -3.214402, 1.334794),
        (1, 'Walking', 5999999999, -2.7513103, 9.615966, 12.4489975),
    )
    with pytest.raises(ValueError):
        parse.split_into_intervals(
            data=given,
            interval_duration_in_nanoseconds=200000000,
            maximum_gap_in_nanoseconds=100000000
        )


def test_split_into_intervals_raises_if_time_repeats():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 400000000, 7.06, 11.35, 0.89),
        (1, 'Jogging', 4500000000, 6.66, 10.0, 11.73),
        (1, 'Jogging', 5000000000, 1.76, 9.85, 1.99),
        (1, 'Jogging', 5490000000, -0.0, -3.214402, 1.334794),
        (1, 'Jogging', 5490000000, -2.7513103, 9.615966, 12.4489975),
    )
    with pytest.raises(ValueError):
        parse.split_into_intervals(
            data=given,
            interval_duration_in_nanoseconds=200000000,
            maximum_gap_in_nanoseconds=100000000
        )
