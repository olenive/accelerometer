import pytest
import numpy as np
from numpy.testing import assert_array_equal

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
    assert result == expected


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
    assert result == expected


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
    assert result == expected


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
    assert result == expected


def test_select_matching_measurements_returns_expected_user_measurements():
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
    result = parse.select_matching_measurements(given, column=0, value=19)
    expected = (
        (19, 'Sitting', 131623411592000, 9.08, -1.38, 1.69),
        (19, 'Sitting', 131623491487000, 9.0, -1.46, 1.73),
    )
    assert result == expected


def test_select_matching_measurements_returns_expected_activity_measurements():
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
    result = parse.select_matching_measurements(given, column=1, value="Walking")
    expected = (
        (33, 'Walking', 49394992294000, 0.84446156, 8.008764, 2.7921712),
        (33, 'Walking', 49395102310000, 1.1168685, 8.62168, 3.7864566),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
    )
    assert result == expected


def test_measurements_by_user_returns_expected_dictionary():
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
    result = parse.measurements_by_user(given)
    assert result == expected


def test_measurements_by_activity_returns_expected_dictionary():
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
    result = parse.measurements_by_activity(given)
    assert result == expected


def test_measurements_by_user_and_activity_returns_dictionary_of_tuples_to_tuples():
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
    result = parse.measurements_by_user_and_activity(given)
    assert result == expected


def test_measurement_is_valid_returns_true_for_normal_looking_data():
    given = (33, 'Jogging', 49183874710000, -0.9942854, 3.0237172, 8.308413)
    assert parse.measurement_is_valid(given)


def test_measurement_is_valid_returns_false_for_float_zeros():
    given = (20, "Walking", 0, 0.0, 0.0, 0.0)
    assert not parse.measurement_is_valid(given)


def test_measurement_is_valid_returns_false_for_integer_zeros():
    given = (20, "Walking", 0, 0, 0, 0.0)
    assert not parse.measurement_is_valid(given)
    

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


def test_next_valid_timepoint_returns_none_if_no_valid_measurements_remain_after_index():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (20, "Walking", 0, 0, 0, 0.0),
        (20, "Walking", 0, 0.0, 0.0, 0.0),
    )
    result = parse.next_valid_timepoint(given, 2)
    assert(result is None)


def test_split_into_intervals_returns_empty_tuple_given_empty_tuple():
    given = ()
    expected = ()
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    assert result == expected


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
    assert result == expected


def test_split_into_intervals_returns_two_expected_intervals_ignoring_repeating_timepoints():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),  # Identical repeating rows in data at time 728192638000
        (1, 'Jogging', 351000000, 1.5, 0, 0),  # Non-identical repeating rows found in data at time 8948102277000.
        (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
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
    assert result == expected


def test_split_into_intervals_handles_this_pathological_case_in_the_data():
    given = (
                (15, "Jogging", 728142284000, 13.14, -10.34, -2.9147544),
                (15, "Jogging", 728192638000, 12.11, -7.93, 3.5276701),
                (15, "Jogging", 728192638000, 12.11, -7.93, 3.5276701),
                (15, "Jogging", 728192638000, 12.11, -7.93, 3.5276701),
                (15, "Jogging", 0, 0, 0, 0.0),
                (15, "Jogging", 0, 0, 0, 0.14982383),
                (15, "Jogging", 728362224000, -0.11, 14.02, 0.14982383),
                (15, "Jogging", 728362224000, -0.11, 14.02, 0.14982383),
                (15, "Jogging", 728362224000, -0.11, 14.02, 0.14982383),
                (15, "Jogging", 728582835000, 0.11, 6.59, -3.9499009),
                (15, "Jogging", 728582835000, 0.11, 6.59, -3.9499009),
                (15, "Jogging", 728632548000, 4.4, 17.08, 5.134871),
                (15, "Jogging", 728682262000, 19.57, 19.57, -8.19945),
    )
    expected = (
        (
            (15, "Jogging", 728142284000, 13.14, -10.34, -2.9147544),
            (15, "Jogging", 728192638000, 12.11, -7.93, 3.5276701),
            (15, "Jogging", 728362224000, -0.11, 14.02, 0.14982383),
            (15, "Jogging", 728582835000, 0.11, 6.59, -3.9499009),
            (15, "Jogging", 728632548000, 4.4, 17.08, 5.134871),
            (15, "Jogging", 728682262000, 19.57, 19.57, -8.19945),
        ),
    )
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=(6 * 10 ** 8),  # 0.6 seconds
        maximum_gap_in_nanoseconds=300000000,  # 0.3 seconds
    )
    assert result == expected


def test_split_into_intervals_handles_repeated_time_at_start_of_series():
    given = (
        (7, 'Downstairs', 208772451722000, 5.94, 7.16, -0.99),
        (7, 'Downstairs', 208772451722000, 5.94, 7.16, -0.99),
        (7, 'Downstairs', 208772561708000, 2.41, 2.15, -3.06),
        (7, 'Downstairs', 208772601625000, 1.08, 5.67, -1.23),
        (7, 'Downstairs', 208772641633000, 3.95, 18.31, 0.91),
    )
    expected = (
        (
            (7, 'Downstairs', 208772451722000, 5.94, 7.16, -0.99),
            (7, 'Downstairs', 208772561708000, 2.41, 2.15, -3.06),
            (7, 'Downstairs', 208772601625000, 1.08, 5.67, -1.23),
            (7, 'Downstairs', 208772641633000, 3.95, 18.31, 0.91),
        ),
    )
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=190000000
    )
    assert result == expected


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
    assert result == expected


def test_split_into_intervals_drops_last_interval_because_gap_to_end_is_too_large():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 260000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 270000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 299000000, 7.06, 11.35, 0.89),
    )
    expected = (
        (
            (1, 'Jogging', 0, 4.48, 14.18, -2.11),
            (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        ),
    )
    # Interval length of 0.2 seconds or 200,000,000 nanoseconds
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    assert result == expected


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
    assert result == expected


def test_intervals_by_user_and_activity_drops_intervals_with_large_gap():
    data = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 251000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 252000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 299000000, 7.06, 11.35, 0.89),
        (2, 'Jogging', 0, 4.48, 14.18, -2.11),
        (2, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (2, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (2, 'Jogging', 140000000, 5.24, 7.21, -5.56),
        (2, 'Jogging', 200000000, 7.27, 5.79, -6.51),
        (2, 'Jogging', 1310000000, 1.61, 12.07, -2.18),
        (2, 'Jogging', 1351000000, 1.5, 17.69, -3.6),
        (2, 'Jogging', 1399000000, 7.06, 11.35, 0.89),
        (2, 'Jogging', 1450000000, 6.66, 10.0, 11.73),
        (2, 'Jogging', 1500000000, 1.76, 9.85, 1.99),
        (2, 'Jogging', 1549000000, -0.0, -3.214402, 1.334794),
        (2, 'Jogging', 1599999999, -2.7513103, 9.615966, 12.4489975),
        (2, 'Walking', 0, 4.48, 14.18, -2.11),
        (2, 'Walking', 50000000, 3.95, 12.26, -2.68),
        (2, 'Walking', 100000000, 3.95, 12.26, -2.68),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 200000000, 5.24, 7.21, -5.56),
        (2, 'Walking', 250000000, 7.27, 5.79, -6.51),
        (2, 'Walking', 310000000, 1.61, 12.07, -2.18),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 450000000, 6.66, 10.0, 11.73),
        (2, 'Walking', 500000000, 1.76, 9.85, 1.99),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 549000000, -0.0, -3.214402, 1.334794),
        (2, 'Walking', 599999999, -2.7513103, 9.615966, 12.4489975),
    )
    intervals_1 = (
        (
            (1, 'Jogging', 0, 4.48, 14.18, -2.11),
            (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        ),
    )
    intervals_2 = (
        (
            (2, 'Jogging', 0, 4.48, 14.18, -2.11),
            (2, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (2, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (2, 'Jogging', 140000000, 5.24, 7.21, -5.56),
            (2, 'Jogging', 200000000, 7.27, 5.79, -6.51),
        ),
        (
            (2, 'Jogging', 1310000000, 1.61, 12.07, -2.18),
            (2, 'Jogging', 1351000000, 1.5, 17.69, -3.6),
            (2, 'Jogging', 1399000000, 7.06, 11.35, 0.89),
            (2, 'Jogging', 1450000000, 6.66, 10.0, 11.73),
            (2, 'Jogging', 1500000000, 1.76, 9.85, 1.99),
        ),
    )
    intervals_3 = (
        (
            (2, 'Walking', 0, 4.48, 14.18, -2.11),
            (2, 'Walking', 50000000, 3.95, 12.26, -2.68),
            (2, 'Walking', 100000000, 3.95, 12.26, -2.68),
            (2, 'Walking', 200000000, 5.24, 7.21, -5.56),
        ),
        (
            (2, 'Walking', 450000000, 6.66, 10.0, 11.73),
            (2, 'Walking', 500000000, 1.76, 9.85, 1.99),
            (2, 'Walking', 549000000, -0.0, -3.214402, 1.334794),
            (2, 'Walking', 599999999, -2.7513103, 9.615966, 12.4489975),
        )
    )
    expected = {
        (1, "Jogging"): intervals_1,
        (1, "Walking"): (),
        (2, "Jogging"): intervals_2,
        (2, "Walking"): intervals_3,
    }
    result = parse.intervals_by_user_and_activity(
        data,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    assert result == expected


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
    assert result == expected


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
    assert result == expected


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
    assert result == expected


def test_split_into_intervals_starts_new_interval_if_time_decreases():
    given = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 160000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 10, 4.48, 14.18, -2.11),
        (1, 'Jogging', 51000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 110000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 161000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 199000000, 5.24, 7.21, -5.56),
    )
    expected = (
        (
            (1, 'Jogging', 0, 4.48, 14.18, -2.11),
            (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (1, 'Jogging', 160000000, 6.05, 9.72, -1.95),
        ),
        (
            (1, 'Jogging', 10, 4.48, 14.18, -2.11),
            (1, 'Jogging', 51000000, 3.95, 12.26, -2.68),
            (1, 'Jogging', 110000000, 6.05, 9.72, -1.95),
            (1, 'Jogging', 161000000, 6.05, 9.72, -1.95),
            (1, 'Jogging', 199000000, 5.24, 7.21, -5.56),
        ),
    )
    # Interval length of 0.2 seconds or 200,000,000 nanoseconds
    result = parse.split_into_intervals(
        data=given,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    assert result == expected


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


def test_intervals_by_user_and_activity_returns_expected_dictionary():
    data = (
        (1, 'Jogging', 0, 4.48, 14.18, -2.11),
        (1, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (1, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (1, 'Jogging', 200000000, 5.24, 7.21, -5.56),
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
        (2, 'Jogging', 0, 4.48, 14.18, -2.11),
        (2, 'Jogging', 50000000, 3.95, 12.26, -2.68),
        (2, 'Jogging', 100000000, 6.05, 9.72, -1.95),
        (2, 'Jogging', 140000000, 5.24, 7.21, -5.56),
        (2, 'Jogging', 200000000, 7.27, 5.79, -6.51),
        (2, 'Jogging', 1310000000, 1.61, 12.07, -2.18),
        (2, 'Jogging', 1351000000, 1.5, 17.69, -3.6),
        (2, 'Jogging', 1399000000, 7.06, 11.35, 0.89),
        (2, 'Jogging', 1450000000, 6.66, 10.0, 11.73),
        (2, 'Jogging', 1500000000, 1.76, 9.85, 1.99),
        (2, 'Jogging', 1549000000, -0.0, -3.214402, 1.334794),
        (2, 'Jogging', 1599999999, -2.7513103, 9.615966, 12.4489975),
        (2, 'Walking', 0, 4.48, 14.18, -2.11),
        (2, 'Walking', 50000000, 3.95, 12.26, -2.68),
        (2, 'Walking', 100000000, 3.95, 12.26, -2.68),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 200000000, 5.24, 7.21, -5.56),
        (2, 'Walking', 250000000, 7.27, 5.79, -6.51),
        (2, 'Walking', 310000000, 1.61, 12.07, -2.18),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 450000000, 6.66, 10.0, 11.73),
        (2, 'Walking', 500000000, 1.76, 9.85, 1.99),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 0, 0, 0, 0.0),
        (2, 'Walking', 549000000, -0.0, -3.214402, 1.334794),
        (2, 'Walking', 599999999, -2.7513103, 9.615966, 12.4489975),
    )
    intervals_1 = (
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
    )
    intervals_2 = (
        (
            (2, 'Jogging', 0, 4.48, 14.18, -2.11),
            (2, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (2, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (2, 'Jogging', 140000000, 5.24, 7.21, -5.56),
            (2, 'Jogging', 200000000, 7.27, 5.79, -6.51),
        ),
        (
            (2, 'Jogging', 1310000000, 1.61, 12.07, -2.18),
            (2, 'Jogging', 1351000000, 1.5, 17.69, -3.6),
            (2, 'Jogging', 1399000000, 7.06, 11.35, 0.89),
            (2, 'Jogging', 1450000000, 6.66, 10.0, 11.73),
            (2, 'Jogging', 1500000000, 1.76, 9.85, 1.99),
        ),
    )
    intervals_3 = (
        (
            (2, 'Walking', 0, 4.48, 14.18, -2.11),
            (2, 'Walking', 50000000, 3.95, 12.26, -2.68),
            (2, 'Walking', 100000000, 3.95, 12.26, -2.68),
            (2, 'Walking', 200000000, 5.24, 7.21, -5.56),
        ),
        (
            (2, 'Walking', 450000000, 6.66, 10.0, 11.73),
            (2, 'Walking', 500000000, 1.76, 9.85, 1.99),
            (2, 'Walking', 549000000, -0.0, -3.214402, 1.334794),
            (2, 'Walking', 599999999, -2.7513103, 9.615966, 12.4489975),
        )
    )
    expected = {
        (1, "Jogging"): intervals_1,
        (1, "Walking"): (),
        (2, "Jogging"): intervals_2,
        (2, "Walking"): intervals_3,
    }
    result = parse.intervals_by_user_and_activity(
        data,
        interval_duration_in_nanoseconds=200000000,
        maximum_gap_in_nanoseconds=100000000
    )
    assert result == expected


def example_intervals():
    intervals_1 = (
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
    )
    intervals_2 = (
        (
            (2, 'Jogging', 0, 4.48, 14.18, -2.11),
            (2, 'Jogging', 50000000, 3.95, 12.26, -2.68),
            (2, 'Jogging', 100000000, 6.05, 9.72, -1.95),
            (2, 'Jogging', 140000000, 5.24, 7.21, -5.56),
            (2, 'Jogging', 200000000, 7.27, 5.79, -6.51),
        ),
        (
            (2, 'Jogging', 1310000000, 1.61, 12.07, -2.18),
            (2, 'Jogging', 1351000000, 1.5, 17.69, -3.6),
            (2, 'Jogging', 1399000000, 7.06, 11.35, 0.89),
            (2, 'Jogging', 1450000000, 6.66, 10.0, 11.73),
            (2, 'Jogging', 1500000000, 1.76, 9.85, 1.99),
        ),
    )
    intervals_3 = (
        (
            (2, 'Walking', 0, 4.48, 14.18, -2.11),
            (2, 'Walking', 50000000, 3.95, 12.26, -2.68),
            (2, 'Walking', 100000000, 3.95, 12.26, -2.68),
            (2, 'Walking', 200000000, 5.24, 7.21, -5.56),
        ),
        (
            (2, 'Walking', 450000000, 6.66, 10.0, 11.73),
            (2, 'Walking', 500000000, 1.76, 9.85, 1.99),
            (2, 'Walking', 549000000, -0.0, -3.214402, 1.334794),
            (2, 'Walking', 599999999, -2.7513103, 9.615966, 12.4489975),
        )
    )
    return {
        (1, "Jogging"): intervals_1,
        (1, "Walking"): (),
        (2, "Jogging"): intervals_2,
        (2, "Walking"): intervals_3,
    }


def test_count_intervals_returns_expected_dictionary():
    data = example_intervals()
    expected = {
        (1, "Jogging"): 2,
        (1, "Walking"): 0,
        (2, "Jogging"): 2,
        (2, "Walking"): 2,
    }
    result = parse.count_intervals(data)
    assert result == expected


def test_count_intervals_per_activity_returns_expected_dictionary():
    data = example_intervals()
    expected = {
        "Jogging": 4,
        "Walking": 2,
    }
    result = parse.count_intervals_per_activity(data)
    assert result == expected


def test_count_intervals_per_user_returns_expected_dictionary():
    data = example_intervals()
    expected = {
        1: 2,
        2: 4,
    }
    result = parse.count_intervals_per_user(data)
    assert result == expected


def test_relative_time_and_accelerations_returns_expected_tuple_of_ndarrays():
    given = (
        (1, 'Jogging', 250000000, 7.27, 5.79, -6.51),
        (1, 'Jogging', 310000000, 1.61, 12.07, -2.18),
        (1, 'Jogging', 351000000, 1.5, 17.69, -3.6),
        (1, 'Jogging', 399000000, 7.06, 11.35, 0.89),
    )
    times = np.array([0, 60000000, 101000000, 149000000])
    x = np.array([7.27, 1.61, 1.5, 7.06])
    y = np.array([5.79, 12.07, 17.69, 11.35])
    z = np.array([-6.51, -2.18, -3.6, 0.89])
    expected = times, x, y, z
    result = parse.relative_time_and_accelerations(given)
    for i in range(4):
        assert_array_equal(result[i], expected[i])


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
    result = parse.train_test_folds(ids, sequence, n)
    assert result == expected
