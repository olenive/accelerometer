import numpy as np
from typing import Tuple, Iterable, Dict, Set, Any, Optional


def file_to_string(file_path: str) -> str:
    with open(file_path, 'r') as myfile:
        return myfile.read()


def raw_data_string_to_timepoint_strings(data: str) -> Tuple[str]:
    """Split raw data string into strings for each time points.

    The raw data file mostly contains one time point per row but there is at least one case of two time points
    in one row.  Semi-colon seems to be a more reliable delimiter than new lines for this file.
    """
    sans_newlines = data.replace('\n', '')
    lines = sans_newlines.split(';')
    # Remove empty lines.
    out = []
    for line in lines:
        if line != '':
            out.append(line)
    return tuple(out)


def timepoint_strings_to_timepoint_tuples(data: Iterable[str]) -> Iterable[Tuple[int, str, int, float, float, float]]:
    def _tuple_of_types(row: str) -> Tuple[int, str, int, float, float, float]:
        x = tuple(v.lstrip().rstrip() for v in row.split(','))  # Remove trailing and leading whitespace.
        return int(x[0]), x[1], int(x[2]), float(x[3]), float(x[4]), float(x[5])

    return tuple(_tuple_of_types(i) for i in data)


def extract_user_set(data: Iterable[Tuple[int, str, int, float, float, float]]) -> Set[int]:
    return set(x[0] for x in data)


def extract_activity_set(data: Iterable[Tuple[int, str, int, float, float, float]]) -> Set[str]:
    return set(x[1] for x in data)


def select_matching_measurements(
        data: Iterable[Tuple[int, str, int, float, float, float]],
        column: int,
        value: Any,
) -> Tuple[Tuple[int, str, int, float, float, float]]:
    """Select time points where a given column matches a value."""
    out = []
    for row in data:
        if row[column] == value:
            out.append(row)
    return tuple(out)


def measurements_by_user(data: Iterable[Tuple[int, str, int, float, float, float]]
                         ) -> Dict[int, Tuple[Tuple[int, str, int, float, float, float]]]:
    """Create a dictionary of user ids to timepoint data."""
    users = extract_user_set(data)
    out = dict()
    for user in users:
        out[user] = select_matching_measurements(data, column=0, value=user)
    return out


def measurements_by_activity(data: Iterable[Tuple[int, str, int, float, float, float]]
                             ) -> Dict[str, Tuple[Tuple[int, str, int, float, float, float]]]:
    """Create a dictionary of activities to timepoint data."""
    activities = extract_activity_set(data)
    out = dict()
    for activity in activities:
        out[activity] = select_matching_measurements(data, column=1, value=activity)
    return out


def measurements_by_user_and_activity(data: Iterable[Tuple[int, str, int, float, float, float]]
                                      ) -> Dict[Tuple[int, str], Tuple[Tuple[int, str, int, float, float, float]]]:
    """Create dictionary mapping user id and activity pairs to relevant timepoint data."""
    users = extract_user_set(data)
    activities = extract_activity_set(data)
    out = dict()
    for user in users:
        user_measurements = select_matching_measurements(data, column=0, value=user)
        for activity in activities:
            out[(user, activity)] = select_matching_measurements(user_measurements, column=1, value=activity)
    return out


def measurement_is_valid(timepoint: Tuple[int, str, int, float, float, float]) -> bool:
    """Function used to filter out data rows that appear to be corrupted."""
    if timepoint[2:] == (0, 0, 0, 0):
        return False
    else:
        return True


def next_valid_timepoint(data: Iterable[Tuple[int, str, int, float, float, float]],
                         starting_index: int) -> Optional[Tuple[int, str, int, float, float, float]]:
    """Return the next valid timepoint after the start index."""
    for timepoint in data[starting_index + 1:]:
        if measurement_is_valid(timepoint):
            return timepoint


def split_into_intervals(
    data: Iterable[Tuple[int, str, int, float, float, float]],
    interval_duration_in_nanoseconds: int,
    maximum_gap_in_nanoseconds: int,
    check_id=True,
    check_activity=True,
) -> Iterable[Tuple[Tuple[int, str, int, float, float, float]]]:
    """Extract intervals of fixed duration from a single series of measurements.

    Ignore measurements that have all zeros for time and acceleration values.
    """
    if check_id:
        ids = extract_user_set(data)
        if len(set(ids)) > 1:
            raise ValueError("Expecting zero or one unique ids but found: {}".format(set(ids)))
    if check_activity:
        activities = extract_activity_set(data)
        if len(set(activities)) > 1:
            raise ValueError("Expecting zero or one unique activities but found: {}".format(set(activities)))
    # noinspection PyTypeChecker
    if len(data) < 2:
        return ()
    time_in_interval = 0
    interval = []
    out = []
    previous_measurement = None
    for measurement in data:
        # Skip invalid measurements.
        if not measurement_is_valid(measurement):
            previous_measurement = measurement
            continue
        # Handle first valid measurement in current interval.
        if len(interval) == 0:
            interval = [measurement]
            previous_measurement = measurement
            continue
        # Ignore repeated time points even if the first measurement is not valid.
        if previous_measurement is not None and previous_measurement[2] == measurement[2]:
            continue
        # Calculate time gap between current and previous timepoint.
        time_gap = measurement[2] - interval[-1][2]

        # Handle time decreasing - indicating the start of a new measurement period.
        if time_gap < 0:
            # Reset interval because a step back in time indicates the start of a new measurement period.
            if interval_duration_in_nanoseconds - time_in_interval < maximum_gap_in_nanoseconds:
                out.append(tuple(interval))
            interval = [measurement]
            time_in_interval = 0
            continue

        # Handle time increasing
        time_in_interval += time_gap
        if time_in_interval <= interval_duration_in_nanoseconds and time_gap > maximum_gap_in_nanoseconds:
            # Reset interval if time gap is too big and is not at end of interval.
            interval = [measurement]
            time_in_interval = 0
        elif time_in_interval > interval_duration_in_nanoseconds:
            # Measurement is past the end of the interval.
            out.append(tuple(interval))
            interval = [measurement]
            time_in_interval = 0
        else:
            interval.append(measurement)
        previous_measurement = measurement
        # Check just in case
        if time_gap <= 0:
            for datum in data:
                print(datum)
            raise ValueError("Expecting time to increase but found: \n{}\n{}".format(
                interval[-1], measurement
            ))
    # Check if final interval should be stored.
    if len(interval) != 0:
        if interval_duration_in_nanoseconds - time_in_interval <= maximum_gap_in_nanoseconds:
            out.append(tuple(interval))
    return tuple(out)


def intervals_by_user_and_activity(
    data: Iterable[Tuple[int, str, int, float, float, float]],
    interval_duration_in_nanoseconds: int,
    maximum_gap_in_nanoseconds: int,
    check_id=True,
    check_activity=True,
) -> Dict[Tuple[int, str], Iterable[Tuple[Tuple[int, str, int, float, float, float]]]]:
    """Create a dictionary mapping user id and activity to measurement intervals of specified duration."""
    out = dict()
    for key, series in measurements_by_user_and_activity(data).items():
        out[key] = split_into_intervals(series, interval_duration_in_nanoseconds, maximum_gap_in_nanoseconds,
                                        check_id=check_id, check_activity=check_activity)
    return out


def count_intervals(intervals: Dict[Tuple[int, str], Iterable[Tuple[Tuple[int, str, int, float, float, float]]]]
                    ) -> Dict[Tuple[int, str], int]:
    out = dict()
    for key, value in intervals.items():
        # noinspection PyTypeChecker
        out[key] = len(value)
    return out


def count_intervals_per_activity(
        intervals: Dict[Tuple[int, str], Iterable[Tuple[Tuple[int, str, int, float, float, float]]]]
) -> Dict[Tuple[int, str], int]:
    activities = extract_activity_set(intervals)
    out = {x: 0 for x in activities}
    for key, value in intervals.items():
        for activity in activities:
            if activity in key:
                # noinspection PyTypeChecker
                out[activity] += len(value)
    return out


def count_intervals_per_user(
        intervals: Dict[Tuple[int, str], Iterable[Tuple[Tuple[int, str, int, float, float, float]]]]
) -> Dict[Tuple[int, str], int]:
    users = extract_user_set(intervals)
    out = {x: 0 for x in users}
    for key, value in intervals.items():
        for user in users:
            if user in key:
                # noinspection PyTypeChecker
                out[user] += len(value)
    return out


def relative_time_and_accelerations(measurements: Iterable[Tuple[int, str, int, float, float, float]]
                                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_times = np.array([v[2] for v in measurements])
    t = raw_times - np.min(raw_times)
    x = np.array([v[3] for v in measurements])
    y = np.array([v[4] for v in measurements])
    z = np.array([v[5] for v in measurements])
    return t, x, y, z


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
