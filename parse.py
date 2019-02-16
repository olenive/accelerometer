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


def timepoint_is_valid(timepoint: Tuple[int, str, int, float, float, float]) -> bool:
    """Function used to filter out data rows that appear to be corrupted."""
    if timepoint[2:] == (0, 0, 0, 0):
        return False
    else:
        return True


def next_valid_timepoint(data: Iterable[Tuple[int, str, int, float, float, float]],
                         starting_index: int) -> Optional[Tuple[int, str, int, float, float, float]]:
    """Return the next valid timepoint after the start index."""
    for timepoint in data[starting_index + 1:]:
        if timepoint_is_valid(timepoint):
            return timepoint


def split_into_intervals(data: Iterable[Tuple[int, str, int, float, float, float]],
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
        if len(set(ids)) != 1:
            raise ValueError("Expecting one unique id but found: {}".format(set(ids)))
    if check_activity:
        activities = extract_activity_set(data)
        if len(set(activities)) != 1:
            raise ValueError("Expecting one unique activity but found: {}".format(set(activities)))
    # noinspection PyTypeChecker
    if len(data) < 2:
        return ()
    time_in_interval = 0
    interval = []
    out = []
    for measurement in data:
        if not timepoint_is_valid(measurement):
            continue
        if len(interval) == 0:  # First valid measurement in current interval.
            interval = [measurement]
            continue
        # Calculate time gap between current and previous timepoint.
        time_gap = measurement[2] - interval[-1][2]
        time_in_interval += time_gap
        if time_gap <= 0:
            raise ValueError("Expecting time to increase but found {} and {}".format(
                measurement[2], interval[-1][2]
            ))
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
        print(interval)
    # Check if final interval should be stored.
    if len(interval) != 0:
        if interval_duration_in_nanoseconds - time_in_interval <= maximum_gap_in_nanoseconds:
            print("\n C:\n", interval)
            out.append(tuple(interval))
    return tuple(out)
