
def read_raw_data(file_path: str) -> str:
    with open(file_path, 'r') as myfile:
        return myfile.read()


def parse_raw_data(data: str) -> List[str]:
    """Split raw data string into time points.

    The raw data file mostly contains one time point per row but there is at least one case of two time points
    in one row.  Semi-colon seems to be a more reliable delimiter than new lines for this file.
    """
    data.replace
