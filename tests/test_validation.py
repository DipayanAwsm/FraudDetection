import pandas as pd
from src.data_validation import validate_columns


def test_validate_columns_success():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert validate_columns(df, ["a", "b"]) is True
