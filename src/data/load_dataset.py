import pandas as pd


def make_dataset(path: str) -> pd.DataFrame:
    dataset: pd.DataFrame = _load_data_from_json(path=path)
    dataset = _cast_column_types(df=dataset)
    return dataset


def _load_data_from_json(path: str) -> pd.DataFrame:
    return pd.read_json(path)


def _cast_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changes datatype "authorId" from integer to string
    """
    if "authorId" in df.columns:
        df["authorId"] = df["authorId"].astype(str)
    return df
