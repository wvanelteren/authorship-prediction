import pandas as pd

COL_NAMES_TO_DROP: list[str] = ["authorName"]


def make_dataset(path: str) -> pd.DataFrame:
    dataset: pd.DataFrame = _load_data_from_json(path=path)
    dataset = _drop_unnecesarry_colums(df=dataset, col_names=COL_NAMES_TO_DROP)
    return dataset


def _load_data_from_json(path: str) -> pd.DataFrame:
    return pd.read_json(path)


def _drop_unnecesarry_colums(
    df: pd.DataFrame, col_names: list[str]
) -> pd.DataFrame:
    for col_name in col_names:
        try:
            df.drop(columns=[col_name])
        except KeyError:
            print(f"column name {col_name} does not exist")
            raise
    return df


def _cast_column_types():
    raise NotImplementedError
