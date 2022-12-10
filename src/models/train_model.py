import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from models.nlp_pipelines import pipelines


class ModelTrainer:
    def __init__(
        self,
        df: pd.DataFrame,
        feature: str,
    ):
        self.X, self.y = self._load_data(df=df, feature=feature)
        self.feature = feature

    def _load_data(self, df: pd.DataFrame, feature: str):
        """
        Load X, y from dataframe
        """
        df = self.remove_unique_target_variable(df=df, column=df["authorId"])
        try:
            X = df[feature]
            y = df["authorId"]
        except KeyError:
            raise
        return X, y

    def train(
        self,
        test_size: float = 0.35,
        random_state: int = 150,
        shuffle=True,
    ):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X=self.X,
            y=self.y,
            stratify=self.y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        return self._run_pipelines(
            X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid
        )

    def _run_pipelines(self, X_train, X_valid, y_train, y_valid) -> pd.DataFrame:
        results: pd.DataFrame = pd.DataFrame()

        for pipeline in pipelines():
            text_clf: Pipeline = pipeline[0]
            clf_name: str = pipeline[1]

            text_clf.fit(X_train, y_train)
            predicted = text_clf.predict(X_valid)
            np.mean(predicted == y_valid)
            test_score: float = text_clf.score(X=X_valid, y=y_valid)

            result = pd.DataFrame([
                {"feature": self.feature, "classifier": clf_name, "score": test_score}
            ])
            results = pd.concat([results, result], ignore_index=True)
        return results

    @staticmethod
    def remove_unique_target_variable(df: pd.DataFrame, column: pd.Series):
        """
        only select authors that occur more than once so we can
        stratify for y in train_test_split
        """
        is_not_unique = column.value_counts() > 1
        df = df[column.isin(is_not_unique[is_not_unique].index)]
        return df
