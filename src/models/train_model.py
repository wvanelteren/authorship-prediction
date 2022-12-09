import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


class ModelTrainer:
    def __init__(
        self,
        df: pd.Dataframe,
        feature: str,
    ):
        """
        Load X, y from dataframe
        """
        self.X, self.y = self._load_data(df=df, feature=feature)

    def _load_data(self, df: pd.DataFrame, feature: str):
        try:
            X = df[feature]
            y = df["authorId"]
        except KeyError:
            raise
        return X, y

    def _train_test_split(self, X, y, test_size, random_state, shuffle):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            stratify=y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        return X_train, X_valid, y_train, y_valid

    def train(
        self,
        test_size: float = 0.35,
        random_state: int = 123,
        shuffle=True,
    ):
        X_train, X_valid, y_train, y_valid = self._train_test_split(
            X=self.X,
            y=self.y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        text_clf = Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                "classifier",
                XGBClassifier(
                    nthread=12,
                    random_state=42,
                    verbosity=1,
                    n_estimators=400,
                    learning_rate=0.1,
                    max_depth=3,
                ),
            ]
        )
        text_clf.fit(X_train, y_train)
        predicted = text_clf.predict(X_valid)
        np.mean(predicted == y_valid)
        test_score: float = text_clf.score(X=X_valid, y=y_valid)
        print(f"Test Accuracy Score: {test_score}")
