import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


from models.nlp_pipelines import linear_svc_pipeline, pipelines


class ModelHandler:
    def __init__(
        self,
        df: pd.DataFrame,
        feature: str,
    ):
        self.X, self.y = self._load_data(df=df, feature=feature)
        self.df = df
        self.feature = feature

    def _load_data(self, df: pd.DataFrame, feature: str):
        """
        Load X, y from dataframe
        """
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
    ) -> pd.DataFrame:
        """
        Trains various models via train_test_split() function
        Returns DataFrame of accuracy scores for the various models
        """
        try:
            df = self.remove_unique_target_variable(
                df=self.df, column=self.df["authorId"]
            )
            X = df[self.feature]
            y = df["authorId"]
        except KeyError:
            raise
        X_train, X_valid, y_train, y_valid = train_test_split(
            X=X,
            y=y,
            stratify=y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        return self._run_pipelines(
            X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid
        )

    def predict(self, test_df: pd.DataFrame):
        """
        Runs the model with the best scoring classifier.
        In our case this is Linear SVC
        """
        X_train = self.X.to_numpy()
        y_train = self.y
        X_test = test_df[self.feature].to_numpy()
        prediction_clf: Pipeline = linear_svc_pipeline()[0]
        prediction_clf.fit(X_train, y_train)
        return prediction_clf.predict(X_test)

    def gridsearch(
        self,
        test_size: float = 0.35,
        random_state: int = 150,
        shuffle=True,
    ):
        """
        Trains various models via train_test_split() function
        Returns DataFrame of accuracy scores for the various models
        """
        try:
            df = self.remove_unique_target_variable(
                df=self.df, column=self.df["authorId"]
            )
            X = df[self.feature]
            y = df["authorId"]
        except KeyError:
            raise
        X_train, X_valid, y_train, y_valid = train_test_split(
            X=X,
            y=y,
            stratify=y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )
        param_grid = {
            "LinearSVC__multi_class": ["ovr", "crammer_singer"],
            "LinearSVC__penalty": ["l2"],
            "LinearSVC__loss": ["hinge", "squared_hinge"],
            "LinearSVC__fit_intercept": [True, False],
            "LinearSVC__tol": [1e-2],
            "LinearSVC__C": [0.5, 0.75, 0.85, 0.95, 1, 1.25, 1.5],
        }
        text_clf = Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("LinearSVC", LinearSVC()),
            ]
        )
        Gridsearch = RandomizedSearchCV(
            estimator=text_clf,
            n_iter=1000,
            param_distributions=param_grid,
            verbose=1,
            cv=2,
        )
        Gridsearchresult = Gridsearch.fit(X_train, y_train)
        print("optimal score: ", Gridsearchresult.best_score_)
        print("optimal score: ", Gridsearchresult.best_params_)

    def _run_pipelines(self, X_train, X_valid, y_train, y_valid) -> pd.DataFrame:
        results: pd.DataFrame = pd.DataFrame()

        for pipeline in pipelines():
            text_clf: Pipeline = pipeline[0]
            clf_name: str = pipeline[1]

            text_clf.fit(X_train, y_train)
            predicted = text_clf.predict(X_valid)
            np.mean(predicted == y_valid)
            test_score: float = text_clf.score(X=X_valid, y=y_valid)

            result = pd.DataFrame(
                [{"feature": self.feature, "classifier": clf_name, "score": test_score}]
            )
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
