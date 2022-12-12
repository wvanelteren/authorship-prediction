from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def linear_svc_pipeline():
    """
    NLP pipeline with Linear SVC classifier
    """
    clf_name = "LinearSVC"
    return (
        Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    LinearSVC(
                        tol=1e-3,
                        fit_intercept=True,
                        multi_class="crammer_singer",
                        C=1e-1,
                        dual=True,
                        loss="squared_hinge",
                    ),
                ),
            ]
        ),
        clf_name,
    )


def sgd_pipeline():
    """
    NLP pipeline with SGD pipeline
    """
    clf_name = "SGD"
    return (
        Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    SGDClassifier(
                        loss="hinge",
                        penalty="l2",
                        alpha=0.0001,
                        random_state=43,
                        max_iter=500,
                        n_jobs=-1,
                        tol=1e-6,
                        learning_rate="optimal",
                        eta0=1,
                        shuffle=True,
                        fit_intercept=True,
                    ),
                ),
            ]
        ),
        clf_name,
    )


def xgb_pipeline():
    """
    NLP pipeline with XGBoost classifier
    """
    clf_name = "XGBoost"
    return (
        Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    XGBClassifier(
                        nthread=12,
                        random_state=42,
                        verbosity=1,
                        n_estimators=400,
                        learning_rate=0.1,
                        max_depth=3,
                    ),
                ),
            ]
        ),
        clf_name,
    )


def bagging_pipeline():
    """
    NLP pipeline with XGBoost classifier
    """
    clf_name = "Bagging"
    return (
        Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    BaggingClassifier(
                        DecisionTreeClassifier(splitter="random", max_leaf_nodes=40),
                        n_estimators=50,
                        max_samples=1.0,
                        bootstrap=True,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        clf_name,
    )


def logistic_regression_pipeline():
    """
    NLP pipeline with Linear SVC classifier
    """
    clf_name = "LogisticRegression"
    return (
        Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    LogisticRegression(
                        n_jobs=-1,
                        multi_class="auto",
                        random_state=42,
                        penalty="l2",
                        solver="saga",
                        class_weight="balanced",
                        max_iter=100,
                        warm_start=False,
                        fit_intercept=True,
                    ),
                ),
            ]
        ),
        clf_name,
    )


def pipelines():
    """
    helper function that returns all NLP models and their names
        -> Most pipelines removed to reduced comp time
    """
    return [linear_svc_pipeline(), sgd_pipeline()]
