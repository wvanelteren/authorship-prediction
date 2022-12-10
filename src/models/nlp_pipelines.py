from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def linear_svc_pipeline():
    clf_name = "LinearSVC"
    return (
        Pipeline(
            [
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                (
                    "svm_clf",
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


def pipelines():
    return [linear_svc_pipeline(), sgd_pipeline()]
