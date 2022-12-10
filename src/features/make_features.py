import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")
stopwords = set(stopwords.words("english"))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # df = _create_cleaned_abstract(df=df)
    # df = _create_lemmed_abstract(df=df)
    # df = _create_cleaned_title(df=df)
    # df = _create_lemmed_title(df=df)
    df = _create_numerical_features_from_abstract(df=df)
    df = _create_features_as_text(df=df)
    # df = _create_features_as_text_clean(df=df)
    return df


def _create_features_as_text(df: pd.DataFrame) -> pd.DataFrame:
    df["featuresAsText"] = (
        df["venue"].astype(str)
        + " "
        + df["abstract"]
        + " "
        + df["title"]
        + " "
        + df["num_unique_words"].astype(str)
        + " "
        + df["num_punctuations"].astype(str)
        + " "
        + df["mean_word_len"].astype(str)
        + " "
        + df["num_stopwords"].astype(str)
        + " "
        + df["num_words"].astype(str)
    )
    return df


def _create_features_as_text_clean(df: pd.DataFrame) -> pd.DataFrame:
    df["featuresAsTextClean"] = (
        df["year"].astype(str)
        + " "
        + df["venue"].astype(str)
        + " "
        + df["abstractCleanLem"]
        + " "
        + df["titleCleanLem"]
        + " "
        + df["num_chars"].astype(str)
        + " "
        + df["num_unique_words"].astype(str)
        + " "
        + df["num_punctuations"].astype(str)
        + " "
        + df["mean_word_len"].astype(str)
        + " "
        + df["num_stopwords"].astype(str)
        + " "
        + df["num_words"].astype(str)
    )
    return df


def _create_cleaned_abstract(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["abstractClean"] = df["abstract"].str.replace("[^[a-zA-Z0-9-]*$]", " ")
        df["abstractClean"] = df["abstractClean"].str.lower()
        df["abstractClean"] = df["abstractClean"].str.strip()
        df["abstractClean"] = df["abstractClean"].str.split()
        df["abstractClean"] = df["abstractClean"].apply(
            lambda x: " ".join([word for word in x if word not in stopwords])
        )
    except KeyError:
        print("Column Abstract does not exist")
    return df


def _create_lemmed_abstract(df: pd.DataFrame) -> pd.DataFrame:
    lemmer = nltk.WordNetLemmatizer()
    try:
        df["abstractCleanLem"] = df["abstractClean"].str.strip()
        df["abstractCleanLem"] = df["abstractCleanLem"].apply(
            lambda x: " ".join([lemmer.lemmatize(word) for word in x])
        )
    except KeyError:
        print("Column Abstract does not exist")
    return df


def _create_cleaned_title(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["titleClean"] = df["title"].str.replace("[^[a-zA-Z0-9-]*$]", " ")
        df["titleClean"] = df["titleClean"].str.lower()
        df["titleClean"] = df["titleClean"].str.strip()
        df["titleClean"] = df["titleClean"].str.split()
        df["titleClean"] = df["titleClean"].apply(
            lambda x: " ".join([word for word in x if word not in stopwords])
        )
    except KeyError:
        print("Column Title does not exist")
    return df


def _create_lemmed_title(df: pd.DataFrame) -> pd.DataFrame:
    lemmer = nltk.WordNetLemmatizer()
    try:
        df["titleCleanLem"] = df["titleClean"].str.strip()
        df["titleCleanLem"] = df["titleCleanLem"].apply(
            lambda x: " ".join([lemmer.lemmatize(word) for word in x])
        )
    except KeyError:
        print("Column Title does not exist")
    return df


def _create_numerical_features_from_abstract(df: pd.DataFrame) -> pd.DataFrame:
    try:
        # Number of words in the text
        df["num_words"] = df["abstract"].apply(lambda x: len(str(x).split()))

        # Number of unique words in the text
        df["num_unique_words"] = df["abstract"].apply(
            lambda x: len(set(str(x).split()))
        )

        # Number of characters in the text
        df["num_chars"] = df["abstract"].apply(lambda x: len(str(x)))

        # Number of stopwords in the text
        df["num_stopwords"] = df["abstract"].apply(
            lambda x: len([w for w in str(x).lower().split() if w in stopwords])
        )

        # Number of punctuations in the text
        df["num_punctuations"] = df["abstract"].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation])
        )

        # Number of title case words in the text
        df["num_words_upper"] = df["abstract"].apply(
            lambda x: len([w for w in str(x).split() if w.isupper()])
        )

        # Number of title case words in the text
        df["num_words_title"] = df["abstract"].apply(
            lambda x: len([w for w in str(x).split() if w.istitle()])
        )

        # Average length of the words in the text
        df["mean_word_len"] = df["abstract"].apply(
            lambda x: np.mean([len(w) for w in str(x).split()])
        )
    except KeyError:
        print("Column Abstract does not exist")
    return df
