## Machine learning assignment
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize

## Assignment is to predict Author (missing) in test.json. We use train.json to train.

with open("data/test.json", "r", encoding="utf-8") as (
    test
):  ##open test file and load json
    test_data = json.load(test)
    for i in test_data[0:1]:
        print(i)  # Print first dictionary in list
    test_df = pd.DataFrame.from_dict(test_data)
    LE = LabelEncoder()
    # for feat in test_df.values():
    #   if feat == "authorId":
    #      test_df[feat] = test_df[feat].apply(LabelEncoder().fit_transform)
    test_df.head()


test_df.isnull().sum()

with open("data/train.json", "r", encoding="utf-8") as (
    train
):  ##open train file and load as json
    train_data = json.load(train)
    for i in train_data[0:1]:
        print(i)  # Print first dictionary in list
    train_df = pd.DataFrame.from_dict(train_data)
    LE = LabelEncoder()
    train_df = train_df.apply(LabelEncoder().fit_transform)
    train_df, validation_df = train_test_split(train_df, test_size=0.2)
    train_df.head()

train_df.isnull().sum()

train_df.info()
train_df.duplicated().sum()
train_df["target"] = train_df["authorId"]
train_df.target.value_counts(normalize=True)
train_df.target.value_counts().plot(kind="bar")
plt.title("Value counts of the target variable")
plt.xlabel("Author ID")
plt.xticks(rotation=0)
plt.ylabel("Count")
plt.show()


# PairGrid object with hue
graph = sns.PairGrid(train_df)
# type of graph for non-diagonal(upper part)
graph = graph.map_upper(sns.scatterplot)
# type of graph for non-diagonal(lower part)
graph = graph.map_lower(sns.kdeplot)
# type of graph for diagonal
graph = graph.map_diag(sns.kdeplot, lw=2)
# to show
plt.show()

sns.PairGrid(train_df, hue="authorId")
plt.show()

train_df.head()
corrmat = train_df.corr()
hm = sns.heatmap(
    corrmat,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 10},
    yticklabels=train_df.columns,
    xticklabels=train_df.columns,
    cmap="Spectral_r",
)
plt.show()

input_cols = train_df.columns
target_cols = ["authorId"]

from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(
    train_df[input_cols],
    train_df[target_cols],
    test_size=0.2,
    shuffle=True,
    random_state=42,
)

##Dummy model neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(
    "Using Tensorflow version {} (git version {})".format(
        tf.version.VERSION, tf.version.GIT_VERSION
    )
)

# Dataframe into NumPy array
X_train.to_numpy()
X_train = np.asarray(X_train).astype(np.int)
y_train = np.asarray(y_train).astype(np.int64)
X_validation.to_numpy()
y_validation = np.asarray(y_validation).astype(np.int64)


X = train[features].to_list()

y = train["target()"].to_list()

tf.convert_to_tensor(
    X_train
)  # I think the problem is with the input of strings. The json is a list of dictionaries.
# One of the keys is authorID -> predict that value
X_train.head()


# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5

# Build the model
def basic_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return basic_model


basic_model = build_model()
basic_model.summary()

# Fit baseline model on training set
acc = []
val_acc = []
loss = []
val_loss = []


history = basic_model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=EPOCHS,
    batch_size=4,
    verbose=0,
)

acc.append(history.history["accuracy"])
val_acc.append(history.history["val_accuracy"])
loss.append(history.history["loss"])
val_loss.append(history.history["val_loss"])
