import json
from collections import OrderedDict

import pandas as pd

from data.load_dataset import make_dataset
from features.make_features import make_features
from models.model_handler import ModelHandler

train_df: pd.DataFrame = make_dataset(path="data/train.json")
test_df: pd.DataFrame = make_dataset(path="data/test.json")

train_df = make_features(train_df)
test_df = make_features(test_df)

model_handler: ModelHandler = ModelHandler(df=train_df, feature="title")
# results = model_handler.train()
# results.to_csv("results.csv", index=False)
predictions = model_handler.predict(test_df=test_df)
predicted = pd.DataFrame({"paperId": [], "authorId": []})
predicted["paperId"], predicted["authorId"] = test_df["paperId"], predictions

with open("predicted.json", "w") as json_file:
    json.dump(
        predicted.to_dict(orient="records", into=OrderedDict()), json_file, indent=4
    )
