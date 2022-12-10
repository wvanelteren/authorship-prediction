import pandas as pd

from data.load_dataset import make_dataset
from features.make_features import make_features
from models.train_model import ModelTrainer

train_df: pd.DataFrame = make_dataset(path="data/train.json")
test_df: pd.DataFrame = make_dataset(path="data/test.json")

train_df = make_features(train_df)
# test_df = make_features(test_df)

model_trainer: ModelTrainer = ModelTrainer(df=train_df, feature="title")
results = model_trainer.train()
results.to_csv("results.csv", index=False)
