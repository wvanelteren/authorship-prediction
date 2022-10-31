# Authorship Prediction

This project is our entry for the authorship prediction competition that functions as the group assignment for the Machine Learning (Fall) course given at Tilburg University. In this challenge, the task is to predict the lead (first) author of scientific papers, based on their metadata.

## Dataset

The dataset contains two files:
* [train.json](./train.json): the metadata including the lead author of all the papers in the training data.
* [test.json](./test.json): the metadata, excluding the lead author, of the papers in the test data.

Both of these files are in the JSON format. The training records specify the lead author (i.e. the first author on the author list) under the key authorID. For the test data this information is missing, as we have to predict it. The other keys have descriptive names indicating the nature of the information: e.g. title, abstract, paperId, venue (where the paper was published), year (date of publication).

Every lead author in the test set occurs at least once in the training data.

## Evaluation metric

The evaluation metric for this task is the accuracy score. Specifically, the following Python code is used for evaluation of the predictions:

```python
import json
from sklearn.metrics import accuracy_score
import numpy as np
import json

def evaluate(gold_path, pred_path):
    gold = { x['paperId']: x['authorId'] for x in json.load(open(gold_path)) }
    pred = { x['paperId']: x['authorId'] for x in json.load(open(pred_path)) }

    y_true = np.array([ gold[key] for key in gold ])
    y_pred = np.array([ pred[key] for key in gold ])

    return accuracy_score(y_true, y_pred)
```

## Methods

TODO

## Setup

Open terminal (mac) or cmd (windows) in the folder that you want to store the project in and do the following:

1. Clone this repository
```properties
git clone https://github.com/wvanelteren/authorship-prediction.git
```

2. Install pip-tools
```properties
pip install pip-tools
```

3. Compile requirements.txt
```properties
pip-compile
```

4. Install all required packages
```properties
pip-sync
```

5. Setup pre-commit (pre-commit is used as formatter to keep the code in the same "style")
```properties
pre-commit install
pre-commit run --all-files
```

## Git-workflow

Do this every time you change or add something significant

1. Add all files to git

```console
git add .
```

2. Commit files to local repository, include a message between the double quotes ("") what you've changed or added

```console
git commit -m "message here"
```

<p> Only do this if you want to update the github repo with your changes:</p>

3. push the changes in your local repository to the github repo:

```console
git push origin main
```
