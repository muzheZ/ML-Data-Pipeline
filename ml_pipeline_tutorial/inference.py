import argparse
import json
import numpy as np
import pandas as pd

from ml_pipeline.datasets.iris import IrisDataset
from ml_pipeline.models.iris_classifier import IrisClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script.", allow_abbrev=False
    )
    parser.add_argument(
        "-a",
        "--artifact-dir",
        type=str,
        help="path to artifact directory",
        required=True,
    )
    args = parser.parse_args()

    # instantiate model object and load the trained model
    model = IrisClassifier()
    model.load(f"{args.artifact_dir}/model.joblib")

    # inference data
    data = {
        # setosa, versicolor, virginica
        "sepal_length": [5.1, 7.0, 5.8],
        "sepal_width": [3.5, 3.2, 3.3],
        "petal_length": [1.4, 4.7, 6.0],
        "petal_width": [0.2, 1.4, 2.5],
    }

    # convert to DataFrame
    df = pd.DataFrame(data)

    # pre-process and feature engineer data
    dataset = IrisDataset()
    dataset.df = df

    # load mean and std from the pre-processing artifact and use it for
    # pre-processing
    preprocessing_artifact = np.load(f"{args.artifact_dir}/preprocessing.npz")
    dataset.preprocess(
        inference=True,
        mean=preprocessing_artifact["mean"],
        std=preprocessing_artifact["std"],
    )

    features = dataset.feature_engineer(
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )

    # get target label encoding
    with open(f"{args.artifact_dir}/encodings.json", "r") as f:
        encodings = json.loads(f.read())

    # perform inference
    y = model.predict(dataset.df)
    for idx, item_pred in enumerate(y):
        species = [k for k, v in encodings.items() if v == item_pred][0]
        print(f"Row {idx}: {species}")