from ml_pipeline.datasets.iris import IrisDataset

if __name__ == "__main__":
    print("Instantiating IrisDataset...")
    dataset = IrisDataset("data/iris.data")

    print("Loading the dataset...")
    dataset.load()

    print("Pre-processing the dataset...")
    dataset.preprocess()
    print("Saving pre-processed data to disk...")
    dataset.save("artifacts", "preprocessed")

    print("Feature engineering the dataset...")
    dataset.feature_engineer(
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    )
    print("Saving feature engineered data to disk...")
    dataset.save("artifacts", "feature_engineered")