import pandas as pd

from numpy.random import default_rng
from omegaconf import OmegaConf

from ml_pipeline.models.iris_classifier import IrisClassifier

if __name__ == "__main__":
    # generate dummy data
    num_rows = 10
    num_columns = 4
    rng = default_rng()
    X = pd.DataFrame(rng.random(size=(num_rows, num_columns)))
    y = pd.Series(rng.random(size=num_rows))

    # classifier and training parameters
    model_params = OmegaConf.create({"penalty": "l2"})
    training_params = OmegaConf.create({"test_split": 0.3})

    # instantiate model object and train the model
    model = IrisClassifier(model_params, training_params, "artifacts")
    idx_train, idx_test = model.train(X, y)
    # save the trained model
    model.save()