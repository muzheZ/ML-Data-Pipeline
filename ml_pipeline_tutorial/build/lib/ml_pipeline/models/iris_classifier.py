import json

from typing import TYPE_CHECKING, Dict, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import load, dump

if TYPE_CHECKING:
    import logging
    import pandas as pd

    from omegaconf import DictConfig

from ml_pipeline.mixins.reporting_mixin import ReportingMixin
from ml_pipeline.mixins.training_mixin import TrainingMixin
from ml_pipeline.model import Model


class IrisClassifier(TrainingMixin, Model, ReportingMixin):
    def __init__(
        self,
        model_params: "DictConfig",
        training_params: "DictConfig",
        artifact_dir: str,
        logger: "logging.Logger" = None,
    ) -> None:
        self.model = LogisticRegression(**model_params)
        self.training_params = training_params
        self.artifact_dir = artifact_dir
        self.logger = logger

    def load(self, model_path: str) -> None:
        self.model = load(model_path)

    def _encode_train_data(
        self, X: "pd.DataFrame" = None, y: "pd.Series" = None
    ) -> Tuple["pd.DataFrame", "pd.Series"]:
        # we are not encoding X because it is not needed for our dataset

        le = LabelEncoder()
        y = le.fit_transform(y)

        # build the encoding dictionary, converting numpy.int64 values to
        # python ints
        self.encodings = dict(
            zip(le.classes_, [int(i) for i in le.transform(le.classes_)])
        )
        self.logger.debug(self.encodings)

        filename = f"{self.artifact_dir}/encodings.json"
        with open(filename, "w") as f:
            json.dump(self.encodings, f)
        self.logger.debug(f"Saved {filename}.")

        return X, y

    def _encode_test_data(
        self, X: "pd.DataFrame" = None, y: "pd.Series" = None
    ) -> Tuple["pd.DataFrame", "pd.Series"]:
        y = y.map(self.encodings)
        return X, y

    def _compute_metrics(
        self, y_true: "pd.Series", y_pred: "pd.Series"
    ) -> Dict:
        self.metrics = {}
        self.metrics["accuracy"] = accuracy_score(y_true, y_pred)
        self.metrics["cm"] = confusion_matrix(y_true, y_pred)

    def create_report(self) -> None:
        self.save_metrics()
        self.plot_confusion_matrix(
            xticklabels=self.encodings.keys(),
            yticklabels=self.encodings.keys(),
        )

    def save(self) -> None:
        filename = f"{self.artifact_dir}/model.joblib"
        dump(self.model, filename)
        self.logger.debug(f"Saved {filename}.")

    def predict(self, X: "pd.DataFrame") -> int:
        return self.model.predict(X)