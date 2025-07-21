from typing import TYPE_CHECKING, Dict, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load, dump

if TYPE_CHECKING:
    import logging
    import pandas as pd

    from omegaconf import DictConfig

from ml_pipeline.mixins.reporting_mixin import ReportingMixin
from ml_pipeline.mixins.training_mixin import TrainingMixin
from ml_pipeline.model import Model


class AutoMPGRegressor(TrainingMixin, Model, ReportingMixin):
    def __init__(
        self,
        model_params: "DictConfig",
        training_params: "DictConfig",
        artifact_dir: str,
        logger: "logging.Logger" = None,
    ) -> None:
        self.model = LinearRegression(**model_params)
        self.training_params = training_params
        self.artifact_dir = artifact_dir
        self.logger = logger

    def load(self, model_path: str) -> None:
        self.model = load(model_path)

    def _encode_train_data(
        self, X: "pd.DataFrame" = None, y: "pd.Series" = None
    ) -> Tuple["pd.DataFrame", "pd.Series"]:
        # in this example, we don't do any encoding
        return X, y

    def _encode_test_data(
        self, X: "pd.DataFrame" = None, y: "pd.Series" = None
    ) -> Tuple["pd.DataFrame", "pd.Series"]:
        # in this example, we don't do any encoding
        return X, y

    def _compute_metrics(
        self, y_true: "pd.Series", y_pred: "pd.Series"
    ) -> Dict:
        self.metrics = {}
        self.metrics["mean_squared_error"] = mean_squared_error(y_true, y_pred)
        self.metrics["r2_score"] = r2_score(y_true, y_pred)

    def create_report(self) -> None:
        self.save_metrics()

    def save(self) -> None:
        filename = f"{self.artifact_dir}/model.joblib"
        dump(self.model, filename)
        self.logger.debug(f"Saved {filename}.")

    def predict(self, X: "pd.DataFrame") -> int:
        return self.model.predict(X)