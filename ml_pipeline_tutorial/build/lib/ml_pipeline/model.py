from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from abc import ABC, abstractmethod


class Model(ABC):
    _model: None  # TODO: model type?

    @property
    def model(self):  # TODO: return type?
        return self._model

    @model.setter
    def model(self, value) -> None:
        self._model = value

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def train(self, X: "pd.DataFrame", y: "pd.Series") -> None:
        pass

    @abstractmethod
    def evaluate(self) -> None:
        pass

    @abstractmethod
    def create_report(self, artifact_path: str) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass