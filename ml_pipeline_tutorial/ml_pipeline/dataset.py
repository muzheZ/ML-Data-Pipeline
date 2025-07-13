from abc import ABC, abstractmethod

import pandas as pd


class Dataset(ABC):
    _name: str
    _df: pd.DataFrame

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value) -> None:
        self._name = value

    @property
    def df(self) -> "pd.DataFrame":
        return self._df

    @df.setter
    def df(self, value) -> None:
        self._df = value

    @abstractmethod
    def load(self) -> None:
        """Implemented in a mixin."""
        pass

    @abstractmethod
    def preprocess(self) -> None:
        pass

    @abstractmethod
    def feature_engineer(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        """Implemented in a mixin."""
        pass