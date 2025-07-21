from typing import TYPE_CHECKING

from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import pandas as pd


class TrainingMixin:
    def _train_test_split(self, idx) -> "pd.Series":
        idx_train, idx_test = train_test_split(
            idx, test_size=self.training_params.test_split, shuffle=True
        )
        return idx_train, idx_test

    def train(self, X: "pd.DataFrame", y: "pd.Series") -> None:
        idx_train, idx_test = self._train_test_split(X.index)
        _, y_train = self._encode_train_data(None, y.loc[idx_train])
        self.model.fit(X.loc[idx_train], y_train)
        return idx_train, idx_test

    def evaluate(self, X: "pd.DataFrame", y_true: "pd.Series") -> None:
        _, y_true = self._encode_test_data(None, y_true)
        y_pred = self.model.predict(X)
        metrics = self._compute_metrics(y_true, y_pred)
        return metrics