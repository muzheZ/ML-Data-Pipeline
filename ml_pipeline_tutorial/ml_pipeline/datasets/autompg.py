from typing import List

from ml_pipeline.dataset import Dataset
from ml_pipeline.mixins.csv_mixin import CSVMixin


class AutoMPGDataset(CSVMixin, Dataset):
    """The AutoMPG dataset.

    Source:
        https://archive.ics.uci.edu/dataset/9/auto+mpg

    Data Attributes:
        1. Fuel consumption in miles per gallon
        2. Number of cylinders
        3. Engine displacement in cubic inches
        4. Horsepower
        5. Weight in lbs
        6. Acceleration, in number of seconds to go from 0 to 60 mph
        7. Model year (YY)
        8. Origin
        9. Car name
    """

    def __init__(self, data_path: str) -> None:
        """Instantiates the dataset object.

        Args:
            data_path (str): Path to the CSV data file.
        """
        self.name = "autompg"
        self.data_path = data_path
        self.columns = [
            "mpg",
            "num_cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "year",
            "origin",
            "name",
        ]

    def preprocess(self) -> None:
        """Pre-processes data."""
        # drop rows for which the value of horsepower is unknown
        self.df = self.df[self.df["horsepower"] != "?"]

    def feature_engineer(self, features: List[str]) -> List[str]:
        """Feature-engineer data.

        Args:
            features (List[str]): List of features to be used in training. Some
                of these features may be removed during the feature-engineering
                process.
        Returns:
            List[str]: Updated list of features to be used in training.
        """
        # we won't do any feature engineering, so nothing to add here
        pass