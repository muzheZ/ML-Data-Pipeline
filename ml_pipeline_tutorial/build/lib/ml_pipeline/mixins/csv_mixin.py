import pandas as pd


class CSVMixin:
    """Mixin for loading and saving CSV files in a Dataset."""

    def load(self) -> None:
        """Loads data into a data frame.

        Raises:
            FileNotFoundError: File not found.
            PermissionError: Insufficient permissions to read file.
            IsADirectoryError: Project config path points to a directory.
        """
        self.df = pd.read_csv(self.data_path, names=self.columns)

    def save(self, artefact_dir: str, suffix: str = "") -> None:
        """Saves data to a CSV file.

        Args:
            artefact_dir (str): Output directory.
            suffix (str): File name suffix.

        Raises:
            PermissionError: Insufficient permissions to write file to path.
        """
        if suffix:
            suffix = f"_{suffix}"

        self.df.to_csv(
            f"{artefact_dir}/{self.name}{suffix}.csv",
            float_format="%.6f",
            index=False,
        )