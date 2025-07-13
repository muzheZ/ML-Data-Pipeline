import matplotlib.pyplot as plt
import seaborn as sns


class ReportingMixin:
    def save_metrics(self) -> None:
        with open(f"{self.artifact_dir}/metrics", "w") as f:
            for key, value in self.metrics.items():
                f.write(f"{key}: {value}\n")

    def plot_confusion_matrix(
        self, annot=True, fmt=".2g", xticklabels="auto", yticklabels="auto"
    ) -> None:
        fig = plt.figure(figsize=(4, 4))
        sns.heatmap(
            self.metrics["cm"],
            square=True,
            annot=annot,
            fmt=fmt,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")

        fig.savefig(f"artifacts/confusion_matrix.png")


# the following is for demonstration purposes
class Model(ReportingMixin):
    def __init__(self):
        self.metrics = {
            "cm": [
                [16, 0, 0],
                [0, 14, 1],
                [0, 1, 13],
            ]
        }


if __name__ == "__main__":
    model = Model()
    model.plot_confusion_matrix()