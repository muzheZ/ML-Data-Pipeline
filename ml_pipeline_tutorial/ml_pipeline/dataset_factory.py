from ml_pipeline.datasets import iris, autompg


class DatasetFactory:
    def __init__(self, dataset_config):
        self.datasets = {
            "iris": {"class": iris.IrisDataset},
            "autompg": {"class": autompg.AutoMPGDataset},
        }
        # get data paths for the registered datasets
        for dataset, config in dataset_config.items():
            if dataset in self.datasets:
                self.datasets[dataset]["path"] = config.path

    def get(self, name: str):
        return self.datasets[name]["class"](
            "data/" + self.datasets[name]["path"]
        )