import argparse
import pathlib
import sys
import time

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging
    import omegaconf

from ml_pipeline import config, dataset_factory, model_factory, utils


def pipeline_task(task_func):
    task_func.is_task = True
    return task_func


class MLPipeline:
    def __init__(
        self: "MLPipeline", project_config_path: str, logger: "logging.Logger"
    ) -> None:
        self.logger = logger

        project_config_path = pathlib.Path(project_config_path)
        self.config = config.Config(
            project_config_path.parent.parent, project_config_path.stem
        )
        self.config.load()

        # build the pipeline by topologically sorting the task DAG
        sorted_tasks = self.topological_sort(self.config.items.project.tasks)
        self.tasks = []
        for task in sorted_tasks:
            try:
                func = getattr(self, task)
            except AttributeError:
                raise Exception(f"'{task}' is not defined in the pipeline.")

            if hasattr(func, "is_task"):
                self.tasks.append(func)
            else:
                raise Exception(f"'{func}' is not a pipeline task.")

        # create a Unix timestamp for the current run
        self.timestamp = int(time.time())

        # create a directory to store training artifacts
        self.artifact_dir = (
            f"artifacts/{self.config.items.project.name}/{self.timestamp}"
        )
        pathlib.Path(self.artifact_dir).mkdir(parents=True)

    def get_dataset(self) -> None:
        self.dataset = dataset_factory.DatasetFactory(
            self.config.items.datasets
        ).get(self.config.items.project.dataset)

    def get_model(self) -> None:
        self.model = model_factory.ModelFactory().get(
            self.config.items.project.model.name,
            self.config.items.project.model.params,
            self.config.items.project.training,
            self.artifact_dir,
            self.logger,
        )

    def topological_sort(self, digraph: "omegaconf.DictConfig"):
        # calculate indegree for all nodes
        indegree = {node: 0 for node in digraph}
        for node in digraph:
            for adjacent_node in digraph[node].next:
                if adjacent_node not in indegree:
                    raise Exception(f"Node '{adjacent_node}' not defined")
                indegree[adjacent_node] += 1

        # get zero-indegree nodes
        zero_indegree_nodes = [node for node in digraph if indegree[node] == 0]

        # sort
        sorted_nodes = []
        while len(zero_indegree_nodes) > 0:
            # add a zero-indegree node to the sorted array
            node = zero_indegree_nodes.pop()
            sorted_nodes.append(node)

            # decrement the indegree of all adjacent nodes
            for adjacent_node in digraph[node].next:
                indegree[adjacent_node] -= 1
                if indegree[adjacent_node] == 0:
                    zero_indegree_nodes.append(adjacent_node)

        if len(sorted_nodes) != len(digraph):
            raise Exception("Tasks do not form a DAG")

        return sorted_nodes

    @pipeline_task
    def load_data(self) -> None:
        self.logger.info("Loading data...")

        self.get_dataset()
        self.dataset.load()
        self.logger.debug(self.dataset.df.head())

        self.logger.info("Loading data done.")

    @pipeline_task
    def preprocess_data(self) -> None:
        self.logger.info("Pre-processing data...")

        self.dataset.preprocess()
        self.logger.debug(self.dataset.df.head())

        # save pre-processed data as an artifact
        self.dataset.save(self.artifact_dir, suffix="preprocessed")

        self.logger.info("Pre-processing data done.")

    @pipeline_task
    def feature_engineer_data(self) -> None:
        self.logger.info("Feature-engineering data...")

        self.config.items.project.features = self.dataset.feature_engineer(
            self.config.items.project.features
        )
        self.logger.debug(self.dataset.df.head())

        # save feature-engineered data as an artifact
        self.dataset.save(self.artifact_dir, suffix="feature_engineered")

        self.logger.info("Feature-engineering data done.")

    @pipeline_task
    def train_model(self) -> None:
        self.logger.info("Training model...")

        # get the model to be trained
        self.get_model()

        # train the model
        self.idx_train, self.idx_test = self.model.train(
            self.dataset.df[self.config.items.project.features],
            self.dataset.df[self.config.items.project.target],
        )

        # save the trained model
        self.model.save()

        self.logger.info("Training model done.")

    @pipeline_task
    def evaluate_model(self) -> None:
        self.logger.info("Evaluating model...")

        self.model.evaluate(
            self.dataset.df[self.config.items.project.features].loc[
                self.idx_test
            ],
            self.dataset.df[self.config.items.project.target].loc[
                self.idx_test
            ],
        )

        self.logger.info("Evaluating model done.")

    @pipeline_task
    def create_report(self) -> None:
        self.logger.info("Creating plots...")

        self.model.create_report()

        self.logger.info("Creating plots done.")

    def run(self) -> None:
        self.logger.info("Commencing pipeline run...")
        self.logger.info(f"artifact directory: {self.artifact_dir}")

        for task in self.tasks:
            task()

        self.logger.info("Pipeline run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Machine learning training pipeline.", allow_abbrev=False
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to project configuration file",
        required=True,
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="run in debug mode"
    )
    args = parser.parse_args()

    logger = utils.Logger("ml-pipeline", debug=args.debug).get()

    try:
        pipeline = MLPipeline(args.config, logger)
        pipeline.run()
    except Exception as error:
        logger.error(error)
        sys.exit(-1)

    print(
        f"Pipeline run complete. See all artifacts in {pipeline.artifact_dir}"
    )