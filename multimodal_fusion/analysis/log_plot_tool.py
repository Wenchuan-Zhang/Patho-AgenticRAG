import yaml
from pymongo import MongoClient


class LogUtil:
    def __init__(
        self,
        algo_config: dict,
        mongo_config_path: str = "multimodal_fusion/config.yaml",
        collection_name: str = "mm_rag",
        flush_interval: int = 10,
    ):
        """
        :param mongo_config_path: Path to YAML containing 'mongo_url'
        :param algo_config: Algorithm parameter dict (unique for each run)
        :param collection_name: MongoDB collection
        :param flush_interval: Number of log calls before flushing to MongoDB
        """
        with open(mongo_config_path, "r") as f:
            mongo_config = yaml.safe_load(f)
        mongo_url = mongo_config["mongo"]["url"]
        if not mongo_url:
            raise ValueError("'mongo_url' not found in config file")

        self.algo_config = algo_config
        self.client = MongoClient(mongo_url)
        self.collection = self.client["emo"][collection_name]

        # Remove existing doc with same algo_config
        self.collection.delete_many(algo_config)

        self.logs = {"fitness": [], "test_score": []}
        self._pending = {"fitness": [], "test_score": []}
        self.generation_count = 0
        self.flush_interval = flush_interval

        # Insert fresh doc
        self.collection.insert_one({**algo_config, "logs": self.logs})

    def log_metrics(self, pop, hall_of_fame):
        best_fitness = max(pop, key=lambda ind: ind.fitness.values[0]).fitness.values[0]
        best_test = hall_of_fame.best_test_score

        self.logs["fitness"].append(best_fitness)
        self.logs["test_score"].append(best_test)
        self._pending["fitness"].append(best_fitness)
        self._pending["test_score"].append(best_test)

        self.generation_count += 1

        if self.generation_count % self.flush_interval == 0:
            print("Config", self.algo_config)
            self.collection.update_one(
                self.algo_config,
                {
                    "$push": {
                        "logs.fitness": {"$each": self._pending["fitness"]},
                        "logs.test_score": {"$each": self._pending["test_score"]},
                    }
                },
            )
            self._pending = {metric: [] for metric in self._pending}

    def log_custom_metric(self, metric_name: str, value):
        if metric_name not in self.logs:
            self.logs[metric_name] = []
            self.collection.update_one(
                self.algo_config, {"$set": {f"logs.{metric_name}": []}}
            )
        self.logs[metric_name].append(value)
        self.collection.update_one(
            self.algo_config, {"$push": {f"logs.{metric_name}": value}}
        )


