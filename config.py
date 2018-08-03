import os
import numpy as np

class Config:
    def __init__(self):
        self._configs = {}
        self._configs["dataset"] = None
        self._configs["sampling_function"] = "kp_detection"

        # Training Config
        self._configs["display"]           = 5
        self._configs["snapshot"]          = 5000
        self._configs["stepsize"]          = 450000
        self._configs["learning_rate"]     = 0.00025
        self._configs["decay_rate"]        = 10
        self._configs["max_iter"]          = 500000
        self._configs["val_iter"]          = 100
        self._configs["batch_size"]        = 1
        self._configs["snapshot_name"]     = None
        self._configs["prefetch_size"]     = 100
        self._configs["weight_decay"]      = False
        self._configs["weight_decay_rate"] = 1e-5
        self._configs["weight_decay_type"] = "l2"
        self._configs["pretrain"]          = None
        self._configs["opt_algo"]          = "adam"
        self._configs["chunk_sizes"]       = None

        # Directories
        self._configs["data_dir"]   = "./data"
        self._configs["cache_dir"]  = "./cache"
        self._configs["config_dir"] = "./config"
        self._configs["result_dir"] = "./results"

        # Split
        self._configs["train_split"] = "trainval"
        self._configs["val_split"]   = "minival"
        self._configs["test_split"]  = "testdev"

        # Rng
        self._configs["data_rng"] = np.random.RandomState(123)
        self._configs["nnet_rng"] = np.random.RandomState(317)

    @property
    def chunk_sizes(self):
        return self._configs["chunk_sizes"]

    @property
    def train_split(self):
        return self._configs["train_split"]

    @property
    def val_split(self):
        return self._configs["val_split"]

    @property
    def test_split(self):
        return self._configs["test_split"]

    @property
    def full(self):
        return self._configs

    @property
    def sampling_function(self):
        return self._configs["sampling_function"]

    @property
    def data_rng(self):
        return self._configs["data_rng"]

    @property
    def nnet_rng(self):
        return self._configs["nnet_rng"]

    @property
    def opt_algo(self):
        return self._configs["opt_algo"]

    @property
    def weight_decay_type(self):
        return self._configs["weight_decay_type"]

    @property
    def prefetch_size(self):
        return self._configs["prefetch_size"]

    @property
    def pretrain(self):
        return self._configs["pretrain"]

    @property
    def weight_decay_rate(self):
        return self._configs["weight_decay_rate"]

    @property
    def weight_decay(self):
        return self._configs["weight_decay"]

    @property
    def result_dir(self):
        result_dir = os.path.join(self._configs["result_dir"], self.snapshot_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    @property
    def dataset(self):
        return self._configs["dataset"]

    @property
    def snapshot_name(self):
        return self._configs["snapshot_name"]

    @property
    def snapshot_dir(self):
        snapshot_dir = os.path.join(self.cache_dir, "nnet", self.snapshot_name)

        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        return snapshot_dir

    @property
    def snapshot_file(self):
        snapshot_file = os.path.join(self.snapshot_dir, self.snapshot_name + "_{}.pkl")
        return snapshot_file

    @property
    def config_dir(self):
        return self._configs["config_dir"]

    @property
    def batch_size(self):
        return self._configs["batch_size"]

    @property
    def max_iter(self):
        return self._configs["max_iter"]

    @property
    def learning_rate(self):
        return self._configs["learning_rate"]

    @property
    def decay_rate(self):
        return self._configs["decay_rate"]

    @property
    def stepsize(self):
        return self._configs["stepsize"]

    @property
    def snapshot(self):
        return self._configs["snapshot"]

    @property
    def display(self):
        return self._configs["display"]

    @property
    def val_iter(self):
        return self._configs["val_iter"]

    @property
    def data_dir(self):
        return self._configs["data_dir"]

    @property
    def cache_dir(self):
        if not os.path.exists(self._configs["cache_dir"]):
            os.makedirs(self._configs["cache_dir"])
        return self._configs["cache_dir"]

    def update_config(self, new):
        for key in new:
            if key in self._configs:
                self._configs[key] = new[key]

system_configs = Config()
