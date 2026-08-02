from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.datasets import make_blobs

from modules.experimental.shape_library import T_YEARS, YEARS, instance_prototypes
from modules.experimental.shapes import GenConfig, make_dataset_from_prototypes


class SyntheticDataGenerator(ABC):
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.ground_truth_labels = None
        self.n_features = None

    @abstractmethod
    def generate(self) -> tuple[pd.DataFrame, np.ndarray]:
        pass


class BlobsSyntheticDataGenerator(SyntheticDataGenerator):
    def generate(self) -> tuple[DataFrame, np.ndarray]:
        x_values, ground_truth_values = make_blobs(**self.kwargs)
        self.ground_truth_labels = ground_truth_values
        self.n_features =  self.kwargs["n_features"]
        return pd.DataFrame(x_values), ground_truth_values


class TimeSeriesSyntheticDataGenerator(SyntheticDataGenerator):
    """Labeled synthetic time series drawn from positioned shape instances.

    Expected kwargs: ``instances`` (list of ``ShapeInstance``) plus the
    ``GenConfig`` fields ``n_per_cluster``, ``sigma``, ``znorm``,
    ``amplitude_jitter``, ``amp_range``, and ``seed``.

    ``BaseClustering.optimal_k_analysis`` mutates these kwargs in place per
    (seed, k); this class absorbs that instead of changing the sweep:
    ``random_state``, when present, overrides ``seed`` -- each injected seed
    is a fresh noise draw over the same fixed prototypes -- and ``centers``
    is ignored, because the ground-truth k is fixed by the instance list.
    Regenerating per candidate k therefore yields identical data at every k.
    """

    _CFG_FIELDS = ("n_per_cluster", "sigma", "znorm", "amplitude_jitter", "amp_range")

    def generate(self) -> tuple[DataFrame, np.ndarray]:
        cfg = GenConfig(T=T_YEARS, **{key: self.kwargs[key]
                                      for key in self._CFG_FIELDS if key in self.kwargs})
        seed = self.kwargs.get("random_state", self.kwargs.get("seed"))
        rng = np.random.default_rng(seed)
        protos = instance_prototypes(self.kwargs["instances"])
        x_values, ground_truth_values = make_dataset_from_prototypes(protos, cfg, rng)
        self.ground_truth_labels = ground_truth_values
        self.n_features = T_YEARS
        return pd.DataFrame(x_values, columns=YEARS), ground_truth_values
