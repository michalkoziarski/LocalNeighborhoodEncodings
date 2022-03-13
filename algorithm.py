from collections import Counter
from typing import Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors


def _get_n_same_class_neighbors_vector(
    X: np.ndarray, y: np.ndarray, k: int
) -> np.ndarray:
    """
    Helper function returning a vector of same-class neighbor counts. i-th element
    of the vector contains the number of observations in the k-neighborhood having
    the same class label as the i-th observation.

    Note that it assumes that there will be no observations with exactly the same
    features, but opposite class label: in that case the results might be distorted.
    """

    n_same_class_neighbors_vector = np.empty(len(y), dtype=np.uint)
    nn = NearestNeighbors(n_neighbors=(k + 1)).fit(X)

    for i, (X_i, y_i) in enumerate(zip(X, y)):
        indices = nn.kneighbors([X_i], return_distance=False)[0, 1:]
        n_same_class_neighbors_vector[i] = np.sum(y[indices] == y_i)

    return n_same_class_neighbors_vector


class LNE:
    def __init__(
        self,
        estimator: BaseEstimator,
        k: int = 5,
        eps: float = 0.0,
        metric: callable = roc_auc_score,
        ratio: float = 1.0,
        random_state=Union[int, np.random.RandomState],
    ):
        self.estimator = estimator
        self.k = k
        self.eps = eps
        self.metric = metric
        self.ratio = ratio
        self.random_state = random_state

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(set(y)) != 2:
            raise ValueError(
                f"{self.__name__} only supports two-class classification, "
                f"received number of classes: {len(set(y))}."
            )

        majority_class = Counter(y).most_common()[0][0]
        minority_class = Counter(y).most_common()[1][0]

        n_same_class_neighbors_vector = _get_n_same_class_neighbors_vector(X, y, self.k)
