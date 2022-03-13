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
    of the vector contains the number of observations in the k-neighborhood of i-th
    observation having the same class label as the i-th observation.

    Note that it assumes that there will be no observations with exactly the same
    features, but opposite class label: in that case the results might be distorted.
    """

    n_same_class_neighbors_vector = np.empty(len(y), dtype=np.uint)
    nn = NearestNeighbors(n_neighbors=(k + 1)).fit(X)

    for i, (X_i, y_i) in enumerate(zip(X, y)):
        indices = nn.kneighbors([X_i], return_distance=False)[0, 1:]
        n_same_class_neighbors_vector[i] = np.sum(y[indices] == y_i)

    return n_same_class_neighbors_vector


def _get_encoding_mask(
    k: int,
    y: np.ndarray,
    n_same_class_neighbors_vector: np.ndarray,
    minority_class: int,
    majority_class: int,
) -> dict:
    """
    Helper for calculating encoding mask, containing the information about which
    types of observation are present in the dataset for both majority and the
    minority class. Used to exclude unavailable types of observations from the
    final neighborhood encoding.
    """
    mask = {"oversampling": {}, "undersampling": {}}

    for i in range(k + 1):
        for resampling, cls in zip(mask.keys(), [minority_class, majority_class]):
            if i in n_same_class_neighbors_vector[y == cls]:
                mask[resampling][i] = True
            else:
                mask[resampling][i] = False

    return mask


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

        self.encoding_mask = None

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(set(y)) != 2:
            raise ValueError(
                f"{self.__name__} only supports two-class classification, "
                f"received number of classes: {len(set(y))}."
            )

        minority_class = Counter(y).most_common()[1][0]
        majority_class = Counter(y).most_common()[0][0]

        n_same_class_neighbors_vector = _get_n_same_class_neighbors_vector(X, y, self.k)

        self.encoding_mask = _get_encoding_mask(
            self.k, y, n_same_class_neighbors_vector, minority_class, majority_class
        )
