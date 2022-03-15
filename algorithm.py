from collections import Counter
from typing import Optional, Union

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

    neighbors_vector = np.empty(len(y), dtype=np.uint)
    nn = NearestNeighbors(n_neighbors=(k + 1)).fit(X)

    for i, (X_i, y_i) in enumerate(zip(X, y)):
        indices = nn.kneighbors([X_i], return_distance=False)[0, 1:]
        neighbors_vector[i] = np.sum(y[indices] == y_i)

    return neighbors_vector


def _get_encoding_mask(
    y: np.ndarray,
    k: int,
    neighbors_vector: np.ndarray,
    minority_class: int,
    majority_class: int,
) -> dict[str, dict[int, bool]]:
    """
    Helper for calculating encoding mask, containing the information about which
    types of observation are present in the dataset for both majority and the
    minority class. Used to exclude unavailable types of observations from the
    final neighborhood encoding.
    """
    mask = {"oversampling": {}, "undersampling": {}}

    for i in range(k + 1):
        for resampling, cls in zip(mask.keys(), [minority_class, majority_class]):
            if i in neighbors_vector[y == cls]:
                mask[resampling][i] = True
            else:
                mask[resampling][i] = False

    return mask


def _get_number_of_unmasked_entries(encoding_mask: dict[str, dict[int, bool]]) -> int:
    """
    Helper counting the number of nested dictionary values in the encoding
    mask set to true.
    """
    return sum([sum(mask.values()) for mask in encoding_mask.values()])


def _individual_to_neighborhood_encoding(
    individual: np.ndarray, encoding_mask: dict[str, dict[int, bool]]
) -> dict[str, dict[int, Optional[float]]]:
    """
    Helper for converting a real-valued vector describing an individual to an
    easier to interpret and operate on dictionary. Uses encoding mask to exclude
    types of observations not available in the particular dataset.
    """
    encoding = {"oversampling": {}, "undersampling": {}}

    n_unmasked_values = _get_number_of_unmasked_entries(encoding_mask)

    if n_unmasked_values != len(individual):
        raise ValueError(
            f"The number of unmasked values ({n_unmasked_values}) "
            f"does not match the length of the individual ({len(individual)}) "
            f"for the mask = {encoding_mask}."
        )

    position = 0

    for resampling in encoding_mask.keys():
        for i, is_present in encoding_mask[resampling].items():
            if is_present:
                encoding[resampling][i] = individual[position]
                position += 1
            else:
                encoding[resampling][i] = None

    return encoding


def _neighborhood_encoding_to_resampling_counts(
    y: np.ndarray,
    neighbors_vector: np.ndarray,
    neighborhood_encoding: dict[str, dict[int, Optional[float]]],
    minority_class: int,
    majority_class: int,
) -> dict[str, dict[int, Optional[int]]]:
    counts = {"oversampling": {}, "undersampling": {}}

    for n_neighbors, value in neighborhood_encoding["undersampling"].items():
        if value is None:
            counts["undersampling"][n_neighbors] = None
        else:
            n_observations = sum(
                (y == majority_class) & (neighbors_vector == n_neighbors)
            )

            counts["undersampling"][n_neighbors] = int(np.round(value * n_observations))

    # TODO : finish for oversampling

    return counts


class LNE:
    def __init__(
        self,
        estimator: BaseEstimator,
        k: int = 5,
        eps: float = 0.0,
        metric: callable = roc_auc_score,
        ratio: float = 1.0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
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

        neighbors_vector = _get_n_same_class_neighbors_vector(X, y, self.k)

        self.encoding_mask = _get_encoding_mask(
            y, self.k, neighbors_vector, minority_class, majority_class
        )
