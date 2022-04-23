from collections import Counter
from typing import Optional, Type, Union

import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
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


def _individual_to_ratio_and_neighborhood_encoding(
    individual: np.ndarray, encoding_mask: dict[str, dict[int, bool]]
) -> tuple[float, dict[str, dict[int, Optional[float]]]]:
    """
    Helper for converting a real-valued vector describing an individual to
    oversampling ratio and an easier to interpret and operate on dictionary.
    Uses encoding mask to exclude types of observations not available in the
    particular dataset.
    """
    encoding = {"oversampling": {}, "undersampling": {}}

    n_unmasked_values = _get_number_of_unmasked_entries(encoding_mask)

    if n_unmasked_values + 1 != len(individual):
        raise ValueError(
            f"The number of unmasked values ({n_unmasked_values}) + 1 (ratio) "
            f"does not match the length of the individual ({len(individual)}) "
            f"for the mask = {encoding_mask}."
        )

    oversampling_ratio = individual[0]

    position = 1

    for resampling in encoding_mask.keys():
        for i, is_present in encoding_mask[resampling].items():
            if is_present:
                encoding[resampling][i] = individual[position]
                position += 1
            else:
                encoding[resampling][i] = None

    return oversampling_ratio, encoding


def _neighborhood_encoding_to_resampling_counts(
    y: np.ndarray,
    neighbors_vector: np.ndarray,
    oversampling_ratio: float,
    neighborhood_encoding: dict[str, dict[int, Optional[float]]],
    minority_class: int,
    majority_class: int,
) -> dict[str, dict[int, Optional[int]]]:
    """
    Helper converting neighborhood encoding to a concrete resampling counts
    for a specific dataset. For undersampling, for each encoding key/value pair
    it simply removes the proportion of observations of a given type (key)
    specified by value. For oversampling, it calculates the total number of
    observations that should be generated based on the provided oversampling
    ratio, and afterwards calculates the proportion for each observation type
    based on the relative values.
    """
    neighborhood_encoding = neighborhood_encoding.copy()

    counts = {"oversampling": {}, "undersampling": {}}

    for n_neighbors, value in neighborhood_encoding["undersampling"].items():
        if value is None:
            counts["undersampling"][n_neighbors] = None
        else:
            n_observations = sum(
                (y == majority_class) & (neighbors_vector == n_neighbors)
            )

            counts["undersampling"][n_neighbors] = int(np.round(value * n_observations))

    n_minority = sum(y == minority_class)
    n_majority = sum(y == majority_class)

    n_oversampling = int(np.round((n_majority - n_minority) * oversampling_ratio))

    total = sum(
        value
        for value in neighborhood_encoding["oversampling"].values()
        if value is not None
    )

    for n_neighbors, value in neighborhood_encoding["oversampling"].items():
        if value is None:
            counts["oversampling"][n_neighbors] = None
        else:
            counts["oversampling"][n_neighbors] = int(
                np.round(value / total * n_oversampling)
            )

    return counts


def _use_counts_to_resample_dataset(
    resampling_counts: dict[str, dict[int, Optional[int]]],
    X: np.ndarray,
    y: np.ndarray,
    *,
    eps: float,
    neighbors_vector: np.ndarray,
    minority_class: int,
    majority_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs a combination of random undersampling and random oversampling
    with noise based on the provided resampling counts dictionary.
    """
    X_, y_ = [], []

    for n_neighbors, count in resampling_counts["oversampling"].items():
        indices = np.where((y == minority_class) & (neighbors_vector == n_neighbors))[0]

        X_.append(X[indices])
        y_.append(y[indices])

        if count is not None and count > 0:
            sample_indices = np.random.choice(indices, size=count, replace=True)
            samples = X[sample_indices]
            samples += np.random.normal(size=samples.shape, scale=eps)

            X_.append(samples)
            y_.append(y[sample_indices])

    for n_neighbors, count in resampling_counts["undersampling"].items():
        if count is None:
            continue

        indices = np.where((y == majority_class) & (neighbors_vector == n_neighbors))[0]

        sample_indices = np.random.choice(
            indices, size=(len(indices) - count), replace=False
        )

        X_.append(X[sample_indices])
        y_.append(y[sample_indices])

    return np.concatenate(X_), np.concatenate(y_)


def _use_individual_to_resample_dataset(
    individual: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    eps: float,
    neighbors_vector: np.ndarray,
    encoding_mask: dict[str, dict[int, bool]],
    minority_class: int,
    majority_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combines previous functions to convert an individual to resampled dataset
    in a single step.
    """
    (
        oversampling_ratio,
        neighborhood_encoding,
    ) = _individual_to_ratio_and_neighborhood_encoding(individual, encoding_mask)

    resampling_counts = _neighborhood_encoding_to_resampling_counts(
        y,
        neighbors_vector,
        oversampling_ratio,
        neighborhood_encoding,
        minority_class,
        majority_class,
    )

    X_, y_ = _use_counts_to_resample_dataset(
        resampling_counts,
        X,
        y,
        eps=eps,
        neighbors_vector=neighbors_vector,
        minority_class=minority_class,
        majority_class=majority_class,
    )

    return X_, y_


def _get_minority_and_majority_class(y: np.ndarray) -> tuple[int, int]:
    minority_class = Counter(y).most_common()[1][0]
    majority_class = Counter(y).most_common()[0][0]

    return minority_class, majority_class


class _LNEProblem(ElementwiseProblem):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        splitting_strategy: str,
        n_splits: int,
        n_repeats: int,
        estimator: BaseEstimator,
        eps: float,
        metric: callable,
        metric_proba: bool,
        neighbors_vector: np.ndarray,
        encoding_mask: dict[str, dict[int, bool]],
    ):
        assert splitting_strategy in ["none", "random"]

        self.X = X
        self.y = y
        self.splitting_strategy = splitting_strategy
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.estimator = estimator
        self.eps = eps
        self.metric = metric
        self.metric_proba = metric_proba
        self.neighbors_vector = neighbors_vector
        self.encoding_mask = encoding_mask

        if splitting_strategy == "none":
            self.folds = [((X, y, neighbors_vector), (X, y))]
        elif splitting_strategy == "random":
            self.folds = []

            for train_index, test_index in RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats
            ).split(X, y):
                self.folds.append(
                    (
                        (X[train_index], y[train_index], neighbors_vector[train_index]),
                        (X[test_index], y[test_index]),
                    )
                )
        else:
            raise NotImplementedError

        self.n_variables = _get_number_of_unmasked_entries(encoding_mask) + 1
        self.minority_class, self.majority_class = _get_minority_and_majority_class(y)

        super().__init__(n_var=self.n_variables, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        scores = []

        for (X_train, y_train, neighbors_vector), (X_test, y_test) in self.folds:
            X_train_, y_train_ = _use_individual_to_resample_dataset(
                x,
                X_train,
                y_train,
                eps=self.eps,
                neighbors_vector=neighbors_vector,
                encoding_mask=self.encoding_mask,
                minority_class=self.minority_class,
                majority_class=self.majority_class,
            )

            if len(np.unique(y_train_)) < 2:
                out["F"] = 1.0

                return

            estimator = clone(self.estimator)
            estimator.fit(X_train_, y_train_)

            if self.metric_proba:
                score = self.metric(y_test, estimator.predict_proba(X_test)[:, 1])
            else:
                score = self.metric(y_test, estimator.predict(X_test))

            scores.append(score)

        out["F"] = 1.0 - np.mean(scores)


class LNE:
    def __init__(
        self,
        estimator: BaseEstimator,
        k: int = 5,
        splitting_strategy: str = "random",
        n_splits: int = 2,
        n_repeats: int = 3,
        algorithm: Type[Algorithm] = DE,
        algorithm_kwargs: Optional[dict] = None,
        eps: float = 0.0,
        metric: callable = roc_auc_score,
        metric_proba: bool = True,
        verbose: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.estimator = estimator
        self.k = k
        self.splitting_strategy = splitting_strategy
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.algorithm = algorithm

        if algorithm_kwargs is None:
            self.algorithm_kwargs = {}
        else:
            self.algorithm_kwargs = algorithm_kwargs

        self.eps = eps
        self.metric = metric
        self.metric_proba = metric_proba
        self.verbose = verbose
        self.random_state = random_state

        self.neighbors_vector = None
        self.encoding_mask = None

    def fit_resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(set(y)) != 2:
            raise ValueError(
                f"{self.__name__} only supports two-class classification, "
                f"received number of classes: {len(set(y))}."
            )

        if self.random_state is not None:
            np.random.seed(self.random_state)

        minority_class, majority_class = _get_minority_and_majority_class(y)

        self.neighbors_vector = _get_n_same_class_neighbors_vector(X, y, self.k)
        self.encoding_mask = _get_encoding_mask(
            y, self.k, self.neighbors_vector, minority_class, majority_class
        )

        problem = _LNEProblem(
            X,
            y,
            splitting_strategy=self.splitting_strategy,
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            estimator=self.estimator,
            eps=self.eps,
            metric=self.metric,
            metric_proba=self.metric_proba,
            neighbors_vector=self.neighbors_vector,
            encoding_mask=self.encoding_mask,
        )
        algorithm = self.algorithm(**self.algorithm_kwargs)
        result = minimize(
            problem, algorithm, seed=self.random_state, verbose=self.verbose
        )

        X_, y_ = _use_individual_to_resample_dataset(
            result.X,
            X,
            y,
            eps=self.eps,
            neighbors_vector=self.neighbors_vector,
            encoding_mask=self.encoding_mask,
            minority_class=minority_class,
            majority_class=majority_class,
        )

        return X_, y_
