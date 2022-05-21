from typing import Optional, Type, Union

import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedStratifiedKFold

from algorithm import (
    _get_encoding_mask,
    _get_minority_and_majority_class,
    _get_n_same_class_neighbors_vector,
    _get_number_of_unmasked_entries,
    _individual_to_ratio_and_neighborhood_encoding,
    _use_counts_to_resample_dataset,
)
from metrics import bac


def _neighborhood_encoding_to_resampling_counts(
    y: np.ndarray,
    neighbors_vector: np.ndarray,
    max_oversampling_proportion: float,
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

    n_minority = sum(y == minority_class)
    n_majority = sum(y == majority_class)

    n_left_undersampling = n_majority - n_minority

    for n_neighbors, value in neighborhood_encoding["undersampling"].items():
        if value is None:
            counts["undersampling"][n_neighbors] = None
        else:
            n_observations = sum(
                (y == majority_class) & (neighbors_vector == n_neighbors)
            )

            count = int(np.round(value * n_observations))

            if count > n_left_undersampling:
                count = n_left_undersampling

            n_left_undersampling -= count

            counts["undersampling"][n_neighbors] = count

    n_oversampling = n_left_undersampling

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


def _use_individual_to_resample_dataset(
    individual: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    oversampler: str,
    max_oversampling_proportion: float,
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
        max_oversampling_proportion,
        oversampling_ratio,
        neighborhood_encoding,
        minority_class,
        majority_class,
    )

    X_, y_ = _use_counts_to_resample_dataset(
        resampling_counts,
        X,
        y,
        oversampler=oversampler,
        eps=eps,
        neighbors_vector=neighbors_vector,
        minority_class=minority_class,
        majority_class=majority_class,
    )

    return X_, y_


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
        oversampler: str,
        max_oversampling_proportion: float,
        eps: float,
        metric: callable,
        metric_proba: bool,
        neighbors_vector: np.ndarray,
        encoding_mask: dict[str, dict[int, bool]],
    ):
        assert splitting_strategy in ["none", "random"]
        assert oversampler in ["ros", "smote"]

        self.X = X
        self.y = y
        self.splitting_strategy = splitting_strategy
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.estimator = estimator
        self.oversampler = oversampler
        self.max_oversampling_proportion = max_oversampling_proportion
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
            try:
                X_train_, y_train_ = _use_individual_to_resample_dataset(
                    x,
                    X_train,
                    y_train,
                    oversampler=self.oversampler,
                    max_oversampling_proportion=self.max_oversampling_proportion,
                    eps=self.eps,
                    neighbors_vector=neighbors_vector,
                    encoding_mask=self.encoding_mask,
                    minority_class=self.minority_class,
                    majority_class=self.majority_class,
                )
            except ValueError:
                out["F"] = 1.0

                return

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

        out["F"] = -np.mean(scores)


class LNE:
    def __init__(
        self,
        estimator: BaseEstimator,
        k: int = 4,
        oversampler: str = "smote",
        max_oversampling_proportion: float = 5.0,
        splitting_strategy: str = "random",
        n_splits: int = 2,
        n_repeats: int = 3,
        algorithm: Type[Algorithm] = DE,
        algorithm_kwargs: Optional[dict] = None,
        eps: float = 0.1,
        metric: callable = bac,
        metric_proba: bool = False,
        verbose: bool = False,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.estimator = estimator
        self.k = k
        self.oversampler = oversampler
        self.max_oversampling_proportion = max_oversampling_proportion
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
        self.solution = None
        self.oversampling_ratio = None
        self.neighborhood_encoding = None
        self.resampling_counts = None

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
            oversampler=self.oversampler,
            max_oversampling_proportion=self.max_oversampling_proportion,
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

        self.solution = result.X

        (
            self.oversampling_ratio,
            self.neighborhood_encoding,
        ) = _individual_to_ratio_and_neighborhood_encoding(
            self.solution, self.encoding_mask
        )

        self.resampling_counts = _neighborhood_encoding_to_resampling_counts(
            y,
            self.neighbors_vector,
            self.max_oversampling_proportion,
            self.oversampling_ratio,
            self.neighborhood_encoding,
            minority_class,
            majority_class,
        )

        X_, y_ = _use_individual_to_resample_dataset(
            self.solution,
            X,
            y,
            oversampler=self.oversampler,
            max_oversampling_proportion=self.max_oversampling_proportion,
            eps=self.eps,
            neighbors_vector=self.neighbors_vector,
            encoding_mask=self.encoding_mask,
            minority_class=minority_class,
            majority_class=majority_class,
        )

        return X_, y_
