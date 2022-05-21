from collections import Counter
from typing import Optional, Type, Union

import numpy as np
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors

from metrics import bac


def _use_individual_to_resample_dataset(
    individual: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    *,
    oversampler: str,
    max_oversampling_proportion: float,
    eps: float,
    minority_class: int,
    majority_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    oversampling_ratio, undersampling_ratio = individual

    n_minority = sum(y == minority_class)
    n_majority = sum(y == majority_class)

    n_oversampling = int(
        np.round(
            (n_majority - n_minority) * oversampling_ratio * max_oversampling_proportion
        )
    )

    n_undersampling = int(np.round(undersampling_ratio * n_majority))

    X_, y_ = [], []
    X_min = X[y == minority_class]

    # oversampling

    nn = NearestNeighbors(n_neighbors=min(6, len(X_min))).fit(X_min)

    indices = np.where(y == minority_class)[0]

    X_.append(X[indices])
    y_.append(y[indices])

    sample_indices = np.random.choice(indices, size=n_oversampling, replace=True)

    if oversampler == "ros":
        samples = X[sample_indices]
        samples += np.random.normal(size=samples.shape, scale=eps)
    elif oversampler == "smote":
        samples = []

        for index, n in Counter(sample_indices).items():
            sample = X[index]
            neighbors = X_min[nn.kneighbors([sample], return_distance=False)[0, 1:]]

            for _ in range(n):
                neighbor = neighbors[np.random.randint(len(neighbors))]
                samples.append(sample + np.random.rand() * (neighbor - sample))

        samples = np.array(samples)
    else:
        raise NotImplementedError

    X_.append(samples)
    y_.append(y[sample_indices])

    # undersampling

    indices = np.where(y == majority_class)[0]

    sample_indices = np.random.choice(
        indices, size=(len(indices) - n_undersampling), replace=False
    )

    X_.append(X[sample_indices])
    y_.append(y[sample_indices])

    return np.concatenate(X_), np.concatenate(y_)


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
        oversampler: str,
        max_oversampling_proportion: float,
        eps: float,
        metric: callable,
        metric_proba: bool,
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

        if splitting_strategy == "none":
            self.folds = [((X, y), (X, y))]
        elif splitting_strategy == "random":
            self.folds = []

            for train_index, test_index in RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats
            ).split(X, y):
                self.folds.append(
                    (
                        (X[train_index], y[train_index]),
                        (X[test_index], y[test_index]),
                    )
                )
        else:
            raise NotImplementedError

        self.n_variables = 2
        self.minority_class, self.majority_class = _get_minority_and_majority_class(y)

        super().__init__(n_var=self.n_variables, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        scores = []

        for (X_train, y_train), (X_test, y_test) in self.folds:
            try:
                X_train_, y_train_ = _use_individual_to_resample_dataset(
                    x,
                    X_train,
                    y_train,
                    oversampler=self.oversampler,
                    max_oversampling_proportion=self.max_oversampling_proportion,
                    eps=self.eps,
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

        self.solution = None

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
        )
        algorithm = self.algorithm(**self.algorithm_kwargs)
        result = minimize(
            problem, algorithm, seed=self.random_state, verbose=self.verbose
        )

        self.solution = result.X

        X_, y_ = _use_individual_to_resample_dataset(
            self.solution,
            X,
            y,
            oversampler=self.oversampler,
            max_oversampling_proportion=self.max_oversampling_proportion,
            eps=self.eps,
            minority_class=minority_class,
            majority_class=majority_class,
        )

        return X_, y_
