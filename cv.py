from collections import Counter
from itertools import product

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


class ResamplingCV:
    def __init__(
        self,
        algorithm,
        classifier,
        metric,
        metric_proba,
        n_splits=2,
        n_repeats=3,
        seed=None,
        **kwargs
    ):
        self.algorithm = algorithm
        self.classifier = classifier
        self.metric = metric
        self.metric_proba = metric_proba
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed
        self.kwargs = kwargs

    def fit_resample(self, X, y):
        best_score = -np.inf
        best_parameters = None

        parameter_combinations = list(
            (dict(zip(self.kwargs, x)) for x in product(*self.kwargs.values()))
        )

        if len(parameter_combinations) == 1:
            return self.algorithm(**parameter_combinations[0]).fit_resample(X, y)

        minority_class = Counter(y).most_common()[-1][0]

        for parameters in parameter_combinations:
            scores = []

            skf = RepeatedStratifiedKFold(
                n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.seed
            )

            for train_idx, test_idx in skf.split(X, y):
                try:
                    X_train, y_train = self.algorithm(**parameters).fit_resample(
                        X[train_idx], y[train_idx]
                    )
                except (ValueError, RuntimeError) as e:
                    scores.append(-np.inf)

                    break
                else:
                    if len(np.unique(y_train)) < 2:
                        scores.append(-np.inf)

                        break

                    classifier = self.classifier.fit(X_train, y_train)

                    if self.metric_proba:
                        proba = classifier.predict_proba(X[test_idx])[
                            :, int(minority_class)
                        ]
                        scores.append(self.metric(y[test_idx], proba))
                    else:
                        predictions = classifier.predict(X[test_idx])
                        scores.append(self.metric(y[test_idx], predictions))

            score = np.mean(scores)

            if score > best_score:
                best_score = score
                best_parameters = parameters

        if best_parameters is None:
            best_parameters = parameter_combinations[0]

        return self.algorithm(**best_parameters).fit_resample(X, y)
