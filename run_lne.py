import argparse
import logging
import pickle
from collections import Counter

import numpy as np
import pandas as pd

import config
import datasets
from algorithm import LNE


def evaluate_trial(classifier_name, eps, k, fold):
    for path in [config.RESULTS_PATH, config.STATS_PATH]:
        path.mkdir(exist_ok=True, parents=True)

    for dataset_name in datasets.names():
        classifiers = config.get_classifiers()
        criteria = config.get_criteria()

        resampler_name = f"LNE({k};{eps:.2f})"

        trial_name = f"{dataset_name}_{fold}_{classifier_name}_{resampler_name}"
        trial_path = config.RESULTS_PATH / f"{trial_name}.csv"

        if trial_path.exists():
            logging.info(f"Skipping {trial_name} (results already present)...")

            continue

        logging.info(f"Evaluating {trial_name}...")

        rows = []

        for criterion_name, criterion in criteria.items():
            logging.info(f"Evaluating for {criterion_name}...")

            dataset = datasets.load(dataset_name)
            classifier = classifiers[classifier_name]

            (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

            resampler = LNE(
                k=k,
                eps=eps,
                estimator=classifier,
                metric=criterion,
                metric_proba=(criterion_name == "AUC"),
                random_state=config.RANDOM_STATE,
            )

            assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

            minority_class = Counter(y_test).most_common()[-1][0]

            try:
                X_train, y_train = resampler.fit_resample(X_train, y_train)
            except RuntimeError:
                continue

            stats_path = config.STATS_PATH / f"{trial_name}_{criterion_name}.p"
            stats = {
                "encoding_mask": resampler.encoding_mask,
                "neighbors_vector": resampler.neighbors_vector,
                "solution": resampler.solution,
                "oversampling_ratio": resampler.oversampling_ratio,
                "neighborhood_encoding": resampler.neighborhood_encoding,
                "resampling_counts": resampler.resampling_counts,
            }

            with open(stats_path, "wb") as f:
                pickle.dump(stats, f)

            clf = classifier.fit(X_train, y_train)
            predictions = clf.predict(X_test)
            proba = clf.predict_proba(X_test)[:, int(minority_class)]

            scoring_functions = config.get_scoring_functions()

            for scoring_function_name in scoring_functions.keys():
                if scoring_function_name == "AUC":
                    score = scoring_functions[scoring_function_name](y_test, proba)
                else:
                    score = scoring_functions[scoring_function_name](
                        y_test, predictions
                    )

                row = [
                    dataset_name,
                    fold,
                    classifier_name,
                    resampler_name,
                    criterion_name,
                    scoring_function_name,
                    score,
                ]
                rows.append(row)

        columns = [
            "Dataset",
            "Fold",
            "Classifier",
            "Resampler",
            "Criterion",
            "Metric",
            "Score",
        ]

        pd.DataFrame(rows, columns=columns).to_csv(trial_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("-classifier_name", type=str, required=True)
    parser.add_argument("-eps", type=float, required=True)
    parser.add_argument("-fold", type=int, required=True)
    parser.add_argument("-k", type=int, required=True)

    args = parser.parse_args()

    evaluate_trial(args.classifier_name, args.eps, args.k, args.fold)
