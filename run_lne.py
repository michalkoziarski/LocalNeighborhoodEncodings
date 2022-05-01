import argparse
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import datasets
import metrics
from algorithm import LNE


def evaluate_trial(k, fold):
    RESULTS_PATH = Path(__file__).parents[0] / "results"
    STATS_PATH = Path(__file__).parents[0] / "stats"
    RANDOM_STATE = 42

    for path in [RESULTS_PATH, STATS_PATH]:
        path.mkdir(exist_ok=True, parents=True)

    for dataset_name in datasets.names():
        classifiers = {
            "CART": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "KNN": KNeighborsClassifier(n_neighbors=1),
            "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            # "MLP": MLPClassifier(random_state=RANDOM_STATE),
        }

        trial_name = f"{dataset_name}_{fold}_LNE_{k}"
        trial_path = RESULTS_PATH / f"{trial_name}.csv"

        if trial_path.exists():
            logging.info(f"Skipping {trial_name} (results already present)...")

            continue

        logging.info(f"Evaluating {trial_name}...")

        rows = []

        for classifier_name in classifiers.keys():
            logging.info(f"Evaluating {classifier_name}...")

            dataset = datasets.load(dataset_name)
            classifier = classifiers[classifier_name]

            (X_train, y_train), (X_test, y_test) = dataset[fold][0], dataset[fold][1]

            resampler = LNE(k=k, estimator=classifier, random_state=RANDOM_STATE)

            assert len(np.unique(y_train)) == len(np.unique(y_test)) == 2

            minority_class = Counter(y_test).most_common()[-1][0]

            try:
                X_train, y_train = resampler.fit_resample(X_train, y_train)
            except RuntimeError:
                continue

            stats_path = (
                STATS_PATH / f"{dataset_name}_{fold}_{classifier_name}_LNE_{k}.p"
            )
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

            scoring_functions = {
                "Precision": metrics.precision,
                "Recall": metrics.recall,
                "AUC": metrics.auc,
                "BAC": metrics.bac,
                "G-mean": metrics.g_mean,
            }

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
                    f"LNE({k})",
                    scoring_function_name,
                    score,
                ]
                rows.append(row)

        columns = ["Dataset", "Fold", "Classifier", "Resampler", "Metric", "Score"]

        pd.DataFrame(rows, columns=columns).to_csv(trial_path, index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("-fold", type=int)
    parser.add_argument("-k", type=int)

    args = parser.parse_args()

    evaluate_trial(args.k, args.fold)
