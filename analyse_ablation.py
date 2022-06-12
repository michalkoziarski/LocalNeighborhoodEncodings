from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import datasets
from analyse_reference import test_friedman_shaffer

CLASSIFIERS = ["CART", "KNN", "SVM"]
METRICS = ["AUC", "BAC", "G-mean", "F-beta"]
METHOD = "LNE(4;smote)"
RESAMPLERS = ["LNE_NRS(4;smote)", "LNE_NTS(4;smote)", METHOD]

P_VALUE = 0.05
PATH = Path(__file__).parent / "results.csv"


def load_final_dict():
    df = pd.read_csv(PATH)
    df = df[df["Resampler"].isin(RESAMPLERS)]

    measurements = OrderedDict()

    for resampler in RESAMPLERS:
        ds = df[df["Resampler"] == resampler]
        measurements[resampler] = []

        for dataset in datasets.names():
            for classifier in CLASSIFIERS:
                for metric in METRICS:
                    scores = ds[
                        (ds["Dataset"] == dataset)
                        & (ds["Classifier"] == classifier)
                        & (ds["Metric"] == metric)
                        & (ds["Criterion"] == metric)
                    ]["Score"]

                    assert len(scores) == 10, len(scores)

                    measurements[resampler].append(np.mean(scores))

    return measurements


if __name__ == "__main__":
    print(" & ".join([""] + RESAMPLERS) + " \\\\")
    print("\\midrule")

    d = load_final_dict()
    ranks, _, corrected_p_values = test_friedman_shaffer(d)

    row = "mean and std"

    for resampler in RESAMPLERS:
        row += f" & {np.mean(d[resampler]):.3f} ± {np.std(d[resampler]):.3f}"

    print(row + " \\\\")

    row = "avg. rank"

    for resampler in RESAMPLERS:
        row += f" & {np.mean(ranks[resampler]):.2f}"

    print(row + " \\\\")

    row = "$p$-value"

    for resampler in RESAMPLERS:
        if resampler == METHOD:
            row += " & n/a"
        else:
            row += f" & {corrected_p_values[METHOD][resampler]}"

    print(row + " \\\\")
