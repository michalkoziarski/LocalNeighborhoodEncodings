from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import config
import datasets
from analyse_reference import test_friedman_shaffer

CLASSIFIERS = ["MLP"]
METRICS = ["AUC", "BAC", "G-mean", "F-beta"]
METHOD = "LNE_proxy(4;smote)"
RESAMPLERS = list(config.get_reference_resamplers().keys()) + [METHOD]
P_VALUE = 0.05
PATH = Path(__file__).parent / "results.csv"


def load_final_dict(classifier, metric):
    df = pd.read_csv(PATH)
    df = df[
        (df["Classifier"] == classifier)
        & (df["Metric"] == metric)
        & (df["Criterion"] == metric)
    ]

    measurements = OrderedDict()

    for resampler in RESAMPLERS:
        measurements[resampler] = []

        for dataset in datasets.names():
            scores = df[(df["Resampler"] == resampler) & (df["Dataset"] == dataset)][
                "Score"
            ]

            assert len(scores) == 10, len(scores)

            measurements[resampler].append(np.mean(scores))

    return measurements


if __name__ == "__main__":
    print(" & ".join(["", "Metric"] + RESAMPLERS) + " \\\\")
    print("\\midrule")

    for classifier in CLASSIFIERS:
        for metric in METRICS:
            if metric == METRICS[0]:
                start = "\\multirow{%d}{*}{%s}" % (len(METRICS), classifier)
            else:
                start = ""

            d = load_final_dict(classifier, metric)
            ranks, _, corrected_p_values = test_friedman_shaffer(d)

            row = [start, metric]

            best_rank = sorted(set(ranks.values()))[0]
            second_best_rank = sorted(set(ranks.values()))[1]

            for resampler in RESAMPLERS:
                rank = ranks[resampler]
                col = "%.2f" % np.round(rank, 2)

                if rank == best_rank:
                    col = "\\textbf{%s}" % col

                if corrected_p_values[METHOD][resampler] <= P_VALUE:
                    if rank < ranks[METHOD]:
                        col = "%s \\textsubscript{--}" % col
                    else:
                        col = "%s \\textsubscript{+}" % col

                row.append(col)

            print(" & ".join(row) + " \\\\")

        if classifier != CLASSIFIERS[-1]:
            print("\\midrule")
