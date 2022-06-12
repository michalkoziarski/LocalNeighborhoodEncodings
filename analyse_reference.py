from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr

import config
import datasets

CLASSIFIERS = ["CART", "KNN", "SVM", "MLP"]
METRICS = ["AUC", "BAC", "G-mean", "F-beta"]
METHOD = "LNE(4;smote)"
RESAMPLERS = list(config.get_reference_resamplers().keys()) + [METHOD]

P_VALUE = 0.05
PATH = Path(__file__).parent / "results.csv"


def test_friedman_shaffer(dictionary):
    df = pd.DataFrame(dictionary)

    columns = df.columns

    pandas2ri.activate()

    importr("scmamp")

    rFriedmanTest = r["friedmanTest"]
    rPostHocTest = r["postHocTest"]

    initial_results = rFriedmanTest(df)
    posthoc_results = rPostHocTest(
        df, test="friedman", correct="shaffer", use_rank=True
    )

    ranks = np.array(posthoc_results[0])[0]
    p_value = initial_results[2][0]
    corrected_p_values = np.array(posthoc_results[2])

    ranks_dict = {col: rank for col, rank in zip(columns, ranks)}
    corrected_p_values_dict = {}

    for outer_col, corr_p_val_vect in zip(columns, corrected_p_values):
        corrected_p_values_dict[outer_col] = {}

        for inner_col, corr_p_val in zip(columns, corr_p_val_vect):
            corrected_p_values_dict[outer_col][inner_col] = corr_p_val

    return ranks_dict, p_value, corrected_p_values_dict


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
