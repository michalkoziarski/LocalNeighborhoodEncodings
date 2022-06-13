from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ranksums

import config
import datasets

CLASSIFIERS = config.get_classifiers().keys()
METRICS = config.get_criteria().keys()
RESAMPLERS = config.get_reference_resamplers().keys()
METHOD = "LNE(4;smote)"

P_VALUE = 0.10
FIGURES_PATH = Path(__file__).parent / "figures"


def get_win_loss_tie_df():
    df = pd.read_csv("results.csv")

    rows = []

    for classifier in CLASSIFIERS:
        for metric in METRICS:
            ds = df[
                (df["Classifier"] == classifier)
                & (df["Metric"] == metric)
                & (df["Criterion"] == metric)
            ]

            for resampler in RESAMPLERS:
                wins, ties, losses = 0, 0, 0

                for dataset in datasets.names():
                    lne = ds[(ds["Dataset"] == dataset) & (ds["Resampler"] == METHOD)][
                        "Score"
                    ].values
                    ref = ds[
                        (ds["Dataset"] == dataset) & (ds["Resampler"] == resampler)
                    ]["Score"].values

                    assert len(lne) == 10
                    assert len(ref) == 10

                    statistic, p_value = ranksums(lne, ref)

                    if p_value <= P_VALUE:
                        if statistic > 0:
                            wins += 1
                        else:
                            losses += 1
                    else:
                        ties += 1

                rows.append([classifier, metric, resampler, wins, ties, losses])

    return pd.DataFrame(
        rows, columns=["Classifier", "Metric", "Resampler", "Wins", "Ties", "Losses"]
    )


def visualize(df):
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    df["Losses"] = df["Wins"] + df["Ties"] + df["Losses"]
    df["Ties"] = df["Wins"] + df["Ties"]

    sns.set_color_codes("bright")

    g = sns.FacetGrid(
        df,
        col="Metric",
        row="Classifier",
        row_order=CLASSIFIERS,
        col_order=METRICS,
        margin_titles=True,
        despine=False,
    )

    g.map(
        sns.barplot,
        "Losses",
        "Resampler",
        order=RESAMPLERS,
        color=sns.color_palette()[3],
    )
    g.map(
        sns.barplot,
        "Ties",
        "Resampler",
        order=RESAMPLERS,
        color="y",
    )
    g.map(
        sns.barplot,
        "Wins",
        "Resampler",
        order=RESAMPLERS,
        color=sns.color_palette()[2],
    )

    g.set(ylabel="", xlabel="")

    plt.savefig(FIGURES_PATH / "win_loss_tie.pdf", bbox_inches="tight")


if __name__ == "__main__":
    df = get_win_loss_tie_df()

    visualize(df)
