from pathlib import Path

import pandas as pd

import config
import datasets


METHOD = "LNE(4;smote)"
RESAMPLERS = list(config.get_reference_resamplers().keys()) + [METHOD]
PATH = Path(__file__).parent / "results"


if __name__ == "__main__":
    df = pd.read_csv(PATH / "results.csv")

    for classifier in df["Classifier"].unique():
        for criterion in df["Criterion"].unique():
            ds = df[
                (df["Classifier"] == classifier)
                & (df["Criterion"] == criterion)
                & (df["Metric"] == criterion)
            ]

            rows = []
            columns = ["Dataset"] + RESAMPLERS[:-1] + ["LNE"]

            for dataset in datasets.names():
                row = [dataset]

                for resampler in RESAMPLERS:
                    dx = ds[(ds["Dataset"] == dataset) & (ds["Resampler"] == resampler)]

                    assert len(dx) == 10

                    row.append(dx["Score"].mean())

                rows.append(row)

            dx = pd.DataFrame(rows, columns=columns)

            avg_row = ["AVERAGE"]
            for resampler in dx.columns[1:]:
                avg_row.append(dx[resampler].mean())
            rows.append(avg_row)

            dx = pd.DataFrame(rows, columns=columns)

            dx.to_csv(PATH / f"{classifier}_{criterion}.csv", index=False)
