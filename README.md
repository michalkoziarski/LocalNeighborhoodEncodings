# Local Neighborhood Encodings for imbalanced data classification

Local Neighborhood Encodings is an evolutionary data resampling strategy combining over- and undersampling, that selects resampling ratios for each type of observation (with the type determined based on the number of neighbors from the same class closest to a given object).

## Project structure

Most important files are as follows:
- `algorithm.py`: implementation of the proposed method, in particular the `LNE` class contains `imblearn`-style resampler.
- `run_*.py`: execute single trial of the experiment based on the command line arguments passed.
- `schedule_*.py`: schedule a separate Slurm job for every possible combination of command line arguments (sufficient to reproduce the results).
- `analyse_*.py`: scripts containing analysis of the results presented in the paper.

## Results

Complete results in a raw format were provided in the `results.csv` file. Additionally, `results` directory contains separate `{CLASSIFIER}_{METRIC}.csv` tables, with results across all datasets.
