import os

RESAMPLERS = [
    "SMOTE",
    "polynom-fit-SMOTE",
    "Lee",
    "SMOBD",
    "G-SMOTE",
    "LVQ-SMOTE",
    "Assembled-SMOTE",
    "SMOTE-TomekLinks",
]


if __name__ == "__main__":
    for fold in range(10):
        for resampler_name in RESAMPLERS:
            command = f"sbatch slurm.sh run_reference.py -fold {fold} -resampler_name {resampler_name}"

            os.system(command)
