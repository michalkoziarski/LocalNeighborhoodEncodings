import os

import config

if __name__ == "__main__":
    for fold in range(10):
        for classifier_name in config.get_classifiers().keys():
            for k in [3, 4, 5, 6]:
                command = (
                    f"sbatch slurm.sh run_lne.py "
                    f"-classifier_name {classifier_name} "
                    f"-oversampler_name smote "
                    f"-fold {fold} -k {k} -eps {0.0}"
                )

                os.system(command)

                for eps in [0.0, 0.01, 0.05, 0.1]:
                    command = (
                        f"sbatch slurm.sh run_lne.py "
                        f"-classifier_name {classifier_name} "
                        f"-oversampler_name ros "
                        f"-fold {fold} -k {k} -eps {eps}"
                    )

                    os.system(command)
