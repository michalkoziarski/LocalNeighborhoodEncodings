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
                    f"-fold {fold} -k {k}"
                )

                os.system(command)
