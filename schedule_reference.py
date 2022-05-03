import os

import config

if __name__ == "__main__":
    for fold in range(10):
        for classifier_name in config.get_classifiers().keys():
            command = (
                f"sbatch slurm.sh run_reference.py "
                f"-classifier_name {classifier_name} -fold {fold}"
            )

            os.system(command)
