import os

import config

if __name__ == "__main__":
    for fold in range(10):
        for classifier_name in config.get_classifiers().keys():
            if classifier_name == "MLP":
                continue

            for ablation_variant in ["NRS", "NTS"]:
                command = (
                    f"sbatch slurm.sh run_ablation.py "
                    f"-classifier_name {classifier_name} "
                    f"-ablation_variant {ablation_variant} "
                    f"-fold {fold} -k 4"
                )

                os.system(command)
