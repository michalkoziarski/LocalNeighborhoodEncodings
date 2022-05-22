import os

if __name__ == "__main__":
    for fold in range(10):
        for classifier_name in ["MLP"]:
            for k in [4]:
                command = (
                    f"sbatch slurm.sh run_proxy.py "
                    f"-classifier_name {classifier_name} "
                    f"-oversampler_name smote "
                    f"-fold {fold} -k {k}"
                )

                os.system(command)
