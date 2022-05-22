import os

if __name__ == "__main__":
    for fold in range(10):
        command = (
            f"sbatch slurm.sh run_proxy.py "
            f"-classifier_name MLP "
            f"-oversampler_name smote "
            f"-fold {fold} -k 4"
        )

        os.system(command)
