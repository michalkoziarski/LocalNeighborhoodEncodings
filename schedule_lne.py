import os

if __name__ == "__main__":
    for fold in range(10):
        for k in [2, 3, 4, 5, 6, 7]:
            command = f"sbatch slurm.sh run_lne.py -fold {fold} -k {k}"

            os.system(command)
