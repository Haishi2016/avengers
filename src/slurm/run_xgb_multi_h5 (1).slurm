#!/bin/bash
#SBATCH --job-name=xgb-multinode
#SBATCH --partition=high_mem
#SBATCH --nodes=1
#SBATCH --qos=normal
#SBATCH --account=is789sp25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=xgb-job-%j-%u.out
#SBATCH --error=xgb-job-%j-%u.err

module purge
module load GCCcore/9.3.0
module load Python/3.8.2-GCCcore-9.3.0

# Activate your virtualenv (or use pip install --user)
source ~/is789sp25_user/project/xgb-env/bin/activate

echo "[INFO] Starting training at: $(date)"
start_time=$(date +%s)

python xgb_dask_multi_h5.py

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "[INFO] Finished at: $(date)"
echo "[INFO] Time taken: $elapsed seconds"

