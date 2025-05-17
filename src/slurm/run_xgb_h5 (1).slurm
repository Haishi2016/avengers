#!/bin/bash
#SBATCH --job-name=xgb-hdf5
#SBATCH --partition=high_mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --account=is789sp25
#SBATCH --output=xgb-hdf5-%j.out
#SBATCH --error=xgb-hdf5-%j.err
#SBATCH --time=0:30:00


module purge
module load GCCcore/9.3.0
module load Python/3.8.2-GCCcore-9.3.0


source ~/is789sp25_user/project/xgb-env/bin/activate

# Start timing
start_time=$(date +%s)
echo "[INFO] Starting training at: $(date)"


python xgb_dask_h5.py

# End timing
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "[INFO] Finished at: $(date)"
echo "[INFO] Time taken: $elapsed seconds"


