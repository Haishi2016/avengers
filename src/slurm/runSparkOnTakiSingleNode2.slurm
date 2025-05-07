#!/bin/bash
#SBATCH --job-name=sparkJob
#SBATCH --partition=high_mem
#SBATCH --nodes=2
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --account=is789sp25
#SBATCH --output=spark-job-%j-%u-slurm.out
#SBATCH --error=spark-job-%j-%u-slurm.out
#SBATCH --time=0:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

# ✅ Load and initialize Conda
module load Anaconda3/2021.05
eval "$(conda shell.bash hook)"
conda activate sparkxgb

# ✅ Confirm correct Python environment
echo "🐍 Python: $(which python)"
python -c "import xgboost; print('✅ XGBoost is available!')"

# ✅ Run Spark job
SPARK_HOME=/umbc/rs/is789sp25/common/software/spark-3.5.0-bin-hadoop3
PYSPARK_PYTHON=/home/hbai1/.conda/envs/sparkxgb/bin/python
PYTHONPATH=/home/hbai1/.conda/envs/sparkxgb/lib/python3.8/site-packages

srun --nodes=1 --ntasks=1 --cpus-per-task=4 \
$SPARK_HOME/bin/spark-submit \
  --master local[4] \
  --conf spark.pyspark.python=$PYSPARK_PYTHON \
  --conf spark.executorEnv.PYTHONPATH=$PYTHONPATH \
  --conf spark.yarn.appMasterEnv.PYTHONPATH=$PYTHONPATH \
  --conf spark.executor.cores=8 \
  --conf spark.executor.instances=1 \
  --packages org.apache.spark:spark-sql_2.12:3.3.0,ml.dmlc:xgboost4j-spark_2.12:1.6.1 \
  /umbc/rs/is789sp25/users/hbai1/data/xgboost-spark-native.py \
  /umbc/rs/is789sp25/users/hbai1/data/Greenland.h5 \
  /umbc/rs/is789sp25/users/hbai1/data/output_100_f2

