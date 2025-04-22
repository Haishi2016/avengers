#!/bin/bash
#SBATCH --job-name=sparkJob
#SBATCH --partition=high_mem
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --qos=normal
#SBATCH --account=is789sp25
#SBATCH --output=spark-job-%j-%u-slurm.out
#SBATCH --error=spark-job-%j-%u-slurm.out
#SBATCH --time=0:30:00

# ✅ Load and initialize Conda - see README.md for more details
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

$SPARK_HOME/bin/spark-submit \
  --conf spark.pyspark.python=$PYSPARK_PYTHON \
  --conf spark.executorEnv.PYTHONPATH=$PYTHONPATH \
  --conf spark.yarn.appMasterEnv.PYTHONPATH=$PYTHONPATH \
  --packages org.apache.spark:spark-sql_2.12:3.3.0,ml.dmlc:xgboost4j-spark_2.12:1.6.1 \
  /umbc/rs/is789sp25/users/hbai1/data/xgboost-spark.py \
  /umbc/rs/is789sp25/users/hbai1/data/Greenland.h5 \
  /umbc/rs/is789sp25/users/hbai1/data/output_100