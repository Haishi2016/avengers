# avengers

Submit Spark job:

```bash
spark-submit xgboost-spark.py <input csv file> <Spark output folder>
# example
# spark-submit xgboost-spark.py /home/hbai/projects/go/src/github.com/Haishi2016/avengers/data_full.csv /home/hbai/projects/go/src/github.com/Haishi2016/avengers/output

```

Plot Spark result:
```bash
python plot.py
```

To build up Conda enviornment for running XGBoost on Spark

```bash
module load Anaconda3/2021.05
conda create -n sparkxgb python=3.9 numpy scipy -c conda-forge -y
conda activate sparkxgb
pip install --no-cache-dir --force-reinstall "xgboost==1.6.1"
conda install -c conda-forge numpy=1.24 pandas=1.5.3 pytables=3.7.0 --force-reinstall
conda install -c conda-forge h5py -y
conda install scikit-learn -y
```