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