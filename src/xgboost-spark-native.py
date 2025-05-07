from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from xgboost.spark import SparkXGBRegressor
import h5py
import pandas as pd
import numpy as np
import os
import sys

features = ['surf_x', 'surf_y', 'surf_vx', 'surf_vy', 'surf_elv', 'surf_SMB']
target = 'track_bed_target'

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: xgboost-sparkdf.py <input_h5_file> <output_dir>", file=sys.stderr)
        sys.exit(-1)

    input_h5_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(input_h5_file, "r") as h5f:
        feature_lengths = [h5f[k].shape[0] for k in features]
        track_bed_length = h5f["track_bed"].shape[1]
        common_length = min(min(feature_lengths), track_bed_length)
        data = {
            key: h5f[key][()].flatten()[:common_length]
            for key in features
        }
        vx = data["surf_vx"]
        vy = data["surf_vy"]
        data["velocity_magnitude"] = np.sqrt(vx**2 + vy**2)
        track_bed = h5f["track_bed"][()]
        if track_bed.ndim == 2 and track_bed.shape[1] >= 2:
            data[target] = track_bed[1, :common_length]
        else:
            raise ValueError("track_bed must be a 2D array with at least 2 columns")
        pdf = pd.DataFrame(data)

    final_features = features + ['velocity_magnitude']

    spark = SparkSession.builder.appName("SparkXGBRegressorExample").getOrCreate()
    sdf = spark.createDataFrame(pdf)
    sdf = sdf.dropna(subset=final_features + [target])

    assembler = VectorAssembler(inputCols=final_features, outputCol="features")
    sdf = assembler.transform(sdf).select("features", col(target).alias("label"))

    train_df, test_df = sdf.randomSplit([0.8, 0.2], seed=42)
    
    train_df = train_df.repartition(32)
    available_cores = spark.sparkContext.defaultParallelism

    xgb = SparkXGBRegressor(
        num_workers=available_cores,               # Number of parallel tasks
        max_depth=7,
        n_estimators=350,
        learning_rate=0.25,
        subsample=0.8,
        min_child_weight=0.25,
        missing=np.nan
    )

    model = xgb.fit(train_df)

    pred_df = model.transform(test_df)
    pred_df.select("label", "prediction").toPandas().to_csv(
        os.path.join(output_dir, "xgboost_spark_predictions.csv"), index=False
    )

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    pred_pd = pred_df.select("label", "prediction").toPandas()
    y_true = pred_pd["label"]
    y_pred = pred_pd["prediction"]
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR^2: {r2:.4f}\n")

    spark.stop()

