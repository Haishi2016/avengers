from xgboost.spark import SparkXGBRegressor
import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, FloatType, IntegerType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
import panda as pd

features = [
    'surf_x', 'surf_y', 'surf_vx', 'surf_vy', 'surf_elv',
    'surf_dhdt', 'surf_SMB', 'velocity_magnitude'
]
target = 'track_bed_target'

h5_df = pd.read_hdf("your_file.h5", key="data")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: xgboost-spark.py <input_h5_file> <output_dir>", file=sys.stderr)
        sys.exit(-1)

    input_h5_file = sys.argv[1]
    output_dir = sys.argv[2]

    pdf = pd.read_hdf(input_h5_file, key='data')

    spark = SparkSession.builder.appName("XGBoostSpark").getOrCreate()
    df = spark.createDataFrame(pdf)

    df = df.withColumn("velocity_magnitude", col("surf_vx")**2 + col("surf_vy")**2)
    df = df.withColumnRenamed("track_bed_target", "label")
    df = df.dropna(subset=features + ["label"])

    assembler = VectorAssembler(inputCols=features, outputCol="raw_features")    
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    preprocessing_pipeline = Pipeline(stages=[assembler, scaler])
    preprocessed_model = preprocessing_pipeline.fit(df)

    total_rows = df.count()
    frac = 0.2
    subset_df = df.limit(int(total_rows * frac))

    train_df, test_df = subset_df.randomSplit([0.8, 0.2], seed=42)
    train_df = preprocessed_model.transform(train_df)
    test_df = preprocessed_model.transform(test_df)

    # # Initialize the XGBoost regressor
    # xgb_regressor = SparkXGBRegressor(
    #     features_col="features",
    #     label_col="label",
    #     num_workers=2,
    # )

    xgb_regressor = SparkXGBRegressor(
        features_col="features",
        label_col="label",
        num_workers=2,
        max_depth=7,
        n_estimators=350,
        learning_rate=0.25,
        subsample=0.8,
        min_child_weight=0.25,
        random_state=42,
    )

    model = xgb_regressor.fit(train_df)

    predictions = model.transform(test_df)

    # Select just the prediction and true label columns
    output_df = predictions.select("label", "prediction", *features)

    # Write to CSV for later analysis
    output_df.write.mode("overwrite").option("header", True).csv(output_dir)
