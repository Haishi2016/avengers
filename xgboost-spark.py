from xgboost.spark import SparkXGBRegressor
import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, FloatType, IntegerType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline

features = [
    'surf_x', 'surf_y', 'surf_vx', 'surf_vy', 'surf_elv',
    'surf_dhdt', 'surf_SMB', 'velocity_magnitude'
]
target = 'track_bed_target'


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: xgboost-spark.py <input_dir> <output_dir>", file=sys.stderr)
        sys.exit(-1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Create a Spark session
    spark = SparkSession.builder.appName("XGBoostSpark").getOrCreate()

    # Define the schema for the DataFrame
    schema = StructType() \
        .add("index", IntegerType()) \
        .add("surf_x", FloatType()) \
        .add("surf_y", FloatType()) \
        .add("surf_vx", FloatType()) \
        .add("surf_vy", FloatType()) \
        .add("surf_elv", FloatType()) \
        .add("surf_dhdt", FloatType()) \
        .add("surf_SMB", FloatType()) \
        .add("track_bed_x", FloatType()) \
        .add("track_bed_y", FloatType()) \
        .add("track_bed_target", FloatType())

    df = spark.read.csv(input_dir, schema=schema, header=True)
    df = df.withColumn("velocity_magnitude", col("surf_vx")**2 + col("surf_vy")**2)

    df = df.withColumnRenamed("track_bed_target", "label")
    df = df.dropna(subset=features + ["label"])
    assembler = VectorAssembler(inputCols=features, outputCol="raw_features")
    
    # Step 2: Standardize features
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)

    # Step 3: Split the dataset
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Step 4: Build preprocessing pipeline (Assembler + Scaler)
    preprocessing_pipeline = Pipeline(stages=[assembler, scaler])
    preprocessed_model = preprocessing_pipeline.fit(df)

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
