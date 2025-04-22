from xgboost import XGBRegressor
import sys
import os
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

features = [
    'surf_x', 'surf_y', 'surf_vx', 'surf_vy', 'surf_elv', 'surf_SMB'
]
target = 'track_bed_target'

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: xgboost-pandas.py <input_h5_file> <output_dir>", file=sys.stderr)
        sys.exit(-1)

    input_h5_file = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    # Read from .h5 using pandas
    with h5py.File(input_h5_file, "r") as h5f:
      feature_lengths = [h5f[k].shape[0] for k in features]
      track_bed_length = h5f["track_bed"].shape[1]
      common_length = min(min(feature_lengths), track_bed_length)
      print("common_length:", common_length)
      data = {
        key: h5f[key][()].flatten()[:common_length]
        for key in features
      }
      vx = data["surf_vx"]
      vy = data["surf_vy"]
      data["velocity_magnitude"] = np.sqrt(vx**2+vy**2)
      track_bed = h5f["track_bed"][()]
      if track_bed.ndim == 2 and track_bed.shape[1] >= 2:
        data[target] = track_bed[1, :common_length]
      else:
        raise ValueError("track_bed must be a 2D array with at least 2 columns")
    pdf = pd.DataFrame(data)
    final_features = features + ["velocity_magnitude"]

    # Limit to 20%
    pdf = pdf.sample(frac=0.2, random_state=42)

    for key in final_features + [target]:
      print(f"{key} NaNs: {np.isnan(data[key]).sum()}")
    X = pdf[final_features].values
    y = pdf[target].values

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train with XGBoost
    model = XGBRegressor(
        max_depth=7,
        n_estimators=350,
        learning_rate=0.25,
        subsample=0.8,
        min_child_weight=0.25,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Save prediction results
    output_df = pd.DataFrame(X_test, columns=final_features)
    output_df["label"] = y_test
    output_df["prediction"] = y_pred
    output_df.to_csv(os.path.join(output_dir, "xgboost_predictions.csv"), index=False)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
      f.write(f"MSE: {mse:.4f}\n")
      f.write(f"MAE: {mae:.4f}\n")
      f.write(f"R^2 score: {r2:.4f}\n")