import os
import time
import numpy as np
import pandas as pd
from xgboost import dask as dxgb
from dask.distributed import Client, LocalCluster
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import multiprocessing


def main():
    print("[INFO] Setting up Dask cluster...")
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)
    print(client)

    # === Load Data ===
    print("[INFO] Loading dataset...")
    df = pd.read_csv("data_full.csv")

    # Drop NaNs
    df = df.dropna()

    # === Feature Engineering ===
    print("[INFO] Engineering features...")
    df["velocity_magnitude"] = np.sqrt(df["surf_vx"]**2 + df["surf_vy"]**2)

    features = [
        "surf_x", "surf_y", "surf_vx", "surf_vy", "surf_elv", "surf_dhdt",
        "surf_SMB", "velocity_magnitude"
    ]
    target = "track_bed_target"

    X = df[features]
    y = df[target]

    # === Dask Conversion ===
    print("[INFO] Converting to Dask arrays...")
    import dask.array as da
    X_dask = da.from_array(X.to_numpy(), chunks="auto")
    y_dask = da.from_array(y.to_numpy(), chunks="auto")

    # === Train XGBoost ===
    print("[INFO] Training XGBoost model...")
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.1,
        "verbosity": 1
    }

    dtrain = dxgb.DaskDMatrix(client, X_dask, y_dask)

    output = dxgb.train(client, params, dtrain, num_boost_round=100)
    booster = output["booster"]  # XGBoost booster

    print("[INFO] Model trained.")

    # === Prediction & Evaluation ===
    print("[INFO] Making predictions...")
    y_pred = dxgb.predict(client, booster, X_dask).compute()

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"[RESULT] RMSE: {rmse:.3f}")
    print(f"[RESULT] MAE: {mae:.3f}")
    print(f"[RESULT] R²: {r2:.3f}")

    # === Shut down cluster ===
    client.shutdown()
    print("[INFO] Done.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()


