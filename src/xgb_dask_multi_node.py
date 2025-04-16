import os
import time
import numpy as np
import pandas as pd
from xgboost import dask as dxgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, wait
import multiprocessing
import dask.array as da


def create_slurm_cluster():
    print("[INFO] Creating SLURMCluster...")
    cluster = SLURMCluster(
        queue="high_mem",
        project="is789sp25",
        cores=4,
        processes=1,
        memory="16GB",
        walltime="00:30:00",
        job_extra=[
            "--qos=normal",
            "--partition=high_mem"
        ]
    )
    cluster.scale(jobs=2)  # Request 2 Dask worker jobs from SLURM
    print("[INFO] SLURMCluster created. Scaling to 2 jobs...")
    return Client(cluster)


def main():
    # Set up Dask cluster
    client = create_slurm_cluster()
    client.wait_for_workers(n_workers=2)
    print("[INFO] Workers ready.")
    print(client)

    # === Load dataset ===
    print("[INFO] Loading dataset...")
    df = pd.read_csv("data_full.csv")
    df = df.dropna()

    # === Feature engineering ===
    print("[INFO] Creating velocity magnitude feature...")
    df["velocity_magnitude"] = np.sqrt(df["surf_vx"]**2 + df["surf_vy"]**2)

    features = [
        "surf_x", "surf_y", "surf_vx", "surf_vy", "surf_elv",
        "surf_dhdt", "surf_SMB", "velocity_magnitude"
    ]
    target = "track_bed_target"

    X = df[features].to_numpy()
    y = df[target].to_numpy()

    # === Convert to Dask arrays ===
    print("[INFO] Converting to Dask arrays...")
    X_dask = da.from_array(X, chunks="auto")
    y_dask = da.from_array(y, chunks="auto")

    # === Dask DMatrix & Training ===
    print("[INFO] Starting training...")
    dtrain = dxgb.DaskDMatrix(client, X_dask, y_dask)

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.1,
        "verbosity": 1
    }

    output = dxgb.train(client, params, dtrain, num_boost_round=100)
    booster = output["booster"]
    print("[INFO] Training complete.")

    # === Evaluation ===
    print("[INFO] Making predictions...")
    y_pred = dxgb.predict(client, booster, X_dask).compute()

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"[RESULT] RMSE: {rmse:.3f}")
    print(f"[RESULT] MAE: {mae:.3f}")
    print(f"[RESULT] R²: {r2:.3f}")

    # === Shutdown ===
    client.shutdown()
    print("[INFO] Dask client shutdown. Done.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

