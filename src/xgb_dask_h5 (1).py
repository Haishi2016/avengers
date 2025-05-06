import os
import time
import numpy as np
import h5py
import dask.array as da
from xgboost import dask as dxgb
from dask.distributed import Client, LocalCluster
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import multiprocessing

def main():
    print("[INFO] Setting up Dask cluster...")
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)
    print(client)

    # === Load HDF5 Data ===
    file_path = "Greenland.h5"  # Update this path as needed

    print("[INFO] Loading variables from .h5...")
    with h5py.File(file_path, 'r') as f:
        # Extract arrays
        surf_x = f["surf_x"][:].ravel()
        surf_y = f["surf_y"][:].ravel()
        surf_vx = f["surf_vx"][:].ravel()
        surf_vy = f["surf_vy"][:].ravel()
        surf_elv = f["surf_elv"][:].ravel()
        surf_dhdt = f["surf_dhdt"][:].ravel()
        surf_SMB = f["surf_SMB"][:].ravel()
        track_bed = f["track_bed"][2, :].ravel()

    # === Derive velocity magnitude
    print("[INFO] Engineering features...")
    velocity_magnitude = np.sqrt(surf_vx**2 + surf_vy**2)

    # === Flatten all variables (shape: (187459428,))
    print("[INFO] Reshaping...")
    X = np.stack([
        surf_x,
        surf_y,
        surf_vx,
        surf_vy,
        surf_elv,
        surf_dhdt,
        surf_SMB,
        velocity_magnitude
    ], axis=1)

    y = track_bed

    min_len = min(len(X), len(y))
    X=X[:min_len]
    y=y[:min_len]

    # === Clean: Remove NaNs
    print("[INFO] Cleaning...")
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    # === Convert to Dask arrays
    print("[INFO] Converting to Dask...")
    X_dask = da.from_array(X, chunks=(100000, X.shape[1]))
    y_dask = da.from_array(y, chunks=(100000,))

    # === Train XGBoost
    print("[INFO] Training XGBoost model...")
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.1,
        "verbosity": 1
    }

    dtrain = dxgb.DaskDMatrix(client, X_dask, y_dask)

    start = time.time()
    output = dxgb.train(client, params, dtrain, num_boost_round=100)
    booster = output["booster"]
    end = time.time()

    print("[INFO] Training complete.")

    # === Predict and Evaluate
    print("[INFO] Evaluating...")
    y_pred = dxgb.predict(client, booster, X_dask).compute()

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"[RESULT] RMSE: {rmse:.3f}")
    print(f"[RESULT] MAE : {mae:.3f}")
    print(f"[RESULT] R²  : {r2:.3f}")
    print(f"[INFO] Time taken: {end - start:.2f} seconds")

    client.shutdown()
    print("[INFO] Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()



