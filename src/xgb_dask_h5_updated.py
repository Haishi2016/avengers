import os
import time
import numpy as np
import h5py
import dask.array as da
from xgboost import dask as dxgb
from dask.distributed import Client, LocalCluster
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import multiprocessing

def main():
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = Client(cluster)
    print(client)

    print("[INFO] Loading .h5 file...")
    file_path = "Greenland.h5"
    with h5py.File(file_path, 'r') as f:
        surf_x = f["surf_x"][:].ravel()
        surf_y = f["surf_y"][:].ravel()
        surf_vx = f["surf_vx"][:].ravel()
        surf_vy = f["surf_vy"][:].ravel()
        surf_elv = f["surf_elv"][:].ravel()
        surf_dhdt = f["surf_dhdt"][:].ravel()
        surf_SMB = f["surf_SMB"][:].ravel()
        track_bed = f["track_bed"][1, :].ravel()  # MATCH SPARK

    velocity_magnitude = np.sqrt(surf_vx**2 + surf_vy**2)

    X = np.stack([
        surf_x, surf_y, surf_vx, surf_vy,
        surf_elv, surf_dhdt, surf_SMB, velocity_magnitude
    ], axis=1)
    y = track_bed

    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]


    np.random.seed(42)
    idx = np.random.choice(len(X), int(1.0 * len(X)), replace=False)
    X = X[idx]
    y = y[idx]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to Dask
    X_train_dask = da.from_array(X_train, chunks=(100000, X_train.shape[1]))
    y_train_dask = da.from_array(y_train, chunks=(100000,))
    X_test_dask = da.from_array(X_test, chunks=(100000, X_test.shape[1]))

    dtrain = dxgb.DaskDMatrix(client, X_train_dask, y_train_dask)

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 7,
        "learning_rate": 0.25,
        "subsample": 0.8,
        "min_child_weight": 0.25,
        "verbosity": 1
    }

    start = time.time()
    output = dxgb.train(client, params, dtrain, num_boost_round=350)
    booster = output["booster"]
    end = time.time()

    # Predict
    dtest = dxgb.DaskDMatrix(client, X_test_dask)
    y_pred = dxgb.predict(client, booster, dtest).compute()

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"[RESULT] RMSE: {rmse:.3f}")
    print(f"[RESULT] MAE : {mae:.3f}")
    print(f"[RESULT] R²  : {r2:.3f}")
    print(f"[INFO] Time taken: {end - start:.2f} seconds")

    client.shutdown()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

