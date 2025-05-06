import time
import numpy as np
import h5py
import dask.array as da
from dask.distributed import Client
from xgboost import dask as dxgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import multiprocessing


def main():
    print("[INFO] Connecting to existing Dask cluster...")
    client = Client()  # Assumes dask-mpi starts scheduler
    print(client)

    file_path = "Greenland.h5"
    print("[INFO] Loading HDF5 dataset...")
    with h5py.File(file_path, 'r') as f:
        surf_x = f["surf_x"][:].ravel()
        surf_y = f["surf_y"][:].ravel()
        surf_vx = f["surf_vx"][:].ravel()
        surf_vy = f["surf_vy"][:].ravel()
        surf_elv = f["surf_elv"][:].ravel()
        surf_dhdt = f["surf_dhdt"][:].ravel()
        surf_SMB = f["surf_SMB"][:].ravel()
        y = f["track_bed"][2, :].ravel()

    print("[INFO] Creating derived velocity magnitude feature...")
    velocity_magnitude = np.sqrt(surf_vx**2 + surf_vy**2)

    X = np.stack([
        surf_x, surf_y, surf_vx, surf_vy,
        surf_elv, surf_dhdt, surf_SMB, velocity_magnitude
    ], axis=1)

    print("[INFO] Cleaning invalid data...")
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    print("[INFO] Converting to Dask arrays...")
    X_dask = da.from_array(X, chunks=(100000, X.shape[1]))
    y_dask = da.from_array(y, chunks=(100000,))

    dtrain = dxgb.DaskDMatrix(client, X_dask, y_dask)

    print("[INFO] Training XGBoost model...")
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.1,
        "verbosity": 1
    }

    start = time.time()
    output = dxgb.train(client, params, dtrain, num_boost_round=100)
    booster = output["booster"]
    end = time.time()

    print("[INFO] Evaluating model...")
    y_pred = dxgb.predict(client, booster, X_dask).compute()

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"[RESULT] RMSE: {rmse:.3f}")
    print(f"[RESULT] MAE : {mae:.3f}")
    print(f"[RESULT] R²  : {r2:.3f}")
    print(f"[INFO] Total training time: {end - start:.2f} seconds")

    client.shutdown()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

