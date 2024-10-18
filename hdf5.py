from pathlib import Path

import h5py
import numpy as np

data_dir_path = Path.home() / "data"
tokam_dir_path = data_dir_path / "tokam2d"

experiment = "interchange_nodriftwave"


file_path = tokam_dir_path / experiment / "data_TOKAM_run_00.h5"

with h5py.File(file_path, "r") as f:
    density = np.array(f["density"])
    potential = np.array(f["potential"])
    time = np.array(f["time"])
    x = np.array(f["x"])
    y = np.array(f["y"])

print(density.shape)
print(potential.shape)
print(time.shape)
print(x.shape)
print(y.shape)
