from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from patrick import PATRICK_DIR_PATH


def load_data(file_name: str, offset_type: str):
    file_path = PATRICK_DIR_PATH / f"input/{file_name}"

    image_array = np.loadtxt(file_path)

    counts, values = np.histogram(image_array, bins=100)
    mode = values[np.argmax(counts)]
    offset_dict = {
        "mean": image_array.mean(),
        "median": np.median(image_array),
        "mode": mode,
        "none": 0.0,
    }
    image_array -= offset_dict[offset_type]

    return np.expand_dims(image_array, axis=0)


def log_dicodile_params(dicodile_kwargs: dict, file_name: str, time_str: str):
    output_dir_path = PATRICK_DIR_PATH / "dicodile_params" / file_name
    output_dir_path.mkdir(parents=True, exist_ok=True)
    file_path = output_dir_path / f"{time_str}.json"
    with Path.open(file_path, "w") as f:
        json.dump(dicodile_kwargs, f, indent=2)


def save_results(d_hat: np.array, z_hat: np.array, file_name: str, time_str: str):
    output_dir_path = PATRICK_DIR_PATH / "learned_dictionaries" / file_name

    output_dir_path.mkdir(parents=True, exist_ok=True)

    d_path = output_dir_path / f"D_hat_{time_str}"
    z_path = output_dir_path / f"z_hat_{time_str}"

    np.save(d_path, d_hat)
    np.save(z_path, z_hat)


def load_dict_and_activations(
    dict_dir_name: str, mode_or_timestamp: str = "latest", verbose: int = 0
):

    dict_path = PATRICK_DIR_PATH / "learned_dictionaries" / dict_dir_name

    if mode_or_timestamp == "latest":
        time_str_list = sorted(
            {
                f"{fpath.stem.split('_')[2]}_{fpath.stem.split('_')[3]}"
                for fpath in dict_path.glob("*")
            }
        )
        time_str = time_str_list[-1]
    else:
        time_str = mode_or_timestamp

    if verbose >= 1:
        print(f"Loaded from: {dict_path}")
        print(f"Timestamp: {time_str}")

    d_hat = np.load(dict_path / f"D_hat_{time_str}.npy")
    z_hat = np.load(dict_path / f"z_hat_{time_str}.npy")

    return d_hat, z_hat
