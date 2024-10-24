import json
from pathlib import Path

import numpy as np


def load_data(experiment: str, frame: int, offset_type: str):

    data_dir_path = Path.home() / "data"
    input_dir_path = data_dir_path / "pattern_detection_tokam/input" / experiment
    file_path = input_dir_path / f"frame_{frame}.txt"

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


def log_dicodile_params(dicodile_kwargs: dict, experiment, frame, time_str):
    data_dir_path = Path.home() / "data"
    pattern_detection_path = data_dir_path / "pattern_detection_tokam"
    output_dir_path = (
        pattern_detection_path / "dicodile_params" / f"{experiment}_frame_{frame}"
    )

    output_dir_path.mkdir(parents=True, exist_ok=True)
    file_path = output_dir_path / f"{time_str}.json"
    with open(file_path) as f:
        json.dump(dicodile_kwargs, f, indent=2)


def save_results(D_hat, z_hat, experiment, frame, time_str):
    data_dir_path = Path.home() / "data"
    pattern_detection_path = data_dir_path / "pattern_detection_tokam"
    output_dir_path = (
        pattern_detection_path / "learned_dictionaries" / f"{experiment}_frame_{frame}"
    )

    output_dir_path.mkdir(parents=True, exist_ok=True)

    D_path = output_dir_path / f"D_hat_{time_str}"
    z_path = output_dir_path / f"z_hat_{time_str}"

    np.save(D_path, D_hat)
    np.save(z_path, z_hat)
