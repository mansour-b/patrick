from pathlib import Path

import numpy as np


def load_dict_and_activations(
    experiment: str, frame: int, mode_or_timestamp: str = "latest"
):

    data_path = Path.home() / "data"
    pattern_detection_path = data_path / "pattern_detection_tokam"
    dict_path = (
        pattern_detection_path / "learned_dictionaries" / f"{experiment}_frame_{frame}"
    )

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

    D_hat = np.load(dict_path / f"D_hat_{time_str}.npy")
    z_hat = np.load(dict_path / f"z_hat_{time_str}.npy")

    return D_hat, z_hat
