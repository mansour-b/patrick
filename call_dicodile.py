import time
from pathlib import Path

import numpy as np
from dicodile import dicodile
from dicodile.update_d.update_d import tukey_window
from dicodile.utils.dictionary import init_dictionary


def load_data(experiment: str, frame: int):

    data_dir_path = Path.home() / "data"
    input_dir_path = data_dir_path / "pattern_detection_tokam/input" / experiment
    file_path = input_dir_path / f"frame_{frame}.txt"

    image_array = np.loadtxt(file_path)
    image_array *= image_array >= image_array.mean()

    return np.expand_dims(image_array, axis=0)


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


if __name__ == "__main__":

    time_str = time.strftime("%y%m%d_%H%M%S")

    n_atoms = 10
    atom_support = (10, 10)

    reg = 0.2  # regularization parameter
    n_iter = 100  # maximum number of iterations
    window = True  # when True, makes sure that the borders of the atoms are 0
    z_positive = True  # when True, requires all activations Z to be positive
    n_workers = 10  # number of workers to be used for computations
    w_world = "auto"  # number of jobs per row
    tol = 1e-3  # tolerance for minimal update size

    experiment = "interchange_nodriftwave"
    frame = 1000

    learnable_image = load_data(experiment, frame)

    D_init = init_dictionary(
        learnable_image, n_atoms=n_atoms, atom_support=atom_support, random_state=60
    )
    tw = tukey_window(atom_support)[
        None, None
    ]  # make sure that the border values are 0
    D_init *= tw

    D_hat, z_hat, pobj, times = dicodile(
        learnable_image,
        D_init,
        reg=reg,
        n_iter=n_iter,
        window=window,
        z_positive=z_positive,
        n_workers=n_workers,
        dicod_kwargs={"max_iter": 10000},
        w_world=w_world,
        tol=tol,
        verbose=1,
    )

    print("[DICOD] final cost : {}".format(pobj))

    save_results(D_hat, z_hat, experiment, frame, time_str)
