import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dicodile import dicodile
from dicodile.update_d.update_d import tukey_window
from dicodile.utils.dictionary import init_dictionary


def load_data():

    data_dir_path = Path.home() / "data/pattern_detection_tokam"

    input_dir_path = data_dir_path / "input"
    file_path_list = sorted(input_dir_path.glob("*"))
    image_array = plt.imread(file_path_list[0])
    learnable_image = image_array[:, :, :-1].transpose([2, 0, 1])

    channel_per_channel_mean = np.reshape(np.mean(learnable_image, (1, 2)), (-1, 1, 1))

    return learnable_image - channel_per_channel_mean


def save_results(D_hat, z_hat, time_str):
    data_dir_path = Path.home() / "data/pattern_detection_tokam"
    output_dir_path = data_dir_path / "learned_dictionaries/first_image"

    output_dir_path.mkdir(parents=True, exist_ok=True)

    D_path = output_dir_path / f"D_hat_{time_str}"
    z_path = output_dir_path / f"z_hat_{time_str}"

    np.save(D_path, D_hat)
    np.save(z_path, z_hat)


if __name__ == "__main__":

    time_str = time.strftime("%y%m%d_%H%M%S")

    n_atoms = 10
    atom_support = (50, 50)

    reg = 0.2  # regularization parameter
    n_iter = 100  # maximum number of iterations
    window = True  # when True, makes sure that the borders of the atoms are 0
    z_positive = True  # when True, requires all activations Z to be positive
    n_workers = 10  # number of workers to be used for computations
    w_world = "auto"  # number of jobs per row
    tol = 1e-3  # tolerance for minimal update size

    learnable_image = load_data()

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

    save_results(D_hat=D_hat, z_hat=z_hat, time_str=time_str)
