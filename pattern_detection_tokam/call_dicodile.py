import time

from dicodile import dicodile
from dicodile.update_d.update_d import tukey_window
from dicodile.utils.dictionary import init_dictionary

from pattern_detection_tokam.load_and_save import load_data, save_results

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
    offset_type = "none"

    learnable_image = load_data(experiment, frame, offset_type)

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
    save_results(D_hat, z_hat, experiment, frame, time_str)
