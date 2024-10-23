import time
from argparse import ArgumentParser

import numpy as np
import wandb
from dicodile import dicodile
from dicodile.update_d.update_d import tukey_window
from dicodile.utils.csc import reconstruct
from dicodile.utils.dictionary import init_dictionary

from patrick.load_and_save import load_data, save_results


def make_parser():
    parser = ArgumentParser()

    parser.add_argument("--n_atoms", type=int, help="number of atoms")
    parser.add_argument("--atom_size", type=int, help="size of the atoms (int)")
    parser.add_argument("--reg", type=float, help="regularization parameter")
    parser.add_argument("--tol", type=float, help="tolerance for minimal update size")
    parser.add_argument(
        "--offset_type",
        choices=["mean", "median", "mode", "none"],
        help="value to be subtracted to the image before the CDL begins",
    )

    parser.add_argument(
        "--n_iter",
        default=100,
        type=int,
        help="maximum number of iterations",
    )
    parser.add_argument(
        "--window",
        default=True,
        type=bool,
        help="when True, makes sure that the borders of the atoms are 0",
    )
    parser.add_argument(
        "--z_positive",
        default=True,
        type=bool,
        help="when True, requires all activations Z to be positive",
    )
    parser.add_argument(
        "--n_workers",
        default=10,
        type=int,
        help="number of workers to be used for computations",
    )
    parser.add_argument(
        "--w_world",
        default=10,
        type=int,
        help="number of jobs per row",
    )

    return parser


def compute_metrics(X, D_hat, z_hat):

    n_atoms = z_hat.shape[0]
    flat_activation_array = z_hat.reshape(n_atoms, -1)

    l0_norm_array = np.linalg.norm(flat_activation_array, ord=0, axis=1)
    l1_norm_array = np.linalg.norm(flat_activation_array, ord=1, axis=1)

    sparsity_l0 = np.mean(l0_norm_array)
    sparsity_l1 = np.mean(l1_norm_array)

    X_hat = reconstruct(z_hat, D_hat)
    estimation_error = np.linalg.norm(X_hat - X)

    return sparsity_l0, sparsity_l1, estimation_error


if __name__ == "__main__":

    parser = make_parser()
    args = parser.parse_args()

    atom_support = (args.atom_size, args.atom_size)

    time_str = time.strftime("%y%m%d_%H%M%S")
    with wandb.init() as run:
        run.name = time_str

    experiment = "interchange_nodriftwave"
    frame = 1000

    learnable_image = load_data(experiment, frame, args.offset_type)

    D_init = init_dictionary(
        learnable_image,
        n_atoms=args.n_atoms,
        atom_support=atom_support,
        random_state=60,
    )
    tw = tukey_window(atom_support)[None, None]
    D_init *= tw  # make sure that the border values are 0

    D_hat, z_hat, pobj, times = dicodile(
        learnable_image,
        D_init,
        reg=args.reg,
        n_iter=args.n_iter,
        window=args.window,
        z_positive=args.z_positive,
        n_workers=args.n_workers,
        dicod_kwargs={"max_iter": 10000},
        w_world=args.w_world,
        tol=args.tol,
        verbose=1,
    )
    save_results(D_hat, z_hat, experiment, frame, time_str)

    sparsity_l0, sparsity_l1, estimation_error = compute_metrics(
        learnable_image, D_hat, z_hat
    )

    wandb.log(
        {
            "sparsity_l0": sparsity_l0,
            "sparsity_l1": sparsity_l1,
            "estimation_error": estimation_error,
            "computation_time": np.sum(times),
        }
    )
