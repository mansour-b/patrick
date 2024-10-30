from argparse import ArgumentParser


def make_dicodile_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--n_atoms",
        type=int,
        default=10,
        help="number of atoms",
    )
    parser.add_argument("--atom_size", type=int, help="size of the atoms (int)")
    parser.add_argument("--reg", type=float, help="regularization parameter")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="tolerance for minimal update size",
    )
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
        "--eps",
        type=float,
        default=1e-5,
        help="stopping value for the alternate minimisation",
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
        "--num_workers_per_row",
        default=4,
        type=int,
        help="number of jobs per row",
    )

    return parser
