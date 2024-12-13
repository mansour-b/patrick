PATteRn dIsCovery Kit (PATRICK)
----------------------------------

Welcome to the Pattern Discovery Kit!

This library provides tools to discover, detect, and track meaningful patterns in physical signals.

Installation
------------

You can install ``patrick`` from the source code by doing the following::

    git clone https://github.com/mansour-b/patrick.git
    cd patrick
    pip install .

Quickstart
==========

Here is an example to present briefly the API:

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from alphacsc import BatchCDL

    # Define the different dimensions of the problem
    n_atoms = 10
    n_times_atom = 50
    n_channels = 5
    n_trials = 10
    n_times = 1000

    # Generate a random set of signals
    X = np.random.randn(n_trials, n_channels, n_times)

    # Learn a dictionary with batch algorithm and rank1 constraints.
    cdl = BatchCDL(n_atoms, n_times_atom, rank1=True)
    cdl.fit(X)

    # Display the learned atoms
    fig, axes = plt.subplots(n_atoms, 2, num="Dictionary")
    for k in range(n_atoms):
        axes[k, 0].plot(cdl.u_hat_[k])
        axes[k, 1].plot(cdl.v_hat_[k])

    axes[0, 0].set_title("Spatial map")
    axes[0, 1].set_title("Temporal map")
    for ax in axes.ravel():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()