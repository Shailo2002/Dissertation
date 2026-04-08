"""
Initialise random starting models and noise hyperparameters for each chain.
Translated from MATLAB initialise_chain.m
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple


def initialise_chains(cfg) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
    """
    Create random initial models, likelihoods, and noise sigmas.

    Returns
    -------
    models : list of ndarray, shape (n_nodes, 2)
        model[:, 0] = depth (log10 m), NaN for surface node
        model[:, 1] = resistivity (log10 ohm-m)
    likelihoods : list of float (empty at init → computed on first step)
    sigmas : list of ndarray
    """
    dtype = cfg.MT.datatype.upper()
    models = []
    likelihoods = []
    sigmas = []

    for _ in range(cfg.nChains):
        # Random number of nodes
        ngrid = cfg.minnodes + np.random.randint(1, 3)   # minnodes+1 or minnodes+2

        # Random depths in the allowed range
        z = cfg.min_z + (cfg.max_z - cfg.min_z) * np.random.rand(ngrid)
        z = np.sort(np.unique(z))

        # Build model matrix: first row is surface node (NaN depth)
        n = len(z)
        r = cfg.min_res_log + (cfg.max_res_log - cfg.min_res_log) * np.random.rand(n + 1)

        model = np.zeros((n + 1, 2))
        model[0, 0] = np.nan
        model[1:, 0] = z
        model[:, 1] = r

        models.append(model)
        likelihoods.append(None)   # computed on first sampler call

        # Initial noise hyperparameters
        method = cfg.inversion_method.upper()
        if method == "MT_DC":
            sigma = np.ones(2 if dtype == "Z" else 3)
        elif method == "MT":
            sigma = np.ones(1 if dtype == "Z" else 2)
        else:  # DC
            sigma = np.ones(1)

        sigmas.append(sigma)

    return models, likelihoods, sigmas
