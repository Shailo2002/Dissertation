"""
Parallel tempering (replica-exchange) between chains.

Translated from MATLAB swap_temperatures.m

Hot chains (T > 1) explore the model space freely; periodic swaps
inject well-mixed states into cold chains.
"""

from __future__ import annotations
import numpy as np
from typing import List


def swap_temperatures(
    likelihoods: List[float],
    models: List[np.ndarray],
    sigmas: List[np.ndarray],
    cfg,
    step_idx: int,
    swap_count: List[np.ndarray],
) -> tuple[List[float], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Attempt parallel-tempering swaps between cold and hot chains.

    Parameters
    ----------
    likelihoods : list of float   (one per chain)
    models : list of ndarray      (one per chain)
    sigmas : list of ndarray      (one per chain)
    cfg : Config
    step_idx : int   (0-based)
    swap_count : list of 1-D ndarrays  (one per chain, length = nsteps)

    Returns
    -------
    Updated (likelihoods, models, sigmas, swap_count).
    """
    temps = np.asarray(cfg.temperature)
    cold_idx = np.where(temps == 1.0)[0]
    hot_idx = np.where(temps != 1.0)[0]

    if len(hot_idx) == 0:
        return likelihoods, models, sigmas, swap_count

    if cfg.jumptype == 0:
        # Randomly pick one cold-hot pair per PT attempt
        if len(cold_idx) == 0 or len(hot_idx) == 0:
            return likelihoods, models, sigmas, swap_count

        n_swap = cfg.nchain_for_PT
        idx1_list = np.random.choice(cold_idx, size=min(n_swap, len(cold_idx)), replace=False)
        idx2_list = np.random.choice(hot_idx,  size=min(n_swap, len(hot_idx)),  replace=False)

        for i1, i2 in zip(idx1_list, idx2_list):
            like1 = likelihoods[i1]
            like2 = likelihoods[i2]
            t1 = temps[i1]
            t2 = temps[i2]

            alpha_swap = min(0.0,
                -like1 / t2 + like2 / t2 - like2 / t1 + like1 / t1)

            if np.log(np.random.rand()) < alpha_swap:
                likelihoods[i1], likelihoods[i2] = likelihoods[i2], likelihoods[i1]
                models[i1], models[i2] = models[i2], models[i1]
                sigmas[i1], sigmas[i2] = sigmas[i2], sigmas[i1]
                swap_count[i1][step_idx] = 1
                swap_count[i2][step_idx] = 1
                print(f"PT swap between chains {i1+1} and {i2+1} "
                      f"(T={t1:.2f} <-> T={t2:.2f})")

    elif cfg.jumptype == 1:
        # Attempt all upper-triangular pairs simultaneously
        n = cfg.nChains
        pairs = _upper_tri_pairs(n)
        for i1, i2 in pairs:
            like1 = likelihoods[i1]
            like2 = likelihoods[i2]
            t1 = temps[i1]
            t2 = temps[i2]
            alpha_swap = min(0.0,
                -like1 / t2 + like2 / t2 - like2 / t1 + like1 / t1)
            if np.log(np.random.rand()) < alpha_swap:
                likelihoods[i1], likelihoods[i2] = likelihoods[i2], likelihoods[i1]
                models[i1], models[i2] = models[i2], models[i1]
                sigmas[i1], sigmas[i2] = sigmas[i2], sigmas[i1]
                swap_count[i1][step_idx] = 1
                swap_count[i2][step_idx] = 1

    return likelihoods, models, sigmas, swap_count


def _upper_tri_pairs(n: int):
    """Return all (i, j) pairs from the strictly upper triangular matrix."""
    pairs = []
    for j in range(1, n):
        for i in range(j):
            pairs.append((i, j))
    return pairs
