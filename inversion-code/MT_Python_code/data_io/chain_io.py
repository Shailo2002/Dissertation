"""
Save and load MCMC chain files.

Saves to numpy .npz format (much faster than MATLAB .mat for Python).
Each chain is saved as:
   <prefix>_chain_<NNN>.npz

The npz file contains all steps' sample dicts stacked into arrays.
"""

from __future__ import annotations
import os
import json
import numpy as np
from typing import List


def _config_to_dict(cfg) -> dict:
    """Serialise Config to a plain dict for JSON storage."""
    import dataclasses
    d = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, np.ndarray):
            d[f.name] = v.tolist()
        elif isinstance(v, (list, tuple)):
            d[f.name] = list(v)
        elif hasattr(v, '__dataclass_fields__'):
            d[f.name] = _config_to_dict(v)
        elif isinstance(v, (bool, int, float, str, type(None))):
            d[f.name] = v
        else:
            try:
                d[f.name] = float(v)
            except Exception:
                d[f.name] = str(v)
    return d


def save_chain(
    results_all: list,          # list[list[dict]]  shape [nChains][nsteps_so_far]
    step_idx: int,              # 0-based current step
    cfg,
    swap_count: list,
    destination_folder: str,
    prefix: str = "MT",
) -> bool:
    """
    Save chain data to disk every 100 steps or at the last step.

    Parameters
    ----------
    results_all : list[list[dict]]
        results_all[ic][is] = samples dict from sampler.run_chain_step
    step_idx : int  (0-based)
    cfg : Config
    swap_count : list of arrays
    destination_folder : str
    prefix : str   'MT' | 'DC' | 'MT_DC'
    """
    one_indexed = step_idx + 1
    if one_indexed % 100 != 0 and one_indexed != cfg.nsteps:
        return True

    os.makedirs(destination_folder, exist_ok=True)

    for ic in range(cfg.nChains):
        chain_samples = results_all[ic]  # list of step-dicts

        # Stack arrays across steps
        steps_done = len(chain_samples)
        if steps_done == 0:
            continue

        # Concatenate each field across steps
        like_list, sigma_list, misfit_list = [], [], []
        ncells_list, model_list, step_list = [], [], []

        for samp in chain_samples:
            like_list.append(samp["like"])
            sigma_list.append(samp["sigma"])
            misfit_list.append(samp["misfit"])
            ncells_list.append(samp["ncells"])
            model_list.append(samp["model"])
            step_list.append(samp["step"])

        fname = os.path.join(
            destination_folder,
            f"{prefix}_TD_Chain_{ic+1:03d}.npz",
        )
        np.savez_compressed(
            fname,
            like=np.concatenate(like_list),
            sigma=np.vstack(sigma_list),
            misfit=np.vstack(misfit_list),
            ncells=np.concatenate(ncells_list),
            model=np.concatenate(model_list, axis=0),
            step=np.concatenate(step_list),
            nsteps_saved=np.array(steps_done),
            swap_count=(swap_count[ic] if swap_count else np.zeros(cfg.nsteps)),
        )

        # Save config as JSON alongside chains
        cfg_path = os.path.join(destination_folder, "config.json")
        if not os.path.exists(cfg_path):
            with open(cfg_path, "w") as f:
                json.dump(_config_to_dict(cfg), f, indent=2, default=str)

    return True


def load_chain(folder: str, chain_idx: int, prefix: str = "MT") -> dict:
    """
    Load a single chain npz file.

    Returns
    -------
    dict with keys: like, sigma, misfit, ncells, model, step, swap_count
    """
    fname = os.path.join(folder, f"{prefix}_TD_Chain_{chain_idx:03d}.npz")
    data = np.load(fname, allow_pickle=False)
    return dict(data)


def load_config(folder: str):
    """
    Re-construct a Config-like namespace from the saved JSON.
    Returns a dict (not a full Config dataclass) for post-processing.
    """
    cfg_path = os.path.join(folder, "config.json")
    with open(cfg_path) as f:
        return json.load(f)
