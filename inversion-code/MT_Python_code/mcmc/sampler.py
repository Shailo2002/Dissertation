"""
Single-chain MCMC sampler (transdimensional Bayesian hierarchical).

Translated from MATLAB bayesain_hier.m

Each call runs one 'step' of nsamples accepted models for one chain.
Returns a Samples dict (mirrors the MATLAB struct).
"""

from __future__ import annotations
import copy
import numpy as np
from typing import Tuple

from mcmc.proposals import (
    propose_type, cell_birth, cell_death, cell_move,
    change_rho, change_noise, change_noise2,
)
from mcmc.likelihood import (
    forward_response, estimate_uncertainty,
    compute_residual, estimate_like_norm,
)


def run_chain_step(
    model: np.ndarray,
    like_current: float,
    sigma: np.ndarray,
    cfg,
    chain_idx: int,
    step_idx: int,
) -> Tuple[dict, np.ndarray, float, np.ndarray]:
    """
    Run one step (nsamples accepted models) for a single chain.

    Parameters
    ----------
    model : ndarray  shape (n, 2)
        Current Voronoi model.
    like_current : float
        Current negative log-likelihood.
    sigma : ndarray
        Current noise hyperparameter(s).
    cfg : Config
        Configuration (with per-chain sigma values already extracted).
    chain_idx : int
        0-based chain index (for logging).
    step_idx : int
        0-based step index (for logging).

    Returns
    -------
    samples : dict
        Collected samples for this step.
    model_out : ndarray
        Updated model after the step.
    like_out : float
        Updated likelihood.
    sigma_out : ndarray
        Updated noise hyperparameters.
    """
    temperature = cfg.temperature[chain_idx]

    # Extract per-chain scalar proposal widths
    sr = cfg.sigma_rho[chain_idx]
    srb = cfg.sigma_rho_birth[chain_idx]
    srd = cfg.sigma_rho_delayed[chain_idx]
    sz = cfg.sigma_loc_z[chain_idx]

    # Build a per-chain config copy with scalar widths
    chain_cfg = copy.copy(cfg)
    chain_cfg.sigma_rho = sr
    chain_cfg.sigma_rho_birth = srb
    chain_cfg.sigma_rho_delayed = srd
    chain_cfg.sigma_loc_z = sz
    chain_cfg.sigma_loc_z_delayed = cfg.sigma_loc_z_delayed[chain_idx]

    dtype = cfg.MT.datatype.upper()

    # ---------------------------------------------------------- #
    # Allocate sample storage
    # ---------------------------------------------------------- #
    ns = cfg.nsamples
    samples = {
        "step":   np.zeros(ns, dtype=np.int32),
        "ncells": np.zeros(ns, dtype=np.int32),
        "model":  np.zeros((ns, cfg.maxnodes, 2)),
        "like":   np.zeros(ns),
        "sigma":  np.zeros((ns, len(sigma))),
        "misfit": None,          # filled below based on method
    }

    method = cfg.inversion_method.upper()
    if method in ("MT", "DC"):
        samples["misfit"] = np.zeros((ns, 1))
    else:
        samples["misfit"] = np.zeros((ns, 3))

    # ---------------------------------------------------------- #
    # Initial forward response + likelihood
    # ---------------------------------------------------------- #
    data_current = forward_response(model, cfg)
    unc_current = estimate_uncertainty(sigma, cfg)
    like_current, nrms = compute_residual(data_current, unc_current, cfg)

    model_proposed = model.copy()
    sigma_proposed = sigma.copy()

    acceptance_count = np.zeros((2, 6), dtype=np.int64)

    # ---------------------------------------------------------- #
    # Main MCMC loop – keep going until nsamples accepted
    # ---------------------------------------------------------- #
    iselect = 0
    while iselect < ns:
        ptype = propose_type(np.random.rand(), cfg.proposal)

        lerr = False
        log_prob = 0.0
        alpha_prior_ratio = 0.0

        if ptype == 1:
            lerr, log_prob, model_proposed = cell_birth(model, chain_cfg)
        elif ptype == 2:
            lerr, log_prob, model_proposed = cell_death(model, chain_cfg)
        elif ptype == 3:
            lerr, model_proposed, _ = cell_move(model, chain_cfg)
        elif ptype == 4:
            lerr, model_proposed, _ = change_rho(model, chain_cfg)
        elif ptype == 5:
            if cfg.log_normal_noise:
                lerr, sigma_proposed, alpha_prior_ratio = change_noise2(sigma, chain_cfg)
            else:
                lerr, sigma_proposed = change_noise(sigma, chain_cfg)

        if not lerr:
            continue

        # ---- Forward model and likelihood ----
        if ptype <= 4:
            data_proposed = forward_response(model_proposed, cfg)
            like_proposed, nrms = compute_residual(data_proposed, unc_current, cfg)
            normalization = 0.0
        else:
            data_proposed = data_current
            unc_proposed = estimate_uncertainty(sigma_proposed, cfg)
            like_proposed, nrms = compute_residual(data_proposed, unc_proposed, cfg)
            normalization, reg_term = estimate_like_norm(
                unc_proposed, unc_current, sigma_proposed, cfg
            )
            if cfg.log_normal_noise:
                like_proposed = like_proposed + reg_term

        if abs(like_proposed) < cfg.eps:
            continue

        # ---- Metropolis-Hastings acceptance ----
        if ptype in (1, 2):     # birth or death
            if cfg.kernel == 0:
                alpha = min(0.0, log_prob + (-like_proposed + like_current + normalization) / temperature)
            else:
                alpha = min(0.0, (-like_proposed + like_current + normalization) / temperature)
        elif ptype == 5:
            if cfg.log_normal_noise:
                log_ratio = np.log(alpha_prior_ratio) if alpha_prior_ratio > 0 else -np.inf
                alpha = min(0.0, (-like_proposed + like_current + normalization + log_ratio) / temperature)
            else:
                alpha = min(0.0, (-like_proposed + like_current + normalization) / temperature)
        else:
            alpha = min(0.0, (-like_proposed + like_current + normalization) / temperature)

        if alpha > np.log(np.random.rand()):
            # Accept
            like_current = like_proposed
            if ptype <= 4:
                model = model_proposed.copy()
                data_current = data_proposed
            else:
                sigma = sigma_proposed.copy()
                unc_current = estimate_uncertainty(sigma, cfg)

            # Store accepted sample
            nn = len(model)
            samples["step"][iselect] = ptype
            samples["ncells"][iselect] = nn
            samples["model"][iselect, :nn, :] = model
            samples["like"][iselect] = like_current
            samples["misfit"][iselect, :] = nrms
            samples["sigma"][iselect, :] = sigma
            acceptance_count[0, ptype - 1] += 1
            iselect += 1
        else:
            acceptance_count[1, ptype - 1] += 1

    # ---- Acceptance rates ----
    total = acceptance_count.sum()
    overall_ar = 100.0 * acceptance_count[0].sum() / max(total, 1)
    per_type_ar = np.zeros(5)
    for pt in range(5):
        denom = acceptance_count[:, pt].sum()
        if denom > 0:
            per_type_ar[pt] = 100.0 * acceptance_count[0, pt] / denom

    samples["acceptance_all"] = np.concatenate([[overall_ar], per_type_ar])
    samples["acceptance_count"] = acceptance_count

    # Console logging (mirrors MATLAB output)
    nrms_print = float(nrms[-1]) if len(nrms) > 1 else float(nrms[0])
    print(
        f"AR [chain step FM_call] {chain_idx+1:2d} {step_idx+1:4d} "
        f"{ns:8d} {total:8d}"
    )
    print(
        f"   [Accp_rate nRMS Layers] {overall_ar:8.2f} {nrms_print:8.2f} {nn:3d}"
    )
    print(
        f"   BDMCN "
        + " ".join(f"{v:8.2f}" for v in per_type_ar)
    )

    return samples, model, like_current, sigma
