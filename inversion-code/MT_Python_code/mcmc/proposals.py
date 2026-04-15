"""
Transdimensional MCMC proposal functions.

Each proposal modifies the current model and returns:
  (lerr, model_proposed, [extra])

  lerr = True  → proposal is within bounds, proceed to acceptance test
  lerr = False → proposal is out-of-bounds, reject immediately

Translated from MATLAB:
  cell_birth.m, cell_death.m, cell_move.m, change_rho.m,
  change_noise.m, change_noise2.m, proposeType.m

Model convention (numpy array, shape (n, 2)):
  col 0 → depth in log10(m); row 0 is always NaN (surface node)
  col 1 → resistivity in log10(ohm-m)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _check_min_thickness(z_nodes_log: np.ndarray, max_z: float,
                          min_thickness: float, logdomain: bool) -> bool:
    """
    Return True if all layers are at least min_thickness metres thick.

    z_nodes_log includes only the internal nodes (not the surface NaN).
    """
    if logdomain:
        z_lin = np.concatenate([[0.0], 10.0 ** z_nodes_log, [10.0 ** max_z]])
    else:
        z_lin = np.concatenate([[0.0], z_nodes_log, [max_z]])
    return float(np.min(np.diff(z_lin))) > min_thickness


def propose_type(u: float, proposal: list) -> int:
    """
    Map a Uniform(0,1) sample to a proposal type.

    Types: 1=birth, 2=death, 3=move, 4=change_rho, 5=change_noise
    """
    if u < proposal[0]:
        return 1
    elif u < proposal[1]:
        return 2
    elif u < proposal[2]:
        return 3
    elif u < proposal[3]:
        return 4
    else:
        return 5


# ------------------------------------------------------------------ #
# Birth
# ------------------------------------------------------------------ #

def cell_birth(model: np.ndarray, cfg) -> Tuple[bool, float, np.ndarray]:
    """
    Add a new Voronoi node at a random depth.

    Returns
    -------
    lerr : bool
    log_prob : float   (Green's ratio; 0 when kernel==1)
    model_new : ndarray
    """
    if len(model) >= cfg.maxnodes:
        return False, 0.0, model

    # Try up to 1000 times to place a node that keeps all layers thick enough
    for _ in range(1000):
        z_new = cfg.min_z + (cfg.max_z - cfg.min_z) * np.random.rand()
        # Internal nodes: model[1:, 0]
        internal = model[1:, 0]
        if cfg.logdomain:
            z_lin = np.sort(np.concatenate([[0.0],
                                            10.0 ** internal,
                                            [10.0 ** z_new],
                                            [10.0 ** cfg.max_z]]))
        else:
            z_lin = np.sort(np.concatenate([[0.0], internal, [z_new], [cfg.max_z]]))
        if np.min(np.diff(z_lin)) > cfg.minimum_layer_thickness:
            break
    else:
        return False, 0.0, model

    # Find nearest existing node to inherit resistivity
    dist = np.abs(model[:, 0] - z_new)
    dist[0] = np.inf            # ignore the NaN surface node
    nearest = int(np.argmin(dist))

    if cfg.kernel == 0:         # Gaussian kernel
        rho_near = model[nearest, 1]
        rho_new = rho_near + cfg.sigma_rho_birth * np.random.randn()
        log_prob = (
            np.log(cfg.sigma_rho_birth * np.sqrt(2 * np.pi))
            + (rho_near - rho_new) ** 2 / (2 * cfg.sigma_rho_birth ** 2)
            - np.log(cfg.max_res_log - cfg.min_res_log)
        )
    else:                       # Prior kernel (uniform)
        rho_new = cfg.min_res_log + np.random.rand() * (cfg.max_res_log - cfg.min_res_log)
        log_prob = 0.0

    if rho_new < cfg.min_res_log or rho_new > cfg.max_res_log:
        return False, 0.0, model

    # Insert the new node at its correct sorted position. The "nearest"
    # neighbour may be above or below z_new, so we can't rely on nearest+1 —
    # doing so produces an unsorted depth array and negative layer thicknesses.
    internal_depths = model[1:, 0]
    insert_at = 1 + int(np.searchsorted(internal_depths, z_new))
    model_new = np.vstack([
        model[:insert_at, :],
        np.array([[z_new, rho_new]]),
        model[insert_at:, :],
    ])
    model_new[0, 0] = np.nan    # surface node is always NaN

    return True, log_prob, model_new


# ------------------------------------------------------------------ #
# Death
# ------------------------------------------------------------------ #

def cell_death(model: np.ndarray, cfg) -> Tuple[bool, float, np.ndarray]:
    """
    Remove a randomly selected Voronoi node.
    """
    if len(model) <= cfg.minnodes:
        return False, 0.0, model

    # Pick any internal node (index 1..n-1)
    indx = np.random.randint(1, len(model))
    rho_remove = model[indx, 1]
    rho_above = model[indx - 1, 1]

    model_new = np.delete(model, indx, axis=0)

    if cfg.kernel == 0:
        log_prob = (
            np.log(1.0 / (cfg.sigma_rho_birth * np.sqrt(2 * np.pi)))
            - (rho_above - rho_remove) ** 2 / (2 * cfg.sigma_rho_birth ** 2)
            + np.log(cfg.max_res_log - cfg.min_res_log)
        )
    else:
        log_prob = 0.0

    return True, log_prob, model_new


# ------------------------------------------------------------------ #
# Move
# ------------------------------------------------------------------ #

def cell_move(model: np.ndarray, cfg) -> Tuple[bool, np.ndarray, int]:
    """
    Perturb the depth of a randomly selected internal node.
    """
    indx = np.random.randint(1, len(model))
    model_new = model.copy()

    z_new = model[indx, 0] + cfg.sigma_loc_z * np.random.randn() * (
        cfg.max_z - cfg.min_z
    ) / 100.0

    if z_new < cfg.min_z or z_new > cfg.max_z:
        return False, model, indx

    model_new[indx, 0] = z_new

    # Re-sort internal nodes by depth; keep surface node fixed
    surface = model_new[0:1, :]
    internal = model_new[1:, :]
    sort_idx = np.argsort(internal[:, 0])
    model_new = np.vstack([surface, internal[sort_idx, :]])

    # Check minimum layer thickness
    internal_z = model_new[1:, 0]
    if not _check_min_thickness(internal_z, cfg.max_z,
                                cfg.minimum_layer_thickness, cfg.logdomain):
        return False, model, indx

    return True, model_new, indx


# ------------------------------------------------------------------ #
# Change resistivity
# ------------------------------------------------------------------ #

def change_rho(model: np.ndarray, cfg) -> Tuple[bool, np.ndarray, int]:
    """
    Perturb the resistivity of a randomly selected node.
    """
    indx = np.random.randint(0, len(model))
    model_new = model.copy()

    rho_new = model[indx, 1] + np.random.randn() * cfg.sigma_rho
    if rho_new < cfg.min_res_log or rho_new > cfg.max_res_log:
        return False, model, indx

    model_new[indx, 1] = rho_new
    return True, model_new, indx


# ------------------------------------------------------------------ #
# Change noise hyperparameter
# ------------------------------------------------------------------ #

def change_noise(sigma: np.ndarray, cfg) -> Tuple[bool, np.ndarray]:
    """
    Perturb the noise scaling hyperparameter (uniform random walk).
    Translated from change_noise.m
    """
    sigma_new = sigma.copy()
    method = cfg.inversion_method.upper()
    dtype = cfg.MT.datatype.upper()

    if method in ("MT", "MT_DC"):
        if dtype == "Z":
            bounds = cfg.sigma_Z
            step = cfg.sigma_noise * (bounds[1] - bounds[0]) / 100.0
            t = sigma[0] + np.random.randn() * step
            if t < bounds[0] or t > bounds[1]:
                return False, sigma
            sigma_new[0] = t
        elif dtype == "APP":
            bounds = cfg.sigma_app_res
            step = cfg.sigma_noise * (bounds[1] - bounds[0]) / 100.0
            t = sigma[0] + np.random.randn() * step
            if t < bounds[0] or t > bounds[1]:
                return False, sigma
            sigma_new[0] = t
        elif dtype == "PHASE":
            bounds = cfg.sigma_phase
            step = cfg.sigma_noise * (bounds[1] - bounds[0]) / 100.0
            t = sigma[1] + np.random.randn() * step
            if t < bounds[0] or t > bounds[1]:
                return False, sigma
            sigma_new[1] = t
        elif dtype == "APP_PHASE":
            bounds_a = cfg.sigma_app_res
            step_a = cfg.sigma_noise * (bounds_a[1] - bounds_a[0]) / 100.0
            t = sigma[0] + np.random.randn() * step_a
            if t < bounds_a[0] or t > bounds_a[1]:
                return False, sigma
            sigma_new[0] = t
            bounds_p = cfg.sigma_phase
            step_p = cfg.sigma_noise * (bounds_p[1] - bounds_p[0]) / 100.0
            t2 = sigma[1] + np.random.randn() * step_p
            if t2 < bounds_p[0] or t2 > bounds_p[1]:
                return False, sigma
            sigma_new[1] = t2

    if method == "DC":
        bounds = cfg.sigma_app_res
        step = cfg.sigma_noise * (bounds[1] - bounds[0]) / 100.0
        t = sigma[0] + np.random.randn() * step
        if t < bounds[0] or t > bounds[1]:
            return False, sigma
        sigma_new[0] = t

    if method == "MT_DC":
        bounds = cfg.sigma_app_res
        step = cfg.sigma_noise * (bounds[1] - bounds[0]) / 100.0
        idx = 2 if dtype == "Z" else 3
        t = sigma[idx - 1] + np.random.randn() * step
        if t < bounds[0] or t > bounds[1]:
            return False, sigma
        sigma_new[idx - 1] = t

    return True, sigma_new


def _lognorm_pdf(x: float, mu: float, sigma: float) -> float:
    if x <= 0:
        return 0.0
    return (1.0 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x) - np.log(mu)) ** 2) / (2 * sigma ** 2)
    )


def change_noise2(sigma: np.ndarray, cfg) -> Tuple[bool, np.ndarray, float]:
    """
    Log-normal noise proposal (used when cfg.log_normal_noise==True).
    Translated from change_noise2.m
    """
    sigma_new = sigma.copy()
    alpha_prior_ratio = 0.0
    method = cfg.inversion_method.upper()
    dtype = cfg.MT.datatype.upper()

    def _propose(s, bounds):
        s_new = s * np.exp(cfg.sigma_noise * np.random.randn())
        s_new = max(bounds[0], min(bounds[1], s_new))
        return s_new

    if method in ("MT", "MT_DC"):
        if dtype == "Z":
            s_new = _propose(sigma[0], cfg.sigma_Z)
            alpha_prior_ratio += _lognorm_pdf(s_new, 1.0, 0.2) / max(
                _lognorm_pdf(sigma[0], 1.0, 0.2), 1e-300
            )
            sigma_new[0] = s_new
        elif dtype == "APP":
            s_new = _propose(sigma[0], cfg.sigma_app_res)
            alpha_prior_ratio += _lognorm_pdf(s_new, 1.0, 0.2) / max(
                _lognorm_pdf(sigma[0], 1.0, 0.2), 1e-300
            )
            sigma_new[0] = s_new
        elif dtype == "PHASE":
            s_new = _propose(sigma[1], cfg.sigma_phase)
            alpha_prior_ratio += _lognorm_pdf(s_new, 1.0, 0.2) / max(
                _lognorm_pdf(sigma[1], 1.0, 0.2), 1e-300
            )
            sigma_new[1] = s_new
        elif dtype == "APP_PHASE":
            s0 = _propose(sigma[0], cfg.sigma_app_res)
            s1 = _propose(sigma[1], cfg.sigma_phase)
            alpha_prior_ratio += (
                _lognorm_pdf(s0, 1.0, 0.2) / max(_lognorm_pdf(sigma[0], 1.0, 0.2), 1e-300)
                + _lognorm_pdf(s1, 1.0, 0.2) / max(_lognorm_pdf(sigma[1], 1.0, 0.2), 1e-300)
            )
            sigma_new[0] = s0
            sigma_new[1] = s1

    if method == "DC":
        s_new = _propose(sigma[0], cfg.sigma_app_res)
        alpha_prior_ratio += _lognorm_pdf(s_new, 1.0, 0.2) / max(
            _lognorm_pdf(sigma[0], 1.0, 0.2), 1e-300
        )
        sigma_new[0] = s_new

    if method == "MT_DC":
        dtype_mt = cfg.MT.datatype.upper()
        idx = 1 if dtype_mt == "Z" else 2
        s_new = _propose(sigma[idx], cfg.sigma_app_res)
        alpha_prior_ratio += _lognorm_pdf(s_new, 1.0, 0.2) / max(
            _lognorm_pdf(sigma[idx], 1.0, 0.2), 1e-300
        )
        sigma_new[idx] = s_new

    return True, sigma_new, alpha_prior_ratio
