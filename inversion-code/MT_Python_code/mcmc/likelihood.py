"""
Forward response computation, likelihood, and uncertainty estimation.

Translated from MATLAB:
  Fwd_response.m, Residual.m, estimate_uncertanity.m, estimate_like_norm.m
"""

from __future__ import annotations
import numpy as np
from forward.mt_forward import mt1d_forward
from forward.dc_forward import dc1d_forward


# ------------------------------------------------------------------ #
# Model → layer arrays
# ------------------------------------------------------------------ #

def model_to_layers(model: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the Voronoi model array to (thicknesses, resistivities).

    The model convention:
      model[0, 0] = NaN  (surface node, depth fixed at 0)
      model[1:, 0] = internal node depths in log10(m)   if logdomain=True
      model[:, 1]  = log10(ohm-m) resistivities

    Returns
    -------
    thicknesses : ndarray
        Layer thicknesses in metres (including a very deep basement layer).
    resistivities : ndarray
        Layer resistivities in ohm-m (linear scale).
    """
    if cfg.logdomain:
        z = np.concatenate([[0.0],
                             10.0 ** model[1:, 0],
                             [10.0 ** cfg.max_z]])
    else:
        z = np.concatenate([[0.0], model[1:, 0], [cfg.max_z]])

    thicknesses = cfg.scale * np.diff(z)           # metres
    resistivities = 10.0 ** model[:, 1]            # ohm-m
    return thicknesses, resistivities


# ------------------------------------------------------------------ #
# Forward response
# ------------------------------------------------------------------ #

def forward_response(model: np.ndarray, cfg) -> dict:
    """
    Compute predicted data for the current model.

    Returns a dict with keys 'MT' and/or 'DC', each containing the
    predicted observables.  Mirrors Fwd_response.m.
    """
    thicknesses, resistivities = model_to_layers(model, cfg)
    dhat = {}

    method = cfg.inversion_method.upper()
    if method in ("MT", "MT_DC"):
        Z, appres, phase = mt1d_forward(resistivities, thicknesses, cfg.MT.period)
        dtype = cfg.MT.datatype.upper()
        mt_out = {}
        if dtype == "Z":
            mt_out["data_Z"] = Z
        elif dtype == "APP":
            mt_out["data_appres"] = np.log10(appres)
        elif dtype == "PHASE":
            mt_out["data_phase"] = phase
        elif dtype == "APP_PHASE":
            mt_out["data_appres"] = np.log10(appres)
            mt_out["data_phase"] = phase
        dhat["MT"] = mt_out

    if method in ("DC", "MT_DC"):
        AB2 = cfg.DC.OA
        rho_log10 = model[:, 1]
        dhat["DC"] = {"data_appres": dc1d_forward(AB2, thicknesses, rho_log10)}

    return dhat


# ------------------------------------------------------------------ #
# Uncertainty
# ------------------------------------------------------------------ #

def estimate_uncertainty(sigma: np.ndarray, cfg) -> dict:
    """
    Scale the data errors by the noise hyperparameter sigma.
    Mirrors estimate_uncertanity.m.
    """
    method = cfg.inversion_method.upper()
    dtype = cfg.MT.datatype.upper()
    unc = {}

    if method in ("MT", "MT_DC"):
        if dtype == "Z":
            unc["MT_Z"] = sigma[0] * cfg.MT.err_Z
        elif dtype == "APP":
            unc["MT_appres"] = sigma[0] * cfg.MT.err_appres
        elif dtype == "PHASE":
            unc["MT_phase"] = sigma[0] * cfg.MT.err_phase
        elif dtype == "APP_PHASE":
            unc["MT_appres"] = sigma[0] * cfg.MT.err_appres
            unc["MT_phase"] = sigma[1] * cfg.MT.err_phase

    if method == "DC":
        unc["DC"] = sigma[0] * cfg.DC.err_appres

    if method == "MT_DC":
        idx_dc = 1 if dtype == "Z" else 2
        unc["DC"] = sigma[idx_dc] * cfg.DC.err_appres

    return unc


# ------------------------------------------------------------------ #
# Likelihood (negative log-likelihood)
# ------------------------------------------------------------------ #

def compute_residual(dhat: dict, uncertainty: dict, cfg) -> tuple[float, np.ndarray]:
    """
    Compute the negative log-likelihood (stored as a positive 'likelihood'
    value so that a smaller value = better fit) and the normalised RMS.

    Mirrors Residual.m.

    Note on convention (same as MATLAB):
      For Z-data: LF = sum(|res/unc|^2)  (factor of 2 already included so
                  acceptance ratio divides by 2×temperature)
      For real data: LF = 0.5 × sum((res/unc)^2)
    """
    method = cfg.inversion_method.upper()
    dtype = cfg.MT.datatype.upper()

    LF_MT = 0.0;  np_MT = 0
    LF_DC = 0.0;  np_DC = 0

    if method in ("MT", "MT_DC"):
        if dtype == "Z":
            res = cfg.MT.dobs_Z - dhat["MT"]["data_Z"]
            err = res / uncertainty["MT_Z"]
            LF_MT = float(np.sum((err.conj() * err).real))
            LF_MT *= 2.0      # factor-of-2 convention (matches MATLAB)
        elif dtype == "APP":
            res = cfg.MT.dobs_appres - dhat["MT"]["data_appres"]
            LF_MT = float(np.sum((res / uncertainty["MT_appres"]) ** 2))
        elif dtype == "PHASE":
            res = cfg.MT.dobs_phase - dhat["MT"]["data_phase"]
            LF_MT = float(np.sum((res / uncertainty["MT_phase"]) ** 2))
        elif dtype == "APP_PHASE":
            res = cfg.MT.dobs_appres - dhat["MT"]["data_appres"]
            LF_MT = float(np.sum((res / uncertainty["MT_appres"]) ** 2))
            res = cfg.MT.dobs_phase - dhat["MT"]["data_phase"]
            LF_MT += float(np.sum((res / uncertainty["MT_phase"]) ** 2))
        np_MT = cfg.MT.ndata

    if method in ("DC", "MT_DC"):
        res = cfg.DC.dobs_appres - dhat["DC"]["data_appres"]
        LF_DC = float(np.sum((res / uncertainty["DC"]) ** 2))
        np_DC = len(cfg.DC.dobs_appres)

    # Combined likelihood and nRMS
    if method == "MT":
        LF = LF_MT if dtype == "Z" else 0.5 * LF_MT
        nrms = np.array([np.sqrt(LF_MT / max(np_MT, 1))])
    elif method == "DC":
        LF = 0.5 * LF_DC
        nrms = np.array([np.sqrt(LF_DC / max(np_DC, 1))])
    else:  # MT_DC
        LF = (0.5 * LF_DC + LF_MT) if dtype == "Z" else 0.5 * (LF_DC + LF_MT)
        nrms = np.array([
            np.sqrt(LF_MT / max(np_MT, 1)),
            np.sqrt(LF_DC / max(np_DC, 1)),
            np.sqrt((LF_MT + LF_DC) / max(np_MT + np_DC, 1)),
        ])

    return LF, nrms


# ------------------------------------------------------------------ #
# Log-normalisation for noise proposal
# ------------------------------------------------------------------ #

def estimate_like_norm(unc_proposed: dict, unc_current: dict,
                       sigma: np.ndarray, cfg) -> tuple[float, float]:
    """
    Compute the log-normalisation factor when the noise hyperparameter changes.
    Mirrors estimate_like_norm.m.
    """
    method = cfg.inversion_method.upper()
    dtype = cfg.MT.datatype.upper()
    norm = 0.0
    reg_term = 0.0

    if method in ("MT", "MT_DC"):
        if dtype == "Z":
            temp = np.log(2.0 * unc_current["MT_Z"] / unc_proposed["MT_Z"])
            norm += float(np.sum(temp))
            reg_term = -0.5 * ((sigma[0] - 1.0) / 0.2) ** 2
        elif dtype == "APP":
            temp = np.log(unc_current["MT_appres"] / unc_proposed["MT_appres"])
            norm += float(np.sum(temp))
            reg_term = -0.5 * ((sigma[0] - 1.0) / 0.2) ** 2
        elif dtype == "PHASE":
            temp = np.log(unc_current["MT_phase"] / unc_proposed["MT_phase"])
            norm += float(np.sum(temp))
            reg_term = -0.5 * ((sigma[0] - 1.0) / 0.2) ** 2
        elif dtype == "APP_PHASE":
            norm += float(np.sum(np.log(
                unc_current["MT_appres"] / unc_proposed["MT_appres"]
            )))
            norm += float(np.sum(np.log(
                unc_current["MT_phase"] / unc_proposed["MT_phase"]
            )))
            reg_term = -0.5 * ((sigma[0] - 1.0) / 0.2) ** 2

    if method in ("DC", "MT_DC"):
        temp = np.log(unc_current["DC"] / unc_proposed["DC"])
        norm += float(np.sum(temp))

    return norm, reg_term
