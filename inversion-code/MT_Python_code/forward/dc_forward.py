"""
1D DC resistivity (Schlumberger array) forward modelling.

Translated from MATLAB DC1DFW.m
Uses the digital linear filter method of Ghosh (1971).
Returns log10(apparent resistivity) to match MATLAB convention.
"""

import numpy as np

# Ghosh (1971) Schlumberger filter coefficients
_FILTER_J1 = -5
_FILTER_J2 = 13
_FILTER_S = -0.14452175
_FILTER_DY = 0.48052648
_FILTER_F = np.array([
    0.00097112, -0.00102152,  0.00906965,  0.01404316,  0.09012000,
    0.30171582,  0.99627084,  1.36908320, -2.99681171,  1.65463068,
   -0.59399277,  0.22329813, -0.10119309,  0.05186135, -0.02748647,
    0.01384932, -0.00599074,  0.00190463, -0.00032160,
])


def _kr_transform(lam: float, thicknesses: np.ndarray, rho_log10: np.ndarray) -> float:
    """
    Koefoed resistivity transform (recursive kernel).

    Parameters
    ----------
    lam : float
        Filter argument (1/lambda).
    thicknesses : ndarray
        Layer thicknesses in metres.
    rho_log10 : ndarray
        Layer resistivities in log10(ohm-m).
    """
    trans = 10.0 ** rho_log10[-1]
    for i in range(len(rho_log10) - 2, -1, -1):
        tlt = np.tanh(thicknesses[i] * lam)
        trans = (trans + 10.0 ** rho_log10[i] * tlt) / (
            1.0 + trans * tlt / 10.0 ** rho_log10[i]
        )
    return trans


def dc1d_forward(AB2: np.ndarray, thicknesses: np.ndarray, rho_log10: np.ndarray) -> np.ndarray:
    """
    Compute 1D DC Schlumberger apparent resistivity using digital linear filters.

    Parameters
    ----------
    AB2 : ndarray, shape (n_data,)
        Half electrode spacing AB/2 in metres.
    thicknesses : ndarray, shape (n_layers,)
        Layer thicknesses in metres.
    rho_log10 : ndarray, shape (n_layers,)
        Layer resistivities in log10(ohm-m).

    Returns
    -------
    r : ndarray, shape (n_data,)
        Apparent resistivity in log10(ohm-m).
    """
    AB2 = np.asarray(AB2, dtype=np.float64)
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    rho_log10 = np.asarray(rho_log10, dtype=np.float64)
    n_filter = _FILTER_J2 - _FILTER_J1 + 1

    r = np.zeros(len(AB2))
    for j, ab in enumerate(AB2):
        off = np.log(ab) + _FILTER_S + _FILTER_DY * (1 - _FILTER_J1)
        r_val = 0.0
        for i in range(n_filter):
            off -= _FILTER_DY
            lam = 1.0 / np.exp(off)
            r_val += _kr_transform(lam, thicknesses, rho_log10) * _FILTER_F[i]
        r[j] = np.log10(r_val)

    return r
