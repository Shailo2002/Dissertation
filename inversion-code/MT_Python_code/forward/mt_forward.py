"""
1D Magnetotelluric forward modelling using the recursive impedance method.

Translated from MATLAB MT1DFW.m
Reference: Weidelt (1972), Wait (1954)

For a stack of N horizontal layers, the surface impedance Z is computed
by recursively propagating from the basement upward through each layer.
"""

import numpy as np


def mt1d_forward(resistivities, thicknesses, periods):
    """
    Compute 1D MT impedance, apparent resistivity, and phase.

    Parameters
    ----------
    resistivities : array-like, shape (n_layers,)
        Layer resistivities in ohm-m (linear scale). Last value is basement.
    thicknesses : array-like, shape (n_layers,)
        Layer thicknesses in metres. Last value is ignored (basement half-space).
    periods : array-like, shape (n_periods,)
        Periods in seconds.

    Returns
    -------
    Z : ndarray, complex, shape (n_periods,)
        MT impedance in ohm  (V/m / A/m = ohm).
    appres : ndarray, shape (n_periods,)
        Apparent resistivity in ohm-m.
    phase : ndarray, shape (n_periods,)
        Impedance phase in degrees.
    """
    mu = 4.0 * np.pi * 1e-7          # magnetic permeability H/m
    omega = 2.0 * np.pi / np.asarray(periods, dtype=np.float64)
    resistivities = np.asarray(resistivities, dtype=np.float64)
    thicknesses = np.asarray(thicknesses, dtype=np.float64)
    n = len(resistivities)

    Z = np.zeros(len(omega), dtype=complex)

    # Extreme RJMCMC proposals can push the exp() argument into overflow /
    # underflow territory. The likelihood layer detects the resulting NaN/Inf
    # and rejects such proposals, so silence the warnings here rather than
    # letting Python's warnings machinery slow every forward call.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for ip, w in enumerate(omega):
            # Basement (half-space) impedance
            Zn = np.sqrt(1j * w * mu * resistivities[-1])
            imp = np.empty(n, dtype=complex)
            imp[-1] = Zn

            # Recurse upward from layer n-2 to 0
            for j in range(n - 2, -1, -1):
                rho_j = resistivities[j]
                h_j = thicknesses[j]

                dj = np.sqrt(1j * w * mu / rho_j)   # induction parameter
                wj = dj * rho_j                       # intrinsic impedance
                ej = np.exp(-2.0 * h_j * dj)         # exponential decay factor

                rj = (wj - imp[j + 1]) / (wj + imp[j + 1])  # reflection coeff
                re = rj * ej
                imp[j] = wj * (1.0 - re) / (1.0 + re)

            Z[ip] = imp[0]

        absZ = np.abs(Z)
        appres = (absZ ** 2) / (mu * omega)
        phase = np.degrees(np.arctan2(Z.imag, Z.real))

    return Z, appres, phase
