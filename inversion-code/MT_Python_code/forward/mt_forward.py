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

    # Extreme RJMCMC proposals can push the exp() argument into overflow /
    # underflow territory. The likelihood layer detects the resulting NaN/Inf
    # and rejects such proposals, so silence the warnings here rather than
    # letting Python's warnings machinery slow every forward call.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        # Vectorised over all frequencies simultaneously: imp has shape (n_periods,)
        # Basement (half-space) impedance for every frequency at once
        imp = np.sqrt(1j * omega * mu * resistivities[-1])

        # Recurse upward from layer n-2 to 0; each iteration is a vectorised
        # numpy operation across all frequencies — eliminates the inner loop.
        for j in range(n - 2, -1, -1):
            rho_j = resistivities[j]
            h_j   = thicknesses[j]
            dj = np.sqrt(1j * omega * mu / rho_j)
            wj = dj * rho_j
            ej = np.exp(-2.0 * h_j * dj)
            rj = (wj - imp) / (wj + imp)
            imp = wj * (1.0 - rj * ej) / (1.0 + rj * ej)

        Z = imp
        absZ = np.abs(Z)
        appres = (absZ ** 2) / (mu * omega)
        phase = np.degrees(np.arctan2(Z.imag, Z.real))

    return Z, appres, phase
