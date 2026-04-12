"""
Create synthetic 1D MT dataset for testing.

Translated from MATLAB: MT_DC_function/create_MT_dataset.m

Usage
-----
  python create_synthetic_data.py
  python create_synthetic_data.py --noise 3 --nperiods 20 --output my_test_data

The script:
  1. Computes the noise-free forward response for a 3-layer model.
  2. Adds Gaussian noise.
  3. Writes MT_data_Z.dat and MT_data.dat (compatible with run_inversion.py).
  4. Plots apparent resistivity and phase.
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from forward.mt_forward import mt1d_forward


def create_synthetic_mt(
    noise_prcnt: float = 3.0,
    nperiods: int = 13,
    period_min: float = 1.0,
    period_max: float = 10_000.0,
    layer_tops: np.ndarray = None,
    layer_rho: np.ndarray = None,
    output_dir: str = ".",
    plot: bool = True,
):
    """
    Generate synthetic MT data.

    Default model (matches MATLAB):
      Layer 1:  0 –  10 km,  100 ohm-m
      Layer 2: 10 –  50 km,   10 ohm-m  (conductor)
      Layer 3: 50+ km,       1000 ohm-m (resistive basement)
    """
    # ---- Default 3-layer model ----
    if layer_tops is None:
        layer_tops = np.array([0, 10_000, 50_000], dtype=float)  # metres
    if layer_rho is None:
        layer_rho = np.array([100.0, 10.0, 1000.0])

    thicknesses = np.diff(np.append(layer_tops, layer_tops[-1] * 1.5))

    # ---- Periods ----
    periods = np.logspace(np.log10(period_min), np.log10(period_max), nperiods)

    # ---- Noise-free forward response ----
    Z_true, appres_true, phase_true = mt1d_forward(layer_rho, thicknesses, periods)

    # ---- Add Gaussian noise to impedance ----
    noise_scale = 0.01 * noise_prcnt
    rng = np.random.default_rng(seed=42)
    Z_noise = rng.standard_normal(nperiods) + 1j * rng.standard_normal(nperiods)
    Z_obs = Z_true + noise_scale * Z_noise * Z_true
    err_Z = noise_scale * np.abs(Z_obs)   # 1-sigma error on |Z|

    # ---- Recompute app-res and phase from noisy Z ----
    mu = 4.0 * np.pi * 1e-7
    omega = 2.0 * np.pi / periods
    absZ = np.abs(Z_obs)
    appres_obs = absZ ** 2 / (mu * omega)
    phase_obs = np.degrees(np.arctan2(Z_obs.imag, Z_obs.real))

    err_appres = 2 * absZ * err_Z / (mu * omega)   # propagated
    err_phase = (err_Z / absZ) * (180.0 / np.pi)

    # ---- Save MT_data_Z.dat ----
    os.makedirs(output_dir, exist_ok=True)
    z_path = os.path.join(output_dir, "MT_data_Z.dat")
    with open(z_path, "w") as fid:
        for i in range(nperiods):
            fid.write(
                f"{periods[i]:12.5f} {Z_obs[i].real:+10.4E} "
                f"{Z_obs[i].imag:+10.4E} {err_Z[i]:+10.4E}\n"
            )
    print(f"Written: {z_path}")

    # ---- Save MT_data.dat (app_phase format) ----
    dat_path = os.path.join(output_dir, "MT_data.dat")
    with open(dat_path, "w") as fid:
        for i in range(nperiods):
            fid.write(
                f"{periods[i]:12.5f} {appres_obs[i]:+10.4E} {err_appres[i]:+10.4E} "
                f"{phase_obs[i]:+10.4E} {err_phase[i]:+10.4E}\n"
            )
    print(f"Written: {dat_path}")

    # ---- Plot ----
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax1, ax2 = axes

        ax1.loglog(periods, appres_true, "k-", label="True (noise-free)", linewidth=2)
        ax1.errorbar(periods, appres_obs, yerr=err_appres,
                     fmt="ro", markersize=5, label="Observed (with noise)")
        ax1.set_ylabel("Apparent Resistivity (ohm-m)")
        ax1.legend(fontsize=9)
        ax1.grid(True, which="both", alpha=0.3)

        ax2.semilogx(periods, phase_true, "k-", linewidth=2)
        ax2.errorbar(periods, phase_obs, yerr=err_phase, fmt="ro", markersize=5)
        ax2.set_ylabel("Phase (degrees)")
        ax2.set_xlabel("Period (s)")
        ax2.grid(True, which="both", alpha=0.3)

        plt.suptitle(f"Synthetic MT Data  ({noise_prcnt}% noise)", fontsize=12)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "synthetic_data_plot.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {fig_path}")

    # ---- Save true_model.json ----
    tm_path = os.path.join(output_dir, "true_model.json")
    with open(tm_path, "w") as fj:
        json.dump({
            "layer_tops": layer_tops.tolist(),
            "layer_rho":  layer_rho.tolist(),
        }, fj, indent=2)
    print(f"Written: {tm_path}")

    # ---- Print true model summary ----
    print("\nTrue model:")
    print(f"  {'Layer':>8}  {'Top (m)':>12}  {'Resistivity (ohm-m)':>22}")
    for i, (z, r) in enumerate(zip(layer_tops, layer_rho)):
        print(f"  {i+1:>8}  {z:>12.0f}  {r:>22.1f}")
    print(f"  (basement below {layer_tops[-1]:.0f} m)")

    return Z_obs, err_Z, appres_obs, phase_obs


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic 1D MT dataset"
    )
    parser.add_argument("--noise",       type=float, default=3.0,
                        help="Noise level in percent (default 3)")
    parser.add_argument("--nperiods",    type=int,   default=13,
                        help="Number of periods (default 13)")
    parser.add_argument("--pmin",        type=float, default=1.0,
                        help="Minimum period in seconds")
    parser.add_argument("--pmax",        type=float, default=10_000.0,
                        help="Maximum period in seconds")
    parser.add_argument("--output",      default=".",
                        help="Output directory for data files")
    parser.add_argument("--no-plot",     action="store_true",
                        help="Skip plotting")
    parser.add_argument("--layer_tops",  nargs="+", type=float, default=None,
                        help="Layer top depths in metres, e.g. --layer_tops 0 25000 80000 200000")
    parser.add_argument("--layer_rho",   nargs="+", type=float, default=None,
                        help="Layer resistivities in ohm-m, e.g. --layer_rho 500 5 5000 20")
    args = parser.parse_args()

    layer_tops = np.array(args.layer_tops) if args.layer_tops else None
    layer_rho  = np.array(args.layer_rho)  if args.layer_rho  else None

    create_synthetic_mt(
        noise_prcnt=args.noise,
        nperiods=args.nperiods,
        period_min=args.pmin,
        period_max=args.pmax,
        layer_tops=layer_tops,
        layer_rho=layer_rho,
        output_dir=args.output,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
