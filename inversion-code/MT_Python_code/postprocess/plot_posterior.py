"""
Plot posterior distribution (3-panel figure).

Equivalent to MATLAB: Process_chains/A_03_plot_posteriori_3_things.m

Panels:
  1. Posterior PDF in depth × log10(resistivity) space
     with 5th/95th percentile contours and optional true model
  2. Interface probability density vs depth
  3. Histogram of number of subsurface layers

Usage
-----
  python postprocess/plot_posterior.py --folder results --prefix MT
  python postprocess/plot_posterior.py --folder results --true_model MT_data.npz
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _true_model_staircase(layer_tops_m: np.ndarray, rho_ohm: np.ndarray, z_max: float):
    """
    Build staircase (x, y) for a 1-D layered model on a depth-vs-rho plot.

    Parameters
    ----------
    layer_tops_m : array  layer top depths in metres, e.g. [0, 10000, 50000]
    rho_ohm      : array  resistivities in ohm-m,     e.g. [100, 10, 1000]
    z_max        : float  maximum depth to draw the basement (metres)

    Returns
    -------
    x : log10(rho) values for the staircase line
    y : depth values (metres, increasing downward)
    """
    x, y = [], []
    n = len(rho_ohm)
    for i in range(n):
        z_top = layer_tops_m[i]
        z_bot = layer_tops_m[i + 1] if i + 1 < n else z_max
        log_rho = np.log10(rho_ohm[i])
        # Go down through the layer at constant rho
        x += [log_rho, log_rho]
        y += [z_top,   z_bot]
        # Step horizontally to next layer's rho at the interface
        if i + 1 < n:
            x += [np.log10(rho_ohm[i + 1])]
            y += [z_bot]
    return np.array(x), np.array(y)


def plot_posterior(
    folder: str,
    prefix: str = "MT",
    true_model_file: str = None,
    true_depths: list = None,
    true_rho: list = None,
    output: str = None,
):
    stat_path = os.path.join(folder, f"{prefix}_TD_Chain_Stat_info.npz")
    proc_path = os.path.join(folder, f"{prefix}_TD_Chain_Processed.npz")

    if not os.path.exists(stat_path):
        print(f"Stat info file not found: {stat_path}")
        print("Run process_chains.py first.")
        return

    # ---- Auto-load true model from results folder if not given ----
    if true_depths is None and true_rho is None:
        tm_path = os.path.join(folder, "true_model.json")
        if os.path.exists(tm_path):
            with open(tm_path) as fj:
                tm = json.load(fj)
            true_depths = tm["layer_tops"]
            true_rho    = tm["layer_rho"]
            print(f"  Auto-loaded true model from {tm_path}")

    S = np.load(stat_path, allow_pickle=True)
    z_plot      = S["z_plot"]
    rho_plot    = S["rho_plot"]
    post_pdf    = S["posterior_pdf"]
    p5          = S["p5"]
    p95         = S["p95"]
    k_samples   = S["k_samples"]
    ngrid       = S["ngrid"]
    dz          = float(S["dz"])
    z_max       = float(S["z_max"])

    # ---- Figure setup ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 8))
    fig.suptitle("1D MT Transdimensional Bayesian Inversion\nPosterior Distribution",
                 fontsize=13)

    # ---- Panel 1: Posterior PDF ----
    ax1 = axes[0]
    pdf_plot = post_pdf.copy()
    pdf_plot[pdf_plot <= 0] = np.nan

    mesh = ax1.pcolormesh(
        rho_plot, z_plot,
        np.log10(pdf_plot + 1e-10),
        shading="auto", cmap="viridis",
    )
    # 5th and 95th percentile contours
    ax1.step(p5,  z_plot, "r-",  linewidth=1.5, where="mid", label="5th/95th %ile")
    ax1.step(p95, z_plot, "r-",  linewidth=1.5, where="mid")

    # ---- Overlay true model if provided ----
    z_true_m = None
    rho_true_ohm = None

    # Option A: passed as --true_depths and --true_rho on command line
    if true_depths is not None and true_rho is not None:
        z_true_m     = np.array(true_depths, dtype=float)
        rho_true_ohm = np.array(true_rho,   dtype=float)

    # Option B: loaded from a .npz file
    elif true_model_file and os.path.exists(true_model_file):
        tm = np.load(true_model_file, allow_pickle=True)
        z_true_m     = tm["z"].astype(float)
        rho_true_ohm = tm["rho"].astype(float)

    if z_true_m is not None and rho_true_ohm is not None:
        xs, ys = _true_model_staircase(z_true_m, rho_true_ohm, z_max)
        ax1.plot(xs, ys, "--w", linewidth=2.5, label="True model")   # white outline
        ax1.plot(xs, ys, "--k", linewidth=1.5)                        # black dashes on top

    ax1.set_xlabel(r"log$_{10}$($\rho$) (ohm-m)", fontsize=11)
    ax1.set_ylabel("Depth (m)", fontsize=11)
    ax1.set_ylim([0, z_max])
    ax1.invert_yaxis()
    ax1.set_xlim([rho_plot[0], rho_plot[-1]])
    ax1.legend(fontsize=9, loc="lower right")
    cbar = plt.colorbar(mesh, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(r"log$_{10}$ (PDF)", fontsize=10)
    ax1.set_title("Posterior PDF", fontsize=11)

    # ---- Panel 2: Interface probability density ----
    ax2 = axes[1]
    k_pdf = np.nansum(k_samples, axis=1) / (np.nansum(k_samples) * dz)
    k_prior = np.nanmean(k_pdf)
    k_pdf[0] = 0.0     # suppress surface interface (always present)

    ax2.plot(k_pdf, z_plot[: len(k_pdf)], linewidth=2, color="steelblue",
             label="Posterior")
    ax2.axvline(k_prior, color="k", linestyle="--", linewidth=1.5, label="Prior")
    # Mark true interface depths as horizontal lines
    if z_true_m is not None:
        for iz, zd in enumerate(z_true_m):
            if zd > 0:   # skip surface
                ax2.axhline(zd, color="r", linestyle="--", linewidth=1.5,
                            label="True interface" if iz == 1 else "")
    ax2.set_xlabel("Probability density", fontsize=11)
    ax2.set_ylabel("Depth (m)", fontsize=11)
    ax2.set_ylim([0, z_max])
    ax2.invert_yaxis()
    ax2.legend(fontsize=9)
    ax2.set_title("Interface Probability", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    # ---- Panel 3: Histogram of number of layers ----
    ax3 = axes[2]
    bins = np.arange(ngrid.min() - 0.5, ngrid.max() + 1.5, 1)
    ax3.hist(ngrid, bins=bins, density=True, color="steelblue",
             edgecolor="white", linewidth=0.5)
    # Mark true number of layers
    if rho_true_ohm is not None:
        ax3.axvline(len(rho_true_ohm), color="r", linestyle="--",
                    linewidth=2, label=f"True = {len(rho_true_ohm)} layers")
        ax3.legend(fontsize=9)
    ax3.set_xlabel("# subsurface layers", fontsize=11)
    ax3.set_ylabel("Probability density", fontsize=11)
    ax3.set_title("Number of Layers", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output is None:
        output = os.path.join(folder, f"{prefix}_posterior.png")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Posterior plot saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot posterior distribution")
    parser.add_argument("--folder",       default="results")
    parser.add_argument("--prefix",       default="MT", choices=["MT", "DC", "MT_DC"])
    parser.add_argument("--true_model",   default=None,
                        help="Path to true model .npz file (optional)")
    parser.add_argument("--true_depths",  nargs="+", type=float, default=None,
                        help="True layer top depths in metres, e.g. --true_depths 0 10000 50000")
    parser.add_argument("--true_rho",     nargs="+", type=float, default=None,
                        help="True layer resistivities in ohm-m, e.g. --true_rho 100 10 1000")
    parser.add_argument("--output",       default=None)
    args = parser.parse_args()
    plot_posterior(args.folder, args.prefix, args.true_model,
                   args.true_depths, args.true_rho, args.output)


if __name__ == "__main__":
    main()
