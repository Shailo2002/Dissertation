"""
Plot chain convergence diagnostics.

Equivalent to MATLAB: Process_chains/A_01_see_chain_convergence.m

Plots:
  1. nRMS vs sample number (log x-axis)
  2. log(likelihood) vs sample number
  3. Noise hyperparameter (sigma/lambda) vs sample number

Usage
-----
  python postprocess/chain_convergence.py --folder results --prefix MT
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_io.chain_io import load_chain, load_config


def plot_convergence(folder: str, prefix: str = "MT", output: str = None):
    """Load all chains and plot convergence."""
    cfg = load_config(folder)
    nChains = cfg["nChains"]
    nsamples = cfg["nsamples"]

    print(f"Processing {prefix} chains (nChains={nChains}) ...")

    # Collect data across all chains
    like_all, nrms_all, sigma_all = [], [], []

    for ic in range(1, nChains + 1):
        try:
            chain = load_chain(folder, ic, prefix)
        except FileNotFoundError:
            print(f"  Chain {ic} not found, skipping")
            continue

        like_all.append(chain["like"])
        nrms_all.append(chain["misfit"][:, 0])   # first nRMS column
        sigma_all.append(chain["sigma"])
        print(f"  Chain {ic}: {len(chain['like'])} samples")

    if not like_all:
        print("No chains found.")
        return

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    total_samples = max(len(x) for x in nrms_all)
    x = np.arange(1, total_samples + 1)

    # Panel 1: nRMS
    ax = axes[0]
    for nrms in nrms_all:
        ax.plot(np.arange(1, len(nrms) + 1), nrms, "-", alpha=0.7, linewidth=0.8)
    ax.axhline(1, color="k", linestyle="--", linewidth=1.5, label="nRMS = 1")
    ax.set_xscale("log")
    ax.set_xlabel("# samples")
    ax.set_ylabel("nRMS")
    ax.set_ylim(-2, 10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Chain Convergence")

    # Panel 2: log(likelihood)
    ax = axes[1]
    for like in like_all:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_like = np.where(like > 0, np.log10(like), np.nan)
        ax.plot(np.arange(1, len(log_like) + 1), log_like, "-", alpha=0.7, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("# samples")
    ax.set_ylabel("log10(likelihood)")
    ax.grid(True, which="both", alpha=0.3)

    # Panel 3: noise hyperparameter
    ax = axes[2]
    for sigma in sigma_all:
        ax.plot(np.arange(1, len(sigma) + 1), sigma[:, 0], "-", alpha=0.7, linewidth=0.8)
        if sigma.shape[1] > 1:
            ax.plot(np.arange(1, len(sigma) + 1), sigma[:, 1], "--", alpha=0.7, linewidth=0.8)
    ax.axhline(1, color="k", linestyle="--", linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("# samples")
    ax.set_ylabel("sigma (noise factor)")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if output is None:
        output = os.path.join(folder, f"{prefix}_convergence.png")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Convergence plot saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot chain convergence")
    parser.add_argument("--folder", default="results", help="Results folder")
    parser.add_argument("--prefix", default="MT",
                        choices=["MT", "DC", "MT_DC"])
    parser.add_argument("--output", default=None, help="Output image path")
    args = parser.parse_args()
    plot_convergence(args.folder, args.prefix, args.output)


if __name__ == "__main__":
    main()
