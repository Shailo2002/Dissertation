"""
Plot noise hyperparameter (sigma/lambda) evolution and distribution.

Equivalent to MATLAB: Process_chains/A_04_plot_noise.m

Usage
-----
  python postprocess/plot_noise.py --folder results --prefix MT
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def plot_noise(folder: str, prefix: str = "MT", output: str = None):
    proc_path = os.path.join(folder, f"{prefix}_TD_Chain_Processed.npz")
    if not os.path.exists(proc_path):
        print(f"Processed file not found: {proc_path}")
        print("Run process_chains.py first.")
        return

    data = np.load(proc_path, allow_pickle=True)
    sigma = data["sigma"]    # shape (n_samples, n_sigma_params)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Panel 1: trace of sigma
    ax1 = axes[0]
    n_samples = sigma.shape[0]
    x = np.arange(1, n_samples + 1)
    ax1.plot(x, sigma[:, 0], "-k", linewidth=0.7, label="sigma_1 (MT)")
    if sigma.shape[1] > 1:
        ax1.plot(x, sigma[:, 1], "-r", linewidth=0.7, label="sigma_2 (DC/phase)")
    ax1.axhline(1.0, color="k", linestyle="--", linewidth=1.2)
    ax1.set_xlabel("# samples")
    ax1.set_ylabel("Lambda (noise factor)")
    ax1.legend(fontsize=9)
    ax1.set_title("Noise Hyperparameter Trace", fontsize=11)
    ax1.grid(alpha=0.3)

    # Panel 2: histogram of sigma
    ax2 = axes[1]
    ax2.hist(sigma[:, 0], bins=50, density=True,
             color="steelblue", edgecolor="white", linewidth=0.5, label="sigma_1")
    if sigma.shape[1] > 1:
        ax2.hist(sigma[:, 1], bins=50, density=True, alpha=0.6,
                 color="tomato", edgecolor="white", linewidth=0.5, label="sigma_2")
    ax2.set_xlabel("Lambda")
    ax2.set_ylabel("p(lambda)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if output is None:
        output = os.path.join(folder, f"{prefix}_noise.png")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Noise plot saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot noise hyperparameter")
    parser.add_argument("--folder", default="results")
    parser.add_argument("--prefix", default="MT", choices=["MT", "DC", "MT_DC"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    plot_noise(args.folder, args.prefix, args.output)


if __name__ == "__main__":
    main()
