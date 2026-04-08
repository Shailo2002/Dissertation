"""
Process MCMC chains: apply burn-in, thinning, binning, and compute posterior PDF.

Equivalent to MATLAB: Process_chains/A_02_process_chains.m

Outputs
-------
  <folder>/<prefix>_TD_Chain_Processed.npz
     Contains z_all, rho_all, like_all, ngrid, step, nrms_all, sigma

  <folder>/<prefix>_TD_Chain_Stat_info.npz
     Contains posterior PDF, percentiles, KL-divergence, interface PDF

Usage
-----
  python postprocess/process_chains.py --folder results --prefix MT
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_io.chain_io import load_chain, load_config


# ------------------------------------------------------------------ #
# KL divergence
# ------------------------------------------------------------------ #

def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """Kullback-Leibler divergence D_KL(P || Q), ignoring zero entries."""
    kld = 0.0
    for p, q in zip(P, Q):
        if p > 0 and q > 0:
            kld += p * (np.log(p) - np.log(q))
    return kld


# ------------------------------------------------------------------ #
# Main processing
# ------------------------------------------------------------------ #

def process_chains(
    folder: str,
    prefix: str = "MT",
    step_discard_frac: float = 0.5,
    chain_thin: int = 25,
    nrms_limit: float = 20.0,
    z_min: float = 0.0,
    z_max: float = 350_000.0,
    dz: float = 1_000.0,
    rho_min: float = -1.0,
    rho_max: float = 6.0,
    drho: float = 0.1,
):
    """
    Apply burn-in and thinning, bin samples, compute posterior PDF.
    """
    cfg = load_config(folder)
    nChains_atT1 = cfg["nChains_atT1"]
    nsamples_per_step = cfg["nsamples"]
    maxnodes = cfg["maxnodes"]
    logdomain = cfg["logdomain"]
    nsteps = cfg["nsteps"]

    step_discard = int(nsteps * step_discard_frac)

    print(f"Processing {prefix} chains ...")
    print(f"  Chains at T=1 : {nChains_atT1}")
    print(f"  nsteps        : {nsteps}")
    print(f"  Samples/step  : {nsamples_per_step}")
    print(f"  Burn-in steps : {step_discard}")
    print(f"  Thinning      : 1/{chain_thin}")

    # ---- Collect thinned samples from cold chains ----
    z_list, rho_list, like_list, ngrid_list, step_list, nrms_list = [], [], [], [], [], []
    sigma_list = []

    for ic in range(1, nChains_atT1 + 1):
        try:
            chain = load_chain(folder, ic, prefix)
        except FileNotFoundError:
            print(f"  Chain {ic} not found, skipping")
            continue

        print(f"  Processing chain {ic} ...")

        # Reshape chain into steps
        total = len(chain["like"])
        # Each step has nsamples_per_step rows
        for istep in range(nsteps):
            if istep < step_discard:
                continue
            i1 = istep * nsamples_per_step
            i2 = min(i1 + nsamples_per_step, total)
            if i1 >= total:
                break

            idx_range = range(i1, i2, chain_thin)
            for i in idx_range:
                ng = int(chain["ncells"][i])
                if logdomain:
                    z_row = np.concatenate([[0.0],
                                            10.0 ** chain["model"][i, 1:ng, 0]])
                else:
                    z_row = np.concatenate([[0.0], chain["model"][i, 1:ng, 0]])

                z_list.append(z_row)
                rho_list.append(chain["model"][i, :ng, 1])
                like_list.append(chain["like"][i])
                ngrid_list.append(ng)
                step_list.append(chain["step"][i])
                nrms_list.append(chain["misfit"][i, :])
                sigma_list.append(chain["sigma"][i, :])

    if not z_list:
        print("No samples collected. Check burn-in / chain settings.")
        return

    z_all = np.array(z_list, dtype=object)
    rho_all = np.array(rho_list, dtype=object)
    like_all = np.array(like_list)
    ngrid = np.array(ngrid_list, dtype=np.int32)
    step_arr = np.array(step_list, dtype=np.int32)
    nrms_all = np.vstack(nrms_list)
    sigma_arr = np.vstack(sigma_list)

    # ---- Filter by nRMS ----
    valid = nrms_all[:, 0] < nrms_limit
    z_all = z_all[valid]
    rho_all = rho_all[valid]
    like_all = like_all[valid]
    ngrid = ngrid[valid]
    step_arr = step_arr[valid]
    nrms_all = nrms_all[valid]
    sigma_arr = sigma_arr[valid]

    nsamples_total = len(ngrid)
    print(f"  Samples after burn-in / thinning / nRMS filter: {nsamples_total}")

    # ---- Save processed arrays ----
    processed_path = os.path.join(folder, f"{prefix}_TD_Chain_Processed.npz")
    np.savez_compressed(
        processed_path,
        z_all=np.array(z_all, dtype=object),
        rho_all=np.array(rho_all, dtype=object),
        like_all=like_all,
        ngrid=ngrid,
        step=step_arr,
        nrms_all=nrms_all,
        sigma=sigma_arr,
    )
    print(f"Saved: {processed_path}")

    # ------------------------------------------------------------------ #
    # Binning: build posterior PDF on a regular depth × resistivity grid
    # ------------------------------------------------------------------ #
    print("Computing posterior PDF ...")

    n_z_bins = int(np.ceil((z_max - z_min) / dz))
    z_plot = z_min + dz / 2.0 + np.arange(n_z_bins) * dz   # bin midpoints

    rho_bins = np.arange(rho_min, rho_max + drho, drho)
    rho_plot = rho_min + drho / 2.0 + np.arange(len(rho_bins) - 1) * drho
    n_rho_bins = len(rho_bins) - 1
    prior = np.full(n_rho_bins, 1.0 / (rho_max - rho_min))

    rho_samples_2d = np.full((n_z_bins, nsamples_total), np.nan)
    k_samples = np.full((n_z_bins, nsamples_total), np.nan)

    for s in range(nsamples_total):
        ng = ngrid[s]
        z = np.append(z_all[s], z_max)     # depth boundaries
        rho = rho_all[s]

        for j in range(ng):
            iz1 = np.searchsorted(z_plot, z[j], side="left")
            iz2 = np.searchsorted(z_plot, z[j + 1], side="left")
            if iz1 < n_z_bins:
                iz2 = min(iz2, n_z_bins)
                rho_samples_2d[iz1:iz2, s] = rho[j]
                k_samples[iz1, s] = 1.0

        if (s + 1) % max(nsamples_total // 10, 1) == 0:
            print(f"  {(s+1)/nsamples_total*100:.0f}% samples binned ...")

    k_samples = k_samples[:-1, :]    # drop last bin (matches MATLAB)

    # ---- Posterior PDF, percentiles, KL divergence ----
    posterior_pdf = np.zeros((n_z_bins, n_rho_bins))
    p5  = np.zeros(n_z_bins)
    p95 = np.zeros(n_z_bins)
    pmean = np.zeros(n_z_bins)
    kld = np.zeros(n_z_bins)

    for iz in range(n_z_bins):
        row = rho_samples_2d[iz, :]
        valid_row = row[~np.isnan(row)]
        if len(valid_row) == 0:
            continue
        counts, _ = np.histogram(valid_row, bins=rho_bins, density=True)
        posterior_pdf[iz, :] = counts
        p5[iz]    = np.nanpercentile(row, 5)
        p95[iz]   = np.nanpercentile(row, 95)
        pmean[iz] = np.nanmean(row)
        kld[iz]   = kl_divergence(counts, prior)

    kld[0] = 0.0      # first interface always present, suppress

    # Shift percentile arrays by one (MATLAB convention)
    p5  = np.concatenate([[p5[0]],  p5[:-1]])
    p95 = np.concatenate([[p95[0]], p95[:-1]])

    # ---- Save stat info ----
    stat_path = os.path.join(folder, f"{prefix}_TD_Chain_Stat_info.npz")
    np.savez_compressed(
        stat_path,
        z_plot=z_plot,
        rho_plot=rho_plot,
        posterior_pdf=posterior_pdf,
        p5=p5, p95=p95, pmean=pmean, kld=kld,
        k_samples=k_samples,
        ngrid=ngrid,
        nrms_all=nrms_all,
        step=step_arr,
        dz=np.array(dz),
        z_max=np.array(z_max),
    )
    print(f"Saved: {stat_path}")
    print("Processing complete.")


def main():
    parser = argparse.ArgumentParser(description="Process MCMC chains")
    parser.add_argument("--folder",     default="results")
    parser.add_argument("--prefix",     default="MT", choices=["MT", "DC", "MT_DC"])
    parser.add_argument("--burnin",     type=float, default=0.5,
                        help="Fraction of steps to discard as burn-in (default 0.5)")
    parser.add_argument("--thin",       type=int,   default=25,
                        help="Thinning factor (default 25)")
    parser.add_argument("--nrms_limit", type=float, default=20.0)
    parser.add_argument("--zmax",       type=float, default=350_000.0)
    parser.add_argument("--dz",         type=float, default=1_000.0)
    args = parser.parse_args()

    process_chains(
        args.folder,
        prefix=args.prefix,
        step_discard_frac=args.burnin,
        chain_thin=args.thin,
        nrms_limit=args.nrms_limit,
        z_max=args.zmax,
        dz=args.dz,
    )


if __name__ == "__main__":
    main()
