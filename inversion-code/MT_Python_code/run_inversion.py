"""
Main entry point for 1D Transdimensional Bayesian MT Inversion.

Equivalent to MATLAB: Bayesian_1D_MTDC_J.m

Usage
-----
  python run_inversion.py                          # uses defaults
  python run_inversion.py --data MT_data_Z.dat     # real data
  python run_inversion.py --nsteps 1000 --nsamples 1000 --nchains 10 --parallel

The --parallel flag runs chains concurrently using multiprocessing,
giving a ~nChains speedup on multi-core machines.

Key speed improvements vs MATLAB:
  1. --parallel  → all chains run simultaneously (multiprocessing)
  2. NumPy vectorised forward model (no MATLAB JIT overhead)
  3. NPZ file I/O is ~10x faster than MATLAB .mat
"""

import argparse
import datetime
import os
import sys
import time
import numpy as np

# Allow running from the MT_Python_code directory
sys.path.insert(0, os.path.dirname(__file__))

from mcmc.config import Config, get_default_config
from mcmc.chain_init import initialise_chains
from mcmc.sampler import run_chain_step
from mcmc.parallel_tempering import swap_temperatures
from data_io.data_reader import read_data
from data_io.chain_io import save_chain


# ------------------------------------------------------------------ #
# Acceptance rate summary  (mirrors MATLAB output)
# ------------------------------------------------------------------ #

def _write_acceptance_summary(results_all: list, cfg, folder: str,
                              elapsed_sec: float | None = None):
    """
    Compute and save acceptance rate summary from in-memory results_all.

    results_all[ic][is_]["acceptance_all"] has 6 values:
      [Overall, Birth, Death, Move, Change_Rho, Change_Noise]
    """
    labels = ["Overall   ", "Birth     ", "Death     ",
              "Move      ", "Change_Rho", "Change_Nse"]

    nChains = cfg.nChains
    nsteps  = cfg.nsteps

    # AR[chain, step, type]  — shape (nChains, nsteps, 6)
    AR = np.zeros((nChains, nsteps, 6))
    for ic in range(nChains):
        for is_, samp in enumerate(results_all[ic]):
            AR[ic, is_, :] = samp["acceptance_all"]

    sep  = "=" * 60
    sep2 = "-" * 50
    sep3 = "-" * 40
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if elapsed_sec is not None:
        h = int(elapsed_sec // 3600)
        m = int((elapsed_sec % 3600) // 60)
        s = elapsed_sec - 3600 * h - 60 * m
        total_time_str = f"{h:d}h {m:02d}m {s:05.2f}s  ({elapsed_sec/60:.2f} min)"
    else:
        total_time_str = "n/a"

    lines = [
        "",
        sep,
        "     ACCEPTANCE RATE SUMMARY (across all chains & steps)",
        sep,
        f"Generated : {now}",
        f"Chains    : {nChains}  |  Steps : {nsteps}",
        f"Total time: {total_time_str}",
        "",
        f"{'Type':<12}  {'Min(%)':>8}  {'Max(%)':>8}  {'Mean(%)':>8}  {'Std(%)':>8}",
        sep2,
    ]
    for k, label in enumerate(labels):
        v = AR[:, :, k].flatten()
        lines.append(
            f"{label:<12}  {v.min():8.2f}  {v.max():8.2f}  {v.mean():8.2f}  {v.std():8.2f}"
        )

    lines += [
        "",
        f"{'Chain':<12}  {'Min(%)':>8}  {'Max(%)':>8}  {'Mean(%)':>8}",
        sep3,
    ]
    for ic in range(nChains):
        v = AR[ic, :, 0]   # overall AR per step for this chain
        lines.append(f"Chain {ic+1:<6d}  {v.min():8.2f}  {v.max():8.2f}  {v.mean():8.2f}")

    lines += ["", sep, ""]

    text = "\n".join(lines)
    print(text)

    os.makedirs(folder, exist_ok=True)
    out_path = os.path.join(folder, "Acceptance_Rate_Summary.txt")
    with open(out_path, "w") as fh:
        fh.write(text)
    print(f"Acceptance rate summary saved to: {out_path}")


# ------------------------------------------------------------------ #
# Temperature schedule helper
# ------------------------------------------------------------------ #

def _auto_temperatures(n: int) -> list:
    """
    Build a sensible PT temperature schedule for n chains.

    Roughly 2/3 cold (T=1) + 1/3 hot with geometrically-spaced temperatures.
    This avoids stuck chains when no explicit schedule is given.
    """
    if n <= 1:
        return [1.0]
    n_cold = max(1, n * 2 // 3)
    n_hot  = n - n_cold
    if n_hot == 0:
        return [1.0] * n
    max_T = max(2.0, 2.0 ** n_hot)   # e.g. 2 hot → max_T=4; 3 hot → 8
    hot_temps = np.geomspace(2.0, max_T, n_hot).tolist()
    return [1.0] * n_cold + hot_temps


# ------------------------------------------------------------------ #
# Worker function (must be at module level for multiprocessing)
# ------------------------------------------------------------------ #

def _worker(args):
    """Run one chain step.  Called by multiprocessing.Pool.map."""
    model, like, sigma, cfg, ic, is_ = args
    return run_chain_step(model, like, sigma, cfg, ic, is_)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="1D Transdimensional Bayesian MT/DC Inversion"
    )
    parser.add_argument("--data",     default="data/synthetic/default_3layer/MT_data_Z.dat",
                        help="MT data file path")
    parser.add_argument("--dcdata",   default=" ",
                        help="DC data file path (optional)")
    parser.add_argument("--method",   default="MT",
                        choices=["MT", "DC", "MT_DC"],
                        help="Inversion method")
    parser.add_argument("--datatype", default="Z",
                        choices=["Z", "app", "phase", "app_phase"],
                        help="MT data type")
    parser.add_argument("--nsteps",   type=int, default=None,
                        help="Number of outer steps (overrides config)")
    parser.add_argument("--nsamples", type=int, default=None,
                        help="Samples per step (overrides config)")
    parser.add_argument("--nchains",     type=int, default=None,
                        help="Total number of chains. Automatically assigns parallel-tempering "
                             "temperatures (~2/3 cold + 1/3 hot). Ignored if --temperatures used.")
    parser.add_argument("--temperatures", nargs="+", type=float, default=None,
                        help="Full temperature schedule, e.g. --temperatures 1 1 1.5 3 10  "
                             "Values=1 are cold chains, >1 are hot chains for parallel tempering.")
    parser.add_argument("--output",     default=None,
                        help="Output folder (default: auto-named under results/)")
    parser.add_argument("--parallel",   action="store_true",
                        help="Run chains in parallel (multiprocessing)")
    parser.add_argument("--true_model", default=None,
                        help="Path to true_model.json from create_synthetic_data.py "
                             "(synthetic only). Copied to results folder so all "
                             "postprocessing steps show the true model automatically.")
    args = parser.parse_args()

    # ---- Auto-generate output folder name if not given ----
    if args.output is None:
        import datetime
        # Derive a short dataset name from the data file path
        # e.g. data/synthetic/craton/MT_data_Z.dat  →  "craton"
        data_parent = os.path.basename(os.path.dirname(os.path.abspath(args.data)))
        timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nsteps_tag  = f"{args.nsteps or 'default'}steps"
        args.output = os.path.join("results", f"{data_parent}_{nsteps_tag}_{timestamp}")
    print(f"Output folder: {args.output}")

    # ---- Build config ----
    cfg = get_default_config()
    cfg.inversion_method = args.method
    cfg.MT.datatype = args.datatype

    if args.nsteps:
        cfg.nsteps = args.nsteps
    if args.nsamples:
        cfg.nsamples = args.nsamples
    if args.temperatures:
        cfg.temperature = args.temperatures        # full PT schedule provided
    elif args.nchains:
        cfg.temperature = _auto_temperatures(args.nchains)
        print(f"Auto PT schedule ({args.nchains} chains): {cfg.temperature}")
    # Re-derive after any changes
    cfg._derive()

    # ---- Load data ----
    cfg = read_data(args.data, args.dcdata, cfg)

    # ---- Copy true model JSON to results folder (synthetic runs) ----
    import shutil
    tm_src = None
    if args.true_model:
        tm_src = args.true_model
    else:
        # Auto-detect: look for true_model.json next to the data file
        data_dir = os.path.dirname(os.path.abspath(args.data))
        auto_tm = os.path.join(data_dir, "true_model.json")
        if os.path.exists(auto_tm):
            tm_src = auto_tm

    if tm_src:
        os.makedirs(args.output, exist_ok=True)
        dest = os.path.join(args.output, "true_model.json")
        shutil.copy(tm_src, dest)
        print(f"True model saved to results: {dest}")

    # ---- Initialise chains ----
    models, likelihoods, sigmas = initialise_chains(cfg)
    swap_count = [np.zeros(cfg.nsteps, dtype=np.int32) for _ in range(cfg.nChains)]
    results_all = [[] for _ in range(cfg.nChains)]   # results_all[ic][is]

    print("\nMCMC Procedure Starts")
    print(f"  Chains      : {cfg.nChains}  (cold T=1: {cfg.nChains_atT1}, hot T>1: {cfg.nChains_atoT})")
    print(f"  Temperatures: {cfg.temperature}")
    print(f"  Steps       : {cfg.nsteps}")
    print(f"  Samples/step: {cfg.nsamples}")
    print(f"  Total samples/chain: {cfg.nsteps * cfg.nsamples:,}")
    print(f"  Parallel    : {args.parallel}\n")

    t_start = time.time()

    def _run_step(is_):
        """Run all chains for one step (sequential)."""
        for ic in range(cfg.nChains):
            samples, m, lf, s = run_chain_step(
                models[ic], likelihoods[ic], sigmas[ic], cfg, ic, is_
            )
            results_all[ic].append(samples)
            models[ic] = m
            likelihoods[ic] = lf
            sigmas[ic] = s

    if args.parallel and cfg.nChains > 1:
        import multiprocessing as mp
        # Create pool once — avoids per-step spawn/teardown overhead
        pool = mp.Pool(processes=min(cfg.nChains, mp.cpu_count()))
    else:
        pool = None

    try:
        for is_ in range(cfg.nsteps):
            # ---- Run all chains for one step ----
            if pool is not None:
                worker_args = [
                    (models[ic], likelihoods[ic], sigmas[ic], cfg, ic, is_)
                    for ic in range(cfg.nChains)
                ]
                results = pool.map(_worker, worker_args)
                for ic, (samples, m, lf, s) in enumerate(results):
                    results_all[ic].append(samples)
                    models[ic] = m
                    likelihoods[ic] = lf
                    sigmas[ic] = s
            else:
                for ic in range(cfg.nChains):
                    samples, m, lf, s = run_chain_step(
                        models[ic], likelihoods[ic], sigmas[ic], cfg, ic, is_
                    )
                    results_all[ic].append(samples)
                    models[ic] = m
                    likelihoods[ic] = lf
                    sigmas[ic] = s

            # ---- Save chains periodically ----
            save_chain(results_all, is_, cfg, swap_count, args.output,
                       prefix=args.method)

            # ---- Parallel tempering swap ----
            if any(t > 1.0 for t in cfg.temperature):
                likelihoods, models, sigmas, swap_count = swap_temperatures(
                    likelihoods, models, sigmas, cfg, is_, swap_count
                )

            elapsed = time.time() - t_start
            pct = (is_ + 1) / cfg.nsteps * 100
            eta = elapsed / (is_ + 1) * (cfg.nsteps - is_ - 1)
            print(f"  Step {is_+1}/{cfg.nsteps} done "
                  f"({pct:.0f}%)  elapsed={elapsed/60:.1f} min  "
                  f"ETA={eta/60:.1f} min\n")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    total_elapsed = time.time() - t_start
    print("MCMC Procedure Ends")
    print(f"Results saved to: {os.path.abspath(args.output)}/")
    print(f"Total time: {total_elapsed/60:.1f} min")

    # ---- Acceptance rate summary ----
    _write_acceptance_summary(results_all, cfg, args.output,
                              elapsed_sec=total_elapsed)

    print("\nNext steps:")
    print(f"  python postprocess/chain_convergence.py --folder {args.output}")
    print(f"  python postprocess/process_chains.py    --folder {args.output}")
    print(f"  python postprocess/plot_posterior.py    --folder {args.output}")
    print(f"  python postprocess/plot_noise.py        --folder {args.output}")
    print(f"  python postprocess/validate_results.py  --folder {args.output} "
          f"--data {args.data}")


if __name__ == "__main__":
    main()
