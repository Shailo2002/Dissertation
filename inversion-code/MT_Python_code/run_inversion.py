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
    parser.add_argument("--data",     default="MT_data_Z.dat",
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
                        help="Number of cold chains (all at T=1). Ignored if --temperatures used.")
    parser.add_argument("--temperatures", nargs="+", type=float, default=None,
                        help="Full temperature schedule, e.g. --temperatures 1 1 1.5 3 10  "
                             "Values=1 are cold chains, >1 are hot chains for parallel tempering.")
    parser.add_argument("--output",   default="results",
                        help="Output folder")
    parser.add_argument("--parallel", action="store_true",
                        help="Run chains in parallel (multiprocessing)")
    args = parser.parse_args()

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
        cfg.temperature = [1.0] * args.nchains     # all cold, no PT
    # Re-derive after any changes
    cfg._derive()

    # ---- Load data ----
    cfg = read_data(args.data, args.dcdata, cfg)

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

    for is_ in range(cfg.nsteps):
        # ---- Run all chains for one step ----
        if args.parallel and cfg.nChains > 1:
            import multiprocessing as mp
            worker_args = [
                (models[ic], likelihoods[ic], sigmas[ic], cfg, ic, is_)
                for ic in range(cfg.nChains)
            ]
            with mp.Pool(processes=min(cfg.nChains, mp.cpu_count())) as pool:
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

    print("MCMC Procedure Ends")
    print(f"Results saved to: {os.path.abspath(args.output)}/")
    print(f"Total time: {(time.time() - t_start)/60:.1f} min")
    print("\nNext steps:")
    print("  python postprocess/chain_convergence.py --folder", args.output)
    print("  python postprocess/process_chains.py    --folder", args.output)
    print("  python postprocess/plot_posterior.py    --folder", args.output)


if __name__ == "__main__":
    main()
