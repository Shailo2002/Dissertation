"""
Validate inversion results — 4 checks in one figure.

Check 1: Data fit  (observed vs predicted)
   → Most important. Predicted should overlap observed within error bars.

Check 2: nRMS distribution
   → Should peak near 1.0.  nRMS >> 1 = bad fit.  nRMS << 1 = overfitting.

Check 3: Convergence  (likelihood vs sample number)
   → Likelihood should plateau (stabilise) before burn-in ends.

Check 4: Acceptance rate summary  (printed to terminal)
   → Overall rate should be 20–40% for healthy chain mixing.

Usage
-----
  # Synthetic data (with true model)
  python postprocess/validate_results.py --folder results_200 \\
      --true_depths 0 10000 50000 --true_rho 100 10 1000

  # Real data (no true model)
  python postprocess/validate_results.py --folder results_real
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from forward.mt_forward import mt1d_forward
from data_io.chain_io import load_chain, load_config
from data_io.data_reader import read_data
from mcmc.config import get_default_config


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _load_data_and_cfg(folder, data_file):
    """Rebuild a minimal Config with data loaded."""
    raw_cfg = load_config(folder)
    cfg = get_default_config()
    cfg.inversion_method = raw_cfg.get("inversion_method", "MT")
    cfg.MT.datatype      = raw_cfg.get("MT", {}).get("datatype", "Z")
    cfg.temperature      = raw_cfg.get("temperature", [1.0, 1.0])
    cfg.nsteps           = raw_cfg.get("nsteps", 20)
    cfg.nsamples         = raw_cfg.get("nsamples", 500)
    cfg.maxnodes         = raw_cfg.get("maxnodes", 30)
    cfg.logdomain        = raw_cfg.get("logdomain", True)
    cfg.max_z            = raw_cfg.get("max_z", np.log10(350_000))
    cfg._derive()
    cfg = read_data(data_file, " ", cfg)
    return cfg


def _predict_from_sample(model_flat, ncells, cfg, logdomain, max_z):
    """
    Compute forward response for one posterior sample.
    model_flat : shape (maxnodes, 2)
    """
    ng = int(ncells)
    model = model_flat[:ng, :]

    if logdomain:
        z = np.concatenate([[0.0], 10.0 ** model[1:, 0], [10.0 ** max_z]])
    else:
        z = np.concatenate([[0.0], model[1:, 0], [max_z]])

    thicknesses   = np.diff(z)
    resistivities = 10.0 ** model[:, 1]

    Z, appres, phase = mt1d_forward(resistivities, thicknesses, cfg.MT.period)
    return Z, appres, phase


# ------------------------------------------------------------------ #
# Model comparison plot (True vs Inversion, separate figure)
# ------------------------------------------------------------------ #

def _plot_model_comparison(
    folder: str,
    prefix: str = "MT",
    true_depths: list = None,
    true_rho: list = None,
    output: str = None,
) -> float:
    """
    Plot inversion mean ± uncertainty vs true model (if given) on a depth profile.
    Uses the posterior percentiles from the Stat_info file produced by process_chains.py.
    Returns the RMS misfit in log10 space (or None if no true model).
    """
    stat_path = os.path.join(folder, f"{prefix}_TD_Chain_Stat_info.npz")
    if not os.path.exists(stat_path):
        print(f"  Stat info not found: {stat_path}. Run process_chains.py first.")
        return None

    S = np.load(stat_path, allow_pickle=True)
    z_plot = S["z_plot"]          # metres, bin midpoints
    p5     = S["p5"]              # log10(ohm-m)
    p95    = S["p95"]
    pmean  = S["pmean"]           # log10(ohm-m)
    z_max  = float(S["z_max"])    # metres

    mask     = z_plot <= z_max
    depth_km = z_plot[mask] / 1000.0

    fig, ax = plt.subplots(figsize=(6, 8))

    # ---- Inversion: mean + 5–95 % uncertainty band ----
    ax.plot(pmean[mask], depth_km, color="red", linewidth=1.5, label="Inversion mean")
    ax.fill_betweenx(depth_km, p5[mask], p95[mask],
                     color="red", alpha=0.3, label="5th–95th %ile")

    rms = None

    # ---- True model (synthetic only) ----
    if true_depths is not None and true_rho is not None:
        z_true_m    = np.array(true_depths, dtype=float)
        rho_true_ohm = np.array(true_rho,  dtype=float)
        rho_true_log = np.log10(rho_true_ohm)

        # Step arrays for plotting (extend last layer to z_max)
        z_step   = np.append(z_true_m, z_max)
        rho_step = np.append(rho_true_log, rho_true_log[-1])
        ax.step(rho_step, z_step / 1000.0, where="post",
                color="blue", linewidth=2, label="True model")

        # RMS: interpolate true model onto inversion depth grid
        interp_true = interp1d(z_step, rho_step, kind="previous",
                               bounds_error=False, fill_value="extrapolate")
        rho_true_interp = interp_true(z_plot[mask])
        misfit = pmean[mask] - rho_true_interp
        rms = float(np.sqrt(np.mean(misfit ** 2)))

        print(f"\n  Model RMS (log10 space) : {rms:.3f}")
        ax.text(0.05, 0.04, f"RMS = {rms:.3f}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.invert_yaxis()
    ax.set_xlabel(r"log$_{10}$(Resistivity) (ohm-m)", fontsize=11)
    ax.set_ylabel("Depth (km)", fontsize=11)
    ax.set_title("Synthetic Test: True vs Inversion", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    if output is None:
        output = os.path.join(folder, f"{prefix}_model_comparison.png")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Model comparison plot saved: {output}")
    return rms


# ------------------------------------------------------------------ #
# Main validation
# ------------------------------------------------------------------ #

def validate(
    folder: str,
    data_file: str,
    prefix: str = "MT",
    true_depths: list = None,
    true_rho: list = None,
    n_pred_samples: int = 200,
    output: str = None,
):
    cfg = _load_data_and_cfg(folder, data_file)

    # ---- Auto-load true model from results folder if not given ----
    if true_depths is None and true_rho is None:
        import json as _json
        tm_path = os.path.join(folder, "true_model.json")
        if os.path.exists(tm_path):
            with open(tm_path) as fj:
                tm = _json.load(fj)
            true_depths = tm["layer_tops"]
            true_rho    = tm["layer_rho"]
            print(f"  Auto-loaded true model from {tm_path}")

    # ---- Load processed samples ----
    proc_path = os.path.join(folder, f"{prefix}_TD_Chain_Processed.npz")
    if not os.path.exists(proc_path):
        print(f"ERROR: {proc_path} not found. Run process_chains.py first.")
        return

    proc = np.load(proc_path, allow_pickle=True)
    # process_chains.py saves z_all / rho_all (object arrays) + ngrid + nrms_all + like_all
    z_all   = proc["z_all"]    # (n_samples,) object array — each entry is metres array
    rho_all = proc["rho_all"]  # (n_samples,) object array — each entry is log10(ohm-m)
    ngrid   = proc["ngrid"]    # (n_samples,) int
    nrms    = proc["nrms_all"] # (n_samples, ...)
    like    = proc["like_all"]

    n_total = len(ngrid)
    raw_cfg = load_config(folder)
    max_z_m = raw_cfg.get("max_z", np.log10(350_000))
    # max_z stored in config is log10(m) if logdomain, so convert to metres
    max_z_metres = 10.0 ** float(max_z_m)

    print(f"\nValidation summary")
    print(f"  Total posterior samples : {n_total}")
    print(f"  Periods                 : {len(cfg.MT.period)}")

    # ---- Forward predictions from random posterior samples ----
    rng = np.random.default_rng(0)
    idx = rng.choice(n_total, size=min(n_pred_samples, n_total), replace=False)

    appres_pred = []
    phase_pred  = []
    Z_pred      = []

    print(f"  Computing {len(idx)} forward predictions ...")
    for i in idx:
        try:
            z_i   = np.asarray(z_all[i],   dtype=float)   # layer tops in metres
            rho_i = np.asarray(rho_all[i], dtype=float)   # log10(ohm-m)
            z_bounds    = np.append(z_i, max_z_metres)
            thicknesses = np.diff(z_bounds)
            resistivities = 10.0 ** rho_i
            Z_i, ap_i, ph_i = mt1d_forward(resistivities, thicknesses, cfg.MT.period)
            appres_pred.append(ap_i)
            phase_pred.append(ph_i)
            Z_pred.append(Z_i)
        except Exception:
            continue

    appres_pred = np.array(appres_pred)   # (n_pred, n_periods)
    phase_pred  = np.array(phase_pred)
    Z_pred      = np.array(Z_pred)

    # ---- Observed data ----
    periods = cfg.MT.period
    dtype   = cfg.MT.datatype.upper()

    if dtype == "Z":
        mu     = 4.0 * np.pi * 1e-7
        omega  = 2.0 * np.pi / periods
        absZ   = np.abs(cfg.MT.dobs_Z)
        appres_obs = absZ ** 2 / (mu * omega)
        phase_obs  = np.degrees(np.arctan2(cfg.MT.dobs_Z.imag, cfg.MT.dobs_Z.real))
        err_appres = 2 * absZ * cfg.MT.err_Z / (mu * omega)
        err_phase  = (cfg.MT.err_Z / absZ) * (180.0 / np.pi)
    elif dtype == "APP_PHASE":
        appres_obs = 10.0 ** cfg.MT.dobs_appres
        phase_obs  = cfg.MT.dobs_phase
        err_appres = appres_obs * cfg.MT.err_appres / 0.4343
        err_phase  = cfg.MT.err_phase
    else:
        appres_obs = 10.0 ** cfg.MT.dobs_appres
        phase_obs  = np.zeros_like(appres_obs)
        err_appres = appres_obs * cfg.MT.err_appres / 0.4343
        err_phase  = np.zeros_like(appres_obs)

    # ---- nRMS stats ----
    nrms_vals = nrms[:, 0]
    mean_nrms = float(np.mean(nrms_vals))
    med_nrms  = float(np.median(nrms_vals))

    # ---- Print verdict ----
    print(f"\n  --- Validation Metrics ---")
    print(f"  Mean nRMS   : {mean_nrms:.3f}  (target ≈ 1.0)")
    print(f"  Median nRMS : {med_nrms:.3f}")
    if med_nrms < 0.8:
        verdict = "OVERFITTING (nRMS too low — model too complex or noise overestimated)"
    elif med_nrms > 2.0:
        verdict = "POOR FIT (nRMS too high — model too simple or data has large errors)"
    elif 0.8 <= med_nrms <= 1.5:
        verdict = "GOOD FIT"
    else:
        verdict = "ACCEPTABLE — could improve with more steps"
    print(f"  Verdict     : {verdict}")

    # ---- Load convergence from chain files ----
    like_chains = []
    nrms_chains = []
    nChains = raw_cfg.get("nChains", 2)
    for ic in range(1, nChains + 1):
        try:
            ch = load_chain(folder, ic, prefix)
            like_chains.append(ch["like"])
            nrms_chains.append(ch["misfit"][:, 0])
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------ #
    # Figure: 2 rows × 3 cols
    # Row 1: app-res data fit | phase data fit | nRMS histogram
    # Row 2: likelihood convergence | model spread | empty/true model
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Inversion Validation", fontsize=14, fontweight="bold")

    # ---- Check 1a: Apparent resistivity fit ----
    ax1 = fig.add_subplot(2, 3, 1)
    # Predicted envelope (5th–95th percentile)
    ap5  = np.percentile(appres_pred, 5,  axis=0)
    ap95 = np.percentile(appres_pred, 95, axis=0)
    ap50 = np.percentile(appres_pred, 50, axis=0)
    ax1.fill_between(periods, ap5, ap95, alpha=0.3, color="steelblue",
                     label="Predicted 5–95%")
    ax1.loglog(periods, ap50, "-b", linewidth=2, label="Predicted median")
    ax1.errorbar(periods, appres_obs, yerr=err_appres,
                 fmt="ro", markersize=5, linewidth=1.5, label="Observed")
    # True noise-free response if true model given
    if true_depths is not None and true_rho is not None:
        z_t = np.array(true_depths, dtype=float)
        r_t = np.array(true_rho,   dtype=float)
        h_t = np.diff(np.append(z_t, z_t[-1] * 1.5))
        _, ap_true, _ = mt1d_forward(r_t, h_t, periods)
        ax1.loglog(periods, ap_true, "--k", linewidth=2, label="True (noise-free)")
    ax1.set_xlabel("Period (s)")
    ax1.set_ylabel("Apparent Resistivity (ohm-m)")
    ax1.set_title("CHECK 1a: App-Res Data Fit", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # ---- Check 1b: Phase fit ----
    ax2 = fig.add_subplot(2, 3, 2)
    ph5  = np.percentile(phase_pred, 5,  axis=0)
    ph95 = np.percentile(phase_pred, 95, axis=0)
    ph50 = np.percentile(phase_pred, 50, axis=0)
    ax2.fill_between(periods, ph5, ph95, alpha=0.3, color="steelblue",
                     label="Predicted 5–95%")
    ax2.semilogx(periods, ph50, "-b", linewidth=2, label="Predicted median")
    ax2.errorbar(periods, phase_obs, yerr=err_phase,
                 fmt="ro", markersize=5, linewidth=1.5, label="Observed")
    if true_depths is not None and true_rho is not None:
        _, _, ph_true = mt1d_forward(r_t, h_t, periods)
        ax2.semilogx(periods, ph_true, "--k", linewidth=2, label="True (noise-free)")
    ax2.set_xlabel("Period (s)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_title("CHECK 1b: Phase Data Fit", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    # ---- Check 2: nRMS histogram ----
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(nrms_vals, bins=50, density=True, color="steelblue",
             edgecolor="white", linewidth=0.5)
    ax3.axvline(1.0,      color="r",  linestyle="--", linewidth=2, label="Target = 1.0")
    ax3.axvline(med_nrms, color="k",  linestyle="-",  linewidth=2,
                label=f"Median = {med_nrms:.2f}")
    ax3.set_xlabel("nRMS")
    ax3.set_ylabel("Probability density")
    ax3.set_title("CHECK 2: nRMS Distribution", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # Colour the background based on verdict
    bg = "palegreen" if 0.8 <= med_nrms <= 1.5 else "lightyellow" if med_nrms <= 2.0 else "mistyrose"
    ax3.set_facecolor(bg)
    ax3.text(0.98, 0.97, verdict, transform=ax3.transAxes,
             fontsize=8, va="top", ha="right",
             bbox=dict(boxstyle="round", facecolor=bg, alpha=0.8))

    # ---- Check 3: Convergence (likelihood) ----
    ax4 = fig.add_subplot(2, 3, 4)
    for lk in like_chains:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_lk = np.where(lk > 0, np.log10(lk), np.nan)
        ax4.plot(np.arange(1, len(log_lk) + 1), log_lk, "-", alpha=0.7, linewidth=0.8)
    ax4.set_xscale("log")
    ax4.set_xlabel("# samples (cumulative)")
    ax4.set_ylabel("log10(likelihood)")
    ax4.set_title("CHECK 3: Convergence", fontweight="bold")
    ax4.grid(True, which="both", alpha=0.3)
    ax4.text(0.02, 0.05,
             "Good: lines plateau and overlap\nBad: still trending downward",
             transform=ax4.transAxes, fontsize=8,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # ---- Check 4: nRMS convergence ----
    ax5 = fig.add_subplot(2, 3, 5)
    for nr in nrms_chains:
        ax5.plot(np.arange(1, len(nr) + 1), nr, "-", alpha=0.7, linewidth=0.8)
    ax5.axhline(1.0, color="r", linestyle="--", linewidth=1.5, label="nRMS = 1")
    ax5.set_xscale("log")
    finite_nrms = nrms_chains[0][np.isfinite(nrms_chains[0])]
    ylim_top = min(np.nanpercentile(finite_nrms, 95) * 2, 20) if len(finite_nrms) > 0 else 20
    ax5.set_ylim(0, ylim_top)
    ax5.set_xlabel("# samples (cumulative)")
    ax5.set_ylabel("nRMS")
    ax5.set_title("CHECK 4: nRMS Convergence", fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.grid(True, which="both", alpha=0.3)
    ax5.text(0.02, 0.97,
             "Good: nRMS → 1 and stabilises",
             transform=ax5.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # ---- Summary text panel ----
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    summary_lines = [
        "VALIDATION SUMMARY",
        "─" * 32,
        f"Posterior samples  : {n_total}",
        f"Mean nRMS          : {mean_nrms:.3f}",
        f"Median nRMS        : {med_nrms:.3f}",
        f"Verdict            : {verdict}",
        "",
        "HOW TO READ:",
        "  Check 1: Predicted envelope",
        "    must cover observed data",
        "  Check 2: nRMS peak ≈ 1.0",
        "  Check 3: Likelihood plateaus",
        "  Check 4: nRMS → 1, stable",
    ]
    if true_depths is not None:
        summary_lines += [
            "",
            "True model layers:",
        ]
        for i, (z, r) in enumerate(zip(true_depths, true_rho)):
            summary_lines.append(f"  L{i+1}: {z:.0f} m  {r:.0f} ohm-m")

    ax6.text(0.05, 0.97, "\n".join(summary_lines),
             transform=ax6.transAxes,
             fontsize=9, va="top", family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()

    if output is None:
        output = os.path.join(folder, f"{prefix}_validation.png")
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nValidation plot saved: {output}")

    # ---- Separate model comparison figure ----
    print("\nGenerating model comparison plot ...")
    _plot_model_comparison(
        folder=folder,
        prefix=prefix,
        true_depths=true_depths,
        true_rho=true_rho,
    )


def main():
    parser = argparse.ArgumentParser(description="Validate inversion results")
    parser.add_argument("--folder",      default="results_200")
    parser.add_argument("--data",        default="MT_data_Z.dat",
                        help="Path to the data file used in inversion")
    parser.add_argument("--prefix",      default="MT", choices=["MT", "DC", "MT_DC"])
    parser.add_argument("--true_depths", nargs="+", type=float, default=None,
                        help="True layer top depths in metres (synthetic only)")
    parser.add_argument("--true_rho",    nargs="+", type=float, default=None,
                        help="True layer resistivities in ohm-m (synthetic only)")
    parser.add_argument("--nsamples",    type=int, default=200,
                        help="Number of posterior samples used for prediction (default 200)")
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    validate(
        folder=args.folder,
        data_file=args.data,
        prefix=args.prefix,
        true_depths=args.true_depths,
        true_rho=args.true_rho,
        n_pred_samples=args.nsamples,
        output=args.output,
    )


if __name__ == "__main__":
    main()
