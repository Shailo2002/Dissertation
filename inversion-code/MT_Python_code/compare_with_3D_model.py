"""
Compare 1D MCMC inversion result for a station against the reference 3D model
(Fine_model_10km_HS.dat from professor's code Jiff_ModEM_AP3DMT).

Usage:
    python compare_with_3D_model.py \
        --results results/real_bot228_1000steps \
        --model /path/to/Jiff_ModEM_AP3DMT/Fine_model_10km_HS.dat \
        --lat -24.3465 --lon 25.5258 \
        --station BOT228

The lat/lon should be for the station you inverted.
For BOT228: lat=-24.3465, lon=25.5258
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


# ──────────────────────────────────────────────────────────────────────────────
# Reference 3D model reader  (mirrors professor's ReadModel.m)
# ──────────────────────────────────────────────────────────────────────────────

def read_modem_model(filepath):
    """Read a ModEM-format resistivity model file.

    Returns
    -------
    rho : ndarray (nx+1, ny+1, nz+1)  log10(resistivity) in ohm-m
    x   : ndarray (nx+1,)  north coordinates  (metres from origin)
    y   : ndarray (ny+1,)  east coordinates   (metres from origin)
    z   : ndarray (nz+1,)  depth coordinates  (metres)
    """
    with open(filepath, "r") as fid:
        # line 1 – comment
        fid.readline()
        # line 2 – dimensions  [nx  ny  nz  TYPE  <LOGE|LINEAR>]
        header = fid.readline().strip()
        parts = header.split()
        nx, ny, nz = int(parts[0]), int(parts[1]), int(parts[2])
        use_loge = "LOGE" in header.upper()

        # Read remaining data as a flat stream of tokens (dx, dy, dz, rho values, origin)
        tokens = []
        for line in fid:
            tokens.extend(line.split())

    tok_iter = iter(tokens)

    def next_float():
        return float(next(tok_iter))

    def read_n(n):
        return np.array([next_float() for _ in range(n)], dtype=float)

    dx = read_n(nx)
    dy = read_n(ny)
    dz = read_n(nz)

    # resistivity values  – stored iz=0..nz-1, iy=0..ny-1, ix=nx-1..0
    rho_raw = np.zeros((nx, ny, nz), dtype=float)
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx - 1, -1, -1):
                rho_raw[ix, iy, iz] = next_float()

    # origin (3 floats)
    origin = read_n(3)

    # coordinate arrays (same indexing as MATLAB cumsum)
    x = np.concatenate([[origin[0]], origin[0] + np.cumsum(dx)])
    y = np.concatenate([[origin[1]], origin[1] + np.cumsum(dy)])
    z = np.concatenate([[origin[2]], origin[2] + np.cumsum(dz)])

    # expand rho to (nx+1, ny+1, nz+1) by repeating last slice
    rho = np.zeros((nx + 1, ny + 1, nz + 1), dtype=float)
    rho[:nx, :ny, :nz] = rho_raw
    rho[nx, :, :] = rho[nx - 1, :, :]
    rho[:, ny, :] = rho[:, ny - 1, :]
    rho[:, :, nz] = rho[:, :, nz - 1]

    # Convert to log10(resistivity) — mirrors ReadModel.m then the negation in the
    # calling script (rho = -squeeze(rho3(...))):
    #   LINEAR: file stores resistivity (ohm-m)  → log10(raw)         = positive
    #   LOGE:   file stores ln(conductivity)      → log10(1/exp(raw))  = positive
    if use_loge:
        sigma = np.exp(rho)               # ln(conductivity) → conductivity
        sigma[sigma > 1e6] = np.nan       # discard air cells
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_log10 = np.log10(1.0 / sigma)  # log10(resistivity)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_log10 = np.log10(rho)     # raw = resistivity → log10(resistivity)

    return rho_log10, x, y, z


# ──────────────────────────────────────────────────────────────────────────────
# Extract 1-D profile from 3-D model at a given lat/lon
# (mirrors the coordinate transform in A1_compare_models_5_slice.m)
# ──────────────────────────────────────────────────────────────────────────────

REF_LAT  = -24.79278   # professor's reference latitude
REF_LONG =  22.41878   # professor's reference longitude
SCALE    = 1000.0      # metres → km

def extract_1d_profile(rho3, x, y, z, lat_stat, lon_stat):
    """Return (rho_1d, z_km) for the nearest grid column to (lat_stat, lon_stat)."""
    # Convert model coords from metres → km → geographic (same as MATLAB)
    x_km = x / SCALE
    y_km = y / SCALE
    x_geo = x_km / 111.0 + REF_LAT
    y_geo = y_km / (111.0 * np.cos(np.radians(REF_LAT))) + REF_LONG
    z_km  = z / SCALE

    # Nearest-grid-point (same as MATLAB: first index >= station coordinate)
    id_lat = np.where(x_geo >= lat_stat)[0]
    id_lon = np.where(y_geo >= lon_stat)[0]

    if id_lat.size == 0 or id_lon.size == 0:
        raise ValueError(
            f"Station lat={lat_stat}, lon={lon_stat} is outside the model extent.\n"
            f"  Model lat range: {x_geo.min():.3f} to {x_geo.max():.3f}\n"
            f"  Model lon range: {y_geo.min():.3f} to {y_geo.max():.3f}"
        )

    ix = id_lat[0]
    iy = id_lon[0]
    rho_1d = rho3[ix, iy, :]  # log10(resistivity) vs depth

    return rho_1d, z_km


# ──────────────────────────────────────────────────────────────────────────────
# Main comparison plot
# ──────────────────────────────────────────────────────────────────────────────

def make_step_curve(rho_1d, z_km):
    """Convert a piecewise-constant depth profile into staircase (x, y) arrays."""
    rr, zz = [], []
    for i, r in enumerate(rho_1d):
        rr += [r, r]
        if i < len(z_km) - 1:
            zz += [z_km[i], z_km[i + 1]]
        else:
            zz += [z_km[i], 2 * z_km[i]]
    return np.array(rr), np.array(zz)


def plot_comparison(results_folder, model_file, lat_stat, lon_stat, station_name):
    # ── load MCMC posterior ───────────────────────────────────────────────────
    stat_file = os.path.join(results_folder, "MT_TD_Chain_Stat_info.npz")
    if not os.path.exists(stat_file):
        raise FileNotFoundError(f"Posterior file not found: {stat_file}")

    posterior = np.load(stat_file)
    z_mcmc_m  = posterior["z_plot"]           # metres
    z_mcmc_km = z_mcmc_m / 1000.0
    pmean     = posterior["pmean"]            # log10(rho) mean
    p5        = posterior["p5"]               # 5th percentile
    p95       = posterior["p95"]              # 95th percentile

    # ── load 3D reference model and extract 1D profile ───────────────────────
    print(f"Reading 3D model: {model_file} ...")
    rho3, x, y, z = read_modem_model(model_file)
    rho_3d, z_3d_km = extract_1d_profile(rho3, x, y, z, lat_stat, lon_stat)

    # ── build staircase for 3D model 1D profile ───────────────────────────────
    # keep only the subsurface (z > 0) and within posterior depth range
    max_depth = z_mcmc_km.max()
    mask_3d   = (z_3d_km > 0) & (z_3d_km <= max_depth)
    rho_3d_plot = rho_3d[mask_3d]
    z_3d_plot   = z_3d_km[mask_3d]

    rr_3d, zz_3d = make_step_curve(rho_3d_plot, z_3d_plot)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 9))

    # MCMC posterior shaded envelope (5th–95th percentile)
    ax.fill_betweenx(z_mcmc_km, p5, p95,
                     alpha=0.35, color="steelblue",
                     label="MCMC 5–95th percentile")

    # MCMC posterior mean
    ax.plot(pmean, z_mcmc_km,
            color="steelblue", linewidth=2, label="MCMC posterior mean")

    # 3D reference model 1D profile
    ax.plot(rr_3d, zz_3d,
            "--k", linewidth=2, label="3D model (RB Inv.)")

    ax.set_xlabel(r"log$_{10}$($\rho$)  [Ω·m]", fontsize=12)
    ax.set_ylabel("Depth  (km)", fontsize=12)
    ax.set_title(f"1-D comparison at station {station_name}\n"
                 f"lat={lat_stat:.4f}°, lon={lon_stat:.4f}°",
                 fontsize=12)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, min(350, max_depth))
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    # x-axis tick labels in ohm-m
    xticks = [0, 1, 2, 3, 4, 5]
    ax.set_xticks(xticks)
    ax.set_xticklabels([r"$10^0$", r"$10^1$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$"])

    out_file = os.path.join(results_folder, f"compare_3Dmodel_{station_name}.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.show()
    print(f"Saved: {out_file}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare MCMC 1D result vs 3D reference model")
    parser.add_argument("--results", required=True,
                        help="Path to results folder (e.g. results/real_bot228_1000steps)")
    parser.add_argument("--model",   required=True,
                        help="Path to Fine_model_10km_HS.dat (professor's 3D model)")
    parser.add_argument("--lat",     type=float, required=True,
                        help="Station latitude  (e.g. -24.3465 for BOT228)")
    parser.add_argument("--lon",     type=float, required=True,
                        help="Station longitude (e.g. 25.5258  for BOT228)")
    parser.add_argument("--station", default="",
                        help="Station name for the plot title (e.g. BOT228)")
    args = parser.parse_args()

    plot_comparison(
        results_folder=args.results,
        model_file=args.model,
        lat_stat=args.lat,
        lon_stat=args.lon,
        station_name=args.station,
    )
