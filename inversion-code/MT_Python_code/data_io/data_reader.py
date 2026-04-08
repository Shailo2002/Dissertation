"""
Read MT and DC data files into the Config object.

File formats (matching MATLAB read_data.m):

MT_data_Z.dat  (4 columns):
   period(s)   Re(Z)   Im(Z)   err_Z

MT_data.dat  (5 columns, app_phase):
   period(s)   app_res(ohm-m)   err_app_res   phase(deg)   err_phase

DC_data.dat  (3 columns):
   AB_half_spacing(m)   app_res(ohm-m)   err_app_res
"""

from __future__ import annotations
import numpy as np
from mcmc.config import Config


def read_data(mt_file: str, dc_file: str, cfg: Config) -> Config:
    """
    Load data into cfg.MT and/or cfg.DC.

    Parameters
    ----------
    mt_file : str   path to MT data file ('' or ' ' to skip)
    dc_file : str   path to DC data file ('' or ' ' to skip)
    cfg : Config

    Returns
    -------
    cfg with data fields populated.
    """
    method = cfg.inversion_method.upper()

    if method in ("MT", "MT_DC") and mt_file.strip():
        data = np.loadtxt(mt_file)
        cfg.MT.period = data[:, 0]
        dtype = cfg.MT.datatype.upper()

        if dtype == "Z":
            cfg.MT.dobs_Z = data[:, 1] + 1j * data[:, 2]
            cfg.MT.err_Z = data[:, 3]
            cfg.MT.ndata = 2 * len(cfg.MT.period)
            print("MT Impedance data loaded")

        elif dtype == "APP":
            cfg.MT.dobs_appres = np.log10(data[:, 1])
            err_lin = data[:, 2]
            cfg.MT.err_appres = 0.4343 * err_lin / data[:, 1]  # convert to log10 err
            cfg.MT.ndata = len(cfg.MT.period)
            print("MT Apparent resistivity data loaded")

        elif dtype == "PHASE":
            cfg.MT.dobs_phase = data[:, 1]
            cfg.MT.err_phase = data[:, 2]
            cfg.MT.ndata = len(cfg.MT.period)
            print("MT Phase data loaded")

        elif dtype == "APP_PHASE":
            cfg.MT.dobs_appres = np.log10(data[:, 1])
            err_lin = data[:, 2]
            cfg.MT.err_appres = 0.4343 * err_lin / data[:, 1]
            cfg.MT.dobs_phase = data[:, 3]
            cfg.MT.err_phase = data[:, 4]
            cfg.MT.ndata = 2 * len(cfg.MT.period)
            print("MT Apparent resistivity and phase data loaded")

        print(f"Total number of periods: {len(cfg.MT.period)}")

    if method in ("DC", "MT_DC") and dc_file.strip():
        data = np.loadtxt(dc_file)
        cfg.DC.OA = data[:, 0]
        cfg.DC.dobs_appres = np.log10(data[:, 1])
        cfg.DC.err_appres = 0.4343 * data[:, 2] / data[:, 1]
        print("DC Apparent resistivity data loaded")
        print(f"Total number of DC data points: {len(cfg.DC.dobs_appres)}")

    return cfg
