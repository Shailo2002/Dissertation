"""
MCMC configuration / hyperparameters.

Equivalent to GetDefaultParameters() in Bayesian_1D_MTDC_J.m
All settings are grouped in a single Config dataclass so they can be
passed around, serialised to JSON, and adjusted from the command line.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class MTConfig:
    """Observational data container for MT."""
    period: np.ndarray = field(default_factory=lambda: np.array([]))
    datatype: str = "Z"           # 'Z' | 'app' | 'phase' | 'app_phase'
    dobs_Z: np.ndarray = None     # complex observed impedance
    err_Z: np.ndarray = None      # impedance error
    dobs_appres: np.ndarray = None
    err_appres: np.ndarray = None
    dobs_phase: np.ndarray = None
    err_phase: np.ndarray = None
    ndata: int = 0


@dataclass
class DCConfig:
    """Observational data container for DC resistivity."""
    OA: np.ndarray = None          # electrode half-spacing AB/2
    dobs_appres: np.ndarray = None
    err_appres: np.ndarray = None


@dataclass
class Config:
    """
    Master configuration object.  Mirrors CData in the MATLAB code.

    Change settings here before calling run_inversion.py.
    """

    # ------------------------------------------------------------------ #
    # Parallel tempering temperatures
    # temperature = 1   → cold chain (target distribution)
    # temperature > 1   → hot chain  (flattened distribution)
    # ------------------------------------------------------------------ #
    temperature: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # Jump type between temperatures during PT swap:
    #   0 = randomly choose any pair
    #   1 = swap all adjacent pairs simultaneously
    jumptype: int = 0

    # ------------------------------------------------------------------ #
    # Model parameterisation
    # ------------------------------------------------------------------ #
    logdomain: bool = True          # True → depths stored in log10(m)
    log_normal_noise: bool = False  # True → log-normal noise proposal
    scale: float = 1.0              # multiply model depths by this (keep 1)

    # Depth limits (in log10 metres when logdomain=True)
    min_z: float = np.log10(10.0)
    max_z: float = np.log10(350_000.0)
    minimum_layer_thickness: float = 1_000.0   # metres

    # Resistivity limits (log10 scale, ohm-m)
    resmin: float = 1.0e-1
    resmax: float = 1.0e5

    # Min / max number of Voronoi cells (layers)
    minnodes: int = 2
    maxnodes: int = 30

    # ------------------------------------------------------------------ #
    # MCMC settings
    # ------------------------------------------------------------------ #
    nsteps: int = 20
    nsamples: int = 1000

    # Proposal type probabilities (cumulative thresholds for u ~ Uniform[0,1)):
    #   birth=10%, death=10%, move=16%, change_rho=52%, change_noise=12%
    # Last value MUST be < 1.0 so that change_noise (ptype 5) is reachable.
    # birth and death are balanced (equal %) to avoid layer-count bias.
    proposal: List[float] = field(default_factory=lambda: [0.10, 0.20, 0.36, 0.88])

    # Kernel type: 0 = Gaussian (for birth/death prob), 1 = prior
    kernel: int = 1
    eps: float = 1.0e-9

    # ------------------------------------------------------------------ #
    # Proposal step sizes  (per chain – lists with one value per chain)
    # ------------------------------------------------------------------ #
    sigma_rho: List[float] = field(default_factory=list)
    sigma_rho_birth: List[float] = field(default_factory=list)
    sigma_rho_delayed: List[float] = field(default_factory=list)
    sigma_loc_z: List[float] = field(default_factory=list)
    sigma_loc_z_delayed: List[float] = field(default_factory=list)

    # Noise hyperparameter bounds [min, max]
    sigma_Z: List[float] = field(default_factory=lambda: [1.0, 5.0])
    sigma_app_res: List[float] = field(default_factory=lambda: [0.8, 5.0])
    sigma_phase: List[float] = field(default_factory=lambda: [0.8, 5.0])
    sigma_noise: float = 0.01   # step size for noise proposal

    # ------------------------------------------------------------------ #
    # Inversion mode
    # ------------------------------------------------------------------ #
    inversion_method: str = "MT"   # 'MT' | 'DC' | 'MT_DC'

    # ------------------------------------------------------------------ #
    # Data containers (filled by io.data_reader.read_data)
    # ------------------------------------------------------------------ #
    MT: MTConfig = field(default_factory=MTConfig)
    DC: DCConfig = field(default_factory=DCConfig)

    # ------------------------------------------------------------------ #
    # Derived fields (computed in post_init)
    # ------------------------------------------------------------------ #
    nChains: int = 0
    nChains_atT1: int = 0
    nChains_atoT: int = 0
    nchain_for_PT: int = 0
    min_res_log: float = 0.0
    max_res_log: float = 0.0
    rho: str = "log10"

    def __post_init__(self):
        self._derive()

    def _derive(self):
        """Compute all quantities that depend on temperature list."""
        temps = np.asarray(self.temperature)
        self.nChains = len(temps)
        self.nChains_atT1 = int(np.sum(temps == 1.0))
        self.nChains_atoT = self.nChains - self.nChains_atT1
        self.nchain_for_PT = min(self.nChains_atT1, self.nChains_atoT)

        self.min_res_log = np.log10(self.resmin)
        self.max_res_log = np.log10(self.resmax)

        # Per-chain proposal widths (cold chains get tighter proposals)
        n1, nT = self.nChains_atT1, self.nChains_atoT
        self.sigma_rho = (
            [0.2] * n1 + list(np.linspace(0.2, 0.4, max(nT, 1)))[:nT]
        )
        self.sigma_rho_birth = (
            [0.4] * n1 + list(np.linspace(0.4, 0.6, max(nT, 1)))[:nT]
        )
        self.sigma_rho_delayed = [s / 4.0 for s in self.sigma_rho]
        self.sigma_loc_z = (
            [2.0] * n1 + list(np.linspace(2.0, 3.0, max(nT, 1)))[:nT]
        )
        self.sigma_loc_z_delayed = [s / 2.0 for s in self.sigma_loc_z]


def get_default_config() -> Config:
    """Return a Config with the same defaults as the MATLAB GetDefaultParameters()."""
    return Config()
