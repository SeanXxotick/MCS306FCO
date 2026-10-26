"""Every setting used by the experiments lives here, so results are reproducible.

Change values in this file (not inside the experiment scripts) and re-run `python run_all.py`.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

SEED = 0  # global random seed (noise generation, network initialisation)

# ---------------------------------------------------------------- synthetic "ground truth"
TRUE_BETA = 0.3     # transmission rate  [1/day]
TRUE_GAMMA = 0.1    # recovery rate      [1/day]  -> R0 = 3, infectious period 10 days
I0 = 0.01           # initial infected fraction
T_DAYS = 100        # length of the synthetic outbreak [days]
NOISE_LEVELS = [0.0, 0.05, 0.10]   # multiplicative Gaussian noise on observed I(t)
NOISE_SEEDS = [0, 1, 2, 3, 4]      # repeated noise draws for the (cheap) classical method

# ---------------------------------------------------------------- RK4
H_DATA = 0.1                          # step size used to generate synthetic data
H_REF = 1e-3                          # "exact" reference solution for the error study
H_LIST = [4, 2, 1, 0.5, 0.25, 0.125]  # step sizes compared in the convergence study
H_FIT_LIST = [5, 2, 1, 0.5, 0.1]      # step sizes used *inside* the classical fit

# ---------------------------------------------------------------- PINN
PINN = dict(
    width=32,          # neurons per hidden layer
    depth=3,           # hidden layers
    lr=1e-3,           # Adam learning rate (the optimiser's "step size")
    n_col=200,         # collocation points where the ODE residual is enforced
    max_epochs=20000,  # hard cap on iterations
    check_every=500,   # how often the stopping rule is evaluated
    tol=1e-4,          # stop when beta and gamma change by < tol (relative) over `check_every` epochs
    w_ode=0.01,        # weight of the physics (ODE residual) loss
    beta0=0.5,         # initial guesses for the learnable parameters
    gamma0=0.2,
)
W_ODE_SWEEP = [1.0, 0.1, 0.01, 0.001]  # physics-loss weights compared in experiment 3
SWEEP_NOISE = 0.05
TORCH_THREADS = 1  # small network: one CPU thread is fastest and keeps runs comparable

# ---------------------------------------------------------------- classical least squares
LSQ_TOL = 1e-10      # ftol = xtol = gtol for scipy.optimize.least_squares
H_FIT = 0.1          # RK4 step size inside the classical fit (synthetic data)

# ---------------------------------------------------------------- real data (OpenDengue)
OPENDENGUE_URL = ("https://raw.githubusercontent.com/OpenDengue/master-repo/main/"
                  "data/releases/V1.3/Spatial_extract_V1_3.zip")
OPENDENGUE_ZIP = DATA_RAW / "Spatial_extract_V1_3.zip"
BULACAN_CSV = DATA_PROCESSED / "bulacan_monthly_1994_2010.csv"
PROVINCE = "PROVINCE OF BULACAN"
SEASONS = [1998, 2001, 2005, 2006, 2008, 2010]  # years with a clear outbreak peak (> 300 cases/month)
SEASON_FIRST_MONTH = 5   # May ... December: the Philippine rainy-season dengue wave
DAYS_PER_MONTH = 30.4
H_REAL = 0.5             # RK4 step size for the real-data model [days]
# Effective infectious period 1/gamma constrained to 7-60 days. Without it the 4-parameter fit can
# collapse to a degenerate solution (gamma at its bound, R0 ~ 1, huge N_eff) that reproduces 8
# monthly points equally well. The range covers the human infectious period (~1 week) plus the
# mosquito stage that a plain SIR model absorbs into gamma.
INFECTIOUS_DAYS_RANGE = (7, 60)
BOOTSTRAP_B = 50         # residual-bootstrap replicates for confidence intervals (2010 season)
REAL_PINN_EPOCHS = 20000
