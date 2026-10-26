"""Reproduce every table and figure in the paper.

    python run_all.py            # all experiments (about 15-30 min on a laptop CPU)
    python run_all.py 1 4        # only experiments 1 and 4

Results go to results/tables/ and results/figures/; a run log with timings and
environment details is written to results/run_log.txt.
"""
import platform
import sys
import time

import matplotlib
import numpy
import pandas
import scipy
import torch

from experiments import exp1_rk4_convergence, exp2_synthetic_recovery, exp3_ode_weight_sweep, exp4_bulacan
from src import config as C

EXPERIMENTS = {
    "1": ("RK4 step size / convergence", exp1_rk4_convergence.main),
    "2": ("Synthetic recovery: PINN vs classical", exp2_synthetic_recovery.main),
    "3": ("PINN physics-loss weight sweep", exp3_ode_weight_sweep.main),
    "4": ("Real data: Bulacan outbreak seasons", exp4_bulacan.main),
}


def main(selected):
    C.TABLES.mkdir(parents=True, exist_ok=True)
    log = [f"Run started {time.strftime('%Y-%m-%d %H:%M:%S')}",
           f"Python {sys.version.split()[0]} on {platform.platform()}",
           f"CPU: {platform.processor()}",
           f"numpy {numpy.__version__}, scipy {scipy.__version__}, pandas {pandas.__version__}, "
           f"matplotlib {matplotlib.__version__}, torch {torch.__version__}",
           f"torch threads: {C.TORCH_THREADS}, seed: {C.SEED}", ""]
    for key in selected:
        name, run = EXPERIMENTS[key]
        print(f"\n=== Experiment {key}: {name} ===")
        start = time.perf_counter()
        run()
        log.append(f"Experiment {key} ({name}): {time.perf_counter() - start:.1f} s")
    log_path = C.ROOT / "results" / ("run_log.txt" if len(selected) == len(EXPERIMENTS)
                                     else f"run_log_{'_'.join(selected)}.txt")
    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n" + "\n".join(log))


if __name__ == "__main__":
    main(sys.argv[1:] or list(EXPERIMENTS))
