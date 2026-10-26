# Physics-Informed Neural Networks for Estimating Dengue Transmission Parameters in Bulacan

MCS 306 Final Course Output. This is a mini-research project comparing a **Physics-Informed Neural Network (PINN)** with a **classical RK4 + least-squares fit** for estimating the transmission rate β and recovery rate γ of an SIR epidemic model. We use synthetic outbreaks and real monthly dengue counts from Bulacan, Philippines (OpenDengue, 1994–2010).

## What's in here

```
src/                     the numerical core (read these first)
  config.py              every setting: seeds, step sizes, tolerances, epochs
  sir.py                 SIR model, RK4 solver, synthetic data, convergence study
  pinn.py                PINN model and training loop (PyTorch)
  classical.py           RK4 + scipy least_squares baseline, bootstrap
  data.py                downloads OpenDengue and extracts Bulacan
  plotting.py            shared figure style
experiments/
  exp1_rk4_convergence.py    RK4 error vs step size h (order of accuracy)
  exp2_synthetic_recovery.py PINN vs classical at 0/5/10% noise
  exp3_ode_weight_sweep.py   PINN stability vs physics-loss weight
  exp4_bulacan.py            real Bulacan outbreak seasons
run_all.py               reproduces every table and figure
tests/                   pytest checks of the numerical core
data/processed/          Bulacan monthly cases (small CSV, committed)
results/tables/          CSV outputs used in the paper
results/figures/         PNG figures used in the paper
paper/                   build_paper.py -> the Word manuscript
presentation/            video script and slide outline
docs/                    study guide (explanations for the panel)
```

## How to run it (Windows, Python 3.11+)

Keep the project in a **short path** such as `C:\dengue_pinn`. Very long Windows paths break Python virtual environments.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

macOS/Linux: use `source .venv/bin/activate`; the rest is the same.

Then:

```bash
python -m pytest -q          # quick checks (~30 s)
python run_all.py            # all experiments (roughly 15-30 min on a laptop CPU)
python run_all.py 1          # just one experiment (1, 2, 3 or 4)
python paper/build_paper.py  # rebuild the Word paper from the latest results
```

Experiment 4 uses `data/processed/bulacan_monthly_1994_2010.csv`. If you delete it, it is rebuilt automatically from the OpenDengue release, which is downloaded once (about 55 MB) into `data/raw/`.

## Reproducibility

- All settings are in `src/config.py`: random seed 0, the RK4 step sizes, PINN learning rate, epochs, stopping tolerance and loss weights.
- Every number and figure in the paper comes from `results/`, and `paper/build_paper.py` reads them directly.
- `results/run_log.txt` records the Python and library versions, the CPU, and the runtime of each experiment.
- PINN training time depends on the machine, but the estimates are deterministic for a given seed and library version.

## Data

**OpenDengue** V1.3 spatial extract (Clarke et al., 2024), licensed CC BY 4.0: <https://opendengue.org>. We use the monthly province-level counts for Bulacan (Central Luzon, Region III), 1994–2010. Province-level counts after 2010 are not available in this release.

No personal or hospital data are used.

## Main limitations

- A plain SIR model ignores the mosquito vector, under-reporting and seasonality. The real-data rates should therefore be read as *effective* rates, not biological constants.
- Monthly data give only 8 points per season, which is too little for the PINN to identify β and γ reliably (see Experiment 4).
