# Team Workflow

The rubric asks for "version history showing collaborative work". Each member makes **one small commit from their own GitHub account**, in their own files. The commits below are safe:

- They don't change any results, so the paper stays consistent.
- No two members edit the same file, so nobody gets a merge conflict.
- All the new tests have been checked and pass.

| Member | Owns | Presents in video | Status |
|---|---|---|---|
| Member 1 (KyHornales) | SIR model and RK4 (`src/sir.py`) | Problem and the SIR model | ✅ Done |
| Member 2 | Classical fitting (`src/classical.py`) | RK4 and the step-size study | ⬜ To do |
| Member 3 | PINN (`src/pinn.py`) | How the PINN works | ⬜ To do |
| Member 4 | Data and experiments (`data/`, `experiments/`) | Results | ⬜ To do |
| Member 5 (OhsirIIII) | Paper and presentation (`paper/`, `presentation/`) | Limitations and conclusion | ✅ Done |

---

## Step 0: one-time setup (everyone)

1. Accept the GitHub collaborator invite (check your email or <https://github.com/notifications>).
2. **Clone the repo; don't `git init` a new folder.** A new folder has no shared history with ours, so your commit can't be merged. Work directly on `main`; there's no need to create a branch. In a terminal:

```bash
git clone https://github.com/SeanXxotick/MCS306FCO.git C:\dengue_pinn
cd C:\dengue_pinn
git config user.name "YourGitHubUsername"
git config user.email "the-email-on-your-github-account"
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q
```

The last command should end with `passed`. The email must match one on your GitHub account, or the commit won't count as yours.

> If you already cloned the repo before 26 Oct 2026, delete that folder and clone again; the history was rewritten.

## Every commit follows the same steps

```bash
git pull
# ...make your edit (see your section below)...
python -m pytest -q
git add <the files you changed>
git commit -m "<your message>"
git push
```

If `git push` is rejected because someone pushed first, run `git pull`, then `git push` again.

---

## ~~Member 1: RK4 explanation + conservation test~~ ✅ Done (KyHornales)

**Edit 1.** In `src/sir.py`, inside the `rk4` docstring, find this line:

```python
    Local error is O(h^5), so the global error at t_end is O(h^4).
```

and add these two lines right below it:

```python
    The weights 1, 2, 2, 1 (divided by 6) are Simpson's rule applied to the slope:
    the start and end slopes count once, the two midpoint slopes count twice.
```

**Edit 2.** Create a new file `tests/test_rk4_properties.py`:

```python
"""Extra RK4 checks: the recovered compartment can only grow."""
import numpy as np

from src.sir import initial_state, rk4


def test_recovered_never_decreases():
    # dR/dt = gamma * I >= 0, so R must never go down
    _, Y = rk4(0.3, 0.1, initial_state(), 100, 0.5)
    assert np.all(np.diff(Y[:, 2]) >= 0)
```

**Commit:**

```bash
git add src/sir.py tests/test_rk4_properties.py
git commit -m "Explain RK4 weights and test that R is non-decreasing"
```

**Be ready to explain:** why R can never decrease (dR/dt = γI ≥ 0), and why the RK4 weights are 1-2-2-1.

---

## Member 2: least-squares explanation + noisy-fit test

**Edit 1.** In `src/classical.py`, replace the one-line docstring of `fit_synthetic`:

```python
    """Estimate (beta, gamma) from daily observations of I(t)."""
```

with:

```python
    """Estimate (beta, gamma) from daily observations of I(t).

    least_squares uses a trust-region method: around the current guess it builds a local
    linear model of the residuals and takes the best step that stays inside a "trust
    region"; the region grows when steps work and shrinks when they don't.
    nfev = number of residual evaluations, i.e. full RK4 simulations.
    """
```

**Edit 2.** Create a new file `tests/test_classical_noise.py`:

```python
"""The classical fit should stay accurate when the data are noisy."""
import pytest

from src import config as C
from src.classical import fit_synthetic
from src.sir import synthetic_data


def test_classical_recovers_noisy_parameters():
    t, I_obs, _, y0 = synthetic_data(noise=0.05, seed=1)
    f = fit_synthetic(t, I_obs, y0)
    assert f["beta"] == pytest.approx(C.TRUE_BETA, rel=0.02)    # within 2%
    assert f["gamma"] == pytest.approx(C.TRUE_GAMMA, rel=0.02)
```

**Commit:**

```bash
git add src/classical.py tests/test_classical_noise.py
git commit -m "Document least-squares solver and add noisy-fit test"
```

**Be ready to explain:** what a residual is, what `nfev` counts, and why each evaluation runs a full RK4 simulation.

---

## Member 3: PINN comments + prediction test

**Edit 1.** In `src/pinn.py`, in the `SIRNet` class, change:

```python
    def forward(self, tau):
        return torch.sigmoid(self.body(tau))
```

to:

```python
    def forward(self, tau):
        # sigmoid squashes the outputs into (0, 1): S, R and the scaled I are all fractions
        return torch.sigmoid(self.body(tau))
```

**Edit 2.** In the same file, inside `train_pinn` (the first function that has these lines, around line 80), add a comment above `log_beta = ...`:

```python
    # store log(beta), log(gamma): exp() of any number is positive, so the rates stay > 0
    log_beta = nn.Parameter(torch.tensor(np.log(p["beta0"])))
```

**Edit 3.** Create a new file `tests/test_pinn_predict.py`:

```python
"""The trained PINN returns one non-negative prediction per time point."""
import numpy as np
import pytest

from src.sir import synthetic_data


def test_pinn_prediction_shape_and_range():
    pytest.importorskip("torch")
    from src.pinn import predict_I, train_pinn
    t, I_obs, _, y0 = synthetic_data(noise=0.0)
    r = train_pinn(t, I_obs, y0, settings=dict(max_epochs=50, check_every=10_000))
    pred = predict_I(r, t)
    assert pred.shape == t.shape
    assert np.all(pred >= 0)
```

**Commit:**

```bash
git add src/pinn.py tests/test_pinn_predict.py
git commit -m "Comment PINN parameterization and add prediction test"
```

**Be ready to explain:** the three parts of the loss, what a collocation point is, and why β is stored as log β.

---

## Member 4: data documentation + results summary

**Edit 1.** Create a new file `data/README.md`:

```markdown
# Data

## processed/bulacan_monthly_1994_2010.csv
Monthly reported dengue cases for the Province of Bulacan (Central Luzon, Region III),
January 1994 – December 2010 (200 months; a few months are missing in 1994, 1995 and 2002).

| Column | Meaning |
|---|---|
| calendar_start_date | first day of the month |
| calendar_end_date | last day of the month |
| Year | calendar year |
| dengue_total | reported dengue cases in that month |
| month | month number (1 = January) |

**Source:** OpenDengue V1.3 spatial extract (Clarke et al., 2024), https://opendengue.org,
licensed CC BY 4.0. Created by `src/data.py`, which downloads the release into `raw/`
(not committed, ~55 MB) and filters the Bulacan rows.

## Synthetic data
Generated on the fly by `src/sir.py` (`synthetic_data`) with β = 0.3, γ = 0.1 and seed 0;
no file is stored.
```

**Edit 2.** In the main `README.md`, add this section right above `## Main limitations`:

```markdown
## Key results

| Experiment | Finding |
|---|---|
| 1. RK4 step size | Observed order of accuracy 3.98–4.01 (theory: 4) |
| 2. Synthetic recovery | Both methods recover β, γ within ~1.3% at 10% noise; the PINN is ~118× slower |
| 3. PINN loss weight | With w_ode = 1 the PINN stalls and hits the 20,000-epoch cap |
| 4. Bulacan 1998–2010 | Classical SIR fits 6 seasons (R² 0.90–0.998, R0 1.29–3.43); the PINN on 2010 is not self-consistent |

Full numbers: `results/tables/`; figures: `results/figures/`.
```

**Commit:**

```bash
git add data/README.md README.md
git commit -m "Add data documentation and key results summary"
```

**Be ready to explain:** where the data come from, why it stops at 2010, and what N_eff means.

---

## Member 5: author names + rebuilt paper

**Edit.** In `paper/build_paper.py`, change:

```python
AUTHORS = ["[Member 1]", "[Member 2]", "[Member 3]", "[Member 4]", "[Member 5]"]
```

to the real names, e.g. `AUTHORS = ["Juan Dela Cruz", "Maria Santos", ...]`. Then rebuild:

```bash
python paper/build_paper.py
```

**Commit:**

```bash
git add paper/build_paper.py paper/PINN_Dengue_Bulacan_Paper.docx
git commit -m "Add author names and rebuild paper"
```

Later, when the slides are done, add them in a second commit:

```bash
git add presentation/
git commit -m "Add presentation slides"
```

**Be ready to explain:** the limitations section and the thesis directions in the conclusion.

---

## Things NOT to change (unless the whole group agrees)

- **Values in `src/config.py`** (seeds, epochs, step sizes, `SEASONS`, `W_ODE_SWEEP`). These change the results. You would then have to rerun `python run_all.py` (15–30 min), rebuild the paper, and update the video numbers. The figure layouts also fit exactly 6 seasons and 4 weights.
- **The Word file, by hand.** `build_paper.py` overwrites it. Do final formatting in Word only as the very last step.
