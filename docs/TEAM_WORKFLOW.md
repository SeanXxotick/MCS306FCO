# Team Workflow

The rubric asks for "version history showing collaborative work", so each member should own one area. Make real commits from your own GitHub account: read the code, improve a comment, or rerun and commit the results.

| Member | Owns | Files | Presents in video |
|---|---|---|---|
| Member 1 | SIR model and RK4 | `src/sir.py`, `experiments/exp1_rk4_convergence.py` | Problem and the SIR model |
| Member 2 | Classical fitting and bootstrap | `src/classical.py` | RK4 and step-size study |
| Member 3 | PINN | `src/pinn.py`, `experiments/exp3_ode_weight_sweep.py` | How the PINN works |
| Member 4 | Experiments and data | `src/data.py`, `experiments/exp2_*.py`, `experiments/exp4_*.py`, `run_all.py` | Results |
| Member 5 | Paper and presentation | `paper/`, `presentation/`, `README.md` | Limitations and conclusion |

## First-time setup (each member)

```bash
git clone https://github.com/SeanXxotick/MCS306FCO.git C:\dengue_pinn
cd C:\dengue_pinn
git config user.name "Your Name"
git config user.email "you@example.com"
```

Then follow the "How to run it" section of the README.

The repository owner must add each member under GitHub, then the repo's **Settings → Collaborators**.

## Everyday loop

```bash
git pull
# ...edit your files...
python -m pytest -q
git add <your files>
git commit -m "Explain the RK4 stage weights in sir.py"
git push
```

Write commit messages that say *what* changed. Don't edit someone else's area without telling them; this avoids merge conflicts.

## Good first commits (easy, and they build understanding)

- **Member 1:** add a sentence to the `rk4` docstring explaining why the weights are 1, 2, 2, 1 (it's Simpson's rule).
- **Member 2:** try `BOOTSTRAP_B = 100` in `config.py`, rerun `python run_all.py 4`, and commit the updated tables.
- **Member 3:** add `0.03` to `W_ODE_SWEEP`, rerun experiment 3, and commit.
- **Member 4:** add another outbreak year to `SEASONS` (for example 2007) and rerun experiment 4.
- **Member 5:** fill in member names in `paper/build_paper.py` (`AUTHORS`), rebuild the paper, and commit.
