# Study Guide: Understanding the Project

Read this before the presentation. Each section answers the questions a panel is likely to ask, using the same names as the code.

---

## 1. The big picture (everyone should know this)

**Problem.** Dengue is a major public-health burden in the Philippines, and Bulacan has outbreaks during the rainy season (July to October). Two numbers summarise an outbreak:

- **β, the transmission rate:** how fast infected people infect susceptible people.
- **γ, the recovery rate:** how fast infected people stop being infectious. 1/γ is the average infectious period in days.
- **R0 = β/γ, the basic reproduction number:** how many people one case infects in a fully susceptible population. If R0 > 1 the outbreak grows.

**Question.** Given case data, how do we *estimate* β and γ? This is called an **inverse problem**. The "forward" problem is: given β and γ, simulate the outbreak.

**Two methods compared.**

1. **Classical.** Guess β and γ, simulate with RK4, compare with the data, and adjust with least squares. Repeat until the fit is best.
2. **PINN.** Train a neural network to fit the data *and* obey the SIR equations at the same time. β and γ are learned like extra weights.

**One-sentence result.** On clean or noisy synthetic data both methods recover β and γ accurately, but the classical method is much faster and has no sensitive tuning knob. On 8 months of real Bulacan data, the PINN's curve fits but its learned β and γ do not reproduce that curve. That is a real limitation and a good thesis direction.

---

## 2. The SIR model (`src/sir.py`)

Everything is a fraction of the population: S + I + R = 1.

```
dS/dt = -β S I          susceptible people get infected
dI/dt =  β S I - γ I    infected: new infections minus recoveries
dR/dt =  γ I            recovered
```

- **β S I** is the infection term. It needs both susceptible (S) and infected (I) people, which is why S and I are multiplied: the "mass action" assumption.
- **γ I** is the recovery term. A fixed fraction of infected people recover per day.
- **Why S + I + R stays 1.** Add the three equations: the right-hand sides cancel, so d(S+I+R)/dt = 0. The code checks this (`max_mass_drift` in Experiment 1).

**Panel question: "Why SIR for dengue? Dengue spreads through mosquitoes."**
Correct, and it's a simplification. A vector-host model (Esteva & Vargas, 1998) adds mosquito compartments. SIR is the simplest model with the right outbreak shape, so β and γ here are *effective* rates that absorb the mosquito stage. That is why the real-data infectious period comes out longer than the clinical ~5–7 days. Extending to a vector-host model is the natural thesis extension.

---

## 3. RK4: the numerical solver (`rk4` in `src/sir.py`)

The ODE cannot be solved in closed form, so we step forward in time with step size **h**:

```
k1 = f(y)
k2 = f(y + h/2 · k1)
k3 = f(y + h/2 · k2)
k4 = f(y + h · k3)
y_next = y + h/6 · (k1 + 2k2 + 2k3 + k4)
```

- It is a weighted average of four slope estimates across the step: start, two midpoints, end.
- **Error:** the local error per step is O(h⁵), and the global error after T/h steps is **O(h⁴)**. Halving h divides the error by 2⁴ = 16.
- **How we verify it (Experiment 1):** solve with h = 4, 2, 1, 0.5, 0.25, 0.125 and compare with a reference solution at h = 0.001. The observed order is p = log(e_h / e_{h/2}) / log 2. We get ≈ 4.0, which matches theory.
- **Experiment 1b:** the step size used *inside* the fitting procedure also matters. A coarse h (5 days) biases the estimates; a fine h removes the bias.

**Panel question: "Why not Euler's method?"**
Euler is O(h), first order. Getting the same accuracy needs vastly more steps. RK4 is the standard trade-off between accuracy and cost per step.

**Panel question: "Why is the h = 0.1 fit error essentially zero?"**
The synthetic data were *generated* with RK4 at h = 0.1, so fitting with the same solver and step reproduces them exactly. This is the "inverse crime": a too-perfect test. That is why we add noise in Experiment 2.

---

## 4. Classical fitting (`src/classical.py`)

**Least squares.** Find (β, γ) minimising Σ (I_model(tᵢ; β, γ) − I_obs(tᵢ))².

- `scipy.optimize.least_squares` uses a **trust-region reflective** algorithm, a Newton-like method that respects bounds (β, γ > 0).
- It needs a residual vector and estimates the Jacobian by finite differences.
- Each "function evaluation" (`nfev`) is one full RK4 simulation.
- **Tolerance:** `ftol`, `xtol` and `gtol` (all 1e-10) stop the iteration when the cost, the parameters or the gradient stop changing.

**Real data (Experiment 4).** The data are *monthly new cases*, not I(t). The model predicts

```
cases in month k = N_eff × (S at start of month − S at end of month)
```

because every drop in S is a new infection.

- **N_eff (effective population)** absorbs under-reporting and prior immunity. It is much smaller than Bulacan's real population, because only a small fraction of real infections are reported, and that is expected.
- We use **multiple starting points** because the problem is non-convex and could get stuck in a local minimum.

**Bootstrap (error estimation).** Resample the fit's residuals, add them back to the fitted curve to make 50 "fake datasets", refit each, and take the 2.5% and 97.5% percentiles. That gives a 95% interval for each parameter. Wide intervals mean the data cannot pin the parameter down.

---

## 5. The PINN (`src/pinn.py`)

**Network.** Input: time τ = t/T in [0, 1]. Output: (S, I, R) through a sigmoid, so it stays in (0, 1). There are 3 hidden layers of 32 tanh neurons.

**The trick: three losses.**

```
L = L_data + w_ode · L_ode + L_ic
```

- **L_data:** the network's I(t) should match the observations.
- **L_ode:** at 200 *collocation points* (times where we check the physics), the network's derivatives must satisfy the SIR equations. The derivatives come from **automatic differentiation** (`torch.autograd.grad`), so no RK4 is needed inside the PINN.
- **L_ic:** the network must start at the known initial state.
- **β and γ** are `nn.Parameter`s stored as log-values, so exp(·) keeps them positive. The optimiser updates them together with the weights; they are the only values in the network that "mean" something.

**Scaling.** I is small (≈ 0.01–0.3), so the network outputs I/s_I, where s_I is the maximum observed I. All outputs are then O(1), which helps training.

**Optimiser: Adam, learning rate 1e-3.** The learning rate is the PINN's "step size". Each epoch is one gradient step.

**Stopping rule (tolerance).** Every 500 epochs we compare β and γ with their values 500 epochs earlier. We stop when both relative changes are below `tol` = 1e-4, otherwise at 20,000 epochs. The reported `iterations` is the stopping epoch.

**Physics-loss weight w_ode (Experiment 3).** With w_ode = 1 the physics term dominates early. The network finds a smooth curve that satisfies the ODE for the *wrong* β and γ and barely fits the data. Smaller weights let the data steer the parameters. This sensitivity is a known PINN difficulty (Wang et al., 2021) and the main practical drawback compared with least squares.

**Panel question: "If the classical method is faster and as accurate, why use a PINN?"**
- PINNs shine when the model is a PDE, when data are scattered or partial, when some compartments are unobserved, or when no fast solver exists.
- For a 2-parameter ODE with good data, the classical method wins, and showing that honestly *is* a result.
- PINNs also give a smooth, differentiable solution as a by-product.

---

## 6. Reading the results

| Experiment | Figure / table | What to say |
|---|---|---|
| 1 | fig1, exp1_*.csv | RK4 is 4th-order as theory predicts; the solver step inside a fit controls bias |
| 2 | fig2, fig3, exp2_summary.csv | Both methods recover β, γ within a few % even at 10% noise; classical is ~100× faster |
| 3 | fig4, exp3_ode_weight_sweep.csv | PINN accuracy depends strongly on w_ode, which is a stability/tuning issue |
| 4 | fig5–7, exp4_*.csv | Real seasons fit well with classical SIR (R0 values, wide bootstrap intervals); the PINN on 8 monthly points is not self-consistent |

Exact numbers are in `results/tables/` and in the paper.

---

## 7. Likely panel questions (quick answers)

- **Why synthetic data?** We know the true answer, so we can measure error. Real data have no ground truth.
- **Why only 1994–2010?** That is what OpenDengue provides at province level for Bulacan. National data are more recent, but they mix many regions with different timing.
- **Is R0 ≈ 1.x–3 believable?** Published dengue R0 estimates commonly fall roughly between 1 and 5, so our values are in a plausible range. But they are effective values from a simplified model.
- **What is a collocation point?** A time where we *evaluate the ODE residual*. It needs no data there; it only enforces the physics.
- **What is automatic differentiation?** PyTorch records every operation and applies the chain rule exactly. It is not a finite-difference approximation.
- **Computational cost?** RK4 is O(T/h) per simulation, and least squares needs only a few dozen simulations. The PINN does thousands of epochs, each a forward and backward pass over 200 points. See the `seconds` columns.
- **What would you do with more time?** Vector-host model, weekly data, seasonality β(t) linked to rainfall, multiple seasons jointly, and a PINN with fewer free parameters or better weighting.
