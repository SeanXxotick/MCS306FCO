# Video Script (target: 8–10 minutes)

The audience is general, so avoid jargon or explain it right away. Each member presents the part they own (see `docs/TEAM_WORKFLOW.md`). Slides are listed as **[SLIDE]**, and figures come from `results/figures/`.

---

## Part 1: The problem (Member 1, ~1.5 min)

**[SLIDE 1: Title]** *Physics-Informed Neural Networks versus Classical Least Squares for Estimating Dengue Transmission Parameters: Bulacan, 1994–2010*, with group names and BulSU.

> Every rainy season, dengue cases in Bulacan climb, and sometimes they explode. In August and September of 2010, more than a thousand cases were reported *each month* in our province.

**[SLIDE 2: fig5_bulacan_overview.png]**

> To respond to an outbreak, health workers want to know two things: how fast is it spreading, and how long do people stay infectious? Mathematicians capture this with the SIR model.

**[SLIDE 3: S → I → R diagram with the three equations]**

> We split the population into Susceptible, Infected and Recovered. Beta, β, is how fast infection spreads. Gamma, γ, is how fast people recover. Their ratio, R-naught, tells us how many people one sick person infects. If it's above 1, the outbreak grows.
>
> The catch is that nobody can measure β and γ directly. We have to work backwards from case counts. That's called an *inverse problem*, and it's what our project is about.

## Part 2: Solving the equations, RK4 (Member 2, ~1.5 min)

**[SLIDE 4: RK4 formula, 4 slopes averaged]**

> These equations can't be solved with pen and paper, so we step forward in time using the Runge–Kutta 4 method. At each step it samples the slope four times and takes a weighted average.

**[SLIDE 5: fig1_rk4_convergence.png]**

> How do we know our solver is right? Theory says that if we halve the step size, the error should shrink 16 times. That's "fourth-order" accuracy. We tested it, and our measured order is 3.98 to 4.01, exactly as predicted.
>
> The right panel shows something practical: the step size used *inside* the fitting procedure matters. Too coarse a step gives biased estimates.

## Part 3: Two ways to estimate β and γ (Member 3, ~2 min)

**[SLIDE 6: side-by-side diagram: Classical vs PINN]**

> **Method 1, classical.** Guess β and γ, simulate with RK4, compare with the data, adjust, and repeat. The computer uses an algorithm called least squares to find the best guess in just a few dozen tries.
>
> **Method 2, the AI approach: a Physics-Informed Neural Network, or PINN.** A neural network learns the shape of the outbreak curve. But we don't just ask it to fit the data. We also punish it whenever its curve breaks the SIR equations. β and γ are learned at the same time, as extra knobs inside the network.

**[SLIDE 7: loss formula L = L_data + w_ode·L_ode + L_ic]**

> So the network is graded on three things: matching the data, obeying the physics, and starting at the right place. The weight w-ode decides how much the physics counts.

## Part 4: Results (Member 4, ~2.5 min)

**[SLIDE 8: fig2_fits.png]**

> First we tested both methods on fake outbreaks where we *know* the true answer: β = 0.3 and γ = 0.1. We added 0, 5 and 10% noise. Both methods recover the true values. At 10% noise the PINN is off by 0.06% for β and 1.3% for γ.

**[SLIDE 9: table from exp2_summary.csv, highlight Time column]**

> But look at the time. The classical method takes about half a second. The PINN takes about a minute: over 100 times slower for the same answer.

**[SLIDE 10: fig4_ode_weight_sweep.png]**

> The PINN is also picky. When the physics weight was too high, β got stuck for 12,000 training steps and never fully converged. Choosing this weight is a real tuning problem.

**[SLIDE 11: fig6_bulacan_seasons.png]**

> Then real data. We fitted six outbreak seasons in Bulacan. The SIR model follows the curves closely, with R-squared from 0.90 to 0.998, and R-naught between about 1.3 and 3.4.

**[SLIDE 12: fig7_bulacan_pinn.png]**

> And the PINN on real data? Its curve fits the eight monthly points almost perfectly. But when we plug the β and γ it learned back into the equations, we get a *different* curve. With only eight data points, the network can bend to fit the data without truly learning the parameters.

## Part 5: Limitations and conclusion (Member 5, ~1.5 min)

**[SLIDE 13: Limitations]**

> Our model is simple. Dengue actually spreads through mosquitoes, which SIR ignores, so our "infectious period" comes out at around three to six weeks instead of one. That's because it absorbs the mosquito's part of the cycle. Our data are monthly and stop in 2010.

**[SLIDE 14: Conclusion]**

> - For a small model and sparse data, the classical method is faster, simpler and just as accurate.
> - PINNs are powerful, but they need more data and careful tuning.
> - Next, for our thesis: a mosquito–human model, weekly data, and rainfall-driven transmission, where a neural network could really earn its place.

**[SLIDE 15: Thank you + repo link]** `https://github.com/SeanXxotick/MCS306FCO`

---

### Recording tips

- Screen-record the slides and add each member's voice. A face cam is optional.
- Show `python run_all.py` running for about 10 seconds to prove the results are reproducible.
- Keep each part within its time; the total should be at most 10 minutes.
- Before recording, check the numbers on the slides against `results/tables/`. If you rerun the experiments, the numbers can change slightly.
