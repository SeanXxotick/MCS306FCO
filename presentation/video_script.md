# Video Script (target: about 6 minutes)

This script matches the slide deck [`PINN_Dengue_Bulacan_Slides.pptx`](PINN_Dengue_Bulacan_Slides.pptx) (11 slides). The same narration is in each slide's speaker notes. The audience is general, so avoid jargon or explain it right away. Figures come from `results/figures/`.

---

## Part 1: The problem (Jian Kyle C. Hornales, ~1.5 min)

**[SLIDE 1: Title]**

> Hello! We are Romar Almoguez, Sean Arkin Dela Cruz, Osyris Fajardo and Jian Kyle Hornales, BS Mathematics students from the College of Science of Bulacan State University. Our project compares an AI method, physics-informed neural networks, with classical least squares for estimating how dengue spreads in Bulacan.

**[SLIDE 2: Dengue in Bulacan (fig5_bulacan_overview.png)]**

> Every rainy season, dengue cases in Bulacan climb, and sometimes they explode. This chart shows the monthly reported cases from 1994 to 2010, from the open OpenDengue database. In 2010, more than a thousand cases were reported in August and again in September: 1,216 in September alone. To respond, health workers want to know two things: how fast is dengue spreading, and how long do people stay infectious?

**[SLIDE 3: The SIR model and the inverse problem]**

> Mathematicians answer this with the SIR model. We split the population into Susceptible, Infected and Recovered. Beta is how fast the infection spreads, and gamma is how fast people recover, so one over gamma is how long they stay infectious. Their ratio, R-naught, is how many people one sick person infects; above 1, the outbreak grows. The catch is that nobody can measure beta and gamma directly. We have to work backwards from case counts. That is called an inverse problem, and it is what our project is about.

## Parts 2–3: Solving the model and two estimation methods (Sean Arkin C. Dela Cruz, ~1.5 min)

**[SLIDE 4: RK4 (fig1_rk4_convergence.png)]**

> These equations have no pen-and-paper solution, so we step forward in time with the fourth-order Runge–Kutta method, or RK4. Each step samples the slope four times and takes a weighted average. To check our solver, we halved the step size again and again. Theory says the error should shrink sixteen times with every halving; that is fourth-order accuracy. Our measured order is between 3.98 and 4.01, exactly as predicted. The right panel shows that the step size inside the fitting matters too: a coarse five-day step biases the estimates by about 0.4 percent, so we fit with a step of 0.1 day.

**[SLIDE 5: Two ways to estimate β and γ]**

> We compare two ways to estimate beta and gamma. Method one is classical: guess beta and gamma, simulate with RK4, compare with the data, and let a least-squares algorithm improve the guess. On our test data it finishes in under a second. Method two is the AI approach: a physics-informed neural network, or PINN. A neural network learns the outbreak curve, and beta and gamma are learned at the same time as extra trainable parameters. Its loss has three parts: fit the data, obey the SIR equations, and start from the right initial state. The weight w-ode sets how much the physics counts.

## Part 4: Results (Romar S. Almoguez, ~2 min)

**[SLIDE 6: Synthetic test (fig2_fits.png)]**

> First, we tested both methods on fake outbreaks where we know the true answer, beta 0.3 and gamma 0.1, with 0, 5 and 10 percent noise. Both methods recover the true values: even at 10 percent noise, every error is below 1.5 percent. But look at the time. The classical method takes about half a second, while the PINN needs thousands of training epochs and about a minute. On average, it is 118 times slower for the same answer.

**[SLIDE 7: Tuning the PINN (fig4_ode_weight_sweep.png)]**

> The PINN is also sensitive to tuning. We trained it with four values of the physics weight. With w-ode equal to 1, beta barely moved for about 12,000 epochs, and training hit the 20,000-epoch limit still 4 percent off. With 0.01, the best value, beta came out almost exact. Choosing this weight is a real tuning problem that the classical method simply does not have.

**[SLIDE 8: Six Bulacan seasons (fig6_bulacan_seasons.png)]**

> Then, real data. We fitted the SIR model to six outbreak seasons in Bulacan, from May to December, using monthly case counts. The model follows the curves closely, with R-squared from 0.903 to 0.998, and R-naught between 1.3 and 3.4. One warning: in 2006, the infectious period hit our lower limit of seven days, so we trust that season less.

**[SLIDE 9: The PINN on real data (fig7_bulacan_pinn.png)]**

> And the PINN on real data? For 2010, its network curve fits the eight monthly points almost perfectly, with an R-squared of 0.9997. But when we put the beta and gamma it learned back into RK4, we get a different curve, with an R-squared of only 0.647. With only eight data points, the network can bend to fit the data without truly learning the parameters. The classical fit gives R-naught 2.72, with a bootstrap 95 percent interval of 1.29 to 4.24.

## Part 5: Limitations and conclusion (Osyris Benedict P. Fajardo, ~1 min)

**[SLIDE 10: What we learned]**

> To wrap up: for a small model and sparse data, the classical method is faster, simpler and more reliable. PINNs work, but they need more data and careful tuning. We also found that the data pin down R-naught much better than the recovery rate gamma, whose interval covers the whole range we allowed. Our study has limits. SIR ignores mosquitoes, so our infectious period of three to six weeks absorbs the mosquito's part of the cycle. Our data are monthly and end in 2010, and the 2006 fit hit a bound. For our thesis, we plan a mosquito–human model with weekly data and rainfall-driven transmission, where a neural network could really earn its place.

**[SLIDE 11: Thank you + repo link]**

> Thank you for watching! Our code, data, figures and paper are all on GitHub at github.com/SeanXxotick/MCS306FCO.

---

### Recording tips

- Screen-record the slides and add each member's voice. A face cam is optional.
- Show `python run_all.py` running for about 10 seconds to prove the results are reproducible.
- Keep each part within its time; the total should be about 6 minutes.
- Before recording, check the numbers on the slides against `results/tables/`. If you rerun the experiments, the numbers can change slightly.

### Editing the slides in Canva

- In Canva, choose **Create a design → Import file** and upload the `.pptx`. Every text box, shape and figure stays editable.
- The fonts are Noto Serif (headings) and Noto Sans (text), which Canva also has. Equations are images, so they look the same everywhere.
- Put the BulSU and College of Science logos over the dashed **LOGO** circles on slides 1 and 11, then delete the circles.
- If you change a number on a slide, change it in the narration too.
