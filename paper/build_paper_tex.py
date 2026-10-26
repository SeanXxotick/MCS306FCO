"""Generate paper/PINN_Dengue_Bulacan_Paper.tex from the experiment CSVs.

LaTeX counterpart of build_paper.py: it reads the same files in results/tables/ and the same
src/config.py, so every number, table and data-dependent sentence in the manuscript is filled
in from your results and stays in sync when you re-run the experiments.

Usage (from the project root, next to build_paper.py):
    python paper/build_paper_tex.py
    cd paper && latexmk -pdf PINN_Dengue_Bulacan_Paper.tex      # or upload to Overleaf

Figures are taken from C.FIGURES (the same PNGs build_paper.py embeds); the \\graphicspath in
the output points there, with a local figures/ folder as fallback (handy for Overleaf).
"""
import math
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

AUTHORS = ["Almoguez, Romar S.", "Dela Cruz, Sean Arkin C.", "Fajardo, Osyris Benedict P.",
           "Hornales, Jian Kyle C."]
AFFILIATION = ["Bachelor of Science in Mathematics major in Computer Science",
               "College of Science, Bulacan State University, Malolos, Bulacan, Philippines"]
COURSE = "MCS 306 Final Course Output"
REPO_URL = "https://github.com/SeanXxotick/MCS306FCO"
OUT = ROOT / "paper" / "PINN_Dengue_Bulacan_Paper.tex"
FIG = C.FIGURES
TAB = C.TABLES

# ------------------------------------------------------------------ load results (same as build_paper.py)
conv = pd.read_csv(TAB / "exp1_rk4_convergence.csv")
fitstep = pd.read_csv(TAB / "exp1_step_size_in_fit.csv")
summ = pd.read_csv(TAB / "exp2_summary.csv")
cls_all = pd.read_csv(TAB / "exp2_classical.csv")
sweep = pd.read_csv(TAB / "exp3_ode_weight_sweep.csv")
seasons = pd.read_csv(TAB / "exp4_seasons.csv")
boot = pd.read_csv(TAB / "exp4_bootstrap_2010.csv").set_index("parameter")
pinn10 = pd.read_csv(TAB / "exp4_pinn_2010.csv")

pinn = summ[summ.method == "PINN"].set_index("noise")
cls = summ[summ.method != "PINN"].set_index("noise")
orders = conv.observed_order.dropna()
speedup = pinn.seconds.mean() / cls.seconds.mean()
best_w = sweep.loc[(sweep.beta_err_pct + sweep.gamma_err_pct).idxmin()]
worst_w = sweep.loc[(sweep.beta_err_pct + sweep.gamma_err_pct).idxmax()]
w_max = sweep.loc[sweep.w_ode.idxmax()]
w_min = sweep.loc[sweep.w_ode.idxmin()]
s2010 = seasons.set_index("season").loc[2010]
lo_days, hi_days = C.INFECTIOUS_DAYS_RANGE
at_bound = seasons[(seasons.infectious_days < lo_days * 1.01) | (seasons.infectious_days > hi_days * 0.99)]
interior = seasons.drop(at_bound.index)
p_cls, p_pinn = pinn10.iloc[0], pinn10.iloc[1]


# ------------------------------------------------------------------ LaTeX formatting helpers
def f(x, d=2):
    return f"{x:.{d}f}"


def sci(x, d):
    """1.698e-05 -> $1.698\\times10^{-5}$"""
    mant, exp = f"{x:.{d}e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def pct(x):
    """Percent value; scientific notation below 0.01% (same rule as build_paper.py)."""
    return rf"{x:.2f}\%" if x >= 0.01 else sci(x, 1) + r"\%"


def esc(text):
    """Escape plain text for LaTeX; N_eff and I0 become math."""
    text = str(text).replace("N_eff", "\x00N").replace("I0", "\x00I")
    for ch, rep in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                    ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")):
        text = text.replace(ch, rep)
    return text.replace("\x00N", r"$\Neff$").replace("\x00I", "$I_0$")


def tabular(label, caption, colspec, header_rows, rows, note=None, tabcolsep=None):
    """Build a booktabs table environment. Cells are already LaTeX."""
    out = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}",
           r"\small"]
    if tabcolsep:
        out.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    out += [rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}", r"\toprule"]
    out += [" & ".join(r_) + r"\\" for r_ in header_rows]
    out.append(r"\midrule")
    out += [" & ".join(r_) + r"\\" for r_ in rows]
    out += [r"\bottomrule", r"\end{tabular}"]
    if note:
        out += ["", r"\smallskip", rf"\parbox{{\textwidth}}{{\footnotesize\textit{{Note.}} {note}}}"]
    out.append(r"\end{table}")
    return "\n".join(out)


def b(h):
    return rf"\textbf{{{h}}}"


# ------------------------------------------------------------------ tables
def table1():
    rows = [
        ["RK4 (data generation)", "step size $h$", f"{C.H_DATA} day"],
        ["RK4 (convergence study)", "step sizes $h$",
         ", ".join(f"{h:g}" for h in C.H_LIST) + f" (reference $h={C.H_REF:g}$)"],
        ["RK4 (inside synthetic fit)", "step size $h$",
         f"{C.H_FIT} day (Exp.\\ 1b varies {', '.join(f'{h:g}' for h in C.H_FIT_LIST)})"],
        ["RK4 (real-data model)", "step size $h$", f"{C.H_REAL} day"],
        ["Least squares", "tolerances ftol, xtol, gtol", rf"$10^{{{round(math.log10(C.LSQ_TOL))}}}$"],
        ["PINN", "Adam learning rate", f"{C.PINN['lr']:g}"],
        ["PINN", "maximum epochs", f"{C.PINN['max_epochs']:,}"],
        ["PINN", "stopping rule",
         rf"relative change of $\beta$ and $\gamma$ over {C.PINN['check_every']} epochs $<{C.PINN['tol']:g}$"],
        ["PINN", r"physics-loss weight $\wode$",
         f"{C.PINN['w_ode']:g} (Exp.\\ 3 varies {', '.join(f'{w:g}' for w in C.W_ODE_SWEEP)})"],
        ["PINN (real data)", "epochs (fixed)", f"{C.REAL_PINN_EPOCHS:,}"],
        ["Bootstrap", "replicates", f"{C.BOOTSTRAP_B}"],
    ]
    return tabular("settings", "Numerical settings used in all experiments.", r"p{3.6cm}p{3.4cm}p{7.6cm}",
                   [[b("Component"), b("Setting"), b("Value")]], rows)


def table2():
    rows = [[f"{r.h:g}", f"{int(r.steps)}", sci(r.global_error, 3),
             "--" if pd.isna(r.observed_order) else f"{r.observed_order:.3f}", sci(r.max_mass_drift, 1)]
            for r in conv.itertuples()]
    return tabular("rk4", r"Global error of the RK4 solver against a reference solution ($h=0.001$).",
                   "rrccc", [[b("Step size $h$"), b("Steps"), b("Global error at $t=100$"),
                              b("Observed order $p$"), r"\textbf{$\max|S+I+R-1|$}"]], rows)


def table3():
    rows = []
    for r in summ.itertuples():
        method = esc(r.method.replace(" (RK4 + least squares)", ""))
        rows.append([method, esc(f"{r.noise:.0%}"), f"{r.beta:.4f} ({pct(r.beta_err_pct)})",
                     f"{r.gamma:.4f} ({pct(r.gamma_err_pct)})", sci(r.traj_rmse, 2),
                     f"{r.iterations:,.0f}", f"{r.seconds:.2f}"])
    note = (f"Classical values are means over {len(C.NOISE_SEEDS)} noise draws (one draw at 0\\% noise); "
            "iterations are function evaluations (each a full RK4 solve). PINN values are from one "
            "training run (seed 0); iterations are epochs until the stopping rule.")
    return tabular("synthetic", r"Recovery of $\beta=0.3$ and $\gamma=0.1$ from synthetic data.",
                   "llccccrr",
                   [[b("Method"), b("Noise"), r"\textbf{$\hat\beta$ (rel.\ error)}",
                     r"\textbf{$\hat\gamma$ (rel.\ error)}", b(r"Traj.\ RMSE"), b("Iter."), b("Time (s)")]],
                   rows, note=note, tabcolsep="4pt")


def table4():
    rows = [[f"{r.w_ode:g}", f"{r.beta:.4f}", pct(r.beta_err_pct), f"{r.gamma:.4f}", pct(r.gamma_err_pct),
             f"{r.epochs:,.0f}", f"{r.seconds:.1f}"] for r in sweep.itertuples()]
    return tabular("weights", rf"PINN estimates at {C.SWEEP_NOISE:.0%} noise for different physics-loss weights.".replace("%", r"\%"),
                   "rcccccr", [[r"$\wode$", r"$\hat\beta$", r"$\beta$ error", r"$\hat\gamma$",
                                r"$\gamma$ error", "Epochs", "Time (s)"]], rows)


def table5():
    rows = [[f"{int(r.season)}", f"{int(r.total_cases):,}", f"{int(r.peak_cases):,}", f"{r.beta:.3f}",
             f"{r.gamma:.3f}", f"{r.R0:.2f}", f"{r.infectious_days:.1f}", f"{r.N_eff:,.0f}", f"{r.R2:.3f}"]
            for r in seasons.itertuples()]
    note = (rf"$1/\hat\gamma$ was constrained to {lo_days}--{hi_days} days; "
            + (f"the {', '.join(str(int(s)) for s in at_bound.season)} estimate lies on the boundary."
               if len(at_bound) else "no estimate lies on the boundary."))
    head = [[b("Season"), b("Cases"), b("Peak"), r"$\hat\beta$", r"$\hat\gamma$", r"$\Rzero$",
             r"$1/\hat\gamma$ (d)", r"$\Neff$", "$R^2$"],
            ["", b("(May--Dec)"), b("month"), "", "", "", "", "", ""]]
    return tabular("seasons", "Classical SIR fits to Bulacan outbreak seasons (May--December).",
                   "crrcccccc", head, rows, note=note, tabcolsep="4pt")


def table6():
    def num(v):
        return f"{v:,.3f}" if v < 100 else f"{v:,.0f}"
    names = [r"$\beta$", r"$\gamma$", r"$\Rzero$", r"$\Neff$"]
    rows = [[n, num(r.estimate), num(r.ci95_low), num(r.ci95_high)] for n, r in zip(names, boot.itertuples())]
    return tabular("boot", rf"Residual-bootstrap 95\% intervals for the 2010 season ({C.BOOTSTRAP_B} replicates).",
                   "lrrr", [[b("Parameter"), b("Estimate"), b(r"95\% CI (lower)"), b(r"95\% CI (upper)")]], rows)


def table7():
    rows = [[esc(r.method), f"{r.beta:.3f}", f"{r.gamma:.3f}", f"{r.R0:.2f}", f"{r.R2_own_curve:.3f}",
             f"{r.R2_rk4_resimulated:.3f}", f"{r.seconds:.1f}"] for r in pinn10.itertuples()]
    note = (r"For the PINN, $\Neff$ and $I_0$ were fixed at the classical estimates so that only $\beta$ and "
            r"$\gamma$ were learned. ``Own curve'' is the network output; ``RK4 re-simulation'' solves the "
            r"SIR model with the PINN's $\beta$ and $\gamma$.")
    return tabular("pinn2010", "Classical fit and PINN on the 2010 season.", "lcccccc",
                   [[b("Method"), r"$\hat\beta$", r"$\hat\gamma$", r"$\Rzero$", r"\textbf{$R^2$ (own curve)}",
                     r"\textbf{$R^2$ (RK4 re-sim.)}", b("Time (s)")]], rows, note=note, tabcolsep="4pt")


# ------------------------------------------------------------------ data-dependent sentences
if not w_max.converged_by_tol:
    W_SENT = (r"With the largest weight ($\wode=" + f"{w_max.w_ode:g}" + r"$) the estimate of $\beta$ stalled on a "
              "plateau far from the true value for most of training before moving, and the run reached the "
              f"{C.PINN['max_epochs']:,}-epoch cap without satisfying the stopping rule. ")
else:
    W_SENT = ""

if len(at_bound):
    names = ", ".join(str(int(s)) for s in at_bound.season)
    BOUNDARY = (f"The {names} season{'s' if len(at_bound) > 1 else ''} ended at the boundary of the constraint, "
                "which signals that the data are not well described by SIR there: in 2006 reported cases fell "
                "abruptly from 504 in September to 110 in October and 17 in November, a drop faster than the "
                "SIR curve can follow with a plausible infectious period, possibly driven by the end of the "
                "rainy season or by changes in reporting. ")
else:
    BOUNDARY = ""

figdir = Path(os.path.relpath(FIG, OUT.parent)).as_posix().rstrip("/") + "/"
boot_r0 = boot.loc["R0"]

TOKENS = {
    "AUTHOR_BLOCK": ("  " + "\\\\\n  ".join(esc(a) for a in AFFILIATION[:0] + AUTHORS) + "\\\\[6pt]\n"
                     + "".join(f"  \\textit{{{esc(a)}}}\\\\\n" for a in AFFILIATION[:-1])
                     + f"  \\textit{{{esc(AFFILIATION[-1])}}}\\\\[6pt]\n"
                     + f"  {esc(COURSE)}\\\\\n  Code and data: \\url{{{REPO_URL}}}"),
    "GRAPHICSPATH": "{" + figdir + "}{figures/}",
    "REPO": REPO_URL,
    "ord_mean": f(orders.mean()), "omin": f(orders.min(), 3), "omax": f(orders.max(), 3),
    "p10_b": pct(pinn.loc[0.10, "beta_err_pct"]), "p10_g": pct(pinn.loc[0.10, "gamma_err_pct"]),
    "c10_b": pct(cls.loc[0.10, "beta_err_pct"]), "c10_g": pct(cls.loc[0.10, "gamma_err_pct"]),
    "nseeds": len(C.NOISE_SEEDS), "speedup": f"{speedup:.0f}",
    "nseasons": len(seasons), "smin": int(seasons.season.min()), "smax": int(seasons.season.max()),
    "r2min3": f(seasons.R2.min(), 3), "r2max3": f(seasons.R2.max(), 3), "r2min2": f(seasons.R2.min()),
    "r0min": f(seasons.R0.min()), "r0max": f(seasons.R0.max()),
    "resim2": f(p_pinn.R2_rk4_resimulated), "resim3": f(p_pinn.R2_rk4_resimulated, 3),
    "own4": f(p_pinn.R2_own_curve, 4),
    "lo": lo_days, "hi": hi_days,
    "tb": C.TRUE_BETA, "tg": C.TRUE_GAMMA, "tr0": f"{C.TRUE_BETA / C.TRUE_GAMMA:.0f}",
    "tper": f"{1 / C.TRUE_GAMMA:.0f}", "ti0": C.I0, "tdays": C.T_DAYS, "hdata": C.H_DATA,
    "season_list": ", ".join(str(int(s)) for s in seasons.season),
    "depth": C.PINN["depth"], "width": C.PINN["width"], "ncol": C.PINN["n_col"],
    "hfit0": f"{fitstep.h_fit.iloc[0]:g}", "fb0": pct(fitstep.beta_rel_err_pct.iloc[0]),
    "fg0": pct(fitstep.gamma_rel_err_pct.iloc[0]), "fb2": pct(fitstep.beta_rel_err_pct.iloc[2]),
    "cls_maxit": int(cls_all.iterations.max()), "cls_sec": f"{cls.seconds.mean():.2f}",
    "pinn_sec": f"{pinn.seconds.mean():.0f}",
    "bw": f"{best_w.w_ode:g}", "bw_b": pct(best_w.beta_err_pct), "bw_g": pct(best_w.gamma_err_pct),
    "ww": f"{worst_w.w_ode:g}", "ww_b": pct(worst_w.beta_err_pct), "ww_g": pct(worst_w.gamma_err_pct),
    "minw": f"{w_min.w_ode:g}", "W_SENT": W_SENT,
    "c2010": f"{int(s2010.total_cases):,}", "b2010": f(s2010.beta, 3), "g2010": f(s2010.gamma, 3),
    "r2010": f(s2010.R0),
    "bootlo": f(boot_r0.ci95_low), "boothi": f(boot_r0.ci95_high),
    "imin": f"{interior.infectious_days.min():.0f}", "imax": f"{interior.infectious_days.max():.0f}",
    "BOUNDARY": BOUNDARY,
    "T1": table1(), "T2": table2(), "T3": table3(), "T4": table4(), "T5": table5(), "T6": table6(), "T7": table7(),
}

TEMPLATE = r'''\documentclass[12pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}              % Times-like text and math
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,array,tabularx}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{caption}
\usepackage[authoryear,round]{natbib}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{url}

\graphicspath{<<GRAPHICSPATH>>}
\setstretch{1.15}
\emergencystretch=2em
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\captionsetup{font=small,labelfont=bf,labelsep=period}

% ---- code listings
\lstset{
  language=Python,
  basicstyle=\ttfamily\footnotesize,
  keywordstyle=\bfseries,
  commentstyle=\itshape\color{gray},
  frame=single,
  framerule=0.3pt,
  backgroundcolor=\color{black!3},
  breaklines=true,
  columns=fullflexible,
  xleftmargin=4pt,
  xrightmargin=4pt,
  aboveskip=6pt,
  belowskip=6pt
}

% ---- shorthand
\newcommand{\Rzero}{R_0}
\newcommand{\Neff}{N_{\mathrm{eff}}}
\newcommand{\wode}{w_{\mathrm{ode}}}

\title{\bfseries Physics-Informed Neural Networks versus Classical Least Squares for Estimating Dengue Transmission Parameters: A Case Study of Bulacan, Philippines (1994--2010)}
\author{%
<<AUTHOR_BLOCK>>}
\date{}

\begin{document}
\maketitle

\begin{abstract}\small
\noindent Estimating how fast a disease spreads is central to outbreak response, yet the transmission and recovery rates of epidemic models cannot be measured directly and must be inferred from case data. This study compares two approaches to this inverse problem for the susceptible--infected--recovered (SIR) model: a classical solve-then-fit method that couples a fourth-order Runge--Kutta (RK4) solver with nonlinear least squares, and a physics-informed neural network (PINN) that learns the solution and the parameters simultaneously by penalizing violations of the differential equations. A step-size study confirms that the RK4 solver converges with observed order <<ord_mean>>, matching the theoretical order of four. On synthetic outbreaks with 10\% observation noise, the PINN recovered the transmission rate $\beta$ and recovery rate $\gamma$ with relative errors of <<p10_b>> and <<p10_g>>, comparable to the classical method (<<c10_b>> and <<c10_g>>, averaged over <<nseeds>> noise draws), but required about <<speedup>> times more computation time and was sensitive to the weight of its physics loss. Applied to monthly dengue counts from Bulacan (OpenDengue, <<nseasons>> outbreak seasons between <<smin>> and <<smax>>), the classical SIR fit reproduced seasonal curves with $R^2$ between <<r2min3>> and <<r2max3>> and effective reproduction numbers between <<r0min>> and <<r0max>>. With only eight monthly observations per season, however, the PINN's learned parameters did not reproduce its own fitted curve ($R^2$ of the re-simulated curve: <<resim2>>). The results suggest that for low-dimensional ODE models with sparse data, classical fitting remains the more reliable and economical tool, while PINNs need richer data or additional constraints to be reliable.

\medskip\noindent\textbf{Keywords:} \textit{dengue, SIR model, physics-informed neural networks, Runge--Kutta, parameter estimation, inverse problems, Bulacan}
\end{abstract}

% =====================================================================
\section{Introduction}

Dengue is a mosquito-borne viral infection and one of the most widespread vector-borne diseases in the world, with an estimated 390 million infections per year \citep{bhatt2013}. In the Philippines, dengue is endemic and peaks during the rainy season, placing recurring pressure on local health systems. Bulacan, a rapidly urbanizing province in Central Luzon (Region III), has experienced repeated outbreaks; reported monthly cases in the province rose above one thousand during the 2010 season \citep[OpenDengue;][]{clarke2024}.

Mathematical models help turn case counts into quantities that guide decisions. The SIR model of \citet{kermack1927} summarizes an outbreak with two rates: the transmission rate $\beta$ and the recovery rate $\gamma$, whose ratio $\Rzero=\beta/\gamma$ (the basic reproduction number) determines whether an outbreak grows. These rates are not observed directly; they must be estimated from data, which makes parameter estimation an inverse problem. The standard approach repeatedly solves the differential equations numerically and adjusts the parameters to minimize the misfit with data.

Physics-informed neural networks \citep[PINNs;][]{raissi2019} offer an artificial-intelligence alternative: a neural network approximates the solution of the differential equations, and the equations themselves enter the training loss, so the network fits the data while obeying the physics. Unknown parameters such as $\beta$ and $\gamma$ are learned alongside the network weights. PINNs have attracted wide interest in scientific machine learning \citep{karniadakis2021}, but they are also known to be difficult to train \citep{wang2021}, and their advantage over classical methods for small epidemic models is not obvious.

This study therefore asks: for estimating SIR parameters from dengue data, how does a PINN compare with a classical RK4 plus least-squares method in accuracy, stability and computational cost? We (1) verify the numerical solver through a step-size convergence study; (2) compare both methods on synthetic outbreaks with known parameters and controlled noise; (3) examine the PINN's sensitivity to its physics-loss weight; and (4) apply both methods to real monthly dengue counts from Bulacan. The significance of the study is practical: it tells local modelers which tool is appropriate for the sparse, aggregated surveillance data typically available at the provincial level, and it identifies what a PINN-based approach would need in order to be useful.

% =====================================================================
\section{Preliminaries}

\subsection{The SIR Epidemic Model}

Let $S(t)$, $I(t)$ and $R(t)$ denote the fractions of a closed population that are susceptible, infected (and infectious) and recovered at time $t$ (in days). The SIR model is the system of ordinary differential equations (ODEs)
\begin{equation}
  \frac{dS}{dt} = -\beta S I, \qquad
  \frac{dI}{dt} = \beta S I - \gamma I, \qquad
  \frac{dR}{dt} = \gamma I
  \label{eq:sir}
\end{equation}
with initial condition $(S,I,R)(0) = (1-I_0,\, I_0,\, 0)$. The term $\beta S I$ models new infections under the mass-action assumption, and $\gamma I$ models recoveries, so $1/\gamma$ is the mean infectious period. Adding the three equations gives $d(S+I+R)/dt = 0$, so $S+I+R=1$ for all $t$. Linearizing $dI/dt$ near $S\approx 1$ gives $dI/dt \approx (\beta-\gamma)I$, so infections grow initially if and only if $\Rzero=\beta/\gamma>1$.

\subsection{The Fourth-Order Runge--Kutta Method}

System~\eqref{eq:sir} has no closed-form solution, so it is solved numerically. Writing $\mathbf{y}=(S,I,R)$ and $d\mathbf{y}/dt = \mathbf{f}(\mathbf{y})$, the classical fourth-order Runge--Kutta (RK4) method advances the solution with a fixed step size $h$ by
\begin{equation}
\begin{aligned}
  \mathbf{k}_1 &= \mathbf{f}(\mathbf{y}_n), &
  \mathbf{k}_2 &= \mathbf{f}\!\left(\mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_1\right),\\
  \mathbf{k}_3 &= \mathbf{f}\!\left(\mathbf{y}_n + \tfrac{h}{2}\mathbf{k}_2\right), &
  \mathbf{k}_4 &= \mathbf{f}\!\left(\mathbf{y}_n + h\,\mathbf{k}_3\right),
\end{aligned}
\label{eq:rk4k}
\end{equation}
\begin{equation}
  \mathbf{y}_{n+1} = \mathbf{y}_n + \frac{h}{6}\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right).
  \label{eq:rk4y}
\end{equation}
RK4 has local truncation error $O(h^5)$ and global error $O(h^4)$ for smooth $\mathbf{f}$ \citep{burden2011}. If $e(h)$ is the global error with step $h$, the observed order of accuracy between two step sizes is
\begin{equation}
  p = \frac{\log\bigl(e(h_1)/e(h_2)\bigr)}{\log(h_1/h_2)},
  \label{eq:order}
\end{equation}
which should approach 4 as $h$ decreases.

\subsection{Nonlinear Least Squares}

Given observations $y_i$ at times $t_i$ and a model prediction $m(t_i;\boldsymbol\theta)$ that depends on parameters $\boldsymbol\theta$, nonlinear least squares chooses $\boldsymbol\theta$ to minimize
\begin{equation}
  \Phi(\boldsymbol\theta) = \tfrac{1}{2}\sum_i r_i(\boldsymbol\theta)^2, \qquad
  r_i(\boldsymbol\theta) = m(t_i;\boldsymbol\theta) - y_i.
  \label{eq:nls}
\end{equation}
We use the trust-region reflective algorithm in SciPy's \texttt{least\_squares} \citep{virtanen2020}, a Gauss--Newton-type method that handles bound constraints (here $\beta,\gamma>0$). Each evaluation of the residual requires a full RK4 solution of \eqref{eq:sir}. Iteration stops when the relative changes in $\Phi$, in $\boldsymbol\theta$, or the gradient norm fall below a tolerance.

\subsection{Neural Networks and Automatic Differentiation}

A fully connected neural network with $L$ hidden layers maps an input $x$ to an output through compositions $z_\ell = \sigma(W_\ell z_{\ell-1} + b_\ell)$, where $W_\ell$ and $b_\ell$ are trainable weights and biases and $\sigma$ is a nonlinear activation (here $\tanh$). Such networks are universal approximators of continuous functions. Training minimizes a loss function with gradient-based optimizers; we use Adam \citep{kingma2015}, whose learning rate plays the role of a step size. Automatic differentiation, as implemented in PyTorch \citep{paszke2019}, computes exact derivatives of the network output with respect to its input and parameters by applying the chain rule to the recorded computation, without finite-difference approximation.

\subsection{Physics-Informed Neural Networks}

A PINN \citep{raissi2019} represents the unknown solution by a network $N(t;w)$ and trains it on a composite loss. For an inverse problem, unknown model parameters are added to the trainable variables. For system~\eqref{eq:sir} with observations of $I$, the loss is
\begin{equation}
  \mathcal{L}(w,\beta,\gamma) = \mathcal{L}_{\mathrm{data}} + \wode\,\mathcal{L}_{\mathrm{ode}} + \mathcal{L}_{\mathrm{ic}},
  \label{eq:pinnloss}
\end{equation}
where $\mathcal{L}_{\mathrm{data}}$ is the mean squared error between the network's $I(t)$ and the data, $\mathcal{L}_{\mathrm{ode}}$ is the mean squared residual of the three equations in \eqref{eq:sir} evaluated by automatic differentiation at a set of collocation points, and $\mathcal{L}_{\mathrm{ic}}$ penalizes deviation from the initial condition. The weight $\wode$ balances fitting the data against satisfying the physics.

\subsection{Bootstrap Error Estimation}

For real data the true parameters are unknown, so uncertainty is estimated by a residual bootstrap: residuals of the fitted model are resampled with replacement and added back to the fitted values to create synthetic datasets, which are refitted. The 2.5th and 97.5th percentiles of the refitted parameters form an approximate 95\% confidence interval.

% =====================================================================
\section{Methods and Applications}

\subsection{Model and Method Description}

\subsubsection*{Problem setup}

Variables: $S$, $I$, $R$ (population fractions) as functions of time $t$ in days. Parameters to estimate: $\beta>0$ and $\gamma>0$ (per day). Derived quantity: $\Rzero=\beta/\gamma$. Assumptions: closed population over one season, homogeneous mixing, constant rates within a season, and no explicit mosquito compartment. Constraints: $\beta,\gamma$ positive (enforced by bounds in least squares and by a logarithmic parameterization, $\beta=\exp(\log\beta)$, in the PINN). For the real data, the effective infectious period $1/\gamma$ is further restricted to <<lo>>--<<hi>> days, i.e.\ $\gamma\in[1/<<hi>>,\,1/<<lo>>]$ per day. Without this constraint, the four-parameter fit to eight monthly points can collapse to a degenerate solution ($\gamma$ at an arbitrary bound, $\Rzero\approx1$ and a very large $\Neff$) that fits the data equally well. The lower limit reflects the roughly one-week human infectious period; the upper limit allows for the mosquito stage that a plain SIR model absorbs into $\gamma$.

\subsubsection*{Synthetic data}

A reference outbreak is generated with $\beta=<<tb>>$, $\gamma=<<tg>>$ ($\Rzero=<<tr0>>$, infectious period <<tper>> days), $I_0=<<ti0>>$, over <<tdays>> days, solved by RK4 with $h=<<hdata>>$. Daily values of $I$ are observed with multiplicative Gaussian noise, $I_{\mathrm{obs}} = I_{\mathrm{true}}(1+\sigma\varepsilon)$ with $\varepsilon\sim N(0,1)$ and $\sigma\in\{0,\,0.05,\,0.10\}$. Because the true parameters are known, estimation error can be measured exactly.

\subsubsection*{Real data}

Monthly reported dengue cases for the Province of Bulacan were extracted from the OpenDengue V1.3 spatial extract \citep{clarke2024}, which provides 200 monthly province-level counts from 1994 to 2010. We selected the <<nseasons>> seasons whose monthly peak exceeded 300 cases (<<season_list>>) and fitted each from May to December (eight months), the period containing the rainy-season wave. Surveillance counts new reported cases rather than the infected fraction, so the observation model is
\begin{equation}
  \text{cases in month } k = \Neff\,\bigl[S(t_{k-1}) - S(t_k)\bigr],
  \label{eq:obs}
\end{equation}
since every decrease in $S$ is a new infection. The effective population $\Neff$ absorbs the reporting rate and the fraction of the population actually susceptible; it is estimated together with $\beta$, $\gamma$ and $I_0$. This makes $\Neff$ much smaller than the province's census population, which is expected when most infections are unreported or occur in a partly immune population.

\subsection{Implementation and Code}

All code is written in Python using NumPy, SciPy, pandas, Matplotlib and PyTorch, and is available in the project repository (\url{<<REPO>>}). All settings (seeds, step sizes, tolerances, epochs) are centralized in \texttt{src/config.py}, and \texttt{run\_all.py} regenerates every table and figure in this paper.

\subsubsection*{Algorithm 1: RK4 solver}

The RK4 step \eqref{eq:rk4k}--\eqref{eq:rk4y} is implemented directly (\texttt{src/sir.py}):
\begin{lstlisting}
def rk4(beta, gamma, y0, t_end, h):
    n = int(round(t_end / h))
    y = np.asarray(y0, dtype=float)
    for k in range(n):
        k1 = sir_rhs(y, beta, gamma)
        k2 = sir_rhs(y + 0.5 * h * k1, beta, gamma)
        k3 = sir_rhs(y + 0.5 * h * k2, beta, gamma)
        k4 = sir_rhs(y + h * k3, beta, gamma)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
\end{lstlisting}

\subsubsection*{Algorithm 2: Classical solve-then-fit}

For a candidate $\boldsymbol\theta=(\beta,\gamma)$, the residual vector is computed by solving \eqref{eq:sir} with RK4 and comparing the model's $I(t_i)$ with the data; \texttt{least\_squares} then updates $\boldsymbol\theta$ (\texttt{src/classical.py}). For the real data, $\boldsymbol\theta=(\beta,\gamma,\log_{10}\Neff,\log_{10}I_0)$ and the residuals follow \eqref{eq:obs}. Because this four-parameter problem is non-convex, the fit is started from six initial guesses and the best solution is kept.
\begin{lstlisting}
def residuals(p):
    t, Y = rk4(p[0], p[1], y0, t_end, h)
    return (Y[::stride, 1] - I_obs) / I_obs.max()

res = least_squares(residuals, p0, bounds=([1e-4, 1e-4], [5, 5]),
                    ftol=1e-10, xtol=1e-10, gtol=1e-10)
\end{lstlisting}

\subsubsection*{Algorithm 3: PINN}

The network maps rescaled time $\tau=t/T\in[0,1]$ to $(S,\,I/s_I,\,R)$ through <<depth>> hidden layers of <<width>> $\tanh$ units and a sigmoid output, where $s_I$ is the maximum observed $I$, so all outputs are of order one. The ODE residuals are evaluated at <<ncol>> equally spaced collocation points using \texttt{torch.autograd.grad}, with the chain rule $d/d\tau = T\,d/dt$. $\beta$ and $\gamma$ are stored as log-values and optimized by Adam together with the weights (\texttt{src/pinn.py}):
\begin{lstlisting}
out = net(tau_c);  S, I_s, R = out[:, :1], out[:, 1:2], out[:, 2:3]
I = I_s * s_I
r_S = d_dtau(S) - T * (-beta * S * I)
r_I = d_dtau(I_s) - T * (beta * S * I - gamma * I) / s_I
r_R = d_dtau(R) - T * (gamma * I)
loss = l_data + w_ode * (r_S**2 + r_I**2 + r_R**2).mean() + l_ic
loss.backward();  opt.step()
\end{lstlisting}

\subsubsection*{Step size, iteration count and tolerance}

Table~\ref{tab:settings} lists the numerical settings used in all experiments. Errors are reported as relative parameter errors $|\hat\theta-\theta|/\theta\times100\%$, and as the root-mean-square error (trajectory RMSE) between the true $I(t)$ and an RK4 re-simulation with the estimated parameters. Runtimes were measured on a laptop CPU (Intel Core i5-13420H) with PyTorch restricted to one thread; environment details are recorded in \texttt{results/run\_log.txt}.

<<T1>>

\subsection{Results and Visualization}

\subsubsection*{Experiment 1: RK4 convergence}

<<T2>>

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig1_rk4_convergence.png}
\caption{(a) Log--log plot of RK4 global error versus step size with a slope-4 reference line. (b) Relative error of the classical estimates on noise-free data as a function of the RK4 step size used inside the fit.}
\label{fig:rk4}
\end{figure}

Table~\ref{tab:rk4} and Figure~\ref{fig:rk4}a show that halving $h$ reduces the error by a factor of about 16; the observed order ranges from <<omin>> to <<omax>>, in agreement with the theoretical order of four. Population is conserved to machine precision at every step size. Figure~\ref{fig:rk4}b shows that the solver step used inside the fit controls a systematic bias: with $h=<<hfit0>>$ days the estimates of $\beta$ and $\gamma$ are off by <<fb0>> and <<fg0>>, while with $h=1$ day the error drops to <<fb2>>. With $h=<<hdata>>$, the same step used to generate the data, the parameters are recovered to machine precision; this is the so-called inverse crime, which is why noise is added in the remaining experiments.

\subsubsection*{Experiment 2: Parameter recovery on synthetic data}

<<T3>>

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig2_fits.png}
\caption{Observed infected fraction (points) with the classical fit (solid) and PINN prediction (dashed) at 0\%, 5\% and 10\% noise.}
\label{fig:fits}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig3_pinn_training.png}
\caption{PINN training at 5\% noise. (a) Data, ODE-residual and total loss per epoch. (b) Estimates of $\beta$ and $\gamma$ during training; dotted lines mark the true values.}
\label{fig:training}
\end{figure}

Both methods recover the parameters accurately (Table~\ref{tab:synthetic}, Figure~\ref{fig:fits}). At 10\% noise the PINN's errors are <<p10_b>> for $\beta$ and <<p10_g>> for $\gamma$, versus <<c10_b>> and <<c10_g>> on average for the classical method. The classical method converges in at most <<cls_maxit>> function evaluations (about <<cls_sec>> s), whereas the PINN needs thousands of epochs (on average <<pinn_sec>> s), roughly <<speedup>> times longer. Figure~\ref{fig:training} shows the PINN's convergence behavior: the data loss falls quickly, the ODE residual is reduced more slowly, and $\beta$ and $\gamma$ settle onto the true values before the stopping rule ends training.

\subsubsection*{Experiment 3: Sensitivity to the physics-loss weight}

<<T4>>

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig4_ode_weight_sweep.png}
\caption{(a) Final relative error of $\beta$ and $\gamma$ (log scale) for each physics-loss weight. (b) The estimate of $\beta$ during training for each weight.}
\label{fig:weights}
\end{figure}

The physics-loss weight changes both the accuracy and the convergence of the PINN (Table~\ref{tab:weights}, Figure~\ref{fig:weights}). The most accurate weight was $\wode=<<bw>>$ (errors <<bw_b>> and <<bw_g>>), while $\wode=<<ww>>$ gave errors of <<ww_b>> and <<ww_g>>. <<W_SENT>>With the smallest weight ($\wode=<<minw>>$) training is fast, but the physics constraint is weak and $\beta$ is less accurate. The classical method has no corresponding tuning parameter.

\subsubsection*{Experiment 4: Dengue seasons in Bulacan}

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig5_bulacan_overview.png}
\caption{Monthly reported dengue cases in Bulacan, 1994--2010 (OpenDengue V1.3).}
\label{fig:monthly}
\end{figure}

<<T5>>

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{fig6_bulacan_seasons.png}
\caption{Reported monthly cases (bars) and classical SIR fits (lines) for six outbreak seasons.}
\label{fig:seasonfits}
\end{figure}

<<T6>>

<<T7>>

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{fig7_bulacan_pinn.png}
\caption{Bulacan 2010: reported cases, classical fit, the PINN network's output, and an RK4 re-simulation using the PINN's estimated $\beta$ and $\gamma$.}
\label{fig:pinn2010}
\end{figure}

The classical SIR model reproduces the shape of each season (Table~\ref{tab:seasons}, Figure~\ref{fig:seasonfits}), with $R^2$ from <<r2min3>> to <<r2max3>> and $\Rzero$ estimates from <<r0min>> to <<r0max>>. For 2010, the largest season (<<c2010>> cases from May to December), the fit gives $\beta=<<b2010>>$ per day, $\gamma=<<g2010>>$ per day and $\Rzero=<<r2010>>$. The bootstrap intervals in Table~\ref{tab:boot} quantify how well eight monthly points constrain the parameters: $\Rzero$ lies between <<bootlo>> and <<boothi>>. On the same season the PINN's network output matches the data ($R^2=<<own4>>$), but solving the SIR model with the PINN's $\beta$ and $\gamma$ gives $R^2=<<resim3>>$ (Table~\ref{tab:pinn2010}, Figure~\ref{fig:pinn2010}), meaning the learned parameters are not consistent with the network's own curve.

\subsection{Analysis and Discussion}

\textbf{Accuracy.} On synthetic data both methods achieve parameter errors of about one percent or less. The classical errors, averaged over several noise draws, grow with the noise level as expected; the PINN errors come from a single training run per noise level and fluctuate rather than growing smoothly, so a lower PINN error in one run should not be read as a systematic advantage. Since both methods fit the same model to the same data, neither can do much better than the information in the noisy observations allows. The classical method's accuracy is limited by the RK4 step used inside the fit (Experiment 1b), which is negligible for $h\le1$ day because of the fourth-order convergence.

\textbf{Stability.} RK4 is stable for the step sizes used, and least squares converged in every synthetic run from the same initial guess used by the PINN. The PINN is less robust: its result depends on the physics-loss weight (Experiment 3). When the physics term dominates, the network first learns a smooth curve that satisfies the ODE for the current, incorrect parameters, and $\beta$ and $\gamma$ then move only slowly; when it is too weak, the data alone determine the curve and the parameters are less constrained. This agrees with known training pathologies of PINNs, where imbalanced loss gradients slow or misdirect learning \citep{wang2021}.

\textbf{Computational cost.} One RK4 solve costs $O(T/h)$ evaluations of $\mathbf{f}$, and the classical fit needs only tens of solves. The PINN's cost per epoch is a forward and backward pass over the collocation and data points, repeated for thousands of epochs. In our experiments the PINN was roughly <<speedup>> times slower. For a two-parameter ODE this cost is not justified; PINNs become attractive for partial differential equations, high-dimensional problems, or when no efficient solver exists.

\textbf{Real data.} The classical SIR model captures the timing and size of Bulacan's outbreaks, but the implied infectious periods ($1/\gamma\approx<<imin>>$--<<imax>> days for the seasons whose estimate lies inside the constraint) are longer than the roughly one-week infectious period of a human dengue case. This is a symptom of model simplification: SIR has no mosquito compartment, so the extrinsic incubation period in the mosquito and the mosquito lifespan are absorbed into $\gamma$. $\beta$ and $\gamma$ should therefore be read as effective seasonal rates rather than biological constants. <<BOUNDARY>>Monthly data also provide only eight observations for four unknowns, and the bootstrap intervals show the resulting uncertainty. The PINN struggled most here: with eight points, the network can interpolate the data without the ODE residual being small enough to pin down $\beta$ and $\gamma$.

\textbf{Practical significance and limitations.} For provincial health offices working with sparse monthly surveillance data, a classical SIR fit is fast, transparent and adequate for describing an outbreak's growth ($\Rzero$) and timing. Our conclusions are limited by: the SIR simplification (no vector, no seasonality in $\beta$, no under-reporting model beyond $\Neff$); data that end in 2010 and are aggregated monthly; one PINN architecture and one training seed per configuration; and treating each season independently.

% =====================================================================
\section{Conclusion}

We compared a physics-informed neural network with a classical RK4 plus least-squares method for estimating SIR transmission parameters. The RK4 solver was verified to converge with fourth-order accuracy. On synthetic outbreaks with up to 10\% noise, both methods recovered $\beta$ and $\gamma$ within a few percent, but the PINN was about <<speedup>> times more expensive and sensitive to its physics-loss weight. On real dengue data from Bulacan, the classical model fitted <<nseasons>> outbreak seasons with $R^2\ge<<r2min2>>$ and effective $\Rzero$ between <<r0min>> and <<r0max>>, while the PINN failed to produce parameters consistent with its own fit from eight monthly observations. For low-dimensional epidemic models with sparse data, classical fitting is the more effective method; PINNs are better suited to richer data or more complex models.

Future work suggested by these findings, and a natural basis for a thesis, includes:
\begin{itemize}[nosep]
  \item replacing SIR with a host--vector dengue model \citep{esteva1998} so that the estimated rates become biologically interpretable;
  \item using weekly or more recent surveillance data (e.g., from the Department of Health) to give the PINN more observations;
  \item letting the transmission rate vary with rainfall and temperature, a setting where a neural network can represent an unknown function $\beta(t)$ that a classical fit cannot easily handle;
  \item fitting several seasons jointly with shared parameters; and
  \item improving PINN training with adaptive loss weighting or two-stage training that fixes the network before estimating $\beta$ and $\gamma$.
\end{itemize}

% =====================================================================
\begin{thebibliography}{99}
\setlength{\itemsep}{3pt}
\small

\bibitem[Bhatt et~al.(2013)]{bhatt2013}
Bhatt, S., Gething, P.~W., Brady, O.~J., Messina, J.~P., Farlow, A.~W., Moyes, C.~L., Drake, J.~M., Brownstein, J.~S., Hoen, A.~G., Sankoh, O., Myers, M.~F., George, D.~B., Jaenisch, T., Wint, G.~R.~W., Simmons, C.~P., Scott, T.~W., Farrar, J.~J., \& Hay, S.~I. (2013).
The global distribution and burden of dengue.
\textit{Nature}, 496(7446), 504--507.
\url{https://doi.org/10.1038/nature12060}

\bibitem[Burden and Faires(2011)]{burden2011}
Burden, R.~L., \& Faires, J.~D. (2011).
\textit{Numerical analysis} (9th ed.).
Brooks/Cole, Cengage Learning.

\bibitem[Clarke et~al.(2024)]{clarke2024}
Clarke, J., Lim, A., Gupte, P., Pigott, D.~M., van Panhuis, W.~G., \& Brady, O.~J. (2024).
A global dataset of publicly available dengue case count data.
\textit{Scientific Data}, 11, Article 296.
\url{https://doi.org/10.1038/s41597-024-03120-7}

\bibitem[Esteva and Vargas(1998)]{esteva1998}
Esteva, L., \& Vargas, C. (1998).
Analysis of a dengue disease transmission model.
\textit{Mathematical Biosciences}, 150(2), 131--151.
\url{https://doi.org/10.1016/S0025-5564(98)10003-2}

\bibitem[Karniadakis et~al.(2021)]{karniadakis2021}
Karniadakis, G.~E., Kevrekidis, I.~G., Lu, L., Perdikaris, P., Wang, S., \& Yang, L. (2021).
Physics-informed machine learning.
\textit{Nature Reviews Physics}, 3(6), 422--440.
\url{https://doi.org/10.1038/s42254-021-00314-5}

\bibitem[Kermack and McKendrick(1927)]{kermack1927}
Kermack, W.~O., \& McKendrick, A.~G. (1927).
A contribution to the mathematical theory of epidemics.
\textit{Proceedings of the Royal Society of London. Series A}, 115(772), 700--721.
\url{https://doi.org/10.1098/rspa.1927.0118}

\bibitem[Kingma and Ba(2015)]{kingma2015}
Kingma, D.~P., \& Ba, J. (2015).
Adam: A method for stochastic optimization.
In \textit{Proceedings of the 3rd International Conference on Learning Representations (ICLR)}.
\url{https://arxiv.org/abs/1412.6980}

\bibitem[Paszke et~al.(2019)]{paszke2019}
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., K{\"o}pf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., \ldots{} Chintala, S. (2019).
PyTorch: An imperative style, high-performance deep learning library.
In \textit{Advances in Neural Information Processing Systems 32} (pp.\ 8024--8035). Curran Associates.

\bibitem[Raissi et~al.(2019)]{raissi2019}
Raissi, M., Perdikaris, P., \& Karniadakis, G.~E. (2019).
Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.
\textit{Journal of Computational Physics}, 378, 686--707.
\url{https://doi.org/10.1016/j.jcp.2018.10.045}

\bibitem[Virtanen et~al.(2020)]{virtanen2020}
Virtanen, P., Gommers, R., Oliphant, T.~E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van~der Walt, S.~J., Brett, M., Wilson, J., Millman, K.~J., Mayorov, N., Nelson, A.~R.~J., Jones, E., Kern, R., Larson, E., \ldots{} SciPy 1.0 Contributors. (2020).
SciPy 1.0: Fundamental algorithms for scientific computing in Python.
\textit{Nature Methods}, 17(3), 261--272.
\url{https://doi.org/10.1038/s41592-019-0686-2}

\bibitem[Wang et~al.(2021)]{wang2021}
Wang, S., Teng, Y., \& Perdikaris, P. (2021).
Understanding and mitigating gradient flow pathologies in physics-informed neural networks.
\textit{SIAM Journal on Scientific Computing}, 43(5), A3055--A3081.
\url{https://doi.org/10.1137/20M1318043}

\end{thebibliography}

\end{document}
'''

def render(template):
    missing = set(re.findall(r"<<(\w+)>>", template)) - set(TOKENS)
    if missing:
        raise KeyError(f"template tokens without a value: {sorted(missing)}")
    return re.sub(r"<<(\w+)>>", lambda m: str(TOKENS[m.group(1)]), template)


OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(render(TEMPLATE), encoding="utf-8")
print(f"Saved {OUT}")
