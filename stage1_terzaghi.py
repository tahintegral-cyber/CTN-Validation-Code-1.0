"""
stage1_terzaghi.py
==================
Validation Stage 1: Crank-Nicolson solver verification against the
analytical Terzaghi 1-D consolidation solution (manuscript Section 3.9.1).

Confirms:
  * pointwise agreement with the analytical solution at three time factors;
  * second-order spatial convergence (observed order ~ 2.0).

Outputs:
  results/stage1_terzaghi.csv      - error metrics at each T_v
  results/stage1_convergence.csv   - grid-convergence study
  figures/stage1_terzaghi.png      - profile comparison + convergence plot

Author : H. Mbah | Licence : MIT
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ctn.analytical import terzaghi_consolidation
from ctn.solver import solve_terzaghi

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")


def run(verbose=True):
    """Run the Terzaghi benchmark and grid-convergence study."""
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    # ---- (a) pointwise accuracy at three time factors --------------------
    time_factors = [0.05, 0.20, 0.50]
    rows = []
    profiles = {}
    for T_v in time_factors:
        zeta, U_num = solve_terzaghi(n_z=200, n_t=2000, T_v=T_v)
        U_ana = terzaghi_consolidation(zeta, T_v, n_terms=300)
        err = np.abs(U_num - U_ana)
        l2 = float(np.sqrt(np.mean(err ** 2)))
        linf = float(np.max(err))
        rows.append((T_v, l2, linf, 100.0 * linf))
        profiles[T_v] = (zeta, U_num, U_ana)
        if verbose:
            print(f"  T_v = {T_v:.2f} :  L2 = {l2:.3e}   "
                  f"Linf = {linf:.3e}  ({100.0*linf:.5f} %)")

    with open(os.path.join(RESULTS, "stage1_terzaghi.csv"), "w") as fh:
        fh.write("T_v,L2_error,Linf_error,Linf_percent\n")
        for T_v, l2, linf, pct in rows:
            fh.write(f"{T_v},{l2:.6e},{linf:.6e},{pct:.6f}\n")

    # ---- (b) grid-convergence study (observed spatial order) -------------
    grids = [25, 50, 100, 200, 400]
    T_ref = 0.20
    errs = []
    for nz in grids:
        zeta, U_num = solve_terzaghi(n_z=nz, n_t=4000, T_v=T_ref)
        U_ana = terzaghi_consolidation(zeta, T_ref, n_terms=300)
        errs.append(float(np.max(np.abs(U_num - U_ana))))

    orders = []
    for i in range(1, len(grids)):
        p = np.log(errs[i - 1] / errs[i]) / np.log(grids[i] / grids[i - 1])
        orders.append(p)

    with open(os.path.join(RESULTS, "stage1_convergence.csv"), "w") as fh:
        fh.write("n_z,Linf_error,observed_order\n")
        fh.write(f"{grids[0]},{errs[0]:.6e},\n")
        for i in range(1, len(grids)):
            fh.write(f"{grids[i]},{errs[i]:.6e},{orders[i-1]:.4f}\n")

    mean_order = float(np.mean(orders))
    if verbose:
        print(f"  grid-convergence observed order = {mean_order:.3f} "
              f"(theoretical 2.00)")

    # ---- (c) figure ------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for T_v in time_factors:
        zeta, U_num, U_ana = profiles[T_v]
        ax1.plot(U_ana, zeta, "-", lw=2, label=f"analytical, $T_v$={T_v}")
        ax1.plot(U_num[::10], zeta[::10], "o", ms=4, mfc="none")
    ax1.set_xlabel("Normalised excess pressure $U$")
    ax1.set_ylabel(r"Normalised depth $\zeta = z/H$")
    ax1.set_title("(a) Terzaghi benchmark: solver vs analytical")
    ax1.invert_yaxis()
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.loglog(grids, errs, "o-", lw=2, color="#1A3C53")
    ref = errs[0] * (np.array(grids, float) / grids[0]) ** (-2.0)
    ax2.loglog(grids, ref, "--", color="grey", label="slope $-2$ (ref.)")
    ax2.set_xlabel("Number of grid intervals $n_z$")
    ax2.set_ylabel(r"$L_\infty$ error")
    ax2.set_title(f"(b) Grid convergence (order = {mean_order:.2f})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "stage1_terzaghi.png"), dpi=200)
    plt.close(fig)

    max_pct = max(r[3] for r in rows)
    return {
        "max_Linf_percent": max_pct,
        "observed_order": mean_order,
        "passed": max_pct < 0.01 and abs(mean_order - 2.0) < 0.15,
    }


if __name__ == "__main__":
    print("Stage 1 - Terzaghi consolidation benchmark")
    summary = run()
    print(f"  -> max Linf error = {summary['max_Linf_percent']:.5f} %")
    print(f"  -> PASSED" if summary["passed"] else "  -> FAILED")
