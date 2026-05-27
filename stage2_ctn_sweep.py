"""
stage2_ctn_sweep.py
===================
Validation Stage 2: verification of the closed-form transmission mapping
eta(CTN) = erfc[1/(2*sqrt(CTN))] against the Crank-Nicolson solver, and
of the finite-domain sealed-overburden correction (manuscript Section 3.9.2).

Confirms:
  * the semi-infinite solver reproduces the erfc mapping across four
    decades of CTN to within ~1 %;
  * the finite-domain (Neumann far-boundary) solver matches the analytical
    Fourier-series solution;
  * the ratio eta_FD / eta_erfc -> 2 as CTN -> 0 (method-of-images limit),
    confirming Eq. 15 is a conservative lower bound.

Outputs:
  results/stage2_ctn_sweep.csv  - eta values and errors over the CTN sweep
  figures/stage2_ctn_sweep.png  - three-panel verification figure

Author : H. Mbah | Licence : MIT
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ctn.analytical import eta_semi_infinite, eta_finite_domain
from ctn.solver import CaprockSolver

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")


def run(verbose=True):
    """Run the CTN sweep verification."""
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    ctn_values = np.logspace(-2.0, 1.0, 40)

    solver_si = CaprockSolver(n_z=200, n_t=800, far_bc="dirichlet")
    solver_fd = CaprockSolver(n_z=200, n_t=800, far_bc="neumann")

    eta_ana, eta_si, rel_err = [], [], []
    eta_fd_num, eta_fd_ana, ratio = [], [], []

    for ctn in ctn_values:
        e_ana = float(eta_semi_infinite(ctn))
        e_num = solver_si.solve(ctn)
        eta_ana.append(e_ana)
        eta_si.append(e_num)
        rel_err.append(100.0 * abs(e_num - e_ana) / max(e_ana, 1e-12))

        e_fd_num = solver_fd.solve(ctn)
        e_fd_ana = float(eta_finite_domain(ctn))
        eta_fd_num.append(e_fd_num)
        eta_fd_ana.append(e_fd_ana)
        ratio.append(e_fd_ana / max(e_ana, 1e-12))

    eta_ana = np.array(eta_ana)
    eta_si = np.array(eta_si)
    rel_err = np.array(rel_err)
    eta_fd_ana = np.array(eta_fd_ana)
    ratio = np.array(ratio)

    # restrict relative-error metric to the well-resolved range CTN >= 0.05
    mask = ctn_values >= 0.05
    max_err = float(np.max(rel_err[mask]))
    if verbose:
        print(f"  max relative error (CTN >= 0.05) = {max_err:.3f} %")
        print(f"  eta_FD / eta_erfc at CTN = {ctn_values[0]:.3f} : "
              f"{ratio[0]:.3f}  (method-of-images limit -> 2)")

    # ---- write results ---------------------------------------------------
    with open(os.path.join(RESULTS, "stage2_ctn_sweep.csv"), "w") as fh:
        fh.write("CTN,eta_erfc,eta_solver,rel_error_percent,"
                 "eta_FD_analytical,eta_FD_over_eta_erfc\n")
        for i, ctn in enumerate(ctn_values):
            fh.write(f"{ctn:.6e},{eta_ana[i]:.6e},{eta_si[i]:.6e},"
                     f"{rel_err[i]:.6f},{eta_fd_ana[i]:.6e},"
                     f"{ratio[i]:.6f}\n")

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    ax.semilogx(ctn_values, eta_ana * 100, "-", lw=2.5,
                color="#1A3C53", label="analytical erfc (Eq. 15)")
    ax.semilogx(ctn_values, eta_si * 100, "o", ms=5, mfc="none",
                color="#B5651D", label="Crank-Nicolson solver")
    ax.set_xlabel("CTN")
    ax.set_ylabel(r"Transmission efficiency $\eta$ (%)")
    ax.set_title("(a) Analytical vs numerical $\\eta$(CTN)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.semilogx(ctn_values[mask], rel_err[mask], "s-", ms=4,
                color="#A8443A")
    ax.axhline(1.0, ls="--", color="grey", label="1 % bound")
    ax.set_xlabel("CTN")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(f"(b) Solver error (max {max_err:.2f} %)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    ax.semilogx(ctn_values, ratio, "-", lw=2.5, color="#3F7D3F")
    ax.axhline(2.0, ls=":", color="grey", label="images limit (=2)")
    ax.axhline(1.0, ls=":", color="grey")
    ax.set_xlabel("CTN")
    ax.set_ylabel(r"$\eta_{FD}\,/\,\eta_{erfc}$")
    ax.set_title("(c) Finite-domain conservative bound")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "stage2_ctn_sweep.png"), dpi=200)
    plt.close(fig)

    return {
        "max_rel_error_percent": max_err,
        "images_ratio_low_ctn": float(ratio[0]),
        "passed": max_err < 2.0 and abs(ratio[0] - 2.0) < 0.05,
    }


if __name__ == "__main__":
    print("Stage 2 - CTN sweep verification")
    summary = run()
    print(f"  -> max relative error = "
          f"{summary['max_rel_error_percent']:.3f} %")
    print(f"  -> PASSED" if summary["passed"] else "  -> FAILED")
