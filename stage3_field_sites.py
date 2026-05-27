"""
stage3_field_sites.py
=====================
Validation Stage 3: retrospective consistency assessment across four
documented CO2 storage sites (manuscript Section 3, Table 3).

For each site the CTN is computed from prior-published parameters alone
(no calibration to observed monitoring outcomes). The resulting regime
classification is compared with the reported field response.

Regime thresholds (manuscript Table 2):
    LOW       : CTN <= 0.1
    MODERATE  : 0.1 < CTN <= 1.0
    HIGH      : CTN > 1.0

Outputs:
  results/stage3_field_sites.csv  - CTN, eta and regime for each site
  figures/stage3_field_sites.png  - sites on the eta-CTN master curve

Author : H. Mbah | Licence : MIT
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ctn.analytical import eta_semi_infinite, ctn_poro
from ctn.sites import SITES, t_inj_seconds

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")


def classify(ctn):
    """Return the operational regime label for a CTN value."""
    if ctn <= 0.1:
        return "LOW"
    if ctn <= 1.0:
        return "MODERATE"
    return "HIGH"


def run(verbose=True):
    """Run the field-site retrospective consistency assessment."""
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    rows = []
    for key, site in SITES.items():
        t_inj = t_inj_seconds(key)
        ctn = ctn_poro(site["D_h"], t_inj, site["H_c"])
        eta = float(eta_semi_infinite(ctn))
        regime = classify(ctn)
        # Stage 3 verifies the poroelastic screening classification.
        # Sites flagged 'coupled' require the thermo-poroelastic term to
        # reach their reported regime (documented in the site 'note').
        target = site["regime_poro"]
        consistent = (regime == target)
        rows.append({
            "key": key, "name": site["name"], "ctn": ctn,
            "eta": eta, "regime": regime,
            "reported": site["regime"], "reported_poro": target,
            "coupled": site["coupled"], "consistent": consistent,
        })
        if verbose:
            flag = "OK" if consistent else "MISMATCH"
            extra = "  (+thermal -> HIGH)" if site["coupled"] else ""
            print(f"  {site['name'][:34]:34s}  CTN_poro = {ctn:10.4e}  "
                  f"{regime:8s}  [{flag}]{extra}")

    # ---- write results ---------------------------------------------------
    with open(os.path.join(RESULTS, "stage3_field_sites.csv"), "w") as fh:
        fh.write("site,name,CTN_poro,eta,computed_regime,"
                 "reported_poro_regime,reported_total_regime,"
                 "thermal_coupled,consistent\n")
        for r in rows:
            fh.write(f"{r['key']},\"{r['name']}\",{r['ctn']:.6e},"
                     f"{r['eta']:.6e},{r['regime']},{r['reported_poro']},"
                     f"{r['reported']},{r['coupled']},{r['consistent']}\n")

    # ---- figure: sites on the eta-CTN master curve -----------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ctn_curve = np.logspace(-4, 1.4, 500)
    eta_curve = eta_semi_infinite(ctn_curve) * 100.0

    ax.axvspan(1e-4, 0.1, color="#E3F0E3", alpha=0.8)
    ax.axvspan(0.1, 1.0, color="#FBF2DA", alpha=0.8)
    ax.axvspan(1.0, 30, color="#F6E0DC", alpha=0.8)
    ax.semilogx(ctn_curve, eta_curve, "-", lw=2.5, color="#1A3C53")

    colour = {"LOW": "#3F7D3F", "MODERATE": "#B8860B", "HIGH": "#A8443A"}
    for r in rows:
        ax.plot(r["ctn"], r["eta"] * 100, "o", ms=10,
                color=colour[r["regime"]], mec="white", mew=1.2, zorder=10)
        ax.annotate(r["name"].split(",")[0], xy=(r["ctn"], r["eta"] * 100),
                    xytext=(0, 12), textcoords="offset points",
                    fontsize=8, ha="center")

    ax.text(0.012, 90, "LOW", fontsize=10, fontweight="bold",
            color="#3F7D3F", ha="center")
    ax.text(0.32, 90, "MODERATE", fontsize=10, fontweight="bold",
            color="#B8860B", ha="center")
    ax.text(5.0, 90, "HIGH", fontsize=10, fontweight="bold",
            color="#A8443A", ha="center")

    ax.set_xlim(1e-4, 30)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Caprock Transmission Number, CTN")
    ax.set_ylabel(r"Transmission efficiency $\eta$ (%)")
    ax.set_title("Stage 3 - field-site retrospective consistency")
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "stage3_field_sites.png"), dpi=200)
    plt.close(fig)

    n_consistent = sum(r["consistent"] for r in rows)
    return {
        "n_sites": len(rows),
        "n_consistent": n_consistent,
        "passed": n_consistent == len(rows),
    }


if __name__ == "__main__":
    print("Stage 3 - field-site retrospective consistency assessment")
    summary = run()
    print(f"  -> {summary['n_consistent']}/{summary['n_sites']} "
          f"sites consistent")
    print(f"  -> PASSED" if summary["passed"] else "  -> FAILED")
