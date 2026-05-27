"""
run_all.py
==========
Master driver for the CTN validation suite. Runs all three validation
stages in sequence and prints a consolidated summary.

Usage
-----
    python run_all.py

Stages
------
  Stage 1 : Terzaghi consolidation benchmark    (solver verification)
  Stage 2 : CTN sweep vs erfc + finite-domain   (mapping verification)
  Stage 3 : field-site retrospective assessment (consistency check)

Each stage writes CSV results to results/ and a figure to figures/.

Author : H. Mbah, Institute for Groundwater Studies,
         University of the Free State, South Africa
Licence : MIT
"""

import sys
import time

import stage1_terzaghi
import stage2_ctn_sweep
import stage3_field_sites


def main():
    t0 = time.time()
    print("=" * 64)
    print("  CTN VALIDATION SUITE")
    print("  Caprock Transmission Number - reproducibility package v1.0.0")
    print("=" * 64)

    print("\n[Stage 1] Terzaghi consolidation benchmark")
    s1 = stage1_terzaghi.run()
    print(f"          max Linf error  : {s1['max_Linf_percent']:.5f} %")
    print(f"          observed order  : {s1['observed_order']:.3f}")

    print("\n[Stage 2] CTN sweep verification")
    s2 = stage2_ctn_sweep.run()
    print(f"          max rel. error  : "
          f"{s2['max_rel_error_percent']:.3f} %")
    print(f"          images ratio    : "
          f"{s2['images_ratio_low_ctn']:.3f}")

    print("\n[Stage 3] Field-site retrospective consistency")
    s3 = stage3_field_sites.run()
    print(f"          consistent      : "
          f"{s3['n_consistent']}/{s3['n_sites']} sites")

    elapsed = time.time() - t0
    all_passed = s1["passed"] and s2["passed"] and s3["passed"]

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("-" * 64)
    print(f"  Stage 1 (Terzaghi benchmark)      : "
          f"{'PASS' if s1['passed'] else 'FAIL'}")
    print(f"  Stage 2 (CTN sweep verification)  : "
          f"{'PASS' if s2['passed'] else 'FAIL'}")
    print(f"  Stage 3 (field-site consistency)  : "
          f"{'PASS' if s3['passed'] else 'FAIL'}")
    print("-" * 64)
    print(f"  OVERALL : {'ALL STAGES PASSED' if all_passed else 'FAILURE'}")
    print(f"  Runtime : {elapsed:.1f} s")
    print("=" * 64)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
