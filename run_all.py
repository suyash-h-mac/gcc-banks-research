#!/usr/bin/env python3
"""
run_all.py  GCC Banking Research
Execute the complete analysis pipeline.
"""
import sys
import os
import time
import importlib

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    start = time.time()

    print("+" + "-" * 58 + "+")
    print("|  GCC BANKING RESEARCH -- FULL PIPELINE                    |")
    print("|  5 Banks x 8 Years (2018-2025)                           |")
    print("|  Emirates NBD | FAB | QNB | Al Rajhi | Mashreq           |")
    print("+" + "-" * 58 + "+")

    # Ensure output directory exists
    os.makedirs(os.path.join(project_root, 'output'), exist_ok=True)

    # Phase 1: Data Cleaning
    print("\nPhase 1: Data Cleaning...")
    mod1 = importlib.import_module('src.01_data_cleaning')
    mod1.main()

    # Phase 2: Ratio Analysis
    print("\nPhase 2: Ratio Analysis...")
    mod2 = importlib.import_module('src.02_ratio_analysis')
    mod2.main()

    # Phase 3: DuPont Decomposition
    print("\nPhase 3: DuPont Decomposition...")
    mod3 = importlib.import_module('src.03_dupont')
    mod3.main()

    # Phase 4: DCF Valuation
    print("\nPhase 4: DCF Valuation...")
    mod4 = importlib.import_module('src.04_dcf_valuation')
    mod4.main()

    # Phase 5: Machine Learning
    print("\nPhase 5: Machine Learning...")
    mod5 = importlib.import_module('src.05_ml_model')
    mod5.main()

    # Phase 6: Tableau Data Preparation
    print("\nPhase 6: Tableau Data Preparation...")
    mod6 = importlib.import_module('src.06_tableau_prep')
    mod6.main()

    elapsed = time.time() - start

    print("\n" + "+" + "-" * 58 + "+")
    print("|  PIPELINE COMPLETE                                       |")
    print(f"|  Total time: {elapsed:.1f}s{' ' * max(0, 44 - len(f'{elapsed:.1f}'))}|")
    print("+" + "-" * 58 + "+")

    print("\nOutput files:")
    output_dir = os.path.join(project_root, 'output')
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        print(f"  - {f:40s} ({size:>8,} bytes)")


if __name__ == '__main__':
    main()
