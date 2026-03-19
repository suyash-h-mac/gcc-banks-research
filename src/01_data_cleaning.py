"""
01_data_cleaning.py  GCC Banking Research
Load, validate, and clean the raw financial dataset.
"""
import pandas as pd
import numpy as np
import os
import sys

def main():
    print("=" * 60)
    print("PHASE 1: DATA CLEANING & VALIDATION")
    print("=" * 60)

    # Paths
    raw_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'gcc_banks_raw.csv')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    clean_path = os.path.join(output_dir, 'gcc_banks_clean.csv')

    # Load
    df = pd.read_csv(raw_path)
    print(f"\n[OK] Loaded raw data: {df.shape[0]} rows x {df.shape[1]} columns")

    # Expected structure
    expected_banks = ['Emirates NBD', 'First Abu Dhabi Bank', 'Qatar National Bank',
                      'Al Rajhi Bank', 'Mashreq Bank']
    expected_years = list(range(2018, 2026))
    expected_rows = len(expected_banks) * len(expected_years)

    # Validate row count
    assert df.shape[0] == expected_rows, \
        f"Expected {expected_rows} rows, got {df.shape[0]}"
    print(f"[OK] Row count verified: {expected_rows} rows (5 banks x 8 years)")

    # Validate banks
    actual_banks = sorted(df['Bank'].unique())
    assert set(actual_banks) == set(expected_banks), \
        f"Missing banks: {set(expected_banks) - set(actual_banks)}"
    print(f"[OK] All 5 banks present: {', '.join(expected_banks)}")

    # Validate years
    for bank in expected_banks:
        bank_years = sorted(df[df['Bank'] == bank]['Year'].unique())
        assert bank_years == expected_years, \
            f"{bank}: missing years {set(expected_years) - set(bank_years)}"
    print(f"[OK] All years 2018-2025 present for each bank")

    # Numeric columns
    numeric_cols = [
        'Total_Assets', 'Net_Loans', 'Gross_Loans', 'Total_Deposits',
        'Total_Equity', 'Net_Interest_Income', 'Total_Operating_Income',
        'Operating_Expenses', 'Net_Profit', 'Tier1_Capital_Ratio',
        'CET1_Ratio', 'Credit_Impaired_Loans'
    ]

    # Type enforcement
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Check for nulls
    null_counts = df[numeric_cols].isnull().sum()
    if null_counts.sum() > 0:
        print("\nWarning: Null values found:")
        print(null_counts[null_counts > 0])
    else:
        print(f"[OK] No null values in any of {len(numeric_cols)} numeric columns")

    # Sanity checks  values should be positive (except possibly Net_Profit during COVID)
    for col in ['Total_Assets', 'Net_Loans', 'Gross_Loans', 'Total_Deposits', 'Total_Equity']:
        neg = df[df[col] <= 0]
        if len(neg) > 0:
            print(f"Warning: Non-positive values in {col}:")
            print(neg[['Bank', 'Year', col]])

    # Verify Net_Loans <= Gross_Loans
    violations = df[df['Net_Loans'] > df['Gross_Loans']]
    if len(violations) > 0:
        print("Warning: Net_Loans > Gross_Loans found:")
        print(violations[['Bank', 'Year', 'Net_Loans', 'Gross_Loans']])
    else:
        print("[OK] Net_Loans <= Gross_Loans for all rows")

    # Sort
    df = df.sort_values(['Bank', 'Year']).reset_index(drop=True)

    # Add currency note
    df.attrs['currency'] = 'AED Millions'
    df.attrs['source'] = 'Annual Reports & Investor Relations (2018-2025)'

    # Summary statistics
    print("\n" + "-" * 60)
    print("SUMMARY STATISTICS (AED Millions)")
    print("-" * 60)
    summary = df.groupby('Bank').agg({
        'Total_Assets': ['min', 'max'],
        'Net_Profit': ['min', 'max'],
        'Tier1_Capital_Ratio': ['min', 'max']
    }).round(1)
    print(summary.to_string())

    # Save
    df.to_csv(clean_path, index=False)
    print(f"\n[OK] Clean data saved to: {clean_path}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    return df


if __name__ == '__main__':
    main()
