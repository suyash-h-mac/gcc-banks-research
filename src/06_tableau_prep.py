"""
06_tableau_prep.py - GCC Banking Research
Consolidate all processed data into a single master CSV for Tableau.
"""
import pandas as pd
import os

def main():
    print("\n" + "="*60)
    print("PHASE 6: TABLEAU DATA PREPARATION")
    print("="*60)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    # 1. Load Datasets
    processed_path = os.path.join(output_dir, 'gcc_banks_processed.csv')
    macro_path = os.path.join(data_dir, 'macro_data.csv')
    ml_path = os.path.join(output_dir, 'ml_predictions.csv')

    if not os.path.exists(processed_path):
        print(f"Error: {processed_path} not found. Run previous phases first.")
        return

    df = pd.read_csv(processed_path)
    macro = pd.read_csv(macro_path) if os.path.exists(macro_path) else None
    ml = pd.read_csv(ml_path) if os.path.exists(ml_path) else None

    # 2. Merge Data
    # Merge with Macro data
    if macro is not None:
        # Avoid duplicate Year columns if already present or merged
        df = df.merge(macro, on='Year', how='left')
        print("[OK] Merged macroeconomic data")

    # Merge with ML predictions
    if ml is not None:
        # ml_predictions.csv already contains the base data + predictions
        # We just need the prediction columns
        pred_cols = ['Bank', 'Year', 'ROE_Predicted_OLS', 'ROE_Predicted_ML']
        ml_subset = ml[pred_cols]
        df = df.merge(ml_subset, on=['Bank', 'Year'], how='left')
        print("[OK] Merged ML predictions")

    # 3. Clean and Rename Columns for Tableau
    # Replace underscores with spaces for better Tableau labels
    rename_map = {col: col.replace('_', ' ') for col in df.columns}
    df = df.rename(columns=rename_map)

    # 4. Save Final Master CSV
    master_path = os.path.join(output_dir, 'tableau_master.csv')
    df.to_csv(master_path, index=False)
    
    print(f"\n[OK] Master dataset saved to: {master_path}")
    print(f"  Total records: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")
    print("="*60)

if __name__ == '__main__':
    main()
