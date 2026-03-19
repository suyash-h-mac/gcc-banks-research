"""
02_ratio_analysis.py  GCC Banking Research
Compute all derived financial ratios from cleaned data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def compute_ratios(df):
    """Compute all financial ratios."""
    # Profitability Ratios
    df['ROE'] = (df['Net_Profit'] / df['Total_Equity']) * 100
    df['ROA'] = (df['Net_Profit'] / df['Total_Assets']) * 100
    df['NIM'] = (df['Net_Interest_Income'] / df['Total_Assets']) * 100

    # Asset Quality
    df['Stage3_Ratio'] = (df['Credit_Impaired_Loans'] / df['Gross_Loans']) * 100

    # Efficiency
    df['Cost_to_Income'] = (df['Operating_Expenses'] / df['Total_Operating_Income']) * 100

    # Liquidity
    df['Loan_to_Deposit'] = (df['Net_Loans'] / df['Total_Deposits']) * 100

    # Additional useful ratios
    df['Equity_to_Assets'] = (df['Total_Equity'] / df['Total_Assets']) * 100
    df['Provision_Coverage'] = ((df['Gross_Loans'] - df['Net_Loans']) / df['Credit_Impaired_Loans']) * 100

    return df


def generate_charts(df, output_dir):
    """Generate publication-quality charts."""
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#1f4e79', '#2e75b6', '#70ad47', '#c55a11', '#7030a0']
    banks = df['Bank'].unique()
    bank_colors = dict(zip(banks, colors))

    # --- Chart 1: ROE Heatmap ---
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = df.pivot_table(values='ROE', index='Bank', columns='Year')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=12,
                linewidths=0.5, ax=ax, cbar_kws={'label': 'ROE (%)'})
    ax.set_title('Return on Equity (ROE %) - GCC Banks 2018-2025',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roe_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] ROE heatmap saved")

    # --- Chart 2: NIM Trend ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for bank in banks:
        data = df[df['Bank'] == bank]
        ax.plot(data['Year'], data['NIM'], marker='o', linewidth=2.5,
                color=bank_colors[bank], label=bank, markersize=6)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('NIM (%)', fontsize=12)
    ax.set_title('Net Interest Margin Trend - GCC Banks 2018-2025',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.set_xticks(range(2018, 2026))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nim_trend.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] NIM trend chart saved")

    # --- Chart 3: Multi-Metric Heatmap (ROE, ROA, NIM, CET1) ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    metrics = [('ROE', 'Return on Equity (%)', 'RdYlGn'),
               ('ROA', 'Return on Assets (%)', 'RdYlGn'),
               ('NIM', 'Net Interest Margin (%)', 'YlOrBr'),
               ('CET1_Ratio', 'CET1 Capital Ratio (%)', 'Blues')]
    for ax, (metric, title, cmap) in zip(axes.flatten(), metrics):
        pivot = df.pivot_table(values=metric, index='Bank', columns='Year')
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap,
                    linewidths=0.5, ax=ax, cbar_kws={'label': ''})
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('')
    plt.suptitle('Key Financial Metrics - GCC Banks 2018-2025',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] Multi-metric heatmap saved")

    # --- Chart 4: Cost-to-Income Comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for bank in banks:
        data = df[df['Bank'] == bank]
        ax.plot(data['Year'], data['Cost_to_Income'], marker='s', linewidth=2.5,
                color=bank_colors[bank], label=bank, markersize=6)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cost-to-Income (%)', fontsize=12)
    ax.set_title('Cost-to-Income Ratio - GCC Banks 2018-2025',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.set_xticks(range(2018, 2026))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_to_income.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] Cost-to-Income chart saved")

    # --- Chart 5: Stage 3 Asset Quality ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for bank in banks:
        data = df[df['Bank'] == bank]
        ax.plot(data['Year'], data['Stage3_Ratio'], marker='^', linewidth=2.5,
                color=bank_colors[bank], label=bank, markersize=6)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Stage 3 Ratio (%)', fontsize=12)
    ax.set_title('Credit Impairment (Stage 3) Ratio - GCC Banks 2018-2025',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.set_xticks(range(2018, 2026))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stage3_ratio.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] Stage 3 ratio chart saved")

    # --- Chart 6: Capital Adequacy Bar ---
    fig, ax = plt.subplots(figsize=(14, 7))
    latest = df[df['Year'] == 2025].sort_values('Bank')
    x = np.arange(len(latest))
    width = 0.35
    bars1 = ax.bar(x - width/2, latest['Tier1_Capital_Ratio'], width,
                   label='Tier 1 Ratio', color='#2e75b6', edgecolor='white')
    bars2 = ax.bar(x + width/2, latest['CET1_Ratio'], width,
                   label='CET1 Ratio', color='#70ad47', edgecolor='white')
    ax.axhline(y=10.5, color='red', linestyle='--', alpha=0.7, label='Basel III Min (10.5%)')
    ax.set_xticks(x)
    ax.set_xticklabels(latest['Bank'], rotation=15, ha='right', fontsize=10)
    ax.set_ylabel('Ratio (%)', fontsize=12)
    ax.set_title('Capital Adequacy - GCC Banks (2025)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=10)
    ax.bar_label(bars1, fmt='%.1f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.1f', padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'capital_adequacy.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] Capital adequacy chart saved")


def main():
    print("\n" + "=" * 60)
    print("PHASE 2: RATIO ANALYSIS")
    print("=" * 60)

    # Load clean data
    clean_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'gcc_banks_clean.csv')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    df = pd.read_csv(clean_path)

    # Compute ratios
    df = compute_ratios(df)

    # Print summary
    ratios = ['ROE', 'ROA', 'NIM', 'Stage3_Ratio', 'Cost_to_Income', 'Loan_to_Deposit']
    print("\nComputed Ratios Summary (2025):")
    print("-" * 80)
    latest = df[df['Year'] == 2025][['Bank'] + ratios].set_index('Bank')
    print(latest.round(2).to_string())

    # Save processed data
    processed_path = os.path.join(output_dir, 'gcc_banks_processed.csv')
    df.to_csv(processed_path, index=False)
    print(f"\n[OK] Processed data saved: {processed_path}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Generate charts
    print("\nGenerating charts...")
    generate_charts(df, output_dir)

    return df


if __name__ == '__main__':
    main()
