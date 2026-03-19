"""
03_dupont.py - GCC Banking Research
DuPont 3-factor decomposition of Return on Equity.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


def dupont_decomposition(df):
    """
    DuPont 3-Factor:
        ROE = Net Margin * Asset Turnover * Equity Multiplier

    Where:
        Net Margin      = Net Profit / Total Operating Income
        Asset Turnover  = Total Operating Income / Total Assets
        Equity Multiplier = Total Assets / Total Equity
    """
    df['DuPont_Net_Margin'] = df['Net_Profit'] / df['Total_Operating_Income']
    df['DuPont_Asset_Turnover'] = df['Total_Operating_Income'] / df['Total_Assets']
    df['DuPont_Equity_Multiplier'] = df['Total_Assets'] / df['Total_Equity']

    # Verify identity: product should approx ROE (as decimal)
    df['DuPont_ROE_Check'] = (df['DuPont_Net_Margin'] *
                               df['DuPont_Asset_Turnover'] *
                               df['DuPont_Equity_Multiplier']) * 100

    # Compute residual to verify accuracy
    if 'ROE' in df.columns:
        df['DuPont_Residual'] = abs(df['DuPont_ROE_Check'] - df['ROE'])

    return df


def generate_dupont_charts(df, output_dir):
    """Generate DuPont decomposition visualizations."""
    banks = df['Bank'].unique()
    colors = {
        'Net Margin': '#1f4e79',
        'Asset Turnover': '#2e75b6',
        'Equity Multiplier': '#70ad47'
    }

    # --- Chart 1: Stacked Bar - DuPont Components by Bank (2025) ---
    fig, ax = plt.subplots(figsize=(14, 8))
    latest = df[df['Year'] == 2025].copy().sort_values('Bank')

    x = np.arange(len(latest))
    width = 0.6

    # Normalize components to show relative contribution
    nm_contrib = np.log(latest['DuPont_Net_Margin'].values)
    at_contrib = np.log(latest['DuPont_Asset_Turnover'].values)
    em_contrib = np.log(latest['DuPont_Equity_Multiplier'].values)
    total_log = nm_contrib + at_contrib + em_contrib

    nm_pct = (nm_contrib / total_log) * 100
    at_pct = (at_contrib / total_log) * 100
    em_pct = (em_contrib / total_log) * 100

    ax.bar(x, nm_pct, width, label='Net Margin', color=colors['Net Margin'], edgecolor='white')
    ax.bar(x, at_pct, width, bottom=nm_pct, label='Asset Turnover',
           color=colors['Asset Turnover'], edgecolor='white')
    ax.bar(x, em_pct, width, bottom=nm_pct + at_pct, label='Equity Multiplier',
           color=colors['Equity Multiplier'], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(latest['Bank'], rotation=15, ha='right', fontsize=11)
    ax.set_ylabel('Contribution to ROE (%)', fontsize=12)
    ax.set_title('DuPont Decomposition - What Drives ROE? (2025)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right')

    # Add ROE annotation on top
    for i, (_, row) in enumerate(latest.iterrows()):
        ax.annotate(f"ROE: {row['DuPont_ROE_Check']:.1f}%",
                    (i, nm_pct[i] + at_pct[i] + em_pct[i] + 1),
                    ha='center', fontsize=9, fontweight='bold', color='#333')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dupont_stacked_2025.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] DuPont stacked bar (2025) saved")

    # --- Chart 2: DuPont Time-Series for Each Bank ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, bank in enumerate(banks):
        ax = axes[idx]
        bdata = df[df['Bank'] == bank].sort_values('Year')

        ax.plot(bdata['Year'], bdata['DuPont_Net_Margin'], 'o-',
                color=colors['Net Margin'], linewidth=2, label='Net Margin')
        ax.plot(bdata['Year'], bdata['DuPont_Asset_Turnover'] * 100, 's-',
                color=colors['Asset Turnover'], linewidth=2, label='Asset Turn. (x100)')
        ax2 = ax.twinx()
        ax2.plot(bdata['Year'], bdata['DuPont_Equity_Multiplier'], '^-',
                 color=colors['Equity Multiplier'], linewidth=2, label='Equity Mult.')
        ax2.set_ylabel('Equity Multiplier', fontsize=9, color=colors['Equity Multiplier'])

        ax.set_title(bank, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Net Margin / Asset Turn.', fontsize=9)
        ax.set_xticks(range(2018, 2026))
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='best')

    # Hide last subplot if odd number of banks
    if len(banks) < len(axes):
        for i in range(len(banks), len(axes)):
            axes[i].set_visible(False)

    plt.suptitle('DuPont Components Over Time - GCC Banks',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dupont_timeseries.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] DuPont time-series charts saved")

    # --- Chart 3: DuPont Component Table (Heatmap) ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    components = [
        ('DuPont_Net_Margin', 'Net Margin (Net Profit / Op. Income)', 'RdYlGn'),
        ('DuPont_Asset_Turnover', 'Asset Turnover (Op. Income / Assets)', 'YlOrBr'),
        ('DuPont_Equity_Multiplier', 'Equity Multiplier (Assets / Equity)', 'PuBu')
    ]
    for ax, (col, title, cmap) in zip(axes, components):
        pivot = df.pivot_table(values=col, index='Bank', columns='Year')
        fmt = '.3f' if col == 'DuPont_Asset_Turnover' else '.2f'
        sns_hm = __import__('seaborn').heatmap(pivot, annot=True, fmt=fmt, cmap=cmap,
                                                linewidths=0.5, ax=ax)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('')

    plt.suptitle('DuPont Component Heatmaps - GCC Banks 2018-2025',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dupont_heatmaps.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] DuPont component heatmaps saved")


def main():
    print("\n" + "=" * 60)
    print("PHASE 3: DUPONT DECOMPOSITION")
    print("=" * 60)

    # Load processed data
    processed_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'gcc_banks_processed.csv')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    df = pd.read_csv(processed_path)

    # Compute DuPont
    df = dupont_decomposition(df)

    # Verification
    print("\nDuPont Identity Verification:")
    print("-" * 60)
    max_residual = df['DuPont_Residual'].max()
    mean_residual = df['DuPont_Residual'].mean()
    print(f"  Max residual: {max_residual:.6f}%")
    print(f"  Mean residual: {mean_residual:.6f}%")
    if max_residual < 0.01:
        print("  [OK] DuPont identity holds within 0.01% for all observations")
    else:
        print("  [Warning] Some residuals exceed 0.01% - check data consistency")

    # Print component summary
    print("\nDuPont Components (2025):")
    print("-" * 80)
    cols = ['Bank', 'DuPont_Net_Margin', 'DuPont_Asset_Turnover',
            'DuPont_Equity_Multiplier', 'DuPont_ROE_Check']
    latest = df[df['Year'] == 2025][cols].set_index('Bank')
    latest.columns = ['Net Margin', 'Asset Turnover', 'Equity Multiplier', 'ROE (%)']
    print(latest.round(4).to_string())

    # Save updated processed data
    df.to_csv(processed_path, index=False)
    print(f"\n[OK] Processed data updated with DuPont columns")

    # Generate charts
    print("\nGenerating DuPont charts...")
    generate_dupont_charts(df, output_dir)

    return df


if __name__ == '__main__':
    main()
