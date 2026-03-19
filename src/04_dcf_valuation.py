"""
04_dcf_valuation.py - GCC Banking Research
Dividend Discount Model valuation for Emirates NBD.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def capm_cost_of_equity(risk_free=0.0425, beta=1.1, erp=0.055, crp=0.008):
    """
    CAPM: Ke = Rf + beta * (ERP + CRP)

    Parameters:
        risk_free: US 10Y Treasury rate (proxy for risk-free)
        beta: ENBD beta from DFM
        erp: Equity Risk Premium (Damodaran global)
        crp: Country Risk Premium (UAE, from Damodaran)
    """
    ke = risk_free + beta * (erp + crp)
    return ke


def dividend_discount_model(dividends_history, roe_avg, payout_ratio, cost_of_equity,
                             terminal_growth, forecast_years=5):
    """
    Two-stage DDM:
        Stage 1: Explicit dividend forecast for n years
        Stage 2: Terminal value via Gordon Growth Model
    """
    # Sustainable growth = ROE  (1 - payout_ratio)
    sustainable_growth = roe_avg * (1 - payout_ratio)

    # Last known dividend
    last_div = dividends_history[-1]

    # Stage 1: Forecast dividends
    forecasted_divs = []
    for t in range(1, forecast_years + 1):
        next_div = last_div * (1 + sustainable_growth) ** t
        forecasted_divs.append(next_div)

    # Terminal value at end of Stage 1
    terminal_div = forecasted_divs[-1] * (1 + terminal_growth)
    terminal_value = terminal_div / (cost_of_equity - terminal_growth)

    # PV of Stage 1 dividends
    pv_divs = sum(d / (1 + cost_of_equity) ** t
                  for t, d in enumerate(forecasted_divs, 1))

    # PV of terminal value
    pv_terminal = terminal_value / (1 + cost_of_equity) ** forecast_years

    # Fair value per share
    fair_value = pv_divs + pv_terminal

    return {
        'fair_value': fair_value,
        'pv_dividends': pv_divs,
        'pv_terminal': pv_terminal,
        'terminal_value': terminal_value,
        'forecasted_dividends': forecasted_divs,
        'sustainable_growth': sustainable_growth,
        'cost_of_equity': cost_of_equity
    }


def sensitivity_analysis(dividends_history, roe_avg, payout_ratio,
                          coe_range, growth_range, forecast_years=5):
    """Generate sensitivity table: CoE vs Terminal Growth."""
    results = np.zeros((len(coe_range), len(growth_range)))

    for i, coe in enumerate(coe_range):
        for j, g in enumerate(growth_range):
            if coe <= g:
                results[i, j] = np.nan  # Invalid: CoE must exceed growth
            else:
                ddm = dividend_discount_model(
                    dividends_history, roe_avg, payout_ratio, coe, g, forecast_years
                )
                results[i, j] = ddm['fair_value']

    return pd.DataFrame(
        results,
        index=[f'{r*100:.0f}%' for r in coe_range],
        columns=[f'{g*100:.0f}%' for g in growth_range]
    )


def main():
    print("\n" + "=" * 60)
    print("PHASE 4: DCF VALUATION - Emirates NBD")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')

    # Load ENBD data
    processed_path = os.path.join(output_dir, 'gcc_banks_processed.csv')
    df = pd.read_csv(processed_path)
    enbd = df[df['Bank'] == 'Emirates NBD'].sort_values('Year')

    # --- ENBD Dividend History (AED per share) ---
    # Source: Emirates NBD investor relations
    dividend_history = {
        2018: 0.40,
        2019: 0.45,
        2020: 0.20,  # Reduced due to COVID
        2021: 0.35,
        2022: 0.60,
        2023: 0.85,
        2024: 1.00,
        2025: 1.10
    }

    dividends = list(dividend_history.values())
    years = list(dividend_history.keys())

    print("\nENBD Dividend History (AED/share):")
    print("-" * 40)
    for y, d in dividend_history.items():
        print(f"  {y}: AED {d:.2f}")

    # --- CAPM Inputs ---
    risk_free = 0.0425   # US 10Y Treasury
    beta = 1.10          # ENBD beta (DFM estimate)
    erp = 0.055          # Equity Risk Premium
    crp = 0.008          # UAE Country Risk Premium

    ke = capm_cost_of_equity(risk_free, beta, erp, crp)

    print(f"\nCAPM Cost of Equity:")
    print(f"  Risk-free rate (Rf):   {risk_free*100:.2f}%")
    print(f"  Beta (b):              {beta:.2f}")
    print(f"  Equity Risk Premium:   {erp*100:.1f}%")
    print(f"  Country Risk Premium:  {crp*100:.1f}%")
    print(f"  Cost of Equity (Ke):   {ke*100:.2f}%")

    # --- DDM Parameters ---
    roe_avg = enbd['ROE'].mean() / 100  # Average ROE as decimal
    payout_ratio = 0.45  # Approximate payout ratio
    terminal_growth = 0.035  # 3.5% long-term growth

    print(f"\nDDM Parameters:")
    print(f"  Average ROE:          {roe_avg*100:.2f}%")
    print(f"  Payout Ratio:         {payout_ratio*100:.0f}%")
    print(f"  Terminal Growth:      {terminal_growth*100:.1f}%")

    # --- Base Case DDM ---
    base_ddm = dividend_discount_model(
        dividends, roe_avg, payout_ratio, ke, terminal_growth
    )

    print(f"\nBase Case Valuation:")
    print(f"  PV of Dividends:     AED {base_ddm['pv_dividends']:.2f}")
    print(f"  PV of Terminal:      AED {base_ddm['pv_terminal']:.2f}")
    print(f"  Fair Value/Share:    AED {base_ddm['fair_value']:.2f}")

    # Current market price (approximate DFM)
    market_price = 21.50  # AED
    upside = (base_ddm['fair_value'] / market_price - 1) * 100

    print(f"\n  Market Price (DFM):  AED {market_price:.2f}")
    print(f"  Implied Upside:      {upside:+.1f}%")
    if upside > 10:
        print(f"   Conclusion: UNDERVALUED")
    elif upside < -10:
        print(f"   Conclusion: OVERVALUED")
    else:
        print(f"   Conclusion: FAIRLY VALUED")

    # --- Sensitivity Analysis ---
    coe_range = [0.09, 0.10, 0.11, 0.12, 0.13]
    growth_range = [0.02, 0.03, 0.04, 0.05]

    sensitivity = sensitivity_analysis(
        dividends, roe_avg, payout_ratio, coe_range, growth_range
    )

    print(f"\nSensitivity Table (Fair Value per Share, AED):")
    print("Cost of Equity (rows) vs Terminal Growth (columns)")
    print("-" * 60)
    print(sensitivity.round(2).to_string())

    # --- Generate Sensitivity Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 7))
    mask = sensitivity.isnull()

    sns.heatmap(sensitivity.astype(float), annot=True, fmt='.1f',
                cmap='RdYlGn', center=market_price,
                linewidths=1, ax=ax, mask=mask,
                cbar_kws={'label': 'Fair Value (AED)'})

    # Highlight cells near market price
    ax.set_xlabel('Terminal Growth Rate', fontsize=12)
    ax.set_ylabel('Cost of Equity', fontsize=12)
    ax.set_title(f'Emirates NBD  DDM Sensitivity Analysis\n'
                 f'(Market Price: AED {market_price:.2f})',
                 fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dcf_sensitivity.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Sensitivity heatmap saved")

    # --- Dividend Growth Chart ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Historical dividends
    ax1.bar(years, dividends, color='#2e75b6', edgecolor='white', width=0.6)
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Dividend per Share (AED)', fontsize=12)
    ax1.set_title('ENBD Historical Dividends', fontsize=13, fontweight='bold')
    ax1.set_xticks(years)
    for i, (y, d) in enumerate(zip(years, dividends)):
        ax1.text(y, d + 0.02, f'{d:.2f}', ha='center', fontsize=9, fontweight='bold')

    # Forecasted dividends
    forecast_years_list = list(range(2026, 2031))
    all_years = years + forecast_years_list
    all_divs = dividends + base_ddm['forecasted_dividends']
    bar_colors = ['#2e75b6'] * len(years) + ['#70ad47'] * len(forecast_years_list)

    ax2.bar(all_years, all_divs, color=bar_colors, edgecolor='white', width=0.6)
    ax2.axvline(x=2025.5, color='red', linestyle='--', alpha=0.7, label='Forecast Start')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Dividend per Share (AED)', fontsize=12)
    ax2.set_title('ENBD Dividend Forecast (DDM)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xticks(all_years)
    ax2.tick_params(axis='x', rotation=45)

    plt.suptitle('Emirates NBD  Dividend History & Forecast',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dcf_dividends.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] Dividend chart saved")

    # Save results
    results_df = pd.DataFrame({
        'Metric': ['Cost of Equity', 'Terminal Growth', 'Sustainable Growth',
                   'PV Dividends', 'PV Terminal', 'Fair Value', 'Market Price',
                   'Upside/Downside', 'Conclusion'],
        'Value': [f'{ke*100:.2f}%', f'{terminal_growth*100:.1f}%',
                  f'{base_ddm["sustainable_growth"]*100:.2f}%',
                  f'AED {base_ddm["pv_dividends"]:.2f}',
                  f'AED {base_ddm["pv_terminal"]:.2f}',
                  f'AED {base_ddm["fair_value"]:.2f}',
                  f'AED {market_price:.2f}',
                  f'{upside:+.1f}%',
                  'Undervalued' if upside > 10 else 'Overvalued' if upside < -10 else 'Fairly Valued']
    })
    results_df.to_csv(os.path.join(output_dir, 'dcf_results.csv'), index=False)
    sensitivity.to_csv(os.path.join(output_dir, 'dcf_sensitivity_table.csv'))
    print("[OK] DCF results and sensitivity table saved")

    return base_ddm, sensitivity


if __name__ == '__main__':
    main()
