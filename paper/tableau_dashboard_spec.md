# Tableau Dashboard Design Specification
## GCC Banking Research - 6-Panel Interactive Dashboard

### Data Source
Import `output/gcc_banks_processed.csv` and `output/dcf_sensitivity_table.csv` into Tableau.

---

## Panel 1: Key Metrics Heatmap
- **Chart Type:** Highlight Table / Heatmap
- **Rows:** Bank (dimension)
- **Columns:** Year (dimension)
- **Values:** Toggle between ROE, ROA, NIM, CET1_Ratio (parameter-driven)
- **Color:** Green-Yellow-Red diverging palette, centered at metric median
- **Annotations:** Display values in cells with 1 decimal place
- **Tooltip:** Bank, Year, selected metric value, plus Total Assets for context

## Panel 2: NIM Trend Over Time
- **Chart Type:** Line Chart with markers
- **X-Axis:** Year (continuous)
- **Y-Axis:** NIM (%)
- **Color:** Bank (5 distinct colors)
- **Reference Lines:** Add horizontal line at 2.0% (industry average reference)
- **Tooltip:** Bank, Year, NIM, Net Interest Income

## Panel 3: DuPont Decomposition (Stacked Bar)
- **Chart Type:** Stacked Bar Chart
- **X-Axis:** Bank
- **Stacking:** Net Margin contribution, Asset Turnover contribution, Equity Multiplier contribution
- **Note:** Normalize the log-decomposition to show % contribution to ROE
- **Color:** Dark blue (Net Margin), Medium blue (Asset Turnover), Green (Equity Multiplier)
- **Annotation:** Show ROE value on top of each bar

## Panel 4: DCF Sensitivity Heatmap
- **Chart Type:** Highlight Table
- **Rows:** Cost of Equity (9%, 10%, 11%, 12%, 13%)
- **Columns:** Terminal Growth Rate (2%, 3%, 4%, 5%)
- **Values:** Fair Value (AED per share)
- **Color:** Green-Red diverging, centered at AED 21.50 (market price)
- **Reference:** Add text annotation showing "Market Price: AED 21.50"

## Panel 5: ML Predicted vs Actual ROE
- **Chart Type:** Scatter Plot
- **X-Axis:** Actual ROE (%)
- **Y-Axis:** Predicted ROE (%) - use ROE_Predicted_OLS
- **Color:** Bank
- **Reference Line:** 45-degree diagonal (perfect prediction)
- **Tooltip:** Bank, Year, Actual ROE, Predicted ROE, Residual
- **Data Source:** `ml_predictions.csv`

## Panel 6: Capital Adequacy Comparison
- **Chart Type:** Grouped Bar Chart
- **X-Axis:** Bank
- **Bars:** Tier1_Capital_Ratio (blue), CET1_Ratio (green)
- **Reference Line:** Basel III minimum at 10.5% (dashed red)
- **Filter:** Year (default to 2025)
- **Tooltip:** Bank, Year, Tier1, CET1

---

## Dashboard Configuration

### Filters (applied across all panels)
1. **Bank Selector:** Multi-select checkbox (default: all 5)
2. **Year Range Slider:** Range slider 2018-2025 (default: full range)
3. **Metric Toggle:** Parameter for Panel 1 heatmap (ROE/ROA/NIM/CET1)

### Layout
- **Size:** 1920 x 1080 (widescreen)
- **Grid:** 3 columns x 2 rows
- **Title:** "GCC Banking Sector Performance Dashboard (2018-2025)"
- **Subtitle:** "Emirates NBD | FAB | QNB | Al Rajhi | Mashreq"

### Color Palette
| Color | Hex | Usage |
|---|---|---|
| Dark Blue | #1f4e79 | ENBD |
| Medium Blue | #2e75b6 | FAB |
| Green | #70ad47 | QNB |
| Orange | #c55a11 | Al Rajhi |
| Purple | #7030a0 | Mashreq |

### Publishing
1. Save to Tableau Public -> `gcc-banking-dashboard`
2. Set auto-refresh to manual (static dataset)
3. Export screenshots for paper appendix
