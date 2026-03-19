A comprehensive financial analysis of five systemically important GCC banks: **Emirates NBD, First Abu Dhabi Bank (FAB), Qatar National Bank (QNB), Al Rajhi Bank,** and **Mashreq Bank**. 

The project spans the pre-COVID stability, pandemic disruption, aggressive monetary tightening (2022-2023), and post-recovery normalization phases.

## Features
- **Data Pipeline**: Automated cleaning and standardization of primary financial data.
- **Ratio Analysis**: Comparative assessment of profitability, asset quality, and capital adequacy.
- **DuPont Decomposition**: Structural analysis of Return on Equity (ROE) drivers.
- **DCF Valuation**: Dividend Discount Model (DDM) for Emirates NBD with sensitivity analysis.
- **Machine Learning**: OLS and Random Forest models to predict ROE using macroeconomic variables (Oil, GDP, Fed Funds, Inflation).
- **Tableau Master Dataset**: Consolidated output for interactive dashboarding.

## Project Structure
```text
 data/               # Raw and macro data files
 output/             # Generated CSVs and Visualization charts
 paper/              # Research paper draft and Tableau specs
 src/                # Analytical pipeline (Phases 1-6)
 run_all.py          # Main execution script
 requirements.txt    # Python dependencies
```

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Tableau Desktop or Tableau Public (for visualization)

### 2. Installation
Install the required analytical libraries:
```bash
pip install -r requirements.txt
```

### 3. Execution
Run the master script to execute all 6 phases of the research pipeline:
```bash
python3 run_all.py
```
This will:
- Clean raw data (`data/`)
- Compute 50+ financial ratios
- Perform DuPont decomposition
- Run DCF valuation for Emirates NBD
- Train and evaluate ML models (OLS/RF)
- Generate a consolidated `output/tableau_master.csv` for dashboarding

---

## Tableau Visualization

To create the interactive dashboard:

1. **Launch Tableau**: Open Tableau Desktop or Tableau Public.
2. **Connect to Data**:
   - Choose **Text File** and select `output/tableau_master.csv`.
   - Optionally, add `output/dcf_sensitivity_table.csv` for the valuation panel.
3. **Dashboard Specs**: Refer to `paper/tableau_dashboard_spec.md` for the exact configuration of the 6 panels:
   - **Panel 1**: Key Metrics Heatmap (ROE/ROA/NIM)
   - **Panel 2**: NIM Trends (2018-2025)
   - **Panel 3**: DuPont Decomposition Stacked Bars
   - **Panel 4**: DCF Sensitivity Heatmap
   - **Panel 5**: ML Predicted vs Actual Scatter
   - **Panel 6**: Capital Adequacy (CET1) Comparison

---

## Research Paper
The full academic draft is available in `paper/gcc_banking_paper.md`. It includes the abstract, methodology, results, and discussion for the 2018-2025 study period.

## License
MIT
