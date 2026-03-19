"""
05_ml_model.py - GCC Banking Research
Machine learning models to predict ROE from macroeconomic variables.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except (ImportError, OSError, Exception):
    HAS_XGBOOST = False
    from sklearn.ensemble import RandomForestRegressor


def prepare_data(processed_path, macro_path):
    """Merge bank data with macro data and compute credit growth."""
    df = pd.read_csv(processed_path)
    macro = pd.read_csv(macro_path)

    # Merge on Year
    merged = df.merge(macro, on='Year', how='left')

    # Compute Credit Growth Rate (YoY loan growth per bank)
    merged = merged.sort_values(['Bank', 'Year'])
    merged['Credit_Growth'] = merged.groupby('Bank')['Gross_Loans'].pct_change() * 100

    # Fill first year NaN with 0
    merged['Credit_Growth'] = merged['Credit_Growth'].fillna(0)

    return merged


def run_ols_regression(X, y, feature_names):
    """Run OLS regression with full diagnostics."""
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return model


def run_ml_model(X, y):
    """Run ML model (XGBoost or Random Forest) with LOO cross-validation."""
    if HAS_XGBOOST:
        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        model_name = "XGBoost"
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            random_state=42
        )
        model_name = "Random Forest"

    # Leave-One-Out CV for small datasets
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)

    # Fit final model on all data for feature importance
    model.fit(X, y)

    return model, y_pred, model_name


def main():
    print("\n" + "=" * 60)
    print("PHASE 5: MACHINE LEARNING - ROE PREDICTION")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    processed_path = os.path.join(output_dir, 'gcc_banks_processed.csv')
    macro_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'macro_data.csv')

    # Prepare data
    merged = prepare_data(processed_path, macro_path)

    # Define features and target
    features = ['Oil_Price_USD', 'UAE_GDP_Growth', 'US_Fed_Funds_Rate',
                'Inflation_Rate_UAE', 'Credit_Growth']
    target = 'ROE'

    # Drop rows with missing values in features/target
    model_data = merged.dropna(subset=features + [target])
    X = model_data[features].values
    y = model_data[target].values

    print(f"\nDataset: {len(model_data)} observations")
    print(f"Features: {', '.join(features)}")
    print(f"Target: {target}")

    # --- Model 1: OLS Regression ---
    print("\n" + "-" * 60)
    print("MODEL 1: OLS LINEAR REGRESSION")
    print("-" * 60)

    ols_model = run_ols_regression(X, y, features)
    print(ols_model.summary().tables[1].as_text())

    print(f"\n  R2:           {ols_model.rsquared:.4f}")
    print(f"  Adjusted R2:  {ols_model.rsquared_adj:.4f}")
    print(f"  F-statistic:  {ols_model.fvalue:.2f}")
    print(f"  F p-value:    {ols_model.f_pvalue:.4f}")

    # OLS predictions
    X_const = sm.add_constant(X)
    ols_pred = ols_model.predict(X_const)

    # --- Model 2: XGBoost / Random Forest ---
    print("\n" + "-" * 60)
    ml_model, ml_pred, model_name = run_ml_model(X, y)
    print(f"MODEL 2: {model_name.upper()}")
    print("-" * 60)

    ml_r2 = r2_score(y, ml_pred)
    ml_mae = mean_absolute_error(y, ml_pred)
    ml_rmse = np.sqrt(mean_squared_error(y, ml_pred))

    print(f"  R2 (LOO-CV):  {ml_r2:.4f}")
    print(f"  MAE:          {ml_mae:.2f}%")
    print(f"  RMSE:         {ml_rmse:.2f}%")

    # Feature importance
    if HAS_XGBOOST:
        importances = ml_model.feature_importances_
    else:
        importances = ml_model.feature_importances_

    feat_imp = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    print(f"\n  Feature Importance ({model_name}):")
    for _, row in feat_imp.iterrows():
        bar = '=' * int(row['Importance'] * 40)
        print(f"    {row['Feature']:25s} {row['Importance']:.4f} {bar}")

    # --- Model Comparison ---
    print("\n" + "-" * 60)
    print("MODEL COMPARISON")
    print("-" * 60)
    comparison = pd.DataFrame({
        'Model': ['OLS Regression', model_name],
        'R2': [ols_model.rsquared, ml_r2],
        'MAE': [mean_absolute_error(y, ols_pred), ml_mae],
        'RMSE': [np.sqrt(mean_squared_error(y, ols_pred)), ml_rmse]
    })
    print(comparison.to_string(index=False))

    # --- Generate Charts ---
    print("\nGenerating ML charts...")

    # Chart 1: Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f4e79', '#2e75b6', '#70ad47', '#c55a11', '#7030a0']
    ax.barh(feat_imp['Feature'], feat_imp['Importance'],
            color=colors[:len(features)], edgecolor='white', height=0.5)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'{model_name} Feature Importance - Predicting Bank ROE',
                 fontsize=14, fontweight='bold', pad=15)
    for i, (_, row) in enumerate(feat_imp.iterrows()):
        ax.text(row['Importance'] + 0.005, i, f'{row["Importance"]:.3f}',
                va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] Feature importance chart saved")

    # Chart 2: Predicted vs Actual ROE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Scatter plot
    ax1.scatter(y, ols_pred, alpha=0.6, color='#2e75b6', s=50, label='OLS', edgecolors='white')
    ax1.scatter(y, ml_pred, alpha=0.6, color='#70ad47', s=50, label=model_name, edgecolors='white')
    min_val, max_val = min(y.min(), 0), max(y.max(), 30)
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
    ax1.set_xlabel('Actual ROE (%)', fontsize=12)
    ax1.set_ylabel('Predicted ROE (%)', fontsize=12)
    ax1.set_title('Predicted vs Actual ROE', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add R2 annotations
    ax1.text(0.05, 0.95, f'OLS R2 = {ols_model.rsquared:.3f}\n{model_name} R2 = {ml_r2:.3f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Residual plot
    ols_residuals = y - ols_pred
    ml_residuals = y - ml_pred
    ax2.scatter(ols_pred, ols_residuals, alpha=0.6, color='#2e75b6', s=50, label='OLS')
    ax2.scatter(ml_pred, ml_residuals, alpha=0.6, color='#70ad47', s=50, label=model_name)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Predicted ROE (%)', fontsize=12)
    ax2.set_ylabel('Residual (%)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Machine Learning Models - ROE Prediction Performance',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] Predicted vs Actual chart saved")

    # Chart 3: OLS Coefficient Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    coefs = pd.DataFrame({
        'Feature': ['Intercept'] + features,
        'Coefficient': ols_model.params,
        'Std Error': ols_model.bse,
        'p-value': ols_model.pvalues
    })
    coefs_no_int = coefs[coefs['Feature'] != 'Intercept'].sort_values('Coefficient')

    bar_colors = ['#70ad47' if p < 0.05 else '#c55a11' if p < 0.10 else '#999999'
                  for p in coefs_no_int['p-value']]
    ax.barh(coefs_no_int['Feature'], coefs_no_int['Coefficient'],
            xerr=coefs_no_int['Std Error'] * 1.96, color=bar_colors,
            edgecolor='white', height=0.5, capsize=3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Coefficient (with 95% CI)', fontsize=12)
    ax.set_title('OLS Regression Coefficients - Drivers of GCC Bank ROE',
                 fontsize=14, fontweight='bold', pad=15)

    # Legend for significance
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#70ad47', label='p < 0.05'),
                       Patch(facecolor='#c55a11', label='p < 0.10'),
                       Patch(facecolor='#999999', label='p  0.10')]
    ax.legend(handles=legend_elements, fontsize=10, title='Significance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ols_coefficients.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  [OK] OLS coefficient chart saved")

    # --- Save Results ---
    coefs.to_csv(os.path.join(output_dir, 'ols_coefficients.csv'), index=False)
    feat_imp.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

    # Save merged dataset with predictions
    model_data = model_data.copy()
    model_data['ROE_Predicted_OLS'] = ols_pred
    model_data['ROE_Predicted_ML'] = ml_pred
    model_data.to_csv(os.path.join(output_dir, 'ml_predictions.csv'), index=False)

    print("\n[OK] All ML results and charts saved")

    return ols_model, ml_model


if __name__ == '__main__':
    main()
