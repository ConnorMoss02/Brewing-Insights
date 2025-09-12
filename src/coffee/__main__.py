#!/usr/bin/env python3
from __future__ import annotations
import warnings
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, FIG_DIR
from .io import load_csv
from .wrangle import (
    trim_fix_country, scale_numeric, ensure_total,
    melt_years_numeric_or_X, melt_years_fy
)
from .plots import (
    save_boxplot_by_country, save_hist, save_line_by_country, lowess_country
)
from .stats_mod import (
    summarize_col, friedman_nemenyi, anova_tukey_lsd, ks_exports_vs_production,
    jackknife_var_ratio, levene_prod_vs_exp, correlations_country_totals
)

# Optional: quiet noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def main():
    # ---- files ----
    f_imp  = DATA_DIR / "Coffee_import.csv"
    f_exp  = DATA_DIR / "Coffee_export.csv"
    f_prod = DATA_DIR / "Coffee_production.csv"

    # ---- load ----
    imp_raw  = load_csv(f_imp)
    exp_raw  = load_csv(f_exp)
    prod_raw = load_csv(f_prod)

    # ---- wrangle: export ----
    exp = trim_fix_country(exp_raw)
    export_countries = ["Brazil","Viet Nam","Colombia","Indonesia","India",
                        "Guatemala","Honduras","Uganda","Mexico","Peru","Ethiopia","Costa Rica"]
    filtered_export = exp[exp["Country"].isin(export_countries)].copy()
    filtered_export_scaled = ensure_total(scale_numeric(filtered_export), "Total_export")
    export_long = melt_years_numeric_or_X(filtered_export_scaled, "Export_Value")
    print("export_long sample:\n", export_long.head())

    # ---- wrangle: import ----
    imp = trim_fix_country(imp_raw)
    import_countries = ["United States of America","Germany","Italy","Japan","France","Spain",
                        "United Kingdom","Belgium","Netherlands","Russian Federation","Poland","Switzerland"]
    filtered_import = imp[imp["Country"].isin(import_countries)].copy()
    filtered_import_scaled = ensure_total(scale_numeric(filtered_import), "Total_import")
    import_long = melt_years_numeric_or_X(filtered_import_scaled, "Import_Value")
    print("import_long sample:\n", import_long.head())

    # ---- wrangle: production ----
    prod = trim_fix_country(prod_raw)
    production_countries = ["Brazil","Viet Nam","Colombia","Indonesia","Ethiopia","India",
                            "Mexico","Guatemala","Honduras","Uganda","Peru"]
    filtered_production = prod[prod["Country"].isin(production_countries)].copy()
    filtered_production_scaled = ensure_total(scale_numeric(filtered_production), "Total_production")
    production_long = melt_years_fy(filtered_production_scaled, "Production_Value")
    print("production_long sample:\n", production_long.head())

    # ---- summaries ----
    summarize_col(filtered_export_scaled,     "Total_export",     "Exports")
    summarize_col(filtered_import_scaled,     "Total_import",     "Imports")
    summarize_col(filtered_production_scaled, "Total_production", "Production")

    # ---- boxplots ----
    save_boxplot_by_country(filtered_import_scaled,     "Total_import",     "Boxplot of Total Imports by Country",     "imports_boxplot.png")
    save_boxplot_by_country(filtered_export_scaled,     "Total_export",     "Boxplot of Total Exports by Country",     "exports_boxplot.png")
    save_boxplot_by_country(filtered_production_scaled, "Total_production", "Boxplot of Total Production by Country",  "production_boxplot.png")

    # ---- lines ----
    save_line_by_country(import_long, "Import_Value", "Coffee Imports Over Time", "Million Kgs of Coffee Imported", "imports_over_time.png")
    save_line_by_country(export_long, "Export_Value", "Coffee Exports Over Time", "Million Kgs of Coffee Exported", "exports_over_time.png")

    # ---- histograms ----
    save_hist(import_long,     "Import_Value",     "Histogram of Imports",     "Import Values",     "hist_imports.png",  binwidth=30)
    save_hist(export_long,     "Export_Value",     "Histogram of Exports",     "Export Values",     "hist_exports.png")
    save_hist(production_long, "Production_Value", "Histogram of Production",  "Production Values", "hist_production.png")

    # ---- stats ----
    friedman_nemenyi(import_long)
    anova_tukey_lsd(import_long)

    exp_vals, prod_vals = ks_exports_vs_production(filtered_export_scaled, filtered_production_scaled)

    print("\n=== Jackknife variance ratio (exports vs production) ===")
    jk = jackknife_var_ratio(exp_vals, prod_vals, alpha=0.05)
    print(jk if jk else "Skipped (not enough data)")

    levene_prod_vs_exp(prod_vals, exp_vals)
    correlations_country_totals(filtered_export_scaled, filtered_production_scaled)

    # ---- smoothing ----
    print("\n=== LOWESS smoothing plots ===")
    lowess_country(import_long, "United States of America", "Import_Value", "lowess_usa_imports.png")
    lowess_country(import_long, "Belgium",                    "Import_Value", "lowess_belgium_imports.png")

    print(f"\nFigures saved to: {FIG_DIR.resolve()}")

if __name__ == "__main__":
    main()
