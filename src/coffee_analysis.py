#!/usr/bin/env python3
# Brewing-Insights: full Python port of your R workflow

from __future__ import annotations
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import pingouin as pg
from statsmodels.nonparametric.smoothers_lowess import lowess

# -------------------- paths --------------------
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "data" / "raw").resolve()
FIG_DIR  = (HERE / ".." / "reports" / "figures").resolve()
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- helpers --------------------
def load_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_csv(p)
    df.replace(-2147483648, np.nan, inplace=True)  # sentinel -> NaN
    return df

def trim_fix_country(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Country" in out.columns:
        out["Country"] = out["Country"].astype(str).str.strip()
        out["Country"] = out["Country"].replace({"Columbia": "Colombia"})
    return out

def scale_numeric(df: pd.DataFrame, factor: float = 1_000_000.0) -> pd.DataFrame:
    out = df.copy()
    num = out.select_dtypes(include=[np.number]).columns
    if len(num) > 0:
        out[num] = out[num] / factor
    return out

def ensure_total(df: pd.DataFrame, total_name: str) -> pd.DataFrame:
    out = df.copy()
    if total_name in out.columns:
        return out

    def is_yearish(c: str) -> bool:
        s = str(c)
        return bool(re.fullmatch(r"\d{4}", s) or re.fullmatch(r"X\d{4}", s) or re.fullmatch(r"\d{4}/\d{2}", s))

    year_cols = [c for c in out.columns if is_yearish(c)]
    if year_cols:
        # strip 'X' if present before summing
        tmp = out[year_cols].copy()
        tmp.columns = [str(c).replace("X", "") for c in tmp.columns]
        nums = tmp.select_dtypes(include=[np.number])
        out[total_name] = nums.sum(axis=1)
    else:
        nums = out.select_dtypes(include=[np.number])
        if not nums.empty:
            out[total_name] = nums.sum(axis=1)
    return out

def melt_years_numeric_or_X(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Melt numeric years (1990) and/or 'X1990' columns into (.., Year, value)."""
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c)) or re.fullmatch(r"X\d{4}", str(c))]
    if not year_cols:
        return pd.DataFrame(columns=[*df.columns, "Year", value_name])  # empty fallback
    id_vars = [c for c in df.columns if c not in year_cols]
    out = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="Year", value_name=value_name)
    out["Year"] = out["Year"].astype(str).str.replace("X", "", regex=False)
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce")
    out = out.dropna(subset=["Year"]).copy()
    out["Year"] = out["Year"].astype(int)
    return out

def melt_years_fy(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Melt '1990/91' style columns; Year = first 4 digits."""
    fy_cols = [c for c in df.columns if re.fullmatch(r"\d{4}/\d{2}", str(c))]
    if not fy_cols:
        return melt_years_numeric_or_X(df, value_name)
    id_vars = [c for c in df.columns if c not in fy_cols]
    out = df.melt(id_vars=id_vars, value_vars=fy_cols, var_name="FY", value_name=value_name)
    out["Year"] = out["FY"].astype(str).str.slice(0, 4).astype(int)
    return out

def summarize_col(df: pd.DataFrame, col: str, label: str):
    if col not in df.columns:
        print(f"[warn] {label}: '{col}' not found")
        return
    print(f"\nSummary — {label} ({col})")
    print(df[col].describe())

def save_boxplot_by_country(df: pd.DataFrame, y_col: str, title: str, fname: str):
    if "Country" not in df.columns or y_col not in df.columns:
        return
    plt.figure(figsize=(10, 5))
    df.boxplot(column=y_col, by="Country", rot=90)
    plt.title(title); plt.suptitle("")
    plt.xlabel("Country"); plt.ylabel(f"{y_col} (scaled)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close()

def save_hist(df: pd.DataFrame, col: str, title: str, xlabel: str, fname: str, binwidth: float | None = None):
    if col not in df.columns:
        return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    plt.figure(figsize=(7, 4))
    if binwidth:
        mn, mx = s.min(), s.max()
        start = np.floor(mn/binwidth)*binwidth
        end   = np.ceil(mx/binwidth)*binwidth + binwidth
        bins = np.arange(start, end, binwidth)
        plt.hist(s, bins=bins, edgecolor="black")
    else:
        plt.hist(s, bins=30, edgecolor="black")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close()

def save_line_by_country(df_long: pd.DataFrame, y_col: str, title: str, ylab: str, fname: str):
    need = {"Country","Year",y_col}
    if not need.issubset(df_long.columns):
        return
    plt.figure(figsize=(10,5))
    for c,g in df_long.sort_values(["Country","Year"]).groupby("Country"):
        plt.plot(g["Year"], g[y_col], marker="o", linewidth=1.2, label=c)
    plt.title(title); plt.xlabel("Year"); plt.ylabel(ylab)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close()

def jackknife_var_ratio(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05):
    X = np.asarray(X, float); X = X[~np.isnan(X)]
    Y = np.asarray(Y, float); Y = Y[~np.isnan(Y)]
    m, n = len(X), len(Y)
    if m < 3 or n < 3:
        return None
    S0 = np.log(np.var(X, ddof=1)); T0 = np.log(np.var(Y, ddof=1))
    Ai, Bj = [], []
    for i in range(m):
        Si = np.log(np.var(np.delete(X, i), ddof=1))
        Ai.append(m*S0 - (m-1)*Si)
    for j in range(n):
        Tj = np.log(np.var(np.delete(Y, j), ddof=1))
        Bj.append(n*T0 - (n-1)*Tj)
    Ai, Bj = np.array(Ai), np.array(Bj)
    Abar, Bbar = Ai.mean(), Bj.mean()
    V1 = ((Ai-Abar)**2).sum()/(m*(m-1)); V2 = ((Bj-Bbar)**2).sum()/(n*(n-1))
    Q = (Abar-Bbar)/np.sqrt(V1+V2)
    z2 = stats.norm.ppf(1 - alpha/2)
    ci = (np.exp((Abar-Bbar) - z2*np.sqrt(V1+V2)),
          np.exp((Abar-Bbar) + z2*np.sqrt(V1+V2)))
    return {"Q": float(Q), "est": float(np.exp(Abar-Bbar)), "ci": (float(ci[0]), float(ci[1]))}

def lowess_country(df_long: pd.DataFrame, country: str, y_col: str, fname: str, frac: float = 0.3):
    d = df_long[df_long["Country"] == country].sort_values("Year")
    if d.empty or y_col not in d.columns: return
    sm = lowess(d[y_col].values, d["Year"].values, frac=frac, return_sorted=True)
    plt.figure(figsize=(8,4))
    plt.plot(d["Year"], d[y_col], "o", label="observed")
    plt.plot(sm[:,0], sm[:,1], "-", label="LOWESS")
    plt.title(f"{country} — {y_col} (smoothed)")
    plt.xlabel("Year"); plt.ylabel(y_col)
    plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close()

# -------------------- load --------------------
f_imp  = DATA_DIR / "Coffee_import.csv"
f_exp  = DATA_DIR / "Coffee_export.csv"
f_prod = DATA_DIR / "Coffee_production.csv"

imp_raw  = load_csv(f_imp)
exp_raw  = load_csv(f_exp)
prod_raw = load_csv(f_prod)

# -------------------- wrangle: export --------------------
exp = trim_fix_country(exp_raw)
last_col_export = exp.columns[-1]
sorted_export = exp.copy()
sorted_export[last_col_export] = pd.to_numeric(sorted_export[last_col_export], errors="coerce")
sorted_export = sorted_export.sort_values(by=last_col_export, ascending=False)

export_countries = ["Brazil","Viet Nam","Colombia","Indonesia","India",
                    "Guatemala","Honduras","Uganda","Mexico","Peru","Ethiopia","Costa Rica"]
filtered_export = exp[exp["Country"].isin(export_countries)].copy()
filtered_export_scaled = scale_numeric(filtered_export)
filtered_export_scaled = ensure_total(filtered_export_scaled, "Total_export")

# long export
export_long = melt_years_numeric_or_X(filtered_export_scaled, "Export_Value")
print("export_long sample:\n", export_long.head())

# -------------------- wrangle: import --------------------
imp = trim_fix_country(imp_raw)
last_col_import = imp.columns[-1]
sorted_import = imp.copy()
sorted_import[last_col_import] = pd.to_numeric(sorted_import[last_col_import], errors="coerce")
sorted_import = sorted_import.sort_values(by=last_col_import, ascending=False)

import_countries = ["United States of America","Germany","Italy","Japan","France","Spain",
                    "United Kingdom","Belgium","Netherlands","Russian Federation","Poland","Switzerland"]
filtered_import = imp[imp["Country"].isin(import_countries)].copy()
filtered_import_scaled = scale_numeric(filtered_import)
filtered_import_scaled = ensure_total(filtered_import_scaled, "Total_import")

# long import
import_long = melt_years_numeric_or_X(filtered_import_scaled, "Import_Value")
print("import_long sample:\n", import_long.head())

# -------------------- wrangle: production --------------------
prod = trim_fix_country(prod_raw)
last_col_prod = prod.columns[-1]
sorted_prod = prod.copy()
sorted_prod[last_col_prod] = pd.to_numeric(sorted_prod[last_col_prod], errors="coerce")
sorted_prod = sorted_prod.sort_values(by=last_col_prod, ascending=False)

production_countries = ["Brazil","Viet Nam","Colombia","Indonesia","Ethiopia","India",
                        "Mexico","Guatemala","Honduras","Uganda","Peru"]
filtered_production = prod[prod["Country"].isin(production_countries)].copy()
filtered_production_scaled = scale_numeric(filtered_production)
filtered_production_scaled = ensure_total(filtered_production_scaled, "Total_production")

# long production (handles 1990/91)
production_long = melt_years_fy(filtered_production_scaled, "Production_Value")
print("production_long sample:\n", production_long.head())

# -------------------- EDA summaries --------------------
summarize_col(filtered_export_scaled,     "Total_export",     "Exports")
summarize_col(filtered_import_scaled,     "Total_import",     "Imports")
summarize_col(filtered_production_scaled, "Total_production", "Production")

# -------------------- Boxplots --------------------
save_boxplot_by_country(filtered_import_scaled,     "Total_import",     "Boxplot of Total Imports by Country",     "imports_boxplot.png")
save_boxplot_by_country(filtered_export_scaled,     "Total_export",     "Boxplot of Total Exports by Country",     "exports_boxplot.png")
save_boxplot_by_country(filtered_production_scaled, "Total_production", "Boxplot of Total Production by Country",  "production_boxplot.png")

# -------------------- Line plots --------------------
save_line_by_country(import_long,     "Import_Value",    "Coffee Imports Over Time",  "Million Kgs of Coffee Imported",  "imports_over_time.png")
save_line_by_country(export_long,     "Export_Value",    "Coffee Exports Over Time",  "Million Kgs of Coffee Exported",  "exports_over_time.png")

# -------------------- Histograms --------------------
save_hist(import_long,     "Import_Value",     "Histogram of Imports",     "Import Values",     "hist_imports.png",  binwidth=30)
save_hist(export_long,     "Export_Value",     "Histogram of Exports",     "Export Values",     "hist_exports.png")
save_hist(production_long, "Production_Value", "Histogram of Production",  "Production Values", "hist_production.png")

# -------------------- Friedman (nonparametric, imports by country, blocks=year) --------------------
print("\n=== Friedman test (imports by country; blocks=Year) ===")
imp_wide = import_long.pivot_table(index="Year", columns="Country", values="Import_Value").dropna()
if imp_wide.shape[1] >= 3 and imp_wide.shape[0] >= 2:
    fried_stat, fried_p = stats.friedmanchisquare(*[imp_wide[c].values for c in imp_wide.columns])
    print(f"Friedman: stat={fried_stat:.3f}, p={fried_p:.4g}")
    # Nemenyi post-hoc (replacement for NSM3::pWNMT follow-up)
    nemenyi = sp.posthoc_nemenyi_friedman(imp_wide)
    print("Nemenyi (head):\n", nemenyi.head())
else:
    print("Skipped (not enough complete blocks).")

# -------------------- ANOVA (parametric) + Tukey + LSD-style --------------------
# --- One-way ANOVA (parametric) + Tukey + LSD-style pairwise ---
print("\n=== One-way ANOVA (imports by country; excluding Germany & USA) ===")
imp_minus2 = import_long[~import_long["Country"].isin(["Germany","United States of America"])].copy()
if not imp_minus2.empty:
    model = smf.ols("Import_Value ~ C(Country)", data=imp_minus2).fit()
    print(anova_lm(model, typ=2))

    # Tukey (alpha=0.01 to mirror your R)
    try:
        tuk = pairwise_tukeyhsd(endog=imp_minus2["Import_Value"],
                                groups=imp_minus2["Country"], alpha=0.01)
        print("\nTukey HSD (alpha=0.01):\n", tuk.summary())
    except Exception as e:
        print("Tukey failed:", e)

    # LSD-style (uncorrected pairwise t-tests) using the new Pingouin API
    try:
        # pairwise_tests is the maintained function; set padjust=None for "LSD-style"
        lsd = pg.pairwise_tests(dv="Import_Value", between="Country",
                                data=imp_minus2, parametric=True,
                                padjust=None, effsize="cohen")

        # Normalize column name for effect size
        if "effsize" in lsd.columns and "cohen-d" not in lsd.columns:
            lsd = lsd.rename(columns={"effsize": "cohen-d"})

        cols = [c for c in ["A", "B", "T", "dof", "p-unc", "cohen-d"] if c in lsd.columns]
        print("\nLSD-style pairwise (head):\n", lsd[cols].head(15))

        # (optional) save a CSV
        TABLE_DIR = (FIG_DIR.parent / "tables")
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        lsd.to_csv(TABLE_DIR / "anova_lsd_pairwise.csv", index=False)
    except Exception as e:
        print("LSD-style pairwise failed:", e)
else:
    print("Skipped (no data after excluding Germany & USA).")

# -------------------- KS test (exports vs production, pooled numerics) --------------------
print("\n=== KS test: exports vs production (pooled) ===")
exp_vals  = filtered_export_scaled.select_dtypes(np.number).to_numpy().ravel()
prod_vals = filtered_production_scaled.select_dtypes(np.number).to_numpy().ravel()
exp_vals  = exp_vals[~np.isnan(exp_vals)]
prod_vals = prod_vals[~np.isnan(prod_vals)]
if len(exp_vals) > 0 and len(prod_vals) > 0:
    ks_stat, ks_p = stats.ks_2samp(exp_vals, prod_vals)
    print(f"KS: stat={ks_stat:.4f}, p={ks_p:.4g}")
else:
    print("Skipped (insufficient numeric data).")

# -------------------- Jackknife variance ratio --------------------
print("\n=== Jackknife variance ratio (exports vs production) ===")
jk = jackknife_var_ratio(exp_vals, prod_vals, alpha=0.05)
print(jk if jk else "Skipped (not enough data)")

# -------------------- Levene’s test (variance homogeneity) --------------------
print("\n=== Levene’s test (production vs exports) ===")
if len(exp_vals) > 1 and len(prod_vals) > 1:
    lev_stat, lev_p = stats.levene(prod_vals, exp_vals, center="mean")
    print(f"Levene: stat={lev_stat:.4f}, p={lev_p:.4g}")
else:
    print("Skipped.")

# -------------------- Correlations (country totals) --------------------
print("\n=== Correlations: Total_production vs Total_export (by country) ===")
subset_export = filtered_export_scaled[["Country","Total_export"]].dropna()
subset_prod   = filtered_production_scaled[["Country","Total_production"]].dropna()
merged = subset_export.merge(subset_prod, on="Country", how="inner")
if not merged.empty:
    tau, tau_p = stats.kendalltau(merged["Total_production"], merged["Total_export"])
    r, r_p     = stats.pearsonr(merged["Total_production"], merged["Total_export"])
    print(f"Kendall tau={tau:.4f}, p={tau_p:.4g}")
    print(f"Pearson r={r:.4f}, p={r_p:.4g}")
else:
    print("No overlap after merge.")

# -------------------- Smoothing (LOWESS; supsmu analogue) --------------------
print("\n=== LOWESS smoothing plots ===")
lowess_country(import_long, "United States of America", "Import_Value", "lowess_usa_imports.png")
lowess_country(import_long, "Belgium",                    "Import_Value", "lowess_belgium_imports.png")

print(f"\nFigures saved to: {FIG_DIR.resolve()}")
