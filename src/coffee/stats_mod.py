from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from .config import TABLE_DIR

# Optional deps
try:
    import scikit_posthocs as sp
    HAS_SCPH = True
except Exception:
    HAS_SCPH = False

try:
    import pingouin as pg
    HAS_PG = True
except Exception:
    HAS_PG = False

def summarize_col(df: pd.DataFrame, col: str, label: str):
    if col not in df.columns:
        print(f"[warn] {label}: '{col}' not found")
        return
    print(f"\nSummary — {label} ({col})")
    print(df[col].describe())

def friedman_nemenyi(import_long: pd.DataFrame):
    print("\n=== Friedman test (imports by country; blocks=Year) ===")
    imp_wide = import_long.pivot_table(index="Year", columns="Country", values="Import_Value").dropna()
    if imp_wide.shape[1] >= 3 and imp_wide.shape[0] >= 2:
        fried_stat, fried_p = stats.friedmanchisquare(*[imp_wide[c].values for c in imp_wide.columns])
        print(f"Friedman: stat={fried_stat:.3f}, p={fried_p:.4g}")
        if HAS_SCPH:
            nemenyi = sp.posthoc_nemenyi_friedman(imp_wide)
            print("Nemenyi (head):\n", nemenyi.head())
            nemenyi.to_csv(TABLE_DIR / "nemenyi_imports.csv")
        else:
            print("Nemenyi skipped (install scikit-posthocs).")
    else:
        print("Skipped (not enough complete blocks).")

def anova_tukey_lsd(import_long: pd.DataFrame):
    print("\n=== One-way ANOVA (imports by country; excluding Germany & USA) ===")
    imp_minus2 = import_long[~import_long["Country"].isin(["Germany","United States of America"])].copy()
    if imp_minus2.empty:
        print("Skipped (no data after exclusion).")
        return
    model = smf.ols("Import_Value ~ C(Country)", data=imp_minus2).fit()
    print(anova_lm(model, typ=2))

    try:
        tuk = pairwise_tukeyhsd(endog=imp_minus2["Import_Value"],
                                groups=imp_minus2["Country"], alpha=0.01)
        print("\nTukey HSD (alpha=0.01):\n", tuk.summary())
    except Exception as e:
        print("Tukey failed:", e)

    if HAS_PG:
        try:
            lsd = pg.pairwise_tests(dv="Import_Value", between="Country",
                                    data=imp_minus2, parametric=True,
                                    padjust=None, effsize="cohen")
            if "effsize" in lsd.columns and "cohen-d" not in lsd.columns:
                lsd = lsd.rename(columns={"effsize": "cohen-d"})
            cols = [c for c in ["A","B","T","dof","p-unc","cohen-d"] if c in lsd.columns]
            print("\nLSD-style pairwise (head):\n", lsd[cols].head(15))
            lsd.to_csv(TABLE_DIR / "anova_lsd_pairwise.csv", index=False)
        except Exception as e:
            print("LSD-style pairwise failed:", e)
    else:
        print("LSD-style pairwise skipped (install pingouin).")

def ks_exports_vs_production(exp_scaled: pd.DataFrame, prod_scaled: pd.DataFrame):
    print("\n=== KS test: exports vs production (pooled) ===")
    exp_vals  = exp_scaled.select_dtypes(np.number).to_numpy().ravel()
    prod_vals = prod_scaled.select_dtypes(np.number).to_numpy().ravel()
    exp_vals  = exp_vals[~np.isnan(exp_vals)]
    prod_vals = prod_vals[~np.isnan(prod_vals)]
    if len(exp_vals) == 0 or len(prod_vals) == 0:
        print("Skipped (insufficient numeric data).")
        return exp_vals, prod_vals
    ks_stat, ks_p = stats.ks_2samp(exp_vals, prod_vals)
    print(f"KS: stat={ks_stat:.4f}, p={ks_p:.4g}")
    return exp_vals, prod_vals

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

def levene_prod_vs_exp(prod_vals, exp_vals):
    print("\n=== Levene’s test (production vs exports) ===")
    if len(exp_vals) > 1 and len(prod_vals) > 1:
        lev_stat, lev_p = stats.levene(prod_vals, exp_vals, center="mean")
        print(f"Levene: stat={lev_stat:.4f}, p={lev_p:.4g}")
    else:
        print("Skipped.")

def correlations_country_totals(exp_scaled: pd.DataFrame, prod_scaled: pd.DataFrame):
    print("\n=== Correlations: Total_production vs Total_export (by country) ===")
    subset_export = exp_scaled[["Country","Total_export"]].dropna()
    subset_prod   = prod_scaled[["Country","Total_production"]].dropna()
    merged = subset_export.merge(subset_prod, on="Country", how="inner")
    if merged.empty:
        print("No overlap after merge."); return
    tau, tau_p = stats.kendalltau(merged["Total_production"], merged["Total_export"])
    r, r_p     = stats.pearsonr(merged["Total_production"], merged["Total_export"])
    print(f"Kendall tau={tau:.4f}, p={tau_p:.4g}")
    print(f"Pearson r={r:.4f}, p={r_p:.4g}")
