from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import FIG_DIR
from statsmodels.nonparametric.smoothers_lowess import lowess

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
