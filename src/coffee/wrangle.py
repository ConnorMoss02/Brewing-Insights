from __future__ import annotations
import re
import numpy as np
import pandas as pd

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
    """Create a total column by summing year columns if it doesn’t exist."""
    out = df.copy()
    if total_name in out.columns:
        return out

    def is_yearish(c: str) -> bool:
        s = str(c)
        return bool(
            re.fullmatch(r"\d{4}", s) or
            re.fullmatch(r"X\d{4}", s) or
            re.fullmatch(r"\d{4}/\d{2}", s)
        )

    year_cols = [c for c in out.columns if is_yearish(c)]
    if year_cols:
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
    """Melt numeric years (1990) and/or 'X1990' columns."""
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c)) or re.fullmatch(r"X\d{4}", str(c))]
    if not year_cols:
        return pd.DataFrame(columns=[*df.columns, "Year", value_name])
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
