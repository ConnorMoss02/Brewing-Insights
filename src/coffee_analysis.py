# ---- Step 0: imports + paths ----
from pathlib import Path
import pandas as pd
import numpy as np
import re 
import matplotlib.pyplot as plt

# Point to repo folders (portable — no hardcoded absolute paths)
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() # If user wants to run in Jupyter, PATH will be cwd()
DATA_DIR = (HERE / ".." / "data" / "raw").resolve() 

# ---- Step 1: read CSVs ----
f_export = DATA_DIR / "Coffee_export.csv"
f_import = DATA_DIR / "Coffee_import.csv"
f_prod   = DATA_DIR / "Coffee_production.csv"

Coffee_export = pd.read_csv(f_export)
Coffee_import = pd.read_csv(f_import)
Coffee_production = pd.read_csv(f_prod)

# Replace sentinel values with NaN
for df in (Coffee_export, Coffee_import, Coffee_production):
    df.replace(-2147483648, np.nan, inplace=True)

print(Coffee_export.head(3))
print(Coffee_import.head(3))
print(Coffee_production.head(3))

# ---- Step 2: Data Wrangling Export ----
last_col_export = Coffee_export.columns[-1] # Last column is where the total export data exists

sorted_export = Coffee_export.copy() # Make a copy to sort by most recent year
sorted_export[last_col_export] = pd.to_numeric(sorted_export[last_col_export], errors="coerce")
sorted_export = sorted_export.sort_values(by=last_col_export, ascending=False)

Coffee_export_small = [
    "Brazil", "Viet Nam", "Columbia", "Colombia", "Indonesia", "India",
    "Guatemala", "Honduras", "Uganda", "Mexico", "Peru", "Ethiopia", "Costa Rica",
]

# Clean up Country and fix the "Columbia" -> "Colombia" issue
export = Coffee_export.copy()
export["Country"] = (
    export["Country"]
    .astype(str)
    .str.strip()
    .replace({"Columbia": "Colombia"})
)

filtered_export = export[export["Country"].isin(Coffee_export_small)].copy()

num_cols = filtered_export.select_dtypes(include=[np.number]).columns # Prep numeric columns to be scaled for accurate comparison
filtered_export_scaled = filtered_export.copy()
filtered_export_scaled[num_cols] = filtered_export_scaled[num_cols] / 1_000_000

# "View" equivalent
print("filtered_export_scaled sample:")
print(filtered_export_scaled.sample(min(5, len(filtered_export_scaled))))

# ---- Step 3: Data Wrangling Import ----
last_col_import = Coffee_import.columns[-1]

# make a DF copy (not a string!)
sorted_import = Coffee_import.copy()
sorted_import[last_col_import] = pd.to_numeric(sorted_import[last_col_import], errors="coerce")
sorted_import = sorted_import.sort_values(by=last_col_import, ascending=False)

Coffee_import_small = [
    "United States of America", "Germany", "France", "Italy", "Spain",
    "Japan", "Belgium", "Canada", "United Kingdom", "Russian Federation", "Netherlands",
]

imp = Coffee_import.copy()
imp["Country"] = imp["Country"].astype(str).str.strip()  # Trim whitespace

filtered_import = imp[imp["Country"].isin(Coffee_import_small)].copy()

# typo fixed: sleect_dtypes -> select_dtypes
num_cols = filtered_import.select_dtypes(include=[np.number]).columns
filtered_import_scaled = filtered_import.copy()
filtered_import_scaled[num_cols] = filtered_import_scaled[num_cols] / 1_000_000

print("filtered_import_scaled sample:")
print(filtered_import_scaled.sample(min(5, len(filtered_import_scaled))))

# ---- Step 4: DW Production ---- 

last_col_prod = Coffee_production.columns[-1]
sorted_consumption = Coffee_production.copy()
sorted_consumption[last_col_prod] = pd.to_numeric(sorted_consumption[last_col_prod], errors="coerce")
sorted_consumption = sorted_consumption.sort_values(by=last_col_prod, ascending=False)

Coffee_production_small = [
    "Brazil", "Viet Nam", "Colombia", "Indonesia", "Ethiopia", "India",
    "Mexico", "Guatemala", "Honduras", "Uganda", "Peru",
]

prod = Coffee_production.copy()
prod["Country"] = prod["Country"].astype(str).str.strip()
filtered_production = prod[prod["Country"].isin(Coffee_production_small)].copy()

num_cols = filtered_production.select_dtypes(include=[np.number]).columns
filtered_production_scaled = filtered_production.copy()
filtered_production_scaled[num_cols] = filtered_production_scaled[num_cols] / 1_000_000

print("filtered_production_scaled sample:")
print(filtered_production_scaled.sample(min(5, len(filtered_production_scaled))))

year_cols = [
    c for c in filtered_production_scaled.columns
    if re.fullmatch(r"\d{4}(?:/\d{2})?", str(c))
]

id_vars = [c for c in filtered_production_scaled.columns if c not in year_cols]

filtered_production_long = filtered_production_scaled.melt(
    id_vars=id_vars,
    value_vars=year_cols,
    var_name="Year",
    value_name="Value",
)

# Convert "1990/91" -> 1990 (take the first 4 digits as the season start year)
filtered_production_long["Year"] = (
    filtered_production_long["Year"]
    .astype(str)
    .str.extract(r"^(\d{4})")  # grab the first 4 digits
)

# Make Year numeric Int64 and drop rows that failed to parse
filtered_production_long["Year"] = pd.to_numeric(filtered_production_long["Year"], errors="coerce").astype("Int64")
filtered_production_long = filtered_production_long.dropna(subset=["Year"]).copy()

print("filtered_production_long sample:")
print(filtered_production_long.head(10))

# --- EDA Summaries ---
def summarize_col(df, colname: str, label: str):
    if colname not in df.columns:
        raise KeyError(f"Column '{colname}' not found in {label}")
    print(f"\nSummary - {label} ({colname})")
    print(df[colname].describe())

summarize_col(filtered_export_scaled, "Total_export", "Exports")
summarize_col(filtered_import_scaled, "Total_import", "Imports")
summarize_col(filtered_production_scaled, "Total_production", "Production")

# --- Visual EDA ---
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def boxplot_by_country(df, y_col, title, fname):
    if "Country" not in df.columns or y_col not in df.columns:
        raise KeyError("Expected columns: 'Country and the y_col you pass in")
    plt.figure(figsize=(10, 5))

    df.boxplot(column=y_col, by="Country", rot=90)
    plt.title(title)
    plt.suptitle("")
    plt.xlabel("Country")
    plt.ylabel(f"{y_col} (scaled)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi = 200, bbox_inches = "tight")
    plt.close()

boxplot_by_country(filtered_import_scaled, "Total_import", "Boxplot of Total Imports by Country", "imports_boxplot.png")
