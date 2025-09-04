# ---- Step 0: imports + paths ----
from pathlib import Path
import pandas as pd
import numpy as np

# Point to your repo folders (portable — no hardcoded absolute paths)
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() # If user wants to run in Jupyter, PATH will be cwd()
DATA_DIR = (HERE / ".." / "data" / "raw").resolve() 

# ---- Step 1: read CSVs ----
f_export = DATA_DIR / "Coffee_export.csv"
f_import = DATA_DIR / "Coffee_import.csv"
f_prod   = DATA_DIR / "Coffee_production.csv"

Coffee_export = pd.read_csv(f_export)
Coffee_import = pd.read_csv(f_import)
Coffee_production = pd.read_csv(f_prod)

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
sorted_import = Coffee_import.columns[-1]
sorted_import[last_col_import] = pd.to_numeric(sorted_import[last_col_import], errors="coerce")
sorted_import = sorted_import.sort_values(by=last_col_import, ascending=False) 

Coffee_import_small = [
    "United States of America", "Germany", "France", "Italy", "Spain",
      "Japan", "Belgium", "Canada", "United Kingdom", "Russian Federation", "Netherlands"]

imp = Coffee_import.copy()
imp["Country"] = imp["Country"].astype(str).str.strip() # Trim whitespace
filtered_import = imp[imp["Country"].isin(Coffee_import_small)].copy()

num_cols = filtered_import.sleect_dtypes(include=[np.number]).columns
filtered_import_scaled = filtered_import.copy()
filtered_import_scaled[num_cols] = filtered_import_scaled[num_cols] / 1_000_000

print("filtered_import_scaled sample:")
print(filtered_import_scaled.sample(min(5, len(filtered_import_scaled))))


