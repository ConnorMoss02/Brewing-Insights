Brewing-Insights

Statistical exploration of global coffee imports, exports, and production.
Loads the datasets, cleans them, runs non-parametric & parametric tests, and saves figures/tables.

Quick start (2–3 minutes)

Fork this repo (top-right on GitHub) and then clone your fork locally:

git clone https://github.com/<your-username>/Brewing-Insights.git
cd Brewing-Insights

Create & activate a virtual environment (recommended):

python -m venv .venv

# macOS/Linux

source .venv/bin/activate

# Windows (PowerShell)

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Add data files (CSV) to data/raw/ with these exact filenames:

Coffee_export.csv
Coffee_import.csv
Coffee_production.csv

(You can use your own CSVs—just keep the same column layout and filenames.)

Run the analysis:

python -m coffee

What you get

Figures saved to: reports/figures/

e.g., imports_boxplot.png, exports_over_time.png, hist_production.png, lowess_usa_imports.png, …

Result tables saved to: reports/tables/

e.g., anova_lsd_pairwise.csv

Repo structure
Brewing-Insights/
├─ data/
│ └─ raw/ # Place CSVs here
├─ docs/
│ └─ Coffee_Project.pdf # Project paper
├─ reports/
│ ├─ figures/ # Plots written here
│ └─ tables/ # CSV result tables here
├─ src/
│ └─ coffee/
│ ├─ **main**.py # Entry point for `python -m coffee`
│ ├─ **init**.py # Package marker
│ ├─ config.py # Paths (data, figures, tables)
│ ├─ io.py # CSV loading helpers
│ ├─ wrangle.py # Cleaning, totals, long-format melts
│ ├─ plots.py # Boxplots, histograms, lines, LOWESS
│ └─ stats_mod.py # Friedman+Nemenyi, ANOVA+Tukey, etc.
├─ .gitignore
├─ requirements.txt
└─ README.md

Requirements

Python 3.9+

Packages listed in requirements.txt (installed in step 3)

Troubleshooting

ModuleNotFoundError → Make sure your venv is activated and you ran pip install -r requirements.txt.

FileNotFoundError → Check that your CSVs are in data/raw/ and named exactly:

Coffee_export.csv, Coffee_import.csv, Coffee_production.csv

Plots don’t show on screen → That’s expected; the script saves figures to reports/figures/. Open them from there.
