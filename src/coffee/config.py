from __future__ import annotations
from pathlib import Path

# Paths
PKG_DIR = Path(__file__).resolve().parent
ROOT    = PKG_DIR.parent.parent  # repo root (…/Brewing-Insights)
DATA_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
FIG_DIR  = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"

# Ensure output dirs exist
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
