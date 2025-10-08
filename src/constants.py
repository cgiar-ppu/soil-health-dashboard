from pathlib import Path

# Resolve project root as the parent of this src directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "Input"
ASSETS_DIR = PROJECT_DIR / "assets"

# File patterns we care about
DATA_FILE_PATTERNS = ("*.csv", "*.xlsx", "*.xls")

# Canonical input file names used by data_loader
MAIN_DATASET_FILENAME = "2025-10-07T13-26_export_SoilHealthClusters.csv"
CLUSTER_KEYWORDS_FILENAME = "2025-10-07T13-25_export_SoilHealth_ClusterKeywords.csv"
CLUSTER_SUMMARIES_DOCX = "Soil Health Cluster Summaries.docx"

# Full paths for convenience (use only for reading; keep paths relative to project root)
MAIN_DATASET_PATH = INPUT_DIR / MAIN_DATASET_FILENAME
CLUSTER_KEYWORDS_PATH = INPUT_DIR / CLUSTER_KEYWORDS_FILENAME
CLUSTER_SUMMARIES_PATH = INPUT_DIR / CLUSTER_SUMMARIES_DOCX
