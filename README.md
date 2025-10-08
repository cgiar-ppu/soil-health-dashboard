CGIAR Soil Health Explorer

Purpose
- Interactive Streamlit app to explore CGIAR Soil Health datasets: filter, visualize, and download records.
- Loads canonical inputs from the project Input directory and provides topic/cluster context when available.

Quick start
- Prerequisites: Python 3.9+
- Install dependencies: pip install -r requirements.txt
- Place input files under the project Input directory: ./Input
  - Required main dataset: ./Input/2025-10-07T13-26_export_SoilHealthClusters.csv
  - Optional cluster keywords: ./Input/2025-10-07T13-25_export_SoilHealth_ClusterKeywords.csv
  - Optional cluster summaries (DOCX): ./Input/Soil Health Cluster Summaries.docx
- Run the app from the project root: streamlit run streamlit_app.py

What the app expects
- Formats: CSV (.csv) and Excel (.xlsx, .xls)
- Columns are used opportunistically. If a column is missing, related UI elements gracefully degrade.
  - Text search over Title and Description (if present)
  - Year is derived from a Year column or any date-like column (publish/issued/created/start)
  - Multi-valued fields (e.g., Country, Partner) may be comma/semicolon separated
  - Common link columns made clickable in tables: PDF link, Evidence 1, URL, Link
  - Topic IDs (numeric) can be enriched with Cluster Name and Cluster Keywords from the optional keywords file

Project layout (relative to project root)
- streamlit_app.py — Streamlit entry point
- src/
  - constants.py — Paths and canonical input filenames used by the app
  - data_loader.py — Input discovery and loaders for main, keywords, and DOCX summaries
  - data_processing.py — Year derivation, text search, explode helpers for countries/partners
  - viz.py — Plotly-based pies, bars, time series, and world map
- assets/styles.css — Optional custom styles applied if present
- Input/ — Place data files here (see “What the app expects”)
- docs/USAGE.md — Usage guide (filters, tabs, chart builder, visual interpretation)
- tests/ — Basic unit tests for core functions
- requirements.txt — Python dependencies

Notes
- Canonical filenames are defined in ./src/constants.py. If the required main dataset is missing, the app will try to load the first file found under ./Input as a fallback.
- All paths above are relative to the project root.
