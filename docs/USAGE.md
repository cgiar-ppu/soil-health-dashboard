USAGE — CGIAR Soil Health Explorer

Data and file expectations
- Place all inputs under the project Input directory: ./Input
  - Required main dataset: ./Input/2025-10-07T13-26_export_SoilHealthClusters.csv
  - Optional cluster keywords: ./Input/2025-10-07T13-25_export_SoilHealth_ClusterKeywords.csv
  - Optional cluster summaries (DOCX): ./Input/Soil Health Cluster Summaries.docx
- Supported formats: CSV (.csv), Excel (.xlsx, .xls)
- Columns are not strictly required; the app adapts:
  - Title, Description (for text search)
  - Year (or any date-like column; the app derives Year when possible)
  - Country/Countries (comma/semicolon-delimited supported)
  - Partner/Partners (comma/semicolon-delimited supported)
  - Type (also tries variants like Output Type, Document Type)
  - Submitter (also tries variants like Submitted By, Submitting Organization)
  - Topic (numeric topic/cluster id). If provided, it can be mapped to Cluster Name and Cluster Keywords using the optional keywords file.

Running locally
- From the project root: streamlit run streamlit_app.py
- Dependencies: pip install -r requirements.txt (Python 3.9+)

Global filters (sidebar)
- Search Title/Description: case-insensitive substring match.
- Year range: shown if Year is derivable; filters records between selected years.
- Countries: multi-select based on exploded Country values.
- Topic / Cluster: multi-select with labels like “Topic N – Cluster Name” when available.
- Type: multi-select using the best-matching type column.
- Submitter: multi-select using the best-matching submitter column.
- Partners: multi-select based on exploded Partner values.
- Reset filters: clears all selections in the current session.

Tabs and features
- Overview
  - KPIs: Total results, Unique countries, Unique partners.
  - Pies: Distribution by Type and by Topic (labels include Cluster Name when available).
- Trends
  - Choose a time field (Year or other detected date columns).
  - Area charts: results over time by Topic and by Type.
- Geography
  - World choropleth by Country (count of records). Top countries bar on the side.
- Clusters
  - Topic catalog with titles like “Topic N – Cluster Name,” keyword chips, and optional per-topic summaries (from DOCX if installed and present).
  - Expander: within each topic, view sample records and a Type-over-time chart.
- Partners
  - Partner frequency bar and a stacked bar of Partners by Topic (if Topic present). Searchable partner table below.
- Builder (Chart Builder)
  - Pick X category, Y aggregation (Count or Sum of a numeric column), optional color (stack), and Top N. Renders a configurable bar.
- Data
  - Paginated table with clickable links for common link fields. Download filtered data as CSV or JSON.

Chart builder controls (details)
- X (category): string or small-cardinality integer columns.
- Y aggregation:
  - Count: number of rows per category.
  - Sum of column: sums a chosen numeric column per category.
- Stack (color): optional second dimension for stacked bars.
- Top N: keep the top categories by total.

How to interpret key visuals
- Distribution pies: show share of records by category (Type or Topic). Use for composition at-a-glance.
- Trends area charts: show volume over time. Peaks indicate periods of higher activity.
- World map: intensity reflects count of records per country. Use alongside the Top countries bar to identify hotspots.
- Partner charts: identify most active partners and their association with topics.

Troubleshooting
- No data shown: ensure at least one file is in ./Input. The app expects the main dataset filename; otherwise, it will try the first file it finds.
- Missing columns: related filters or charts may be hidden; this is expected.
- DOCX summaries not parsed: ensure python-docx is installed (in requirements) and the file exists at ./Input/Soil Health Cluster Summaries.docx.
- Links not clickable: only certain columns are treated as links (PDF link, Evidence 1, URL, Link), and only valid http(s) links are rendered.

All paths above are relative to the project root.
