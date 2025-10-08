Test plan

Automated unit tests (pytest)
- data_processing.clean_column_names
  - Input: columns with spaces, punctuation, and mixed case
  - Expect: lowercase, underscores, no leading/trailing underscores
- data_processing.basic_summary
  - Input: small DataFrame
  - Expect: correct row/column counts, columns list, and positive memory usage

Manual checks
- App boots without errors: streamlit run streamlit_app.py
- With no files in ./Input, app shows instructions and no errors
- Add a small CSV to ./Input, select it, verify summary and preview render
- Add an Excel file to ./Input, verify it loads and previews
- Toggle between files to confirm UI updates without errors
