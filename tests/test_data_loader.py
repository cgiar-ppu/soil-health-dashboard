from pathlib import Path
import types
import pandas as pd
import pytest

import src.data_loader as dl


def test_load_main_dataset_success():
    df = dl.load_main_dataset()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 0


def test_load_cluster_keywords_success():
    df = dl.load_cluster_keywords()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] >= 1


def test_load_cluster_summaries_success():
    data = dl.load_cluster_summaries()
    assert isinstance(data, dict)
    assert set(["overall", "topics", "message"]).issubset(data.keys())
    assert isinstance(data["topics"], dict)


def test_missing_main_dataset_error(monkeypatch):
    missing = Path("Input/nonexistent_main.csv")
    monkeypatch.setattr(dl, "MAIN_DATASET_PATH", missing, raising=False)
    with pytest.raises(FileNotFoundError) as ei:
        dl.load_main_dataset()
    msg = str(ei.value)
    assert "Expected path:" in msg
    assert "Input" in msg


def test_missing_keywords_error(monkeypatch):
    missing = Path("Input/nonexistent_keywords.csv")
    monkeypatch.setattr(dl, "CLUSTER_KEYWORDS_PATH", missing, raising=False)
    with pytest.raises(FileNotFoundError) as ei:
        dl.load_cluster_keywords()
    msg = str(ei.value)
    assert "Expected path:" in msg
    assert "Input" in msg


def test_required_columns_added():
    # Use a column name that is extremely unlikely to exist
    missing_col = "___unlikely_missing_col___"
    df = dl.load_main_dataset(required_columns=(missing_col,))
    assert missing_col in df.columns
    assert df[missing_col].isna().all()


def test_summaries_without_python_docx(monkeypatch):
    # Force the internal parser to simulate missing dependency
    def _raise_import_error(path: Path):
        raise ImportError("python-docx is not installed")

    monkeypatch.setattr(dl, "_parse_docx_summaries", _raise_import_error, raising=True)
    result = dl.load_cluster_summaries()
    assert isinstance(result, dict)
    assert result["overall"] == ""
    assert result["topics"] == {}
    assert isinstance(result.get("message"), str)


def test_list_input_files_defaults_and_patterns():
    files = dl.list_input_files()
    names = {p.name for p in files}
    # CSVs should be present
    assert "2025-10-07T13-26_export_SoilHealthClusters.csv" in names
    assert "2025-10-07T13-25_export_SoilHealth_ClusterKeywords.csv" in names
    # DOCX should not be included by default CSV/XLS patterns
    assert "Soil Health Cluster Summaries.docx" not in names
