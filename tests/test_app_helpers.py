import pandas as pd
import pytest

st = pytest.importorskip("streamlit")  # ensure streamlit is available for importing the app

import streamlit_app as app


def test_split_keywords_dedup_and_order():
    text = "soil; water, Soil\nhealth;  ;  WATER"
    out = app._split_keywords(text)
    # dedup case-insensitively, preserve first occurrences order
    assert out == ["soil", "water", "health"]


def test_sanitize_link_values_handles_invalid():
    df = pd.DataFrame({
        "PDF link": ["http://a", "https://b", "ftp://c", "", None, "nan"],
        "URL": ["https://x", "notaurl", "http://y", "<NA>", "None", "http://z"],
    })
    clean = app._sanitize_link_values(df, ["PDF link", "URL"])
    assert list(clean["PDF link"]) == ["http://a", "https://b", None, None, None, None]
    assert list(clean["URL"]) == ["https://x", None, "http://y", None, None, "http://z"]


essential_cols = ("Title", "Description", "Year", "Countries", "Topic", "Type", "Submitter", "Partners")


def _sample_df():
    return pd.DataFrame({
        "Title": ["Soil Health", "Water Quality", "Soil nutrients"],
        "Description": ["Great soil project", "River study", "NPK in soils"],
        "Year": [2019, 2021, 2020],
        "Countries": ["USA; Canada", "France", "USA"],
        "Topic": [1, 2, 1],
        "Type": ["Report", "Dataset", "Report"],
        "Submitter": ["NASA", "CIMMYT", "FAO"],
        "Partners": ["ACME; NASA", "CIRAD", "NASA"],
        "Result ID": ["R1", "R2", "R3"],
    })


def test_get_first_and_all_present():
    df = pd.DataFrame({"Type": ["x"], "output type": ["y"], "Other": [1]})
    first = app._get_first_present(df, ["Output Type", "Type of Output", "Type"])
    assert first in ("Type", "output type") and df.columns.tolist().count(first) == 1
    allp = app._get_all_present(df, ["Type", "Output Type", "Missing"])
    assert set(allp) <= set(df.columns)
    assert ("Type" in allp) or ("output type" in allp)


def test_safe_unique_and_row_id():
    ser = pd.Series([" b ", None, "", "a", "b"]) 
    uniq = app._safe_unique(ser)
    assert uniq == ["a", "b"]

    df = pd.DataFrame({"x": [10, 20, 30]})
    out = app._ensure_row_id(df)
    assert "_row_id" in out.columns
    assert list(out["_row_id"]) == [0, 1, 2]


def test_build_topic_display_map_uses_cluster_name():
    df = pd.DataFrame({
        "Topic": [1, 2, 3],
        "Cluster Name": ["Soils", "Water", None],
    })
    mapping = app._build_topic_display_map(df)
    assert mapping[1].startswith("Topic 1") and "Soils" in mapping[1]
    assert mapping[2].startswith("Topic 2") and "Water" in mapping[2]
    assert mapping[3] == "Topic 3"


def test_detect_time_columns_includes_year_and_named():
    df = pd.DataFrame({
        "Year": [2020, 2021],
        "createdDate": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "Publish Date": ["2023-05-01", "2023-05-02"],
        "Other": [1, 2],
    })
    cols = app._detect_time_columns(df)
    assert "Year" in cols
    assert "createdDate" in cols
    assert "Publish Date" in cols


def test_apply_global_filters_composition():
    df = _sample_df()
    filters = {
        "search": "soil",
        "year_range": (2019, 2020),
        "countries": ["USA"],
        "topics": [1],
        "types": ["Report"],
        "submitters": ["NASA", "FAO"],
        "partners": ["NASA"],
    }
    out = app._apply_global_filters(df, filters)
    # Expect rows 0 and 2 match search, year range, country=USA, topic=1, type=Report; 
    # after partners filter with NASA, both rows 0 and 2 still match
    assert set(out.index.tolist()) == {0, 2}


def test_apply_global_filters_handles_missing_columns_gracefully():
    df = pd.DataFrame({"Title": ["A"], "x": [1]})
    filters = {
        "search": "a",
        "year_range": (2000, 2025),
        "countries": ["USA"],
        "topics": [1],
        "types": ["Report"],
        "submitters": ["NASA"],
        "partners": ["ACME"],
    }
    out = app._apply_global_filters(df, filters)
    # Should not error and at least keep the matching text search row
    assert len(out) == 1


def test_safe_link_md_basic():
    assert app._safe_link_md("https://example.com", label="Open") == "[Open](https://example.com)"
    assert app._safe_link_md("ftp://example.com") == ""
    assert app._safe_link_md(None) == ""
