from __future__ import annotations
import re
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


# ------------------------------
# Column utilities
# ------------------------------

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized, snake_case column names.
    - Lowercase
    - Strip
    - Replace non-alphanumeric with underscore
    - Collapse multiple underscores and strip leading/trailing underscores
    """

    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    df2 = df.copy()
    df2.columns = [_norm(str(c)) for c in df2.columns]
    return df2


def basic_summary(df: pd.DataFrame) -> dict:
    mem = int(df.memory_usage(index=True, deep=True).sum())
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "memory_usage_bytes": mem,
    }


# ------------------------------
# Multi-value parsing utilities
# ------------------------------

def parse_multi_values(text: object, delimiters: str = ",;") -> List[str]:
    """Parse a delimited string (commas/semicolons by default) into a list of values.

    Rules:
    - Split on any character in `delimiters` (default: comma and semicolon)
    - Trim surrounding whitespace for each token
    - Drop empty tokens
    - If `text` is list/tuple/set, coerce elements to strings and apply trim/drop logic
    - Missing/NA/None/NaN/empty-string inputs return an empty list
    - Deduplicate tokens case-insensitively while preserving first-seen order

    Args:
        text: Input to parse (string or list-like)
        delimiters: Characters to split on

    Returns:
        List[str]: Clean list of tokens
    """
    # Handle list-like early
    if isinstance(text, (list, tuple, set)):
        items = [str(x) for x in text]
    else:
        if text is None:
            return []
        try:
            if pd.isna(text):  # type: ignore[arg-type]
                return []
        except Exception:
            # Some objects may raise in pd.isna; ignore and proceed to casting
            pass
        s = str(text)
        if not s or not s.strip():
            return []
        pat = f"[{re.escape(delimiters)}]"
        items = re.split(pat, s)

    # Normalize: trim, drop empties, dedupe (case-insensitive preserve order)
    out: List[str] = []
    seen_ci: set[str] = set()
    for it in items:
        t = str(it).strip()
        if not t:
            continue
        key = t.casefold()
        if key in seen_ci:
            continue
        seen_ci.add(key)
        out.append(t)
    return out


def explode_multi_value_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Explode a potentially multi-valued text column into long-form rows.

    Steps:
    - Parse the specified column to a list-of-strings using parse_multi_values()
    - pandas.DataFrame.explode to one value per row
    - Drop NA/empty values and trim whitespace

    Behavior when column missing: returns a copy of the input unchanged.

    Args:
        df: Input DataFrame
        column: Column name to explode

    Returns:
        Long-form DataFrame with one value per row for `column`.
    """
    if column not in df.columns:
        return df.copy()

    df2 = df.copy()
    df2[column] = df2[column].apply(lambda x: parse_multi_values(x))
    out = df2.explode(column, ignore_index=True)
    if column in out.columns:
        ser = out[column].astype("string").str.strip()
        out = out.loc[ser.notna() & (ser != "")].copy()
        out[column] = ser.loc[out.index]
    return out.reset_index(drop=True)


# Back-compat wrappers (kept for existing callers/tests)

def normalize_delimited_column(
    df: pd.DataFrame,
    column: str,
    delimiters: str = ",;",
) -> pd.DataFrame:
    """Normalize a delimited text column into lists of trimmed values.

    Deprecated in favor of parse_multi_values + assign/apply.
    Retained as a convenience/back-compat wrapper used elsewhere in the app.

    Returns a copy of the DataFrame with the normalized column. If the column
    doesn't exist, the input is returned unchanged.
    """
    if column not in df.columns:
        return df.copy()

    df2 = df.copy()
    df2[column] = df2[column].apply(lambda v: parse_multi_values(v, delimiters=delimiters))
    return df2


def explode_multivalue(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Back-compat wrapper for explode_multi_value_column."""
    return explode_multi_value_column(df, column)


# ------------------------------
# Data preparation utilities (Year derivation)
# ------------------------------

_DATE_LIKE_COL_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"^year$", re.IGNORECASE),
    re.compile(r"date", re.IGNORECASE),
    re.compile(r"publish|publication|published|issued", re.IGNORECASE),
    re.compile(r"created|start|begin", re.IGNORECASE),
)


def _find_column_case_insensitive(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Return the first matching column name from candidates (case-insensitive)."""
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _find_columns_by_pattern(df: pd.DataFrame, patterns: Iterable[re.Pattern[str]]) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        sc = str(c)
        if any(p.search(sc) for p in patterns):
            cols.append(c)
    # Prioritize 'Year' exact if found
    cols_sorted: List[str] = sorted(
        cols,
        key=lambda x: (0 if str(x).strip().lower() == "year" else 1, str(x).lower()),
    )
    return cols_sorted


def _extract_year_from_series(s: pd.Series) -> pd.Series:
    """Attempt to extract a 4-digit year from a Series of mixed types.

    Strategy:
    - If numeric, coerce to integer years where reasonable (1900-2100), else NA
    - Else try pandas.to_datetime(...).dt.year
    - Else regex search for 4-digit year pattern

    Returns a pandas Series of dtype Int64 (nullable integer)
    """
    # If numeric-like
    s_num = pd.to_numeric(s, errors="coerce")
    # When source already numbers, keep as year if in plausible range
    year_from_num = s_num.where((s_num >= 1500) & (s_num <= 3000))

    # Try datetime parsing
    try:
        dt = pd.to_datetime(s, errors="coerce")
    except Exception:
        dt = pd.to_datetime(pd.Series([pd.NA] * len(s)), errors="coerce")
    year_from_dt = dt.dt.year

    # Regex extract 4-digit year
    s_str = s.astype("string")
    year_regex = s_str.str.extract(r"((?:1[5-9]|2[0-9])\d{2})", expand=False)
    year_from_regex = pd.to_numeric(year_regex, errors="coerce")

    # Combine preferring explicit numeric, then datetime, then regex
    year = year_from_num
    year = year.fillna(year_from_dt)
    year = year.fillna(year_from_regex)

    return year.astype("Int64")


def derive_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a 'Year' column (nullable Int64) derived.

    Rules:
    - If a 'Year' column already exists, coerce to integer year (Int64)
    - Else search columns whose names look like dates/publication
      and derive a year using datetime parsing or regex
    - If no candidates yield a year, add 'Year' filled with <NA>

    This function never raises due to missing columns; it degrades gracefully.
    """
    df2 = df.copy()

    # Prefer exact 'Year' if present
    year_col = _find_column_case_insensitive(df2, ["Year"])
    if year_col is not None:
        df2["Year"] = _extract_year_from_series(df2[year_col])
        return df2

    # Find candidate date-like columns
    candidates = _find_columns_by_pattern(df2, _DATE_LIKE_COL_PATTERNS)
    year_series = None
    for c in candidates:
        y = _extract_year_from_series(df2[c])
        if y.notna().any():
            year_series = y
            break
    if year_series is None:
        year_series = pd.Series([pd.NA] * len(df2), index=df2.index, dtype="Int64")

    df2["Year"] = year_series.astype("Int64")
    return df2


# ------------------------------
# Long-form helpers for Countries/Partners
# ------------------------------

def _find_first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # Case-insensitive check
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _standardize_proper_noun(s: pd.Series) -> pd.Series:
    # Title-case words, but keep acronyms (all-caps) as is
    def _fmt(x: object) -> object:
        if pd.isna(x):
            return x
        t = str(x).strip()
        if not t:
            return pd.NA
        # If it's all uppercase and short, keep as-is
        if t.isupper() and len(t) <= 5:
            return t
        return t.title()

    out = s.astype("string").map(_fmt)
    return out


def build_country_long(df: pd.DataFrame) -> pd.DataFrame:
    """Return a country-level exploded DataFrame for charts/maps.

    Heuristics:
    - Detect a country column among common variants, e.g. 'Country', 'Countries'
    - Split by comma/semicolon, trim, explode
    - Drop blank entries

    Returns the exploded DataFrame with a 'Country' column. If no country
    column is found, returns the input unchanged.
    """
    country_col = _find_first_present(
        df,
        [
            "Country",
            "Countries",
            "Country/Region",
            "Geography",
            "Country of Focus",
        ],
    )
    if country_col is None:
        return df.copy()

    tmp = normalize_delimited_column(df, country_col)
    tmp = explode_multivalue(tmp, country_col)

    # Standardize column name
    if country_col != "Country":
        tmp = tmp.rename(columns={country_col: "Country"})

    # Standardize text
    tmp["Country"] = _standardize_proper_noun(tmp["Country"]).astype("string")

    # Drop empties again
    tmp = tmp[tmp["Country"].notna() & (tmp["Country"].str.len() > 0)].copy()
    tmp.reset_index(drop=True, inplace=True)
    return tmp


def build_partner_long(df: pd.DataFrame) -> pd.DataFrame:
    """Return a partner-level exploded DataFrame for charts.

    Detects columns that may contain partners/org names and explodes them
    into long form under the column name 'Partner'. If none found, returns
    the input unchanged.
    """
    partner_col = _find_first_present(
        df,
        [
            "Partner",
            "Partners",
            "Organization",
            "Organizations",
            "Collaborators",
            "Institutions",
        ],
    )
    if partner_col is None:
        return df.copy()

    tmp = normalize_delimited_column(df, partner_col)
    tmp = explode_multivalue(tmp, partner_col)

    if partner_col != "Partner":
        tmp = tmp.rename(columns={partner_col: "Partner"})

    tmp["Partner"] = _standardize_proper_noun(tmp["Partner"]).astype("string")
    tmp = tmp[tmp["Partner"].notna() & (tmp["Partner"].str.len() > 0)].copy()
    tmp.reset_index(drop=True, inplace=True)
    return tmp


# ------------------------------
# Topic/keywords mapping
# ------------------------------

def map_topics(df: pd.DataFrame, keywords_df: pd.DataFrame) -> pd.DataFrame:
    """Join cluster keywords/names by Topic number.

    The function attempts to find the topic id columns in both dataframes
    (looking for variants like 'Topic', 'topic', 'Topic #', 'Cluster'),
    and the descriptive columns in the keywords dataframe
    ('Cluster Name', 'Cluster Keywords').

    Returns a new DataFrame with 'Cluster Name' and 'Cluster Keywords' added
    where possible. Missing inputs/columns are handled gracefully by returning
    the original rows (with NA for unmapped fields).
    """
    if df is None or keywords_df is None:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def _find_topic_col(frame: pd.DataFrame) -> Optional[str]:
        # Try exact names first
        exact = _find_column_case_insensitive(frame, ["Topic", "Topic #", "Topic Number", "Cluster", "Cluster #"])
        if exact:
            return exact
        # Fallback to fuzzy search
        for c in frame.columns:
            cl = str(c).lower()
            if "topic" in cl or cl == "cluster" or cl.startswith("cluster "):
                return c
        return None

    left = df.copy()
    right = keywords_df.copy()

    left_topic = _find_topic_col(left)
    right_topic = _find_topic_col(right)
    if left_topic is None or right_topic is None:
        # Cannot join; ensure columns exist but empty
        if "Cluster Name" not in left.columns:
            left["Cluster Name"] = pd.NA
        if "Cluster Keywords" not in left.columns:
            left["Cluster Keywords"] = pd.NA
        return left

    # Identify name/keywords columns on right
    name_col = _find_column_case_insensitive(right, ["Cluster Name", "Name", "ClusterName"]) or "Cluster Name"
    keywords_col = _find_column_case_insensitive(right, ["Cluster Keywords", "Keywords", "ClusterKeywords"]) or "Cluster Keywords"

    # Build a tidy right table with standardized columns
    rt = right[[right_topic] + [c for c in [name_col, keywords_col] if c in right.columns]].copy()
    rt = rt.rename(columns={right_topic: "Topic", name_col: "Cluster Name", keywords_col: "Cluster Keywords"})

    # Coerce Topic to numeric for safe join
    rt["Topic"] = pd.to_numeric(rt["Topic"], errors="coerce").astype("Int64")

    # Left side standardization
    left = left.rename(columns={left_topic: "Topic"})
    left["Topic"] = pd.to_numeric(left["Topic"], errors="coerce").astype("Int64")

    # Drop duplicate topic rows in rt keeping first
    rt = rt.drop_duplicates(subset=["Topic"], keep="first")

    merged = left.merge(rt, on="Topic", how="left")

    return merged


# ------------------------------
# General text search
# ------------------------------

def text_search(
    df: pd.DataFrame,
    query: Optional[str],
    columns: Sequence[str] = ("Title", "Description"),
) -> pd.DataFrame:
    """Case-insensitive partial text search across specified columns.

    Args:
        df: DataFrame to filter
        query: Search string. If falsy/empty, returns df unchanged
        columns: Columns to search (searched only if present). If none of the
                 provided columns exist, returns an empty DataFrame with the
                 same columns as df.

    Returns:
        Filtered DataFrame containing rows where any searched column contains
        the query substring (case-insensitive). Handles missing columns
        gracefully.
    """
    if not query:
        return df.copy()

    # Restrict to existing columns
    present_cols = [c for c in columns if c in df.columns]
    if not present_cols:
        # Nothing to search against
        return df.iloc[0:0].copy()

    pat = re.escape(str(query))
    mask = pd.Series(False, index=df.index)
    for c in present_cols:
        ser = df[c].astype("string")
        mask = mask | ser.str.contains(pat, case=False, na=False)

    return df.loc[mask].copy()
