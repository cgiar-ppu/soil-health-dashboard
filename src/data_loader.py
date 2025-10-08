from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Optional, Tuple
import logging
import re

import pandas as pd

from .constants import (
    INPUT_DIR,
    DATA_FILE_PATTERNS,
    MAIN_DATASET_PATH,
    CLUSTER_KEYWORDS_PATH,
    CLUSTER_SUMMARIES_PATH,
)

# Optional Streamlit cache decorator (degrades to no-op outside Streamlit)
try:  # pragma: no cover - import guard
    import streamlit as st  # type: ignore
    _cache_data = st.cache_data
except Exception:  # pragma: no cover - degrade gracefully if streamlit missing
    def _cache_data(func=None, **kwargs):  # type: ignore
        """No-op cache decorator used when Streamlit is unavailable."""
        if func is not None:
            return func

        def decorator(f):
            return f

        return decorator


logger = logging.getLogger(__name__)


def list_input_files(patterns: Iterable[str] | None = None) -> List[Path]:
    """List files in the Input directory matching provided patterns.

    Args:
        patterns: Iterable of glob patterns (e.g., ["*.csv"]). If None, uses DATA_FILE_PATTERNS.

    Returns:
        List of Path objects for matching files (deduplicated, sorted within each pattern).
    """
    patterns = tuple(patterns) if patterns is not None else DATA_FILE_PATTERNS
    input_dir = INPUT_DIR
    input_dir.mkdir(exist_ok=True)  # Ensure it exists

    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(input_dir.glob(pat)))
    # Deduplicate while preserving order
    seen = set()
    unique_files: list[Path] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    return unique_files


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from CSV or Excel using pandas.

    Args:
        path: File path to load.

    Returns:
        pandas.DataFrame

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the file extension is unsupported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"File not found: {p}\n"
            f"Expected file at: {p}. Ensure it exists under the project Input/ directory."
        )

    suffix = p.suffix.lower()
    if suffix == ".csv":
        # Let pandas infer; fallback to utf-8-sig
        try:
            df = pd.read_csv(p)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="utf-8-sig")
        return df
    elif suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _ensure_required_columns(df: pd.DataFrame, required: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Ensure required columns exist; add missing ones as empty if needed.

    Args:
        df: Input dataframe.
        required: Iterable of required column names.

    Returns:
        DataFrame with at least the required columns. Missing columns are added with NA values.
    """
    if not required:
        return df
    missing = [c for c in required if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = pd.NA
        logger.warning("Missing expected columns %s were added as empty.", missing)
    return df


@_cache_data(show_spinner=False)
def load_main_dataset(required_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load and cache the primary dataset from Input/.

    Reads Input/2025-10-07T13-26_export_SoilHealthClusters.csv into a DataFrame.

    Args:
        required_columns: Optional iterable of column names to enforce. Missing ones will be created as NA.

    Returns:
        pandas.DataFrame (cached by Streamlit when available)

    Raises:
        FileNotFoundError: with guidance if the expected file is missing.
    """
    path = MAIN_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Primary dataset not found.\n"
            f"Expected path: {path}\n"
            "Please place '2025-10-07T13-26_export_SoilHealthClusters.csv' under the Input/ folder."
        )
    df = load_dataset(path)
    return _ensure_required_columns(df, required_columns)


@_cache_data(show_spinner=False)
def load_cluster_keywords(required_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Load and cache the cluster keywords dataset from Input/.

    Reads Input/2025-10-07T13-25_export_SoilHealth_ClusterKeywords.csv into a DataFrame.

    Args:
        required_columns: Optional iterable of column names to enforce. Missing ones will be created as NA.

    Returns:
        pandas.DataFrame (cached by Streamlit when available)

    Raises:
        FileNotFoundError: with guidance if the expected file is missing.
    """
    path = CLUSTER_KEYWORDS_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Cluster keywords file not found.\n"
            f"Expected path: {path}\n"
            "Please place '2025-10-07T13-25_export_SoilHealth_ClusterKeywords.csv' under the Input/ folder."
        )
    df = load_dataset(path)
    return _ensure_required_columns(df, required_columns)


def _parse_docx_summaries(docx_path: Path) -> Tuple[str, Dict[int, str]]:
    """Parse the DOCX summaries file into overall and per-topic summaries.

    The parser is tolerant of headings such as:
      - "Overall" or "Overall Summary"
      - "Topic 1", "Topic 1:", "Cluster 1 - <title>" etc.

    Args:
        docx_path: Path to the DOCX file.

    Returns:
        A tuple: (overall_summary_text, topic_summaries_dict)
    """
    try:  # optional import
        import docx  # type: ignore
    except Exception as e:  # pragma: no cover - exercised only when dependency missing
        raise ImportError(
            "python-docx is not installed. Install 'python-docx' to parse summaries."
        ) from e

    document = docx.Document(docx_path)
    lines: list[str] = []
    for p in document.paragraphs:
        txt = (p.text or "").strip()
        if txt:
            lines.append(txt)

    overall_parts: list[str] = []
    topics: Dict[int, list[str]] = {}

    cur_topic: Optional[int] = None
    in_overall = False

    topic_pat = re.compile(r"^(?:topic|cluster)\s*(\d+)\b[:\-]?\s*(.*)$", re.IGNORECASE)
    overall_pat = re.compile(r"^(overall(?:\s+summary)?)\b[:\-]?\s*(.*)$", re.IGNORECASE)

    for ln in lines:
        m_overall = overall_pat.match(ln)
        m_topic = topic_pat.match(ln)

        if m_overall:
            in_overall = True
            cur_topic = None
            rest = (m_overall.group(2) or "").strip()
            if rest:
                overall_parts.append(rest)
            continue

        if m_topic:
            in_overall = False
            cur_topic = int(m_topic.group(1))
            topics.setdefault(cur_topic, [])
            rest = (m_topic.group(2) or "").strip()
            if rest:
                topics[cur_topic].append(rest)
            continue

        # Normal content line
        if in_overall or cur_topic is None:
            overall_parts.append(ln)
        else:
            topics.setdefault(cur_topic, []).append(ln)

    overall_text = "\n".join([s for s in (s.strip() for s in overall_parts) if s])
    topic_texts: Dict[int, str] = {k: "\n".join([s for s in (s.strip() for s in v) if s]) for k, v in topics.items()}

    return overall_text, topic_texts


@_cache_data(show_spinner=False)
def load_cluster_summaries() -> Dict[str, object]:
    """Load and cache overall and per-topic cluster summaries from a DOCX file.

    Uses python-docx if available; if not installed, degrades gracefully by returning
    an object with a clear message and empty summaries so the app can continue.

    Returns:
        A dict containing:
            - "overall": str (may be empty if not parsed)
            - "topics": Dict[int, str]
            - "message": Optional[str] explanatory message when parsing couldn't be performed

    Raises:
        FileNotFoundError: if the DOCX file is missing, with guidance on expected path.
    """
    path = CLUSTER_SUMMARIES_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Cluster summaries DOCX not found.\n"
            f"Expected path: {path}\n"
            "Please place 'Soil Health Cluster Summaries.docx' under the Input/ folder."
        )

    try:
        overall, topics = _parse_docx_summaries(path)
        return {"overall": overall, "topics": topics, "message": None}
    except ImportError as e:
        # Dependency missing; degrade gracefully
        msg = (
            "python-docx not installed; returning empty summaries. "
            "Install 'python-docx' to enable DOCX parsing."
        )
        logger.warning("%s", msg)
        return {"overall": "", "topics": {}, "message": str(e)}
