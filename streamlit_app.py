alerefrom __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re
import html

import pandas as pd
import numpy as np
import streamlit as st
# Optional: capture Plotly click events. Fallback to no-op if package missing.
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
except Exception:  # pragma: no cover
    def plotly_events(fig, **kwargs):  # type: ignore
        st.plotly_chart(fig, use_container_width=True)
        return []

from src.constants import INPUT_DIR, ASSETS_DIR
from src.data_loader import (
    list_input_files,
    load_dataset,
    load_main_dataset,
    load_cluster_keywords,
    load_cluster_summaries,
)
from src.data_processing import (
    basic_summary,
    derive_year_column,
    build_country_long,
    build_partner_long,
    map_topics,
    text_search,
)
from src.viz import (
    show_dataframe,
    pie_distribution,
    configurable_bar,
    time_series,
    world_map_countries,
)
import plotly.express as px  # Add this

COOL_SEQUENCE = px.colors.sequential.Blues_r
WARM_SEQUENCE = px.colors.sequential.Oranges_r

# After imports and before helpers
TOPIC_TITLES = {
    -1: "Soil Agronomic Practices",
    0: "Producer Training Honduras",
    1: "No-Till Soil Management",
    2: "Farmer Agroecology Training",
    3: "Conservation Crop Yields",
    4: "Tunisia Livestock Forage",
    5: "Rice Conservation Scenarios",
    6: "Kenya Agroecological Resources",
    7: "Fertilizer Nutrient Advisory",
    8: "Rice Yield Management",
    9: "Soil Salinity Mapping",
    10: "Acid Cropland Mitigation",
    11: "Bangladesh Organic Vermicompost",
    12: "India Natural Farming",
    13: "Regenerative Soil Health",
    14: "Lime Acidity Management",
    15: "AMF Disease Resistance",
    16: "Nature Positive Solutions",
    17: "Microbial Biodiversity Longan",
    18: "Digital Soil Platform",
    19: "Wheat Fertility Production",
    20: "Sustainable Livestock Strategies",
    21: "Women Soil Empowerment"
}

# --------------
# Helpers
# --------------

def _timestamp_str() -> str:
    """Return a filesystem-safe timestamp string for filenames."""
    return pd.Timestamp.now().strftime("%Y-%m-%dT%H-%M-%S")


def _sanitize_link_values(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Return a copy with invalid/missing link values set to None for given columns.
    A valid link is considered to start with http:// or https://.
    """
    if df is None or df.empty:
        return df
    clean = df.copy()
    for c in columns:
        if c in clean.columns:
            ser = clean[c].astype(str)
            ser = ser.where(ser.str.startswith("http"), None)
            ser = ser.where(~ser.isin(["nan", "NaN", "None", "", "<NA>"]), None)
            clean[c] = ser
    return clean


def _safe_link_md(url: Optional[str], label: Optional[str] = None) -> str:
    """Return a Markdown link string for a URL with basic sanitization.

    - Only http/https URLs are allowed; otherwise returns an empty string.
    - Label defaults to the URL host/path if not provided.
    - Escapes label for HTML safety.
    """
    if not url or not isinstance(url, str):
        return ""
    if not (url.startswith("http://") or url.startswith("https://")):
        return ""
    lab = label if (label and isinstance(label, str)) else url
    lab_safe = html.escape(lab)
    # Note: Streamlit Markdown sanitizes by default; keep simple format
    return f"[{lab_safe}]({url})"


def _load_styles():
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_all_sources() -> Dict[str, object]:
    """Load primary dataset plus cluster keywords and summaries.
    Returns a dict with graceful fallbacks if some optional files are missing.
    """
    out: Dict[str, object] = {"data": pd.DataFrame(), "keywords": pd.DataFrame(), "summaries": {}}
    # Main dataset (required for most of the app)
    try:
        df = load_main_dataset()
    except FileNotFoundError:
        # Fallback: list any other input files the user may have placed
        files = list_input_files()
        df = pd.DataFrame()
        if files:
            try:
                df = load_dataset(files[0])
            except Exception:
                pass
    out["data"] = df

    # Keywords (optional)
    try:
        kw = load_cluster_keywords()
    except FileNotFoundError:
        kw = pd.DataFrame()
    out["keywords"] = kw

    # Summaries (optional)
    try:
        summaries = load_cluster_summaries()
    except FileNotFoundError:
        summaries = {"overall": "", "topics": {}, "message": "Summaries DOCX not found."}
    out["summaries"] = summaries
    return out


def _get_first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _get_all_present(df: pd.DataFrame, candidates: Sequence[str]) -> List[str]:
    out = []
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            out.append(c)
        elif c.lower() in lower_map:
            out.append(lower_map[c.lower()])
    return out


def _safe_unique(series: pd.Series) -> List[str]:
    if series is None:
        return []
    ser = series.dropna().astype(str).str.strip()
    ser = ser[ser != ""]
    vals = sorted(ser.unique().tolist())
    return vals


def _ensure_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add a stable row id to help join exploded filters back to base records."""
    if "_row_id" not in df.columns:
        df = df.copy()
        df["_row_id"] = np.arange(len(df))
    return df


def _build_topic_display_map(df_with_topics: pd.DataFrame) -> Dict[int, str]:
    """Return mapping from Topic number (Int) to 'Topic N – Cluster Name' label."""
    if "Topic" not in df_with_topics.columns:
        return {}
    labels: Dict[int, str] = {}
    # Extract numeric topic if string
    topic_series = pd.to_numeric(df_with_topics["Topic"].str.extract(r'Topic (\d+) -', expand=False), errors="coerce").dropna().astype(int)
    unique_topics = sorted(topic_series.unique().tolist())
    for t in unique_topics:
        title = TOPIC_TITLES.get(t, "Unnamed")
        labels[t] = f"Topic {t} - {title}"
    return labels


def _detect_time_columns(df: pd.DataFrame) -> List[str]:
    """Heuristically detect date/time-like columns including Year.
    - Include 'Year' if present (will be converted)
    - Include any columns with datetime dtype
    - Include columns whose name contains 'date', 'publish', 'issued', 'created', 'start'
    """
    out: List[str] = []
    if "Year" in df.columns:
        out.append("Year")
    # datetime dtypes
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            out.append(c)
    # name patterns
    name_patterns = ["date", "publish", "issued", "created", "start"]
    for c in df.columns:
        cl = str(c).lower()
        if any(p in cl for p in name_patterns):
            if c not in out:
                out.append(c)
    # Deduplicate preserve order
    seen = set()
    ordered = []
    for c in out:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


# --------------
# Filtering
# --------------

def _apply_global_filters(
    df: pd.DataFrame,
    filters: Dict[str, object],
) -> pd.DataFrame:
    """Apply all filters to a copy of df. Handles optional columns gracefully."""
    if df is None or df.empty:
        return df
    out = _ensure_row_id(df)

    # Text search
    query = (filters.get("search") or "").strip()
    if query:
        # Search over Title/Description where present
        out = text_search(out, query, columns=["Title", "Description"]) if set(["Title", "Description"]).intersection(out.columns) else out

    # Year range
    yr_min, yr_max = filters.get("year_range", (None, None))
    if yr_min is not None and yr_max is not None and "Year" in out.columns:
        mask = out["Year"].between(yr_min, yr_max, inclusive="both")
        out = out.loc[mask]

    # Countries multiselect (using exploded approach)
    sel_countries: List[str] = filters.get("countries") or []
    if sel_countries:
        exploded = build_country_long(out)
        if "Country" in exploded.columns:
            good_ids = exploded.loc[exploded["Country"].isin(sel_countries), "_row_id"].unique().tolist()
            out = out[out["_row_id"].isin(good_ids)]

    # Topic filter
    sel_topics: List[int] = filters.get("topics") or []
    if sel_topics and "Topic" in out.columns:
        # Parse number from string
        out["__topic_num"] = pd.to_numeric(out["Topic"].str.extract(r'Topic (\d+) -', expand=False), errors="coerce").astype("Int64")
        out = out[out["__topic_num"].isin(sel_topics)]
        out = out.drop(columns=["__topic_num"])

    # Type filter
    type_col = _get_first_present(out, ["Type", "Output Type", "Document Type", "Type of Output"]) or None
    sel_types: List[str] = filters.get("types") or []
    if sel_types and type_col:
        out = out[out[type_col].astype(str).isin(sel_types)]

    # Submitter filter
    sub_col = _get_first_present(out, ["Submitter", "Submitted By", "Submitting Organization", "Submitting organisation"]) or None
    sel_submitters: List[str] = filters.get("submitters") or []
    if sel_submitters and sub_col:
        out = out[out[sub_col].astype(str).isin(sel_submitters)]

    # Partners filter (exploded)
    sel_partners: List[str] = filters.get("partners") or []
    if sel_partners:
        p_long = build_partner_long(out)
        if "Partner" in p_long.columns:
            good_ids = p_long.loc[p_long["Partner"].isin(sel_partners), "_row_id"].unique().tolist()
            out = out[out["_row_id"].isin(good_ids)]

    return out


# --------------
# Cluster helpers (keywords+summaries UX)
# --------------

def _split_keywords(text: Optional[str]) -> List[str]:
    """Split a delimited keywords string into tokens (commas/semicolons/newlines)."""
    if not text:
        return []
    s = str(text)
    parts = [p.strip() for p in re.split(r"[,;\n]", s) if p and p.strip()]
    # De-dup preserve order
    seen = set()
    out: List[str] = []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def _render_keyword_chips(keywords: List[str]):
    if not keywords:
        return
    chips = " ".join([f"<span class='chip'>{html.escape(k)}</span>" for k in keywords])
    st.markdown(f"<div class='chip-container'>{chips}</div>", unsafe_allow_html=True)


# --------------
# Selection handling for charts -> details
# --------------

def _detect_multivalue_role(df: pd.DataFrame, col: Optional[str]) -> str:
    """Return role of a column: 'partner', 'country', or 'simple'."""
    if not col:
        return 'simple'
    # Resolve actual column name present in df
    lower_map = {str(c).lower(): c for c in df.columns}
    col_real = lower_map.get(str(col).lower(), col)
    partner_candidates = ["Partner", "Partners", "Organization", "Organizations", "Collaborators", "Institutions"]
    country_candidates = ["Country", "Countries", "Country/Region", "Geography", "Country of Focus"]
    if any(str(col_real).lower() == str(c).lower() for c in partner_candidates):
        return 'partner'
    if any(str(col_real).lower() == str(c).lower() for c in country_candidates):
        return 'country'
    return 'simple'


def _filter_records_by_selection(df: pd.DataFrame, x_col: str, x_value: object, color_col: Optional[str] = None, color_value: Optional[object] = None) -> pd.DataFrame:
    """Filter records matching the selection from a bar chart.

    Handles multi-value columns for Partners/Countries via exploded long tables.
    """
    base = df.copy()
    rid_col = "_row_id" if "_row_id" in base.columns else None

    def filter_on(col: str, val: object, frame: pd.DataFrame) -> pd.DataFrame:
        role = _detect_multivalue_role(frame, col)
        if role == 'partner':
            long = build_partner_long(frame if rid_col else _ensure_row_id(frame))
            ids = set(long.loc[long["Partner"].astype(str) == str(val), "_row_id"].tolist()) if "Partner" in long.columns else set()
            out = frame[(frame["_row_id"].isin(ids))] if ids else frame.iloc[0:0]
            return out
        elif role == 'country':
            long = build_country_long(frame if rid_col else _ensure_row_id(frame))
            ids = set(long.loc[long["Country"].astype(str) == str(val), "_row_id"].tolist()) if "Country" in long.columns else set()
            out = frame[(frame["_row_id"].isin(ids))] if ids else frame.iloc[0:0]
            return out
        else:
            # Simple equality filter (string compare)
            return frame[frame[col].astype(str) == str(val)] if col in frame.columns else frame.iloc[0:0]

    out = filter_on(x_col, x_value, base)
    if color_col and color_value is not None:
        out = filter_on(color_col, color_value, out)
    return out


def _aggregate_lists_by_row(df: pd.DataFrame, column: str, out_name: str) -> pd.DataFrame:
    """Explode a multi-value column and aggregate back to a '; ' joined list per _row_id."""
    tmp = df.copy()
    if "_row_id" not in tmp.columns:
        tmp = _ensure_row_id(tmp)
    if column not in tmp.columns:
        return pd.DataFrame({"_row_id": tmp["_row_id"], out_name: pd.NA}).drop_duplicates("_row_id")
    if out_name == "Partner":
        long = build_partner_long(tmp)
        val_col = "Partner"
    elif out_name == "Country":
        long = build_country_long(tmp)
        val_col = "Country"
    else:
        # generic: split by commas/semicolons
        from src.data_processing import normalize_delimited_column, explode_multivalue
        long = normalize_delimited_column(tmp, column)
        long = explode_multivalue(long, column)
        val_col = column
    if val_col not in long.columns:
        return pd.DataFrame({"_row_id": tmp["_row_id"], out_name: pd.NA}).drop_duplicates("_row_id")
    agg = long.groupby("_row_id")[val_col].apply(lambda s: "; ".join(sorted(pd.Series(s).dropna().astype(str).unique()))).reset_index(name=out_name)
    return agg


def _prepare_record_view(df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact records table with key fields and link columns."""
    if df is None or df.empty:
        return df
    work = df.copy()
    if "_row_id" not in work.columns:
        work = _ensure_row_id(work)

    # Derive Partners and Countries joined lists
    partners_join = _aggregate_lists_by_row(work, _get_first_present(work, ["Partners", "Partner"]) or "Partners", out_name="Partner")
    countries_join = _aggregate_lists_by_row(work, _get_first_present(work, ["Countries", "Country"]) or "Countries", out_name="Country")

    # Join back
    cols_keep = [c for c in ["_row_id", "Title", "Result ID", "ResultId", "ID", "Id"] if c in work.columns]
    sub_col = _get_first_present(work, ["Submitter", "Submitted By", "Submitting Organization", "Submitting organisation"]) or None
    if sub_col and sub_col not in cols_keep:
        cols_keep.append(sub_col)
    link_cols = [c for c in ["PDF link", "Evidence 1"] if c in work.columns]
    cols_keep += [c for c in link_cols if c not in cols_keep]

    base = work[cols_keep].copy() if cols_keep else work[['_row_id']].copy()
    base = base.merge(partners_join, on="_row_id", how="left")
    base = base.merge(countries_join, on="_row_id", how="left")

    # Rename ID to a single 'ID' column if present
    id_col = _get_first_present(base, ["Result ID", "ResultId", "ID", "Id"]) or None
    if id_col and id_col != "ID":
        base = base.rename(columns={id_col: "ID"})
    elif id_col is None:
        base["ID"] = pd.NA

    # Ensure column order
    cols_order = [c for c in ["Title", "ID", sub_col or "Submitter", "Partner", "Country"] if c in base.columns]
    cols_order += [c for c in ["PDF link", "Evidence 1"] if c in base.columns]
    # Keep only ordered columns when possible
    view = base[cols_order] if cols_order else base

    # Sanitize links
    view = _with_clickable_links(view)
    return view


def _render_selection_details(header: str, df: pd.DataFrame):
    """Render a details panel for selected results with link columns."""
    st.markdown(header)
    if df is None or df.empty:
        st.info("No matching records for this selection.")
        return
    show_dataframe(df, caption=f"{len(df)} records")


def _bar_chart_with_details(fig, base_df: pd.DataFrame, x_col: str, color_col: Optional[str] = None, key: Optional[str] = None, header_prefix: str = "Details"):
    """Render a Plotly bar figure with click-to-details behavior.
    Returns the last selection dict (if any).
    """
    events = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=key or f"evt_{x_col}_{color_col or 'none'}")
    selection = events[0] if events else None
    if selection:
        x_val = selection.get("x") or selection.get("label")
        color_val = None
        if color_col is not None:
            try:
                cnum = selection.get("curveNumber")
                if cnum is not None:
                    # Access the trace name which is the color category
                    color_val = fig.data[cnum].name  # type: ignore[attr-defined]
            except Exception:
                color_val = None
        subset = _filter_records_by_selection(base_df, x_col=x_col, x_value=x_val, color_col=color_col, color_value=color_val)
        view = _prepare_record_view(subset)
        title_bits = [f"{x_col} = {x_val}"]
        if color_col and color_val is not None:
            title_bits.append(f"{color_col} = {color_val}")
        _render_selection_details(f"**{header_prefix}:** " + ", ".join(title_bits), view)
    else:
        st.caption("Tip: Click a bar segment to open matching records below.")
    return selection

# --------------
# Tabs content
# --------------

def _tab_insights(df: pd.DataFrame):
    st.subheader("Insights")
    if df is None or df.empty:
        st.info("No data to display with the current filters.")
        return

    # 1) Top Partners by result count
    p_long = build_partner_long(df)
    col1, col2 = st.columns([2, 2])
    with col1:
        st.caption("Top 10 partners by result count")
        if "Partner" in p_long.columns and not p_long.empty:
            p_counts = p_long.groupby("Partner", dropna=False).size().reset_index(name="Count")
            p_counts = p_counts.sort_values("Count", ascending=False).head(10)
            fig = configurable_bar(p_counts, x="Partner", y_agg={"Count": "sum:Count"}, color_discrete_sequence=COOL_SEQUENCE)
            _bar_chart_with_details(fig, base_df=df, x_col="Partner", color_col=None, key="ins_partners", header_prefix="Partner selection")
        else:
            fig = configurable_bar(pd.DataFrame({"Partner": [], "Count": []}), x="Partner", y_agg={"Count": "sum:Count"})
            st.plotly_chart(fig, use_container_width=True)

    # 2) Results by Country (map and/or bar)
    c_long = build_country_long(df)
    with col2:
        st.caption("Results by country (top 15)")
        if "Country" in c_long.columns and not c_long.empty:
            c_counts = c_long.groupby("Country", dropna=False).size().reset_index(name="Count")
            c_counts = c_counts.sort_values("Count", ascending=False).head(15)
            fig = configurable_bar(c_counts, x="Country", y_agg={"Count": "sum:Count"}, color_discrete_sequence=WARM_SEQUENCE)
            _bar_chart_with_details(fig, base_df=df, x_col="Country", color_col=None, key="ins_countries", header_prefix="Country selection")
        else:
            fig = configurable_bar(pd.DataFrame({"Country": [], "Count": []}), x="Country", y_agg={"Count": "sum:Count"})
            st.plotly_chart(fig, use_container_width=True)

    # Optional wide map if countries available
    if "Country" in c_long.columns and not c_long.empty:
        st.caption("Global distribution of results")
        c_counts_full = c_long.groupby("Country", dropna=False).size().reset_index(name="Count")
        st.plotly_chart(world_map_countries(c_counts_full, country_col="Country", value_col="Count"), use_container_width=True, key="insights_map")

    # 3) Results over time by Submitter
    st.divider()
    st.caption("Results over time by Submitter")
    sub_col = _get_first_present(df, ["Submitter", "Submitted By", "Submitting Organization", "Submitting organisation"]) or None
    time_options = _detect_time_columns(df)
    if sub_col and time_options:
        work = df.copy()
        tcol = time_options[0]
        if tcol == "Year" and "Year" in work.columns:
            years = pd.to_numeric(work["Year"], errors="coerce")
            work = work.loc[years.notna()].copy()
            work["__time"] = pd.to_datetime(years.astype(int).astype(str) + "-01-01", errors="coerce")
        else:
            work["__time"] = pd.to_datetime(work[tcol], errors="coerce")
        work = work.dropna(subset=["__time"]) if "__time" in work.columns else work
        st.plotly_chart(time_series(work, time_col="__time", category=sub_col, metric="count", kind="line"), use_container_width=True)
    else:
        # Graceful empty chart
        st.plotly_chart(time_series(pd.DataFrame({"__time": [], sub_col or "Submitter": []}), time_col="__time", category=sub_col or "Submitter", metric="count", kind="line"), use_container_width=True)

    # 4) Stacked bar example using common fields
    st.divider()
    st.caption("Stacked bar: Type by Topic (top 20 categories)")
    x_col = _get_first_present(df, ["Type", "Output Type", "Document Type", "Type of Output"]) or None
    color_col = "Topic Label" if "Topic Label" in df.columns else "Topic" if "Topic" in df.columns else None
    if x_col:
        fig = configurable_bar(df, x=x_col, y_agg={"Count": "count"}, color=color_col, top_n=20)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback: Partners by Type if available
        if "Partner" in p_long.columns and not p_long.empty:
            agg = p_long.copy()
            if x_col is None:
                # try to detect a type-like column from original df merged via _row_id
                type_guess = _get_first_present(df, ["Type", "Output Type", "Document Type", "Type of Output"]) or None
            else:
                type_guess = x_col
            if type_guess and "_row_id" in agg.columns and "_row_id" in df.columns:
                merged = agg.merge(df[["_row_id", type_guess]], on="_row_id", how="left")
                fig = configurable_bar(merged, x="Partner", y_agg={"Count": "count"}, color=type_guess, top_n=20)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.plotly_chart(configurable_bar(pd.DataFrame({"Partner": []}), x="Partner", y_agg={"Count": "count"}), use_container_width=True)
        else:
            st.plotly_chart(configurable_bar(pd.DataFrame({"Category": []}), x="Category", y_agg={"Count": "count"}), use_container_width=True)

def _quick_filter_chips(values: List[str], key_prefix: str, session_key: str, max_show: int = 20):
    """Render a row of clickable chips to quickly apply filters to session state.

    Clicking a chip toggles its presence in the multiselect session state.
    """
    if not values:
        return
    cols = st.columns(min(len(values), max_show))
    for i, v in enumerate(values[:max_show]):
        with cols[i]:
            pressed = st.button(v, key=f"{key_prefix}_{i}")
            if pressed:
                cur = set(st.session_state.get(session_key, []) or [])
                if v in cur:
                    cur.remove(v)
                else:
                    cur.add(v)
                st.session_state[session_key] = sorted(cur)
                st.rerun()


def _tab_overview(df: pd.DataFrame):
    st.subheader("Overview")
    if df.empty:
        st.info("No data to display with the current filters.")
        return

    # KPIs
    df = df.drop_duplicates(subset=['_row_id'])
    c_long = build_country_long(df)
    p_long = build_partner_long(df)
    total_results = len(df)
    unique_countries = int(c_long["Country"].nunique()) if "Country" in c_long.columns else 0
    unique_partners = int(p_long["Partner"].nunique()) if "Partner" in p_long.columns else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total results", f"{total_results:,}")
    with m2:
        st.metric("Unique countries", f"{unique_countries:,}")
    with m3:
        st.metric("Unique partners", f"{unique_partners:,}")

    st.divider()

    # Top distributions: Type, Topic (if present)
    colA, colB = st.columns(2)
    with colA:
        type_col = _get_first_present(df, ["Type", "Output Type", "Document Type", "Type of Output"]) or None
        if type_col:
            st.caption(f"Distribution by {type_col}")
            fig = pie_distribution(df, category=type_col, top_n=12, color_discrete_sequence=COOL_SEQUENCE)
            fig.update_traces(hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra>Destacado: Categoría de tipo</extra>")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Type column not found")
    with colB:
        if "Topic" in df.columns:
            st.caption("Distribution by Topic")
            st.plotly_chart(pie_distribution(df, category="Topic", top_n=12, color_discrete_sequence=WARM_SEQUENCE), use_container_width=True)
        else:
            st.caption("Topic column not found")


def _tab_trends(df: pd.DataFrame):
    st.subheader("Trends")
    if df.empty:
        st.info("No data to display with the current filters.")
        return

    # Time field selection
    time_options = _detect_time_columns(df)
    if not time_options:
        st.info("No time/date columns detected.")
        return

    tcol = st.selectbox("Time field", options=time_options, index=0, help="Choose the time column used for trends")

    # Prepare a working copy ensuring time col is present/parsable
    work = df.copy()
    if tcol == "Year":
        # Convert Year -> datetime (Jan 1) for plotting
        # Some rows may be NA; drop NA for plotting
        year_series = pd.to_numeric(work["Year"], errors="coerce")
        work = work.loc[year_series.notna()].copy()
        work["__time"] = pd.to_datetime(year_series.astype(int).astype(str) + "-01-01", errors="coerce")
    else:
        work["__time"] = pd.to_datetime(work[tcol], errors="coerce")
        work = work.dropna(subset=["__time"])  # keep only valid times

    # Trends by Topic
    cat_topic = "Topic Label" if "Topic Label" in work.columns else ("Topic" if "Topic" in work.columns else None)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Results over time by Topic")
        st.plotly_chart(time_series(work, time_col="__time", category=cat_topic, metric="count", kind="area", color_discrete_sequence=WARM_SEQUENCE), use_container_width=True)
    with col2:
        # Trends by Type
        type_col = _get_first_present(work, ["Type", "Output Type", "Document Type", "Type of Output"]) or None
        st.caption("Results over time by Type")
        st.plotly_chart(time_series(work, time_col="__time", category=type_col, metric="count", kind="area", color_discrete_sequence=COOL_SEQUENCE), use_container_width=True)


def _tab_geography(df: pd.DataFrame):
    st.subheader("Geography")
    if df.empty:
        st.info("No data to display with the current filters.")
        return

    df = df.drop_duplicates(subset=['_row_id'])
    c_long = build_country_long(df)
    if "Country" not in c_long.columns or c_long.empty:
        st.info("No country data available.")
        return

    counts = c_long.groupby("Country", dropna=False).size().reset_index(name="Count")

    map_col, bar_col = st.columns([2, 1])
    with map_col:
        try:
            if counts.empty:
                fig = empty_figure("No country data")
            else:
                fig = world_map_countries(counts, country_col="Country", value_col="Count")
            st.plotly_chart(fig, use_container_width=True, key="geo_map")
        except Exception as e:
            st.error(f"Error rendering map: {str(e)}")
            st.plotly_chart(empty_figure("Map error"), use_container_width=True)

    with bar_col:
        st.caption("Top countries")
        try:
            top = counts.sort_values("Count", ascending=False).head(20)
            if top.empty:
                fig = empty_figure("No data")
            else:
                fig = configurable_bar(top, x="Country", y_agg={"Count": "sum:Count"})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering bar chart: {str(e)}")
            st.plotly_chart(empty_figure("Chart error"), use_container_width=True)


def _tab_clusters(df: pd.DataFrame, summaries: Dict[str, object]):

    st.subheader("Clusters")
    if df.empty or "Topic" not in df.columns:
        st.info("No Topic/Cluster data available.")
        return

    # Notice if DOCX summaries are not available
    if summaries and isinstance(summaries, dict):
        msg = summaries.get("message")
        if msg:
            st.warning(f"Cluster summaries unavailable: {msg}\n\nShowing keywords from CSV where available.")

    # Build compact per-topic catalog
    name_col = _get_first_present(df, ["Cluster Name"]) or None
    kw_col = _get_first_present(df, ["Cluster Keywords"]) or None

    topic_vals = pd.to_numeric(df["Topic"].str.extract(r'Topic (\d+) -', expand=False), errors="coerce").dropna().astype(int)
    if topic_vals.empty:
        st.info("No valid Topic IDs found.")
        return

    unique_topics = sorted(topic_vals.unique().tolist())
    topic_summaries: Dict[int, str] = (summaries or {}).get("topics", {}) or {}

    catalog: List[Dict[str, object]] = []
    for t in unique_topics:
        # Name
        if name_col:
            name_series = df.loc[df["Topic"] == t, name_col].dropna().astype(str).str.strip()
            name = name_series.iloc[0] if not name_series.empty else ""
        else:
            name = ""
        # Keywords string and list
        if kw_col:
            kw_series = df.loc[df["Topic"] == t, kw_col].dropna().astype(str)
            kw_text = kw_series.iloc[0] if not kw_series.empty else ""
        else:
            kw_text = ""
        kw_list = _split_keywords(kw_text)
        # Summary
        summary_text = topic_summaries.get(t) or ""

        catalog.append(
            {
                "topic": int(t),
                "title": f"Topic {int(t)} - {TOPIC_TITLES.get(int(t), item.get('name', 'Unnamed'))}"
                if name
                else f"Topic {int(t)}",
                "name": name,
                "keywords_text": kw_text,
                "keywords": kw_list,
                "summary": summary_text,
            }
        )

    # Search box across name/keywords/summary
    q = st.text_input("Search clusters (name, keywords, summary)", value="")
    def _match(item: Dict[str, object], qtext: str) -> bool:
        if not qtext:
            return True
        ql = qtext.lower().strip()
        hay = " ".join([
            str(item.get("title", "")),
            str(item.get("name", "")),
            str(item.get("keywords_text", "")),
            str(item.get("summary", "")),
        ]).lower()
        return ql in hay

    q_norm = (q or "").strip()
    if q_norm:
        filtered_items = [it for it in catalog if _match(it, q_norm)]
    else:
        filtered_items = catalog

    st.caption(f"Showing {len(filtered_items)} of {len(catalog)} clusters")

    # Render cards for each cluster
    for item in filtered_items:
        with st.container():
            title = f"Topic {int(item['topic'])} - {TOPIC_TITLES.get(int(item['topic']), item.get('name', 'Unnamed'))}"
            st.markdown(f"### {title}")
            if item.get("name"):
                st.caption(f"Cluster Name: {item['name']}")
            # Keyword chips
            if item["keywords"]:
                _render_keyword_chips(item["keywords"])  # chips under the header
            else:
                st.caption("No keywords available.")
            # Summary text
            if item["summary"]:
                st.markdown("**Summary**")
                st.write(item["summary"])
            else:
                st.caption("No summary available for this cluster.")

            # Optional details per cluster
            with st.expander("View records and trends"):
                sub = df[df["Topic"] == item["topic"]]
                # Top Types over time
                st.markdown("**Top Types over time**")
                tcols = _detect_time_columns(sub)
                work = sub.copy()
                if tcols:
                    t = tcols[0]
                    if t == "Year":
                        yrs = pd.to_numeric(work["Year"], errors="coerce")
                        work = work.loc[yrs.notna()].copy()
                        work["__time"] = pd.to_datetime(yrs.astype(int).astype(str) + "-01-01", errors="coerce")
                    else:
                        work["__time"] = pd.to_datetime(work[t], errors="coerce")
                        work = work.dropna(subset=["__time"]) 
                else:
                    work["__time"] = pd.NaT

                type_col = _get_first_present(work, ["Type", "Output Type", "Document Type", "Type of Output"]) or None
                st.plotly_chart(time_series(work, time_col="__time", category=type_col, metric="count", kind="area"), use_container_width=True, key=f"cluster_time_series_{item['topic']}")

                # Sample records
                st.markdown("**Sample records**")
                show_dataframe(sub.head(50))
        st.divider()


def _tab_partners(df: pd.DataFrame):
    st.subheader("Partners")
    if df.empty:
        st.info("No data to display with the current filters.")
        return

    p_long = build_partner_long(df)
    if "Partner" not in p_long.columns or p_long.empty:
        st.info("No partner data available.")
        return

    counts = p_long.groupby("Partner", dropna=False).size().reset_index(name="Count")
    counts = counts.sort_values("Count", ascending=False)

    col1, col2 = st.columns([2, 2])
    with col1:
        st.caption("Partner frequency")
        st.plotly_chart(configurable_bar(counts.head(30), x="Partner", y_agg={"Count": "sum:Count"}), use_container_width=True)
    with col2:
        # Partner by Topic stacked bar
        if "Topic Label" in p_long.columns:
            agg = p_long.groupby(["Partner", "Topic Label"], dropna=False).size().reset_index(name="Count")
            st.caption("Partners by Topic")
            st.plotly_chart(configurable_bar(agg, x="Partner", y_agg={"Count": "sum:Count"}, color="Topic Label", top_n=25), use_container_width=True)
        elif "Topic" in p_long.columns:
            agg = p_long.groupby(["Partner", "Topic"], dropna=False).size().reset_index(name="Count")
            st.plotly_chart(configurable_bar(agg, x="Partner", y_agg={"Count": "sum:Count"}, color="Topic", top_n=25), use_container_width=True)

    # Searchable partner list
    st.divider()
    q = st.text_input("Search partners", value="")
    if q:
        filt = counts[counts["Partner"].astype(str).str.contains(q, case=False, na=False)]
    else:
        filt = counts
    show_dataframe(filt)


def _tab_partners_countries(df: pd.DataFrame):
    st.subheader("Partners & Countries")
    if df.empty:
        st.info("No data to display with the current filters.")
        return

    p_long = build_partner_long(df)
    c_long = build_country_long(df)

    # Partners section
    st.markdown("### Partners")
    if "Partner" in p_long.columns and not p_long.empty:
        p_counts = p_long.groupby("Partner", dropna=False).size().reset_index(name="Count").sort_values("Count", ascending=False)
        st.plotly_chart(configurable_bar(p_counts, x="Partner", y_agg={"Count": "sum:Count"}, top_n=25, color_discrete_sequence=COOL_SEQUENCE), use_container_width=True)
        st.caption("Quick filter by top partners")
        _quick_filter_chips(p_counts["Partner"].head(12).astype(str).tolist(), key_prefix="chip_partner", session_key="flt_partners")

        with st.expander("Browse results by partner"):
            sel_partner = st.selectbox("Select a partner", options=p_counts["Partner"].astype(str).tolist())
            st.button("Apply filter", on_click=lambda: _apply_partner_filter(sel_partner), key="apply_partner_filter")
            try:
                if sel_partner is not None:
                    # Case-insensitive match
                    ids = set(p_long.loc[p_long["Partner"].str.lower() == str(sel_partner).lower(), "_row_id"].tolist())
                    sub = df[df["_row_id"].isin(ids)].copy()
                else:
                    sub = pd.DataFrame()
                q = st.text_input("Search within results (Title/Description)", value="")
                if q:
                    sub = text_search(sub, q, columns=["Title", "Description"]) if set(["Title", "Description"]).intersection(sub.columns) else sub
                if sub.empty:
                    st.info("No records found for this partner.")
                else:
                    show_dataframe(_with_clickable_links(sub))
            except Exception as e:
                st.error(f"Error loading records: {str(e)}")

    st.divider()

    # Countries section
    st.markdown("### Countries")
    if "Country" in c_long.columns and not c_long.empty:
        c_counts = c_long.groupby("Country", dropna=False).size().reset_index(name="Count").sort_values("Count", ascending=False)
        st.plotly_chart(configurable_bar(c_counts, x="Country", y_agg={"Count": "sum:Count"}, top_n=25, color_discrete_sequence=WARM_SEQUENCE), use_container_width=True)
        st.caption("Quick filter by top countries")
        _quick_filter_chips(c_counts["Country"].head(12).astype(str).tolist(), key_prefix="chip_country", session_key="flt_countries")

        with st.expander("Browse results by country"):
            sel_country = st.selectbox("Select a country", options=c_counts["Country"].astype(str).tolist())
            st.button("Apply filter", on_click=lambda: _apply_country_filter(sel_country), key="apply_country_filter")
            try:
                if sel_country is not None:
                    # Case-insensitive match
                    ids = set(c_long.loc[c_long["Country"].str.lower() == str(sel_country).lower(), "_row_id"].tolist())
                    sub = df[df["_row_id"].isin(ids)].copy()
                else:
                    sub = pd.DataFrame()
                q = st.text_input("Search within results (Title/Description)", value="", key="q_country")
                if q:
                    sub = text_search(sub, q, columns=["Title", "Description"]) if set(["Title", "Description"]).intersection(sub.columns) else sub
                if sub.empty:
                    st.info("No records found for this country.")
                else:
                    show_dataframe(_with_clickable_links(sub))
            except Exception as e:
                st.error(f"Error loading records: {str(e)}")


def _apply_partner_filter(value: str):
    cur = set(st.session_state.get("flt_partners", []) or [])
    cur.add(value)
    st.session_state["flt_partners"] = sorted(cur)
    st.rerun()


def _apply_country_filter(value: str):
    cur = set(st.session_state.get("flt_countries", []) or [])
    cur.add(value)
    st.session_state["flt_countries"] = sorted(cur)
    st.rerun()


def _with_clickable_links(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with link columns sanitized. Used in tables."""
    link_columns = [c for c in ["PDF link", "Evidence 1", "URL", "Link"] if c in df.columns]
    return _sanitize_link_values(df.copy(), link_columns)


def _tab_builder(df: pd.DataFrame):
    st.subheader("Bar Builder")
    if df.empty:
        st.info("No data to display with the current filters.")
        return

    # Candidate categorical columns (object/string or small cardinality)
    cat_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
    # Also include a few likely ID-ish small integer columns
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() < 100:
            cat_cols.append(c)
    # De-duplicate while preserving type and stable order
    seen = set()
    cat_cols = [c for c in cat_cols if (str(c) not in seen and not seen.add(str(c)))]

    x = st.selectbox("X (categorical)", options=sorted(cat_cols, key=str), index=0 if cat_cols else None)

    # Metric selection: Count rows or Distinct by id
    id_candidates = ["Result ID", "ResultId", "ID", "Id", "Record ID", "_row_id"]
    present_ids = _get_all_present(df, id_candidates)
    default_id = present_ids[0] if present_ids else "_row_id"

    metric_mode = st.radio("Y metric", options=["Count rows", "Distinct by column"], horizontal=True)
    if metric_mode == "Distinct by column":
        id_col = st.selectbox("Column for distinct count", options=present_ids or ["_row_id"], index=0)
        y_agg = {"Distinct": f"nunique:{id_col}"}
    else:
        y_agg = {"Count": "count"}

    color_opt = st.selectbox("Stack by (optional)", options=["<None>"] + sorted(cat_cols, key=str))
    color = None if color_opt == "<None>" else color_opt

    top_n = st.number_input("Top N categories (by total)", min_value=0, value=25, step=5)

    if x:
        fig = configurable_bar(df, x=x, y_agg=y_agg, color=color, top_n=int(top_n) if top_n else None)
        _bar_chart_with_details(fig, base_df=df, x_col=x, color_col=color, key="builder_chart", header_prefix="Selection")


def _tab_data(df: pd.DataFrame):
    st.subheader("Data")
    if df.empty:
        st.info("No data to display with the current filters.")
        return

    # Pagination controls
    n = len(df)
    page_size = st.selectbox("Rows per page", options=[25, 50, 100, 200], index=0)
    total_pages = max(1, int(np.ceil(n / page_size)))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start = (page - 1) * page_size
    end = start + page_size

    # Clickable link columns
    link_columns = []
    for cand in ["PDF link", "Evidence 1", "URL", "Link"]:
        if cand in df.columns:
            link_columns.append(cand)

    view_df = df.iloc[int(start):int(end)].copy()
    # Sanitize links for rendering
    view_df = _sanitize_link_values(view_df, link_columns)

    col_config = {}
    try:
        for c in link_columns:
            display_text = "Open" if c in {"PDF link", "Evidence 1"} else c
            col_config[c] = st.column_config.LinkColumn(c, display_text=display_text)
    except Exception:
        # Older Streamlit versions: skip column config
        col_config = {}

    st.dataframe(view_df, use_container_width=True, hide_index=True, column_config=col_config or None)

    # Downloads of filtered data
    st.divider()
    c1, c2 = st.columns(2)
    ts = _timestamp_str()
    csv_name = f"filtered_data_{ts}.csv"
    json_name = f"filtered_data_{ts}.json"
    with c1:
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=csv_name,
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download JSON",
            data=df.to_json(orient="records").encode("utf-8"),
            file_name=json_name,
            mime="application/json",
        )


# --------------
# App
# --------------

def main():
    st.set_page_config(page_title="CGIAR Soil Health Explorer", layout="wide")
    _load_styles()

    st.title("CGIAR Soil Health Explorer")
    st.caption("Interactive dashboard with global filters, trends, and downloads.")

    # Data loading
    sources = _load_all_sources()
    df_raw: pd.DataFrame = sources.get("data", pd.DataFrame())
    kw_df: pd.DataFrame = sources.get("keywords", pd.DataFrame())
    summaries: Dict[str, object] = sources.get("summaries", {})

    if df_raw is None or df_raw.empty:
        st.info("No input files found. Place .csv or .xlsx in the 'Input' folder at the project root.")
        return

    # Prepare dataset: derive Year; map topics
    df = derive_year_column(df_raw)
    if not kw_df.empty:
        df = map_topics(df, kw_df)
    df = _ensure_row_id(df)
    df["_row_id"] = pd.Series(range(len(df))).values  # Reset to unique
    # Remove any duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    # Add Topic Label if not present
    if "Topic Label" not in df.columns and "Topic" in df.columns:
        df["Topic"] = df["Topic"].apply(lambda t: f"Topic {t} - {TOPIC_TITLES.get(t, 'Unnamed')}" if pd.notna(t) else pd.NA)

    # Sidebar filters
    st.sidebar.header("Global filters")

    # Reset filters
    if st.sidebar.button("Reset filters"):
        for k in list(st.session_state.keys()):
            if k.startswith("flt_"):
                del st.session_state[k]
        st.rerun()

    # Text search
    search = st.sidebar.text_input("Search Title/Description", key="flt_search")

    # Year range if available
    year_col_present = "Year" in df.columns and df["Year"].notna().any()
    if year_col_present:
        yr_min = int(pd.to_numeric(df["Year"], errors="coerce").dropna().min())
        yr_max = int(pd.to_numeric(df["Year"], errors="coerce").dropna().max())
        yr_range = st.sidebar.slider("Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max), key="flt_year")
    else:
        yr_range = (None, None)
        st.sidebar.caption("Year column not available")

    # Countries multiselect (exploded)
    c_long_all = build_country_long(df)
    countries_opts = _safe_unique(c_long_all["Country"]) if "Country" in c_long_all.columns else []
    countries_sel = st.sidebar.multiselect("Countries", options=countries_opts, key="flt_countries")

    # Topic/Cluster multiselect
    topic_labels = _build_topic_display_map(df)
    topics_sel_labels = st.sidebar.multiselect(
        "Topic / Cluster",
        options=[topic_labels[t] for t in sorted(topic_labels.keys())],
        key="flt_topics",
    ) if topic_labels else []
    topics_sel: List[int] = []
    if topics_sel_labels:
        inv = {v: k for k, v in topic_labels.items()}
        topics_sel = [inv[x] for x in topics_sel_labels]

    # Type multiselect
    type_col = _get_first_present(df, ["Type", "Output Type", "Document Type", "Type of Output"]) or None
    type_sel = st.sidebar.multiselect("Type", options=_safe_unique(df[type_col]) if type_col else [], key="flt_types") if type_col else []

    # Submitter multiselect
    sub_col = _get_first_present(df, ["Submitter", "Submitted By", "Submitting Organization", "Submitting organisation"]) or None
    sub_sel = st.sidebar.multiselect("Submitter", options=_safe_unique(df[sub_col]) if sub_col else [], key="flt_submitters") if sub_col else []

    # Partners multiselect (exploded)
    p_long_all = build_partner_long(df)
    partners_opts = _safe_unique(p_long_all["Partner"]) if "Partner" in p_long_all.columns else []
    partners_sel = st.sidebar.multiselect("Partners", options=partners_opts, key="flt_partners")

    # Compose filters dict
    filters = {
        "search": search,
        "year_range": yr_range,
        "countries": countries_sel,
        "topics": topics_sel,
        "types": type_sel,
        "submitters": sub_sel,
        "partners": partners_sel,
    }

    # Apply filters
    filtered = _apply_global_filters(df, filters)

    # Tabs
    tabs = st.tabs(["Overview", "Insights", "Trends", "Geography", "Clusters", "Partners & Countries", "Builder", "Data"])

    with tabs[0]:
        _tab_overview(filtered)

    with tabs[1]:
        _tab_insights(filtered)

    with tabs[2]:
        _tab_trends(filtered)

    with tabs[3]:
        _tab_geography(filtered)

    with tabs[4]:
        _tab_clusters(filtered, summaries)

    with tabs[5]:
        _tab_partners_countries(filtered)

    with tabs[6]:
        _tab_builder(filtered)

    with tabs[7]:
        _tab_data(filtered)

if __name__ == "__main__":
    main()
