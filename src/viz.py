from __future__ import annotations
from typing import Dict, Optional, Tuple, List

import pandas as pd

# Optional imports to keep this module importable without heavy deps at runtime
try:  # Plotly for figures
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised only when plotly missing
    px = None  # type: ignore
    go = None  # type: ignore

try:  # Streamlit used only for convenience helpers; optional
    import streamlit as st  # type: ignore
except ImportError:  # pragma: no cover
    st = None  # type: ignore


def _ensure_plotly():
    """Ensure plotly is available, raising a clear error if not.

    Returns:
        Tuple: (px, go) plotly modules.
    """
    if px is None or go is None:
        raise ImportError(
            "Plotly is required for visualization functions. Please install 'plotly'."
        )
    return px, go


def _empty_figure(title: str = "No data"):
    """Return an empty Plotly Figure with a centered annotation."""
    _, go_mod = _ensure_plotly()
    fig = go_mod.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=None,
        annotations=[
            dict(
                text=title,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="#888"),
            )
        ],
    )
    return fig


def empty_figure(title: str = "No data"):
    """Public helper returning a minimal empty Plotly figure.

    Useful when a chart has no data after filtering; keeps layout consistent
    and avoids runtime errors in the Streamlit app.
    """
    return _empty_figure(title)


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
            # Replace typical NA representations and non-http values with None
            ser = ser.where(ser.str.startswith("http"), None)
            ser = ser.where(~ser.isin(["nan", "NaN", "None", "", "<NA>"]), None)
            clean[c] = ser
    return clean


def show_dataframe(df: pd.DataFrame, caption: str | None = None):
    """Render a DataFrame in Streamlit with basic link rendering when possible.

    - Uses LinkColumn for common link fields (PDF link, Evidence 1, URL, Link).
    - Missing/invalid links are shown as empty cells.
    """
    if st is None:
        return
    if df is None:
        return

    link_candidates = ["PDF link", "Evidence 1", "URL", "Link"]
    present_links = [c for c in link_candidates if c in df.columns]
    view_df = _sanitize_link_values(df, present_links)

    col_config: Dict[str, object] = {}
    try:
        for c in present_links:
            # Display a short label for the link; keep column label as-is
            display_text = "Open" if c in {"PDF link", "Evidence 1"} else c
            col_config[c] = st.column_config.LinkColumn(c, display_text=display_text)
    except Exception:
        # Fallback: no column config support in older Streamlit
        col_config = {}

    st.dataframe(view_df, use_container_width=True, hide_index=True, column_config=col_config or None)
    if caption:
        st.caption(caption)


def configurable_bar(
    df: pd.DataFrame,
    x: str,
    y_agg: Dict[str, str] | None = None,
    color: Optional[str] = None,
    top_n: Optional[int] = None,
):
    """Build a configurable bar chart using Plotly.

    Args:
        df: Input DataFrame. Can be pre-filtered; empty frames are handled.
        x: Column to use for the x-axis (categorical).
        y_agg: Aggregation spec dict mapping a resulting metric name to either:
            - "count": count of rows per group; ignores key name except for label.
            - "sum:<col>": sum of the given numeric column name per group.
            - "sum" with key as the numeric column name (legacy form).
            - "nunique:<col>" to compute distinct count of a column per group.
            If None, defaults to {"Count": "count"}.
        color: Optional column name for color grouping (stacked bars).
        top_n: If provided, keep only the top N categories by the metric on x.

    Returns:
        plotly.graph_objects.Figure
    """
    _ensure_plotly()

    if df is None or len(df) == 0 or x not in df.columns:
        return _empty_figure()

    # Parse y aggregation
    if not y_agg:
        metric_name, mode, target_col = "Count", "count", None
    else:
        # Accept forms: {"Count": "count"} or {"value": "sum"} or {"Sum": "sum:col"} or {"Distinct": "nunique:col"}
        if len(y_agg) != 1:
            raise ValueError("y_agg must be a single-item dict")
        (key, val) = next(iter(y_agg.items()))
        if val == "count":
            metric_name, mode, target_col = key, "count", None
        elif isinstance(val, str) and val.startswith("sum:"):
            metric_name, mode, target_col = key, "sum", val.split(":", 1)[1]
        elif val == "sum":
            metric_name, mode, target_col = key, "sum", key
        elif isinstance(val, str) and (val.startswith("nunique:") or val.startswith("distinct:")):
            metric_name, mode, target_col = key, "nunique", val.split(":", 1)[1]
        else:
            raise ValueError("y_agg value must be 'count', 'sum', 'sum:<col>', or 'nunique:<col>'")

    group_cols = [x]
    if color and color in df.columns:
        group_cols.append(color)
    elif color:
        # color provided but missing in df; ignore silently to avoid errors
        color = None

    g = df.groupby(group_cols, dropna=False)
    if mode == "count":
        agg = g.size().reset_index(name=metric_name)
    elif mode == "sum":
        if target_col is None or target_col not in df.columns:
            return _empty_figure(f"Column to sum not found: {target_col}")
        agg = g[target_col].sum(min_count=1).reset_index(name=metric_name)
    else:  # nunique
        if target_col is None or target_col not in df.columns:
            return _empty_figure(f"Column for distinct count not found: {target_col}")
        agg = g[target_col].nunique(dropna=True).reset_index(name=metric_name)

    # Determine top_n categories at x level (overall across colors)
    if top_n is not None and top_n > 0:
        totals = agg.groupby(x)[metric_name].sum().nlargest(top_n)
        mask = agg[x].isin(totals.index)
        agg = agg[mask]

    # Sort bars by metric descending
    order = (
        agg.groupby(x)[metric_name].sum().sort_values(ascending=False).index.tolist()
    )

    try:
        fig = px.bar(
            agg,
            x=x,
            y=metric_name,
            color=color,
            category_orders={x: order} if order else None,
            text_auto=True,
        )
    except TypeError:
        # Older plotly versions may not support text_auto
        fig = px.bar(
            agg,
            x=x,
            y=metric_name,
            color=color,
            category_orders={x: order} if order else None,
        )
    y_title = metric_name if mode == "count" else (f"Sum of {target_col}" if mode == "sum" else f"Distinct {target_col}")
    fig.update_layout(bargap=0.2, xaxis_title=x, yaxis_title=y_title)

    # Helpful hover tooltips encouraging clicks; keeps default vars
    hover_tmpl = (
        "<b>%{x}</b><br>" +
        f"{y_title}: %{{y}}<br>" +
        ("Group: %{fullData.name}<br>" if color else "") +
        "<extra>Click to view matching records</extra>"
    )
    try:
        fig.update_traces(hovertemplate=hover_tmpl)
    except Exception:
        # If update_traces fails (older versions), ignore
        pass

    return fig


def pie_distribution(
    df: pd.DataFrame,
    category: str,
    top_n: int = 15,
    other_bucket: bool = True,
):
    """Pie chart for category distribution based on row counts.

    Args:
        df: Input DataFrame.
        category: Column name to aggregate.
        top_n: Keep the top N categories by count.
        other_bucket: If True, aggregate the remaining items into an 'Other' slice.

    Returns:
        plotly.graph_objects.Figure
    """
    _ensure_plotly()

    if df is None or category not in df.columns or len(df) == 0:
        return _empty_figure()

    counts = df[category].value_counts(dropna=False)
    if top_n and top_n > 0:
        top = counts.nlargest(top_n)
        remainder = counts.drop(top.index)
        if other_bucket and remainder.sum() > 0:
            data = pd.concat([top, pd.Series({"Other": remainder.sum()})])
        else:
            data = top
    else:
        data = counts

    plot_df = data.reset_index()
    plot_df.columns = [category, "Count"]

    fig = px.pie(plot_df, names=category, values="Count")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(legend_title=category)
    return fig


def time_series(
    df: pd.DataFrame,
    time_col: str,
    category: Optional[str] = None,
    metric: str = "count",
    kind: str = "line",
    normalize: bool = False,
):
    """Time series chart for counts or sums per time bucket.

    Args:
        df: Input DataFrame.
        time_col: Name of the datetime column.
        category: Optional category column for multiple series (colored lines/areas).
        metric: Either "count" or "sum:<col>" to sum a numeric column.
        kind: "line" or "area".
        normalize: If True and category provided, convert to percent share per time (0-100).
            If True and no category, normalize the single series to 0-100 by its max.

    Returns:
        plotly.graph_objects.Figure
    """
    _ensure_plotly()

    if df is None or time_col not in df.columns or len(df) == 0:
        return _empty_figure()

    # Prepare time column (floor to day to avoid over-fragmentation)
    ts = df.copy()
    ts[time_col] = pd.to_datetime(ts[time_col], errors="coerce").dt.floor("D")
    ts = ts.dropna(subset=[time_col])
    if ts.empty:
        return _empty_figure()

    # Determine aggregation
    if metric == "count":
        value_col = "Count"
        group_cols = [time_col] + ([category] if category and category in ts.columns else [])
        agg = ts.groupby(group_cols, dropna=False).size().reset_index(name=value_col)
    elif isinstance(metric, str) and metric.startswith("sum:"):
        sum_col = metric.split(":", 1)[1]
        if sum_col not in ts.columns:
            return _empty_figure(f"Column to sum not found: {sum_col}")
        value_col = f"Sum {sum_col}"
        group_cols = [time_col] + ([category] if category and category in ts.columns else [])
        agg = ts.groupby(group_cols, dropna=False)[sum_col].sum(min_count=1).reset_index(name=value_col)
    else:
        return _empty_figure("Unsupported metric. Use 'count' or 'sum:<col>'")

    # Normalize if requested
    if normalize:
        if category and category in agg.columns:
            # Percent share per time across categories
            totals = agg.groupby(time_col)[value_col].transform("sum").replace(0, pd.NA)
            agg[value_col] = (agg[value_col] / totals) * 100
            y_title = "Share (%)"
        else:
            max_v = agg[value_col].max()
            if pd.isna(max_v) or max_v == 0:
                agg[value_col] = 0
            else:
                agg[value_col] = (agg[value_col] / max_v) * 100
            y_title = "Normalized (0-100)"
    else:
        y_title = value_col

    chart_fn = px.area if kind == "area" else px.line
    fig = chart_fn(
        agg,
        x=time_col,
        y=value_col,
        color=category if category and category in agg.columns else None,
    )
    fig.update_layout(xaxis_title=time_col, yaxis_title=y_title)
    return fig


def world_map_countries(
    df_country_counts: pd.DataFrame,
    country_col: str = "Country",
    value_col: str = "Count",
):
    """Choropleth world map using country names.

    Args:
        df_country_counts: DataFrame with country and count/value columns.
        country_col: Column containing country names.
        value_col: Column with numeric values to map.

    Returns:
        plotly.graph_objects.Figure
    """
    _ensure_plotly()

    required = {country_col, value_col}
    if df_country_counts is None or not required.issubset(df_country_counts.columns):
        return _empty_figure()

    fig = px.choropleth(
        df_country_counts,
        locations=country_col,
        locationmode="country names",
        color=value_col,
        hover_name=country_col,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        coloraxis_colorbar=dict(title=value_col),
    )
    return fig
