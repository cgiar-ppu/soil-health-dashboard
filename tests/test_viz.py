import pytest

plotly = pytest.importorskip("plotly")

import pandas as pd

from src.viz import (
    configurable_bar,
    pie_distribution,
    time_series,
    world_map_countries,
)


def test_configurable_bar_count_basic():
    df = pd.DataFrame({
        'cat': ['A', 'A', 'B', 'C', 'C', 'C']
    })
    fig = configurable_bar(df, x='cat')
    # Should return a plotly graph_objects.Figure with at least one trace
    assert hasattr(fig, 'data')
    assert len(fig.data) >= 1


def test_configurable_bar_sum_with_color():
    df = pd.DataFrame({
        'cat': ['A', 'A', 'B', 'B'],
        'grp': ['X', 'Y', 'X', 'Y'],
        'val': [1, 2, 3, 4],
    })
    fig = configurable_bar(df, x='cat', y_agg={'Value': 'sum:val'}, color='grp')
    assert hasattr(fig, 'data')
    assert len(fig.data) >= 1


def test_configurable_bar_distinct_count():
    df = pd.DataFrame({
        'cat': ['A', 'A', 'B', 'B', 'B'],
        'id': [1, 1, 2, 3, 3],
    })
    # Distinct id per category should yield A->1, B->2
    fig = configurable_bar(df, x='cat', y_agg={'Distinct': 'nunique:id'})
    assert hasattr(fig, 'data')
    assert len(fig.data) >= 1


def test_pie_distribution_with_other_bucket():
    df = pd.DataFrame({'cat': ['A'] * 5 + ['B'] * 3 + ['C']})
    fig = pie_distribution(df, category='cat', top_n=1, other_bucket=True)
    assert hasattr(fig, 'data')
    assert len(fig.data) == 1
    labels = list(fig.data[0]['labels'])
    assert 'A' in labels
    assert 'Other' in labels


def test_time_series_count_by_category():
    df = pd.DataFrame({
        'ts': pd.to_datetime([
            '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02',
        ]),
        'cat': ['X', 'Y', 'X', 'Y']
    })
    fig = time_series(df, time_col='ts', category='cat', metric='count', kind='line')
    # Expect separate traces per category (X, Y)
    assert len(fig.data) == 2


def test_world_map_countries_basic():
    df = pd.DataFrame({
        'Country': ['United States', 'Canada', 'France'],
        'Count': [10, 5, 7]
    })
    fig = world_map_countries(df)
    assert hasattr(fig, 'data')
    assert len(fig.data) >= 1
