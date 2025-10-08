import math
import pandas as pd
from src.data_processing import (
    clean_column_names,
    basic_summary,
    derive_year_column,
    parse_multi_values,
    explode_multi_value_column,
    normalize_delimited_column,
    explode_multivalue,
    map_topics,
    build_country_long,
    build_partner_long,
    text_search,
)


essential_nan = pd.NA


def test_clean_column_names_basic():
    df = pd.DataFrame({
        'Column A': [1, 2],
        'column-B': [3, 4],
        '  MIXED  Case  ': [5, 6],
        'weird@@name!!': [7, 8],
        'spaces   and   tabs': [9, 10],
    })
    out = clean_column_names(df)
    assert list(out.columns) == [
        'column_a', 'column_b', 'mixed_case', 'weird_name', 'spaces_and_tabs'
    ]


def test_basic_summary_counts_and_columns():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    s = basic_summary(df)
    assert s['n_rows'] == 3
    assert s['n_cols'] == 2
    assert s['columns'] == ['a', 'b']
    assert isinstance(s['memory_usage_bytes'], int) and s['memory_usage_bytes'] > 0


def test_derive_year_column_prefers_existing_and_parses_dates():
    # Prefer existing Year column
    df = pd.DataFrame({'Year': [2020.0, '2019', None], 'Title': ['a', 'b', 'c']})
    out = derive_year_column(df)
    assert 'Year' in out.columns
    yrs = list(out['Year'].astype('Int64'))
    assert yrs[0] == 2020 and yrs[1] == 2019 and pd.isna(yrs[2])

    # Parse date-like column when Year missing
    df2 = pd.DataFrame({'Publication Date': ['2021-06-01', 'Mar 2019', None]})
    out2 = derive_year_column(df2)
    assert list(out2['Year'].astype('Int64')) == [2021, 2019, pd.NA]

    # Graceful when nothing usable
    df3 = pd.DataFrame({'X': [1, 2]})
    out3 = derive_year_column(df3)
    assert 'Year' in out3.columns
    assert list(out3['Year']) == [pd.NA, pd.NA]


def test_parse_multi_values_varied_delimiters_and_edge_cases():
    # Mixed delimiters and spaces
    s = 'USA; Canada,  Mexico ; ; , Brazil '
    assert parse_multi_values(s) == ['USA', 'Canada', 'Mexico', 'Brazil']

    # Deduplicate case-insensitively, preserve first occurrence casing
    s2 = 'usa, USA, Usa, canada, CANADA'
    out = parse_multi_values(s2)
    assert out[0] == 'usa' and 'USA' not in out[1:]
    assert out[1] == 'canada' and 'CANADA' not in out[2:]

    # None / NaN / empty -> []
    assert parse_multi_values(None) == []
    assert parse_multi_values(float('nan')) == []
    assert parse_multi_values('   ') == []

    # Already list-like; trims and dedupes
    assert parse_multi_values([' A ', 'B', 'b', '']) == ['A', 'B']


def test_explode_multi_value_column_basic():
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'Partners': ['acme ; NASA ; ;', None, '  ']
    })
    exploded = explode_multi_value_column(df, 'Partners')
    assert list(exploded['id']) == [1, 1]
    assert list(exploded['Partners']) == ['acme', 'NASA']

    # Missing column returns unchanged
    out2 = explode_multi_value_column(df, 'Missing')
    assert out2.equals(df)


def test_normalize_and_explode_multivalue():
    df = pd.DataFrame({'id': [1, 2, 3], 'Countries': ['USA; Canada, Mexico', None, '  ']})
    norm = normalize_delimited_column(df, 'Countries')
    assert norm.loc[0, 'Countries'] == ['USA', 'Canada', 'Mexico']
    assert norm.loc[1, 'Countries'] == []
    long = explode_multivalue(df, 'Countries')
    assert list(long['Countries']) == ['USA', 'Canada', 'Mexico']
    assert list(long['id']) == [1, 1, 1]


def test_map_topics_joins_cluster_name_and_keywords():
    left = pd.DataFrame({'Cluster': [1, 2, 3], 'Title': ['a', 'b', 'c']})
    right = pd.DataFrame({
        'Topic': [1, 2],
        'Cluster Name': ['Soils', 'Water'],
        'Cluster Keywords': ['soil; earth', 'water; hydro']
    })
    out = map_topics(left, right)
    # Check first two mapped values match expected
    assert list(out['Cluster Name'].iloc[:2]) == ['Soils', 'Water']
    assert list(out['Cluster Keywords'].iloc[:2]) == ['soil; earth', 'water; hydro']
    # For the third row, ensure NA rather than a string
    assert pd.isna(out['Cluster Name'].iloc[2])


def test_build_country_long_and_partner_long():
    # Country
    df_c = pd.DataFrame({'id': [1, 2], 'Countries': ['USA; canada', '']})
    c_long = build_country_long(df_c)
    assert list(c_long['Country']) == ['USA', 'Canada']
    assert list(c_long['id']) == [1, 1]

    # Partner
    df_p = pd.DataFrame({'Partners': ['acme; NASA', None], 'x': [1, 2]})
    p_long = build_partner_long(df_p)
    assert list(p_long['Partner']) == ['Acme', 'NASA']
    assert list(p_long['x']) == [1, 1]


def test_text_search_behaviour():
    df = pd.DataFrame({
        'Title': ['Soil Health', 'Water quality'],
        'Description': ['Great soil project', ''],
        'x': [1, 2]
    })
    # Case-insensitive search
    out = text_search(df, 'soil')
    assert list(out['Title']) == ['Soil Health']

    # Empty query returns original
    out2 = text_search(df, '')
    assert out2.equals(df)

    # Missing columns => empty result with same columns
    out3 = text_search(df, 'abc', columns=["MissingCol"])  # type: ignore
    assert out3.empty and list(out3.columns) == list(df.columns)
