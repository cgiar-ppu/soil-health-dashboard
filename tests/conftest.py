import pytest

# Clear Streamlit cache-decorated functions between tests if available
try:
    import src.data_loader as dl
except Exception:  # pragma: no cover - if module import fails, nothing to clear
    dl = None  # type: ignore


@pytest.fixture(autouse=True)
def clear_streamlit_caches():
    if dl is not None:
        for fn_name in ("load_main_dataset", "load_cluster_keywords", "load_cluster_summaries"):
            fn = getattr(dl, fn_name, None)
            if fn is not None:
                try:
                    # streamlit.cache_data-decorated functions expose .clear()
                    fn.clear()  # type: ignore[attr-defined]
                except Exception:
                    # No-op if clear not supported or not decorated
                    pass
    yield
