import pytest

def test_import():
    try:
        import scimes
        scimes_importable = True
    except ImportError:
        scimes_importable = False
    assert scimes_importable, 'Cannot import SCIMES at all!'

def test_function_import():
    # we want to be able to import functions like this
    from scimes import mat_smooth

    # test for all of them
    import scimes
    from scimes.scimes import __all__ as API
    for name in API:
        assert getattr(scimes, name) is not None, f'Cannot `from scimes import {name}`'


    from scimes.plotting import dendroplot_clusters
