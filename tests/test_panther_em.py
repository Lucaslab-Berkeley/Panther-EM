import panther_em


def test_imports_with_version():
    assert isinstance(panther_em.__version__, str)
