import importlib


def test_streamlit_app_importable():
    module = importlib.import_module('erlang_calculator')
    assert hasattr(module, 'run_app')
