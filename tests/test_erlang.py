import numpy as np
import pandas as pd
from erlang_calculator import X, CHAT, BL


def test_erlang_b_basic():
    assert abs(X.erlang_b(5, 10) - 0.01838457) < 1e-6


def test_erlang_c_basic():
    assert abs(X.erlang_c(5, 10) - 0.03610536) < 1e-6


def test_service_level_and_asa():
    sl = CHAT.service_level(5, 10, 180, 20)
    asa = CHAT.asa(5, 10, 180)
    assert abs(sl - 0.97928443) < 1e-6
    assert abs(asa - 1.29979293) < 1e-6


def test_required_agents():
    agents = CHAT.required_agents(5, 180, 0.8, 20)
    assert isinstance(agents, int)
    assert agents >= 5


def test_sensitivity_dataframe():
    df = BL.sensitivity(np.linspace(4, 8, 5), agents=8, aht=180, target=20)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["traffic", "service_level"]
    assert len(df) == 5


def test_monte_carlo_series():
    series = BL.monte_carlo(6, agents=8, aht=180, target=20, iters=50)
    assert isinstance(series, pd.Series)
    assert len(series) == 50
