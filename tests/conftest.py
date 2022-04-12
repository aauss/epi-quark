import numpy as np
import pandas as pd
import pytest

from epiquark import ScoreCalculator, Timeliness, TimeSpaciness


@pytest.fixture
def paper_example_dfs(shared_datadir) -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    return cases, signals


@pytest.fixture
def paper_example_score(shared_datadir) -> ScoreCalculator:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    return ScoreCalculator(cases, signals)


@pytest.fixture
def paper_example_timespaciness(shared_datadir) -> TimeSpaciness:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    return TimeSpaciness(cases, signals)


@pytest.fixture
def paper_example_timeliness(shared_datadir) -> Timeliness:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    return Timeliness(cases, signals)


def compare_dicts_with_nas(result, expected) -> None:
    for k, v in expected.items():
        np.testing.assert_equal(result[k], v)
