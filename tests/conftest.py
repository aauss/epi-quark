import numpy as np
import pandas as pd
import pytest

from epiquark import EpiMetrics, Score


@pytest.fixture
def paper_example_score(shared_datadir) -> Score:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    return Score(cases, signals)


@pytest.fixture
def paper_example_epimetric(shared_datadir) -> EpiMetrics:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    return EpiMetrics(cases, signals)


def compare_dicts_with_nas(result, expected) -> None:
    for k, v in expected.items():
        np.testing.assert_equal(result[k], v)
