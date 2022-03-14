import numpy as np
import pandas as pd

from epiquark import EpiMetrics


def test_timeliness(paper_example_epimetric: EpiMetrics) -> None:
    timeliness = paper_example_epimetric.timeliness("x2", 4)
    timeliness_expected = {"one": 0.0, "three": 0.0, "two": 0.0}
    timeliness == timeliness_expected


def test_calc_delay() -> None:
    delay_3 = pd.DataFrame({"value_cases": [0, 0, 0], "value_signals": [0, 0, 1]})
    assert 3 == EpiMetrics._calc_delay(delay_3)
    delay_3 = pd.DataFrame({"value_cases": [0, 0, 1], "value_signals": [0, 0, 0]})
    assert 3 == EpiMetrics._calc_delay(delay_3)
    delay_3 = pd.DataFrame({"value_cases": [0, 0, 0], "value_signals": [0, 0, 0]})
    assert 3 == EpiMetrics._calc_delay(delay_3)
    delay_3 = pd.DataFrame({"value_cases": [0, 1, 0], "value_signals": [1, 0, 0]})
    assert 3 == EpiMetrics._calc_delay(delay_3)

    delay_2 = pd.DataFrame({"value_cases": [1, 0, 0], "value_signals": [0, 0, 1]})
    assert 2 == EpiMetrics._calc_delay(delay_2)
    delay_2 = pd.DataFrame({"value_cases": [1, 0, 1], "value_signals": [0, 0, 1]})
    assert 2 == EpiMetrics._calc_delay(delay_2)

    delay_1 = pd.DataFrame({"value_cases": [0, 1, 1], "value_signals": [0, 0, 1]})
    assert 1 == EpiMetrics._calc_delay(delay_1)
    delay_1 = pd.DataFrame({"value_cases": [1, 1, 1], "value_signals": [0, 1, 1]})
    assert 1 == EpiMetrics._calc_delay(delay_1)

    delay_0 = pd.DataFrame({"value_cases": [0, 1, 0], "value_signals": [0, 1, 1]})
    assert 0 == EpiMetrics._calc_delay(delay_0)
    delay_0 = pd.DataFrame({"value_cases": [1, 1, 0], "value_signals": [1, 1, 1]})
    assert 0 == EpiMetrics._calc_delay(delay_0)


def test_time_masking(shared_datadir, paper_example_epimetric) -> None:
    time_mask = paper_example_epimetric._time_mask("x2").reset_index(drop=True)
    expected = pd.read_csv(shared_datadir / "paper_example/time_masking.csv")
    pd.testing.assert_frame_equal(time_mask, expected)

    # case where no cases appear on one label
    cases = paper_example_epimetric.cases
    cases.loc[cases["data_label"] == "one", "value"] = 0
    time_mask = paper_example_epimetric._time_mask("x2").reset_index(drop=True)
    expected = pd.read_csv(shared_datadir / "paper_example/time_masking.csv")
    expected.loc[expected["data_label"] == "one", "time_mask"] = 0
    pd.testing.assert_frame_equal(time_mask, expected)


def test_timespace_weighting(shared_datadir, paper_example_epimetric: EpiMetrics) -> None:
    gauss_weights = paper_example_epimetric.timespace_weighting(["x1", "x2"])
    expected = pd.read_csv(shared_datadir / "paper_example/gauss_weights.csv")
    pd.testing.assert_frame_equal(gauss_weights, expected)

    gauss_weights = paper_example_epimetric.timespace_weighting(["x1", "x2"], np.diag(np.ones(2)))
    expected = pd.read_csv(shared_datadir / "paper_example/gauss_weights.csv")
    pd.testing.assert_frame_equal(gauss_weights, expected)

    gauss_weights = paper_example_epimetric.timespace_weighting(["x1", "x2"], time_axis="x2")
    expected = pd.read_csv(shared_datadir / "paper_example/gauss_weights_timemask.csv")
    pd.testing.assert_frame_equal(gauss_weights, expected)
