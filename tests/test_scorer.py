import json

import numpy as np
import pandas as pd
import pytest
from scorer import Score
from scorer.api import ThreshRequired, check_threshs, score
from sklearn import metrics


@pytest.fixture
def paper_example_score(shared_datadir) -> Score:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    return Score(cases, signals)


def test_non_case_imputation(shared_datadir, paper_example_score: Score) -> None:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    imputed = paper_example_score._impute_non_case(cases)

    imputed_expected = pd.read_csv(shared_datadir / "paper_example/non_case_imputed_long.csv")
    pd.testing.assert_frame_equal(imputed, imputed_expected, check_dtype=False)


def test_signal_imputation(shared_datadir, paper_example_score: Score) -> None:
    cases = pd.read_csv(shared_datadir / "paper_example/non_case_imputed_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/signals_long.csv")

    imputed = paper_example_score._impute_signals(signals, cases, "min")
    imputed_expected = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    pd.testing.assert_frame_equal(imputed, imputed_expected, check_dtype=False)


def test_p_di_given_x(shared_datadir, paper_example_score: Score) -> None:
    p_di_given_x = paper_example_score._p_di_given_x()
    p_di_given_x_expected = pd.read_csv(shared_datadir / "paper_example/p_di_given_x.csv")
    pd.testing.assert_frame_equal(p_di_given_x, p_di_given_x_expected, check_dtype=False)


def test_p_hat_sj_given_x(shared_datadir, paper_example_score: Score) -> None:
    p_hat_sj_given_x = paper_example_score._p_hat_sj_given_x()
    p_hat_sj_given_x_expected = pd.read_csv(
        shared_datadir / "paper_example/p_hat_sj_given_x_long.csv"
    )
    pd.testing.assert_frame_equal(p_hat_sj_given_x, p_hat_sj_given_x_expected, check_dtype=False)


def test_p_hat_di_given_sj_x(shared_datadir, paper_example_score: Score) -> None:
    p_hat_di_given_sj_x = paper_example_score._p_hat_di_given_sj_x()
    p_hat_di_given_sj_x_expected = pd.read_csv(
        shared_datadir / "paper_example/p_hat_di_given_sj_x.csv"
    )
    str_cols = list(p_hat_di_given_sj_x.select_dtypes(exclude="number").columns)
    pd.testing.assert_frame_equal(
        p_hat_di_given_sj_x.sort_values(by=str_cols).reset_index(drop=True),
        p_hat_di_given_sj_x_expected.sort_values(by=str_cols).reset_index(drop=True),
        check_dtype=False,
    )


def test_p_hat_di(shared_datadir, paper_example_score: Score) -> None:
    p_hat_di = (
        paper_example_score._p_hat_di().sort_values(by=["x1", "x2", "d_i"]).reset_index(drop=True)
    )
    p_hat_di_expected = (
        pd.read_csv(shared_datadir / "paper_example/p_hat_di.csv")
        .sort_values(by=["x1", "x2", "d_i"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        p_hat_di,
        p_hat_di_expected,
        check_dtype=False,
    )


def test_eval_df(shared_datadir, paper_example_score: Score) -> None:
    eval_df = paper_example_score._eval_df()
    eval_df_expected = pd.read_csv(shared_datadir / "paper_example/eval_df.csv")
    pd.testing.assert_frame_equal(eval_df, eval_df_expected, check_dtype=False)


def test_score(paper_example_score: Score) -> None:
    scores = paper_example_score.calc_score(metrics.f1_score, 1 / 2, 1 / 5)
    assert scores == {
        "endemic": 0.6153846153846154,
        "non_case": 1.0,
        "one": 0.7499999999999999,
        "three": 0.0,
        "two": 0.22222222222222224,
    }


def test_case_data_error(shared_datadir) -> None:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")

    cases_with_nans = cases.copy()
    cases_with_nans.loc[2, "value"] = pd.NA
    with pytest.raises(ValueError, match="Cases DataFrame must not contain any NaN values."):
        Score(cases_with_nans, signals)

    cases_negative = cases.copy()
    cases_negative.at[2, "value"] = -1
    with pytest.raises(ValueError, match="Case counts must be non-negative, whole numbers."):
        Score(cases_negative, signals)

    cases_float = cases.copy()
    cases_float.loc[:, "value"] = cases_float.loc[:, "value"].astype(float)
    with pytest.raises(ValueError, match="Case counts must be non-negative, whole numbers."):
        Score(cases_float, signals)

    cases_non_case_label = cases.copy()
    cases_non_case_label.at[0, "data_label"] = "non_case"
    with pytest.raises(
        ValueError,
        match="Please remove entries with label 'non_cases' from cases DataFrame. This label is included automatically and therefore internally reseverd.",
    ):
        Score(cases_non_case_label, signals)


def test_signal_coord_error(shared_datadir) -> None:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")

    signals.at[0, "x1"] = 6
    with pytest.raises(
        ValueError, match="Coordinats of 'signals' must be a subset of coordinats of 'cases'."
    ):
        Score(cases, signals)


def test_signals_float_error(shared_datadir) -> None:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")

    signals_negative_value = signals.copy()
    signals_negative_value.at[0, "value"] = -1.2
    with pytest.raises(
        ValueError, match="'values' in signal DataFrame must be floats between 0 and 1."
    ):
        Score(cases, signals_negative_value)

    signals_high_values = signals.copy()
    signals_high_values.at[0, "value"] = 2.0
    with pytest.raises(
        ValueError, match="'values' in signal DataFrame must be floats between 0 and 1."
    ):
        Score(cases, signals_high_values)

    signals_not_float = signals.copy()
    signals_not_float.loc[:, "value"] = 1
    with pytest.raises(
        ValueError, match="'values' in signal DataFrame must be floats between 0 and 1."
    ):
        Score(cases, signals_not_float)


def test_signals_equal_number_error(shared_datadir) -> None:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")

    signals = signals.iloc[1:]
    with pytest.raises(
        ValueError, match="Each coordinate must contain the same amount of signals."
    ):
        Score(cases, signals)


def test_class_based_conf_mat(paper_example_score: Score) -> None:
    confusion_matrix = paper_example_score.class_based_conf_mat()
    confusion_matrix_expected = {
        "endemic": [[16, 4], [1, 4]],
        "non_case": [[13, 0], [0, 12]],
        "one": [[20, 2], [0, 3]],
        "three": [[18, 5], [2, 0]],
        "two": [[17, 4], [3, 1]],
    }
    assert json.dumps(confusion_matrix, sort_keys=True) == json.dumps(
        confusion_matrix_expected, sort_keys=True
    )

    confusion_matrix_weighted = paper_example_score.class_based_conf_mat(weighted=True)
    confusion_matrix_weighted_expected = {
        "endemic": [[20, 4], [1, 4]],
        "non_case": [[13, 0], [0, 12]],
        "one": [[20, 2], [0, 3]],
        "three": [[19, 5], [2, 0]],
        "two": [[17, 4], [3, 1]],
    }
    assert json.dumps(confusion_matrix_weighted, sort_keys=True) == json.dumps(
        confusion_matrix_weighted_expected, sort_keys=True
    )


def test_thresh_check_class() -> None:
    tr = ThreshRequired(p_thresh=True, p_hat_thresh=True)
    with pytest.raises(
        ValueError, match=f"This metric requires p_thresh and requires p_hat_thresh."
    ):
        tr.check_threshs_correct(None, 0.4)

    tr = ThreshRequired(p_thresh=True, p_hat_thresh=False)
    with pytest.raises(
        ValueError, match=f"This metric requires p_thresh and must not contain p_hat_thresh."
    ):
        tr.check_threshs_correct(None, 0.4)

    tr = ThreshRequired(p_thresh=False, p_hat_thresh=True)
    with pytest.raises(
        ValueError, match=f"This metric must not contain p_thresh and requires p_hat_thresh."
    ):
        tr.check_threshs_correct(1, 0.4)

    tr = ThreshRequired(p_thresh=False, p_hat_thresh=False)
    with pytest.raises(
        ValueError,
        match=f"This metric must not contain p_thresh and must not contain p_hat_thresh.",
    ):
        tr.check_threshs_correct(None, 0.4)

    tr = ThreshRequired(p_thresh=True, p_hat_thresh=True)
    tr.check_threshs_correct(1, 0.4)


def test_thresh_check_func():
    check_threshs("f1", p_thresh=1, p_hat_thresh=0.4)
    with pytest.raises(KeyError):
        check_threshs("not a metric", p_thresh=1, p_hat_thresh=0.4)


def test_scorer_api_no_weighting(shared_datadir, paper_example_score: Score) -> None:
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "f1",
        0.5,
        0.2,
    ) == {
        "endemic": 0.6153846153846154,
        "non_case": 1.0,
        "one": 0.7499999999999999,
        "three": 0.0,
        "two": 0.22222222222222224,
    }
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "sensitivity",
            0.5,
            0.2,
        )
        == {"endemic": 0.8, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.25}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "recall",
            0.5,
            0.2,
        )
        == {"endemic": 0.8, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.25}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "tpr",
            0.5,
            0.2,
        )
        == {"endemic": 0.8, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.25}
    )
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "specificity",
        0.5,
        0.2,
    ) == {
        "endemic": 0.8,
        "non_case": 1.0,
        "one": 0.9090909090909091,
        "three": 0.782608695652174,
        "two": 0.8095238095238095,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "tnr",
        0.5,
        0.2,
    ) == {
        "endemic": 0.8,
        "non_case": 1.0,
        "one": 0.9090909090909091,
        "three": 0.782608695652174,
        "two": 0.8095238095238095,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "matthews",
        0.5,
        0.2,
    ) == {
        "endemic": 0.5144957554275266,
        "non_case": 1.0,
        "one": 0.7385489458759964,
        "three": -0.14744195615489714,
        "two": 0.0545544725589981,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "brier",
        0.5,
    ) == {
        "endemic": 0.17,
        "non_case": 0.05,
        "one": 0.07333333333333333,
        "three": 0.08666666666666666,
        "two": 0.15333333333333335,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "auc",
        0.5,
    ) == {
        "endemic": 0.7800000000000001,
        "non_case": 1.0,
        "one": 0.9545454545454545,
        "three": 0.6086956521739131,
        "two": 0.47023809523809523,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "r2",
    ) == {
        "endemic": -0.013702460850111953,
        "non_case": 0.7996794871794872,
        "one": 0.3055555555555556,
        "three": -0.7795138888888888,
        "two": -0.17753623188405832,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "mse",
    ) == {
        "endemic": 0.1611111111111111,
        "non_case": 0.05,
        "one": 0.07333333333333333,
        "three": 0.04555555555555555,
        "two": 0.13,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "mae",
    ) == {
        "endemic": 0.20666666666666664,
        "non_case": 0.1,
        "one": 0.17333333333333337,
        "three": 0.1533333333333333,
        "two": 0.23333333333333336,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "fpr",
        0.5,
        0.2,
    ) == {
        "endemic": 0.2,
        "non_case": 0.0,
        "one": 0.09090909090909091,
        "three": 0.21739130434782608,
        "two": 0.19047619047619047,
    }
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "fnr",
            0.5,
            0.2,
        )
        == {"endemic": 0.2, "non_case": 0.0, "one": 0.0, "three": 1.0, "two": 0.75}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "precision",
            0.5,
            0.2,
        )
        == {"endemic": 0.5, "non_case": 1.0, "one": 0.6, "three": 0.0, "two": 0.2}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "ppv",
            0.5,
            0.2,
        )
        == {"endemic": 0.5, "non_case": 1.0, "one": 0.6, "three": 0.0, "two": 0.2}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "npv",
            0.5,
            0.2,
        )
        == {"endemic": 0.9411764705882353, "non_case": 1.0, "one": 1.0, "three": 0.9, "two": 0.85}
    )


def test_scorer_api_case_weighting(shared_datadir, paper_example_score: Score) -> None:
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "f1",
            0.5,
            0.2,
            weights="cases",
        )
        == {"endemic": 0.8888888888888888, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.4}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "sensitivity",
            0.5,
            0.2,
            weights="cases",
        )
        == {"endemic": 0.8888888888888888, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.25}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "recall",
            0.5,
            0.2,
            weights="cases",
        )
        == {"endemic": 0.8888888888888888, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.25}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "tpr",
            0.5,
            0.2,
            weights="cases",
        )
        == {"endemic": 0.8888888888888888, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.25}
    )

    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "specificity",
        0.5,
        0.2,
        weights="cases",
    )
    expected = {"endemic": 0.0, "non_case": np.nan, "one": np.nan, "three": np.nan, "two": np.nan}
    _compare_dicts_with_nas(result, expected)

    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "tnr",
        0.5,
        0.2,
        weights="cases",
    )
    expected = {"endemic": 0.0, "non_case": np.nan, "one": np.nan, "three": np.nan, "two": np.nan}
    _compare_dicts_with_nas(result, expected)

    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "matthews",
            0.5,
            0.2,
            weights="cases",
        )
        == {"endemic": -0.1111111111111111, "non_case": 0.0, "one": 0.0, "three": 0.0, "two": 0.0}
    )
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "brier",
        0.5,
        weights="cases",
    ) == {
        "endemic": 0.3,
        "non_case": 0.10416666666666667,
        "one": 0.4444444444444445,
        "three": 0.6944444444444445,
        "two": 0.7847222222222223,
    }

    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "auc",
        0.5,
        weights="cases",
    )
    expected = {"endemic": 0.5, "non_case": np.nan, "one": np.nan, "three": np.nan, "two": np.nan}
    _compare_dicts_with_nas(result, expected)

    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "r2",
        weights="cases",
    ) == {
        "endemic": -5.944444444444444,
        "non_case": 0.0,
        "one": 0.0,
        "three": -32.000000000000014,
        "two": -12.629629629629632,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "mse",
        weights="cases",
    ) == {
        "endemic": 0.2777777777777778,
        "non_case": 0.10416666666666667,
        "one": 0.4444444444444445,
        "three": 0.20370370370370372,
        "two": 0.638888888888889,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "mae",
        weights="cases",
    ) == {
        "endemic": 0.4666666666666667,
        "non_case": 0.20833333333333334,
        "one": 0.6666666666666666,
        "three": 0.4444444444444445,
        "two": 0.75,
    }
    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "fpr",
        0.5,
        0.2,
        weights="cases",
    )
    expected = {"endemic": 1.0, "non_case": np.nan, "one": np.nan, "three": np.nan, "two": np.nan}
    _compare_dicts_with_nas(result, expected)
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "fnr",
            0.5,
            0.2,
            weights="cases",
        )
        == {"endemic": 0.1111111111111111, "non_case": 0.0, "one": 0.0, "three": 1.0, "two": 0.75}
    )
    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "precision",
        0.5,
        0.2,
        weights="cases",
    )
    expected = {
        "endemic": 0.8888888888888888,
        "non_case": 1.0,
        "one": 1.0,
        "three": np.nan,
        "two": 1.0,
    }
    _compare_dicts_with_nas(result, expected)
    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "ppv",
        0.5,
        0.2,
        weights="cases",
    )
    expected = {
        "endemic": 0.8888888888888888,
        "non_case": 1.0,
        "one": 1.0,
        "three": np.nan,
        "two": 1.0,
    }
    _compare_dicts_with_nas(result, expected)
    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "npv",
        0.5,
        0.2,
        weights="cases",
    )
    expected = {"endemic": 0.0, "non_case": np.nan, "one": np.nan, "three": 0.0, "two": 0.0}
    _compare_dicts_with_nas(result, expected)


def test_scorer_api_timespace_weighting(shared_datadir, paper_example_score: Score) -> None:
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "f1",
        0.5,
        0.2,
        weights="timespace",
        gauss_dims=["x1"],
        time_axis="x2",
    ) == {
        "endemic": 0.6401730435463963,
        "non_case": 1.0,
        "one": 0.8040975332158111,
        "three": 0.0,
        "two": 0.2520523228826373,
    }
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "sensitivity",
            0.5,
            0.2,
            weights="timespace",
            gauss_dims=["x1"],
            time_axis="x2",
        )
        == {"endemic": 0.764115797871024, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.25}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "recall",
            0.5,
            0.2,
            weights="timespace",
            gauss_dims=["x1"],
            time_axis="x2",
        )
        == {"endemic": 0.764115797871024, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.25}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "tpr",
            0.5,
            0.2,
            weights="timespace",
            gauss_dims=["x1"],
            time_axis="x2",
        )
        == {"endemic": 0.764115797871024, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.25}
    )

    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "specificity",
        0.5,
        0.2,
        gauss_dims=["x1"],
        time_axis="x2",
    ) == {
        "endemic": 0.8,
        "non_case": 1.0,
        "one": 0.9090909090909091,
        "three": 0.782608695652174,
        "two": 0.8095238095238095,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "tnr",
        0.5,
        0.2,
        gauss_dims=["x1"],
        time_axis="x2",
    ) == {
        "endemic": 0.8,
        "non_case": 1.0,
        "one": 0.9090909090909091,
        "three": 0.782608695652174,
        "two": 0.8095238095238095,
    }

    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "matthews",
        0.5,
        0.2,
        weights="timespace",
        gauss_dims=["x1"],
        time_axis="x2",
    ) == {
        "endemic": 0.527579512894754,
        "non_case": 1.0,
        "one": 0.7687461985833044,
        "three": -0.1720992466804028,
        "two": 0.07080661892053153,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "brier",
        0.5,
        weights="timespace",
        gauss_dims=["x1"],
        time_axis="x2",
    ) == {
        "endemic": 0.14499248666797407,
        "non_case": 0.0505799891010867,
        "one": 0.1069915189009573,
        "three": 0.14508347169779873,
        "two": 0.179749369637523,
    }

    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "auc",
        0.5,
        weights="timespace",
        gauss_dims=["x1"],
        time_axis="x2",
    )
    expected = {
        "endemic": 0.7921447900305028,
        "non_case": 1.0,
        "one": 0.939463690698793,
        "three": 0.5596406702450042,
        "two": 0.4800873364045545,
    }
    _compare_dicts_with_nas(result, expected)

    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "r2",
        weights="timespace",
        gauss_dims=["x1"],
        time_axis="x2",
    ) == {
        "endemic": 0.21861769964021116,
        "non_case": 0.7973268665903567,
        "one": 0.32884147528534324,
        "three": -0.18093896486554217,
        "two": -0.16042389293765646,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "mse",
        weights="timespace",
        gauss_dims=["x1"],
        time_axis="x2",
    ) == {
        "endemic": 0.1333385609183281,
        "non_case": 0.0505799891010867,
        "one": 0.1069915189009573,
        "three": 0.0579654094964463,
        "two": 0.15106763035595444,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "mae",
        weights="timespace",
        gauss_dims=["x1"],
        time_axis="x2",
    ) == {
        "endemic": 0.1864166710366832,
        "non_case": 0.1011599782021734,
        "one": 0.21157654307874668,
        "three": 0.19253843867078885,
        "two": 0.2518907794255543,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "fpr",
        0.5,
        0.2,
        gauss_dims=["x1"],
        time_axis="x2",
    ) == {
        "endemic": 0.2,
        "non_case": 0.0,
        "one": 0.09090909090909091,
        "three": 0.21739130434782608,
        "two": 0.19047619047619047,
    }
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "fnr",
            0.5,
            0.2,
            gauss_dims=["x1"],
            time_axis="x2",
        )
        == {"endemic": 0.2, "non_case": 0.0, "one": 0.0, "three": 1.0, "two": 0.75}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "precision",
            0.5,
            0.2,
            gauss_dims=["x1"],
            time_axis="x2",
        )
        == {"endemic": 0.5, "non_case": 1.0, "one": 0.6, "three": 0.0, "two": 0.2}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "ppv",
            0.5,
            0.2,
            gauss_dims=["x1"],
            time_axis="x2",
        )
        == {"endemic": 0.5, "non_case": 1.0, "one": 0.6, "three": 0.0, "two": 0.2}
    )
    assert (
        score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "npv",
            0.5,
            0.2,
            gauss_dims=["x1"],
            time_axis="x2",
        )
        == {"endemic": 0.9411764705882353, "non_case": 1.0, "one": 1.0, "three": 0.9, "two": 0.85}
    )


def test_scorer_api_errors(shared_datadir, paper_example_score: Score) -> None:
    with pytest.raises(ValueError, match="weights must be None, 'cases', or 'timespace'."):
        assert score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "mse",
            weights="not a weighting strategy",
        )


def _compare_dicts_with_nas(result, expected) -> None:
    for k, v in expected.items():
        np.testing.assert_equal(result[k], v)
