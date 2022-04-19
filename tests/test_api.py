import numpy as np
import pandas as pd
import pytest

from epiquark.api import _check_threshs, _ThreshRequired, conf_matrix, score, timeliness

from .conftest import compare_dicts_with_nas


def test_thresh_check_class() -> None:
    tr = _ThreshRequired(p_thresh=True, p_hat_thresh=True)
    with pytest.raises(
        ValueError, match="This metric requires p_thresh and requires p_hat_thresh."
    ):
        tr.check_threshs_correct(None, 0.4)

    tr = _ThreshRequired(p_thresh=True, p_hat_thresh=False)
    with pytest.raises(
        ValueError, match="This metric requires p_thresh and must not contain p_hat_thresh."
    ):
        tr.check_threshs_correct(None, 0.4)

    tr = _ThreshRequired(p_thresh=False, p_hat_thresh=True)
    with pytest.raises(
        ValueError, match="This metric must not contain p_thresh and requires p_hat_thresh."
    ):
        tr.check_threshs_correct(1, 0.4)

    tr = _ThreshRequired(p_thresh=False, p_hat_thresh=False)
    with pytest.raises(
        ValueError,
        match="This metric must not contain p_thresh and must not contain p_hat_thresh.",
    ):
        tr.check_threshs_correct(None, 0.4)

    tr = _ThreshRequired(p_thresh=True, p_hat_thresh=True)
    tr.check_threshs_correct(1, 0.4)


def test_thresh_check_func():
    _check_threshs("f1", p_thresh=1, p_hat_thresh=0.4)
    with pytest.raises(KeyError):
        _check_threshs("not a metric", p_thresh=1, p_hat_thresh=0.4)


def test_scorer_api_no_weighting(shared_datadir) -> None:
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
        "two": 0.25,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "sensitivity",
        0.5,
        0.2,
    ) == {"endemic": 0.8, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 1 / 3}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "recall",
        0.5,
        0.2,
    ) == {"endemic": 0.8, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 1 / 3}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "tpr",
        0.5,
        0.2,
    ) == {"endemic": 0.8, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 1 / 3}
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
        "three": 0.7916666666666666,
        "two": 0.8181818181818182,
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
        "three": 0.7916666666666666,
        "two": 0.8181818181818182,
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
        "three": -0.10206207261596575,
        "two": 0.12309149097933272,
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
        "three": 0.06,
        "two": 0.12666666666666665,
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
        "three": 0.6041666666666666,
        "two": 0.4242424242424242,
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
        "three": 0.20833333333333334,
        "two": 0.18181818181818182,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "fnr",
        0.5,
        0.2,
    ) == {"endemic": 0.2, "non_case": 0.0, "one": 0.0, "three": 1.0, "two": 2 / 3}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "precision",
        0.5,
        0.2,
    ) == {"endemic": 0.5, "non_case": 1.0, "one": 0.6, "three": 0.0, "two": 0.2}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "ppv",
        0.5,
        0.2,
    ) == {"endemic": 0.5, "non_case": 1.0, "one": 0.6, "three": 0.0, "two": 0.2}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "npv",
        0.5,
        0.2,
    ) == {"endemic": 0.9411764705882353, "non_case": 1.0, "one": 1.0, "three": 0.95, "two": 0.9}


def test_scorer_api_case_weighting(shared_datadir) -> None:
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "f1",
        0.5,
        0.2,
        weighting="cases",
    ) == {"endemic": 0.8888888888888888, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 0.5}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "sensitivity",
        0.5,
        0.2,
        weighting="cases",
    ) == {"endemic": 0.8888888888888888, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 1 / 3}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "recall",
        0.5,
        0.2,
        weighting="cases",
    ) == {"endemic": 0.8888888888888888, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 1 / 3}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "tpr",
        0.5,
        0.2,
        weighting="cases",
    ) == {"endemic": 0.8888888888888888, "non_case": 1.0, "one": 1.0, "three": 0.0, "two": 1 / 3}

    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "specificity",
        0.5,
        0.2,
        weighting="cases",
    )
    expected = {"endemic": 0.0, "non_case": np.nan, "one": np.nan, "three": 1.0, "two": 1.0}
    compare_dicts_with_nas(result, expected)

    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "tnr",
        0.5,
        0.2,
        weighting="cases",
    )
    expected = {"endemic": 0.0, "non_case": np.nan, "one": np.nan, "three": 1.0, "two": 1.0}
    compare_dicts_with_nas(result, expected)

    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "matthews",
        0.5,
        0.2,
        weighting="cases",
    ) == {"endemic": -0.1111111111111111, "non_case": 0.0, "one": 0.0, "three": 0.0, "two": 1 / 3}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "brier",
        0.5,
        weighting="cases",
    ) == {
        "endemic": 0.3,
        "non_case": 0.10416666666666667,
        "one": 0.4444444444444445,
        "three": 0.47222222222222227,
        "two": 0.6180555555555556,
    }

    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "auc",
        0.5,
        weighting="cases",
    )
    expected = {"endemic": 0.5, "non_case": np.nan, "one": np.nan, "three": 0.5, "two": 1 / 3}
    compare_dicts_with_nas(result, expected)

    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "r2",
        weighting="cases",
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
        weighting="cases",
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
        weighting="cases",
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
        weighting="cases",
    )
    expected = {"endemic": 1.0, "non_case": np.nan, "one": np.nan, "three": 0.0, "two": 0.0}
    compare_dicts_with_nas(result, expected)
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "fnr",
        0.5,
        0.2,
        weighting="cases",
    ) == {"endemic": 0.1111111111111111, "non_case": 0.0, "one": 0.0, "three": 1.0, "two": 2 / 3}
    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "precision",
        0.5,
        0.2,
        weighting="cases",
    )
    expected = {
        "endemic": 0.8888888888888888,
        "non_case": 1.0,
        "one": 1.0,
        "three": np.nan,
        "two": 1.0,
    }
    compare_dicts_with_nas(result, expected)
    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "ppv",
        0.5,
        0.2,
        weighting="cases",
    )
    expected = {
        "endemic": 0.8888888888888888,
        "non_case": 1.0,
        "one": 1.0,
        "three": np.nan,
        "two": 1.0,
    }
    compare_dicts_with_nas(result, expected)
    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "npv",
        0.5,
        0.2,
        weighting="cases",
    )
    expected = {"endemic": 0.0, "non_case": np.nan, "one": np.nan, "three": 1 / 3, "two": 1 / 3}
    compare_dicts_with_nas(result, expected)


def test_scorer_api_timespace_weighting(shared_datadir) -> None:
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "f1",
        0.5,
        0.2,
        weighting="timespace",
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {
        "endemic": 0.6401730435463963,
        "non_case": 1.0,
        "one": 0.8040975332158111,
        "three": 0.0,
        "two": 0.2883980180668917,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "sensitivity",
        0.5,
        0.2,
        weighting="timespace",
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {
        "endemic": 0.764115797871024,
        "non_case": 1.0,
        "one": 1.0,
        "three": 0.0,
        "two": 0.33333333333333337,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "recall",
        0.5,
        0.2,
        weighting="timespace",
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {
        "endemic": 0.764115797871024,
        "non_case": 1.0,
        "one": 1.0,
        "three": 0.0,
        "two": 0.33333333333333337,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "tpr",
        0.5,
        0.2,
        weighting="timespace",
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {
        "endemic": 0.764115797871024,
        "non_case": 1.0,
        "one": 1.0,
        "three": 0.0,
        "two": 0.33333333333333337,
    }

    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "specificity",
        0.5,
        0.2,
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {
        "endemic": 0.8,
        "non_case": 1.0,
        "one": 0.9090909090909091,
        "three": 0.7916666666666666,
        "two": 0.8181818181818182,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "tnr",
        0.5,
        0.2,
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {
        "endemic": 0.8,
        "non_case": 1.0,
        "one": 0.9090909090909091,
        "three": 0.7916666666666666,
        "two": 0.8181818181818182,
    }

    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "matthews",
        0.5,
        0.2,
        weighting="timespace",
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {
        "endemic": 0.527579512894754,
        "non_case": 1.0,
        "one": 0.7687461985833044,
        "three": -0.1159204500759278,
        "two": 0.14727817191256395,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "brier",
        0.5,
        weighting="timespace",
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {
        "endemic": 0.14499248666797407,
        "non_case": 0.0505799891010867,
        "one": 0.1069915189009573,
        "three": 0.08857445837800257,
        "two": 0.14697023903001608,
    }

    result = score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "auc",
        0.5,
        weighting="timespace",
        time_space_weighting={"x1": 1},
        time_axis="x2",
    )
    expected = {
        "endemic": 0.7921447900305028,
        "non_case": 1.0,
        "one": 0.939463690698793,
        "three": 0.554117121682121,
        "two": 0.43076069413363327,
    }
    compare_dicts_with_nas(result, expected)

    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "r2",
        weighting="timespace",
        time_space_weighting={"x1": 1},
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
        weighting="timespace",
        time_space_weighting={"x1": 1},
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
        weighting="timespace",
        time_space_weighting={"x1": 1},
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
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {
        "endemic": 0.2,
        "non_case": 0.0,
        "one": 0.09090909090909091,
        "three": 0.20833333333333334,
        "two": 0.18181818181818182,
    }
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "fnr",
        0.5,
        0.2,
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {"endemic": 0.2, "non_case": 0.0, "one": 0.0, "three": 1.0, "two": 0.6666666666666666}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "precision",
        0.5,
        0.2,
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {"endemic": 0.5, "non_case": 1.0, "one": 0.6, "three": 0.0, "two": 0.2}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "ppv",
        0.5,
        0.2,
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {"endemic": 0.5, "non_case": 1.0, "one": 0.6, "three": 0.0, "two": 0.2}
    assert score(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "npv",
        0.5,
        0.2,
        time_space_weighting={"x1": 1},
        time_axis="x2",
    ) == {"endemic": 0.9411764705882353, "non_case": 1.0, "one": 1.0, "three": 0.95, "two": 0.9}


def test_scorer_api_errors(shared_datadir) -> None:
    with pytest.raises(ValueError, match="weighting must be None, 'cases', or 'timespace'."):
        assert score(
            pd.read_csv("tests/data/paper_example/cases_long.csv"),
            pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
            "mse",
            weighting="not a weighting strategy",
        )


def test_conf_matrix_api(shared_datadir) -> None:
    confusion_matrix = conf_matrix(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        0.5,
        0.2,
    )

    expected = {
        "endemic": np.array([[16, 4], [1, 4]]),
        "non_case": np.array([[13, 0], [0, 12]]),
        "one": np.array([[20, 2], [0, 3]]),
        "three": np.array([[19, 5], [1, 0]]),
        "two": np.array([[18, 4], [2, 1]]),
    }

    np.testing.assert_equal(confusion_matrix, expected)


def test_timeliness_api():
    output = timeliness(
        pd.read_csv("tests/data/paper_example/cases_long.csv"),
        pd.read_csv("tests/data/paper_example/imputed_signals_long.csv"),
        "x2",
        2,
    )
    expected = {"one": 0.0, "three": 0.0, "two": 0.0}
    assert output == expected
