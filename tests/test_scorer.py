import numpy as np
import pandas as pd
import pytest
from sklearn import metrics

from epiquark import ScoreCalculator


def test_non_case_imputation(shared_datadir, paper_example_score: ScoreCalculator) -> None:
    cases = pd.read_csv(shared_datadir / "paper_example/cases_long.csv")
    imputed = paper_example_score._impute_non_case(cases)

    imputed_expected = pd.read_csv(shared_datadir / "paper_example/non_case_imputed_long.csv")
    pd.testing.assert_frame_equal(imputed, imputed_expected, check_dtype=False)


def test_p_d_given_x(shared_datadir, paper_example_score: ScoreCalculator) -> None:
    p_d_given_x = paper_example_score._p_d_given_x()
    p_d_given_x_expected = pd.read_csv(shared_datadir / "paper_example/p_d_given_x.csv")
    pd.testing.assert_frame_equal(p_d_given_x, p_d_given_x_expected, check_dtype=False)


def test_p_s_given_x(shared_datadir, paper_example_score: ScoreCalculator) -> None:
    p_s_given_x = paper_example_score._p_s_given_x()
    p_s_given_x_expected = pd.read_csv(shared_datadir / "paper_example/p_s_given_x_long.csv")
    pd.testing.assert_frame_equal(p_s_given_x, p_s_given_x_expected, check_dtype=False)


def test_p_d_given_s(shared_datadir, paper_example_score: ScoreCalculator) -> None:
    p_d_given_s_x = paper_example_score._p_d_given_s()
    p_d_given_s_x_expected = pd.read_csv(shared_datadir / "paper_example/p_d_given_s.csv")
    str_cols = list(p_d_given_s_x.select_dtypes(exclude="number").columns)
    pd.testing.assert_frame_equal(
        p_d_given_s_x.sort_values(by=str_cols).reset_index(drop=True),
        p_d_given_s_x_expected.sort_values(by=str_cols).reset_index(drop=True),
        check_dtype=False,
    )


def test_p_hat_d(shared_datadir, paper_example_score: ScoreCalculator) -> None:
    p_hat_d = (
        paper_example_score._p_hat_d().sort_values(by=["x1", "x2", "d"]).reset_index(drop=True)
    )
    p_hat_d_expected = (
        pd.read_csv(shared_datadir / "paper_example/p_hat_d.csv")
        .sort_values(by=["x1", "x2", "d"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        p_hat_d,
        p_hat_d_expected,
        check_dtype=False,
    )


def test_eval_df(shared_datadir, paper_example_score: ScoreCalculator) -> None:
    eval_df = paper_example_score._eval_df()
    eval_df_expected = pd.read_csv(shared_datadir / "paper_example/eval_df.csv")
    pd.testing.assert_frame_equal(eval_df, eval_df_expected, check_dtype=False)


def test_score(paper_example_score: ScoreCalculator) -> None:
    scores = paper_example_score.calc_score(metrics.f1_score, 1 / 2, 1 / 5)
    assert scores == {
        "endemic": 0.6153846153846154,
        "non_case": 1.0,
        "one": 0.7499999999999999,
        "three": 0.0,
        "two": 0.25,
    }


def test_check_no_nans_exist(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    cases.loc[2, "value"] = pd.NA
    with pytest.raises(ValueError, match="Cases DataFrame must not contain any NaN values."):
        ScoreCalculator(cases, signals)


def test_check_cases_pos_ints(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    cases_negative = cases.copy()
    cases_negative.at[2, "value"] = -1
    with pytest.raises(ValueError, match="Case counts must be non-negative, whole numbers."):
        ScoreCalculator(cases_negative, signals)

    cases_float = cases.copy()
    cases_float.loc[:, "value"] = cases_float.loc[:, "value"].astype(float)
    with pytest.raises(ValueError, match="Case counts must be non-negative, whole numbers."):
        ScoreCalculator(cases_float, signals)


def test_check_non_cases_not_include(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    cases.at[0, "data_label"] = "non_case"
    with pytest.raises(
        ValueError,
        match=(
            "Please remove entries with label 'non_cases' from cases DataFrame. "
            "This label is included automatically and therefore internally reserved."
        ),
    ):
        ScoreCalculator(cases, signals)


def test_check_non_cases_not_included(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    cases = cases.loc[cases.loc[:, "data_label"] != "endemic"]
    with pytest.raises(
        ValueError,
        match=("Please add the label 'endemic' to your cases DataFrame."),
    ):
        ScoreCalculator(cases, signals)


def test_check_data_label_consitency(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    cases_missing_label = cases.copy()
    cases_missing_label = cases_missing_label.iloc[1:]
    with pytest.raises(
        ValueError,
        match=(
            "The set of all data labels in the cases DataFrame "
            "must equal the available data labels per cell"
        ),
    ):
        ScoreCalculator(cases_missing_label, signals)

    cases_additional_label = cases.copy()
    cases_additional_label = cases_additional_label.append(
        pd.DataFrame({"x1": [0.0], "x2": [0.0], "data_label": ["additional_label"], "value": [5]})
    )
    with pytest.raises(
        ValueError,
        match=(
            "The set of all data labels in the cases DataFrame "
            "must equal the available data labels per cell"
        ),
    ):
        ScoreCalculator(cases_additional_label, signals)


def test_signal_empty_coord_error(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    signals.loc[(signals.loc[:, "x1"] == 0) & (signals.loc[:, "x2"] == 0), "value"] = 0
    with pytest.raises(
        ValueError,
        match=("At least one signal per coordinate has to be non-zero in the signals DataFrame."),
    ):
        ScoreCalculator(cases, signals)


def test_check_coords_points_are_case_subset(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    missing_coords = signals[~((signals["x1"] == 0) & (signals["x2"] == 0))]
    with pytest.raises(
        ValueError,
        match=("Coordinates of cases must be subset of signals' coordinates"),
    ):
        ScoreCalculator(cases, missing_coords)


def test_signal_missing_labels(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    no_endemic_label = signals.loc[signals.loc[:, "signal_label"] != "endemic"]
    with pytest.raises(
        ValueError,
        match="Signals DataFrame must contain 'endemic' and 'non_case' signal_label.",
    ):
        ScoreCalculator(cases, no_endemic_label)

    no_non_case_label = signals.loc[signals.loc[:, "signal_label"] != "non_case"]
    with pytest.raises(
        ValueError,
        match="Signals DataFrame must contain 'endemic' and 'non_case' signal_label.",
    ):
        ScoreCalculator(cases, no_non_case_label)


def test_signals_float_error(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    signals_negative_value = signals.copy()
    signals_negative_value.at[0, "value"] = -1.2
    with pytest.raises(
        ValueError, match="'values' in signals DataFrame must be floats between 0 and 1."
    ):
        ScoreCalculator(cases, signals_negative_value)

    signals_high_values = signals.copy()
    signals_high_values.at[0, "value"] = 2.0
    with pytest.raises(
        ValueError, match="'values' in signals DataFrame must be floats between 0 and 1."
    ):
        ScoreCalculator(cases, signals_high_values)

    signals_not_float = signals.copy()
    signals_not_float.loc[:, "value"] = 1
    with pytest.raises(
        ValueError, match="'values' in signals DataFrame must be floats between 0 and 1."
    ):
        ScoreCalculator(cases, signals_not_float)


def test_check_signal_label_consitency(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    signals = signals.iloc[1:]
    with pytest.raises(
        ValueError,
        match=(
            "The set of all signal labels in the signals DataFrame "
            "must equal the available signals labels per cell."
        ),
    ):
        ScoreCalculator(cases, signals)

    signals = signals.append(
        pd.DataFrame(
            {"x1": [0.0], "x2": [0.0], "signal_label": ["additional_label"], "value": [0.5]}
        )
    )
    with pytest.raises(
        ValueError,
        match=(
            "The set of all signal labels in the signals DataFrame "
            "must equal the available signals labels per cell."
        ),
    ):
        ScoreCalculator(cases, signals)


def test_check_case_coords_subset_of_signal_coords(paper_example_dfs) -> None:
    cases, signals = paper_example_dfs

    signals = signals.rename(columns={"x1": "x3"})
    with pytest.raises(
        KeyError,
        match=(
            "Not all coordinate columns of the cases DataFrame are contained in the "
            "signals DataFrame."
        ),
    ):
        ScoreCalculator(cases, signals)


def test_conf_mat(paper_example_score: ScoreCalculator) -> None:
    confusion_matrix = paper_example_score.calc_score(
        metrics.confusion_matrix, p_thresh=0.5, p_hat_thresh=0.2
    )
    confusion_matrix_expected = {
        "endemic": np.array([[16, 4], [1, 4]]),
        "non_case": np.array([[13, 0], [0, 12]]),
        "one": np.array([[20, 2], [0, 3]]),
        "three": np.array([[19, 5], [1, 0]]),
        "two": np.array([[18, 4], [2, 1]]),
    }
    np.testing.assert_equal(confusion_matrix, confusion_matrix_expected)

    confusion_matrix_weighted = paper_example_score.calc_score(
        metrics.confusion_matrix, weighting="cases", p_thresh=0.5, p_hat_thresh=0.2
    )
    confusion_matrix_weighted_expected = {
        "endemic": np.array([[0, 1], [1, 8]]),
        "non_case": np.array([[0, 0], [0, 12]]),
        "one": np.array([[0, 0], [0, 3]]),
        "three": np.array([[1, 0], [2, 0]]),
        "two": np.array([[1, 0], [2, 1]]),
    }
    np.testing.assert_equal(confusion_matrix_weighted, confusion_matrix_weighted_expected)
