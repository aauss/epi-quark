import pandas as pd

from epiquark.utils import impute_signals


def test_signal_imputation(shared_datadir) -> None:
    cases = pd.read_csv(shared_datadir / "paper_example/non_case_imputed_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/signals_long.csv")
    imputed = impute_signals(signals, cases, coords=["x1", "x2"])

    imputed_expected = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    pd.testing.assert_frame_equal(imputed, imputed_expected, check_dtype=False)

    cases = pd.read_csv(shared_datadir / "paper_example/non_case_imputed_long.csv")
    signals = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    imputed = impute_signals(signals, cases, coords=["x1", "x2"])

    imputed_expected = pd.read_csv(shared_datadir / "paper_example/imputed_signals_long.csv")
    pd.testing.assert_frame_equal(imputed, imputed_expected, check_dtype=False)
