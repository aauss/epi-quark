from typing import Callable, Optional
from warnings import warn

import numpy as np
import pandas as pd


def impute_signals(
    signals: pd.DataFrame,
    cases: pd.DataFrame,
    coords: list[str],
    agg_function: Optional[str] = "min",
) -> pd.DataFrame:
    """Calculates signals for endemic and non cases when they are missing."""

    assigns = _column_imputation(signals)
    aggs = dict.fromkeys(list(assigns), agg_function)
    if len(assigns) > 0:
        non_case_info = (
            cases.query("data_label=='non_case'")
            .rename(columns={"value": "non_case"})
            .drop(columns="data_label")
        )  # non_case information needed for column imputation
        missing_signals = (
            signals.merge(
                non_case_info,
                on=coords,
                how="right",
            )
            .assign(**assigns)
            .groupby(coords)
            .agg(aggs)
            .reset_index()
            .melt(id_vars=coords, var_name="signal_label")
        )
        return pd.concat([signals, missing_signals], ignore_index=True)
    else:
        return signals


def _column_imputation(signals: pd.DataFrame) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    assigns = {}
    if not signals["signal_label"].str.contains("endemic").any():
        assigns["endemic"] = lambda x: (1 - x["value"]) * np.logical_xor(x["non_case"], 1)
        warn("endemic is missing and is being imputed.")

    if not signals["signal_label"].str.contains("non_case").any():
        assigns["non_case"] = lambda x: (1 - x["value"]) * x["non_case"]
        warn("non_case is missing and is being imputed.")
    return assigns
