from itertools import product
from typing import Optional, Union

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from scipy.stats import multivariate_normal


class _DataLoader:
    """A class to read, check, and impute data used in this package."""

    def __init__(self, cases: pd.DataFrame, signals: pd.DataFrame) -> None:
        self.cases = cases
        self.signals = signals
        self.MUST_HAVE_LABELS = {"endemic", "non_case"}
        self.COORDS = self._extract_coords(cases)
        self.cases = self._prepare_cases(cases)
        self.signals = self._check_signals_correctness(signals, self.cases)
        self.SIGNALS_LABELS = self.signals["signal_label"].unique()
        self.DATA_LABELS = self.cases["data_label"].unique()

    def _extract_coords(self, cases: pd.DataFrame) -> list[str]:
        return list(cases.columns[~cases.columns.isin(["data_label", "value"])])

    def _prepare_cases(self, cases: pd.DataFrame) -> pd.DataFrame:
        cases_correct = self._check_cases_correctness(cases)
        return self._impute_non_case(cases_correct)

    def _check_cases_correctness(self, cases: pd.DataFrame) -> pd.DataFrame:
        self._check_no_nans_exist(cases)
        self._check_non_cases_not_included(cases)
        self._check_endemic_exists(cases)
        self._check_cases_pos_ints(cases)
        self._check_data_label_consitency(cases)
        return cases

    def _check_no_nans_exist(self, cases) -> None:
        if cases.isna().any(axis=None):
            raise ValueError("Cases DataFrame must not contain any NaN values.")

    def _check_non_cases_not_included(self, cases) -> None:
        if "non_case" in cases["data_label"].values:
            raise ValueError(
                (
                    "Please remove entries with label 'non_cases' from cases DataFrame. "
                    "This label is included automatically and therefore internally reserved."
                )
            )

    def _check_endemic_exists(self, cases) -> None:
        if "endemic" not in cases["data_label"].values:
            raise ValueError(("Please add the label 'endemic' to your cases DataFrame."))

    def _check_cases_pos_ints(self, cases) -> None:
        if not (pd.api.types.is_integer_dtype(cases["value"]) and (cases["value"] >= 0).all()):
            raise ValueError("Case counts must be non-negative, whole numbers.")

    def _check_data_label_consitency(self, cases) -> None:
        unique_data_label_not_eq_data_labels_per_coord = cases.groupby(self.COORDS)[
            "data_label"
        ].apply(lambda x: list(x) != list(cases["data_label"].unique()))
        if unique_data_label_not_eq_data_labels_per_coord.any():
            raise ValueError(
                (
                    "The set of all data labels in the cases DataFrame "
                    "must equal the available data labels per cell."
                )
            )

    def _impute_non_case(self, cases: pd.DataFrame) -> pd.DataFrame:
        """Imputes case numbers for non_case class.

        Args:
            data: DataFrame with case numbers but without information on non_cases.

        Returns:
            data DataFrame with imputed non_case numbers.
        """
        non_cases = (
            cases.groupby(self.COORDS)
            .agg({"value": "sum"})
            .assign(value=lambda x: np.where(x["value"] == 0, 1, 0), data_label="non_case")
            .reset_index()
        )
        return pd.concat([cases, non_cases], ignore_index=True)

    def _check_signals_correctness(
        self,
        signals: pd.DataFrame,
        cases: pd.DataFrame,
    ) -> pd.DataFrame:
        self._check_case_coords_subset_of_signal_coords(signals)
        self._check_coords_points_are_case_subset(signals, cases)
        self._check_signal_label_consistency(signals)
        self._check_signals_floats_between_zero_one(signals)
        self._check_must_have_signals(signals)
        self._check_empty_cells(signals)
        return signals

    def _check_case_coords_subset_of_signal_coords(self, signals) -> None:
        try:
            signals[self.COORDS]
        except KeyError:
            raise KeyError(
                (
                    "Not all coordinate columns of the cases DataFrame are contained in the "
                    f"signals DataFrame. It must contain {self.COORDS}"
                )
            )

    def _check_coords_points_are_case_subset(self, signals, cases) -> None:
        unique_case_coords = set(cases[self.COORDS].apply(tuple, axis=1))
        unique_signale_coords = set(signals[self.COORDS].apply(tuple, axis=1))
        if not (unique_case_coords - unique_signale_coords == set()):
            raise ValueError("Coordinates of cases must be subset of signals' coordinates")

    def _check_signal_label_consistency(self, signals) -> None:
        unique_signal_label_not_eq_signal_labels_per_coord = signals.groupby(self.COORDS)[
            "signal_label"
        ].apply(lambda x: list(x) != list(signals["signal_label"].unique()))
        if unique_signal_label_not_eq_signal_labels_per_coord.any():
            raise ValueError(
                (
                    "The set of all signal labels in the signals DataFrame "
                    "must equal the available signals labels per cell."
                )
            )

    def _check_signals_floats_between_zero_one(self, signals) -> None:
        if not (
            pd.api.types.is_float_dtype(signals["value"])
            and ((1 >= signals["value"]) & (signals["value"] >= 0)).all()
        ):
            raise ValueError("'values' in signals DataFrame must be floats between 0 and 1.")

    def _check_must_have_signals(self, signals) -> None:
        if ("endemic" not in signals["signal_label"].values) or (
            "non_case" not in signals["signal_label"].values
        ):
            raise ValueError(
                "Signals DataFrame must contain 'endemic' and 'non_case' signal_label."
            )

    def _check_empty_cells(self, signals) -> None:
        if (signals.groupby(self.COORDS).agg({"value": "sum"}).values == 0).any():
            raise ValueError(
                (
                    "At least one signal per coordinate has to be non-zero "
                    "in the signals DataFrame."
                )
            )


class _ScoreBase(_DataLoader):
    """Class that contains main logic to calculate p(d) and p^(d)."""

    def __init__(self, cases, signals) -> None:
        super().__init__(cases, signals)

    def _eval_df(self) -> pd.DataFrame:
        """Creates DataFrame with p(d| x) and p^(d| x)"""
        return self._p_d_given_x().merge(
            self._p_hat_d(),
            on=self.COORDS + ["d"],
        )

    def _p_hat_d(self) -> pd.DataFrame:
        """Calculates p^(d| x) = sum( p(d|s, x) p(s, x) )"""
        p_hat_d = self._p_d_given_s().merge(
            self._p_s_given_x(),
            on="s",
        )
        p_hat_d.loc[:, "p(d,s|x)"] = p_hat_d["posterior"] * p_hat_d["prior"]
        p_hat_d = (
            p_hat_d.groupby(self.COORDS + ["d"])
            .agg({"p(d,s|x)": sum})
            .rename(columns={"p(d,s|x)": "p^(d)"})
            .reset_index()
        )
        return p_hat_d

    def _p_d_given_x(self) -> pd.DataFrame:
        return self.cases.assign(
            value=lambda x: x.value / x.groupby(self.COORDS)["value"].transform("sum").values
        ).rename(columns={"data_label": "d", "value": "p(d)"})

    def _p_s_given_x(self) -> pd.DataFrame:
        """p (s|x) = w(s, x) / sum_s (w(s,x))"""
        return (
            self.signals.assign(
                prior=lambda x: (x.value / x.groupby(self.COORDS)["value"].transform("sum")),
                s=lambda x: x["signal_label"],
            )
            .drop(columns=["signal_label", "value"])
            .loc[:, self.COORDS + ["prior", "s"]]
        )

    def _p_d_given_s(self) -> pd.DataFrame:
        """Calculates p(d|s) which depends on the use case."""
        signal_per_diseases = list(
            product(
                self.DATA_LABELS,
                self.SIGNALS_LABELS,
            ),
        )
        df = pd.DataFrame(
            signal_per_diseases,
            columns=["d", "s"],
        )

        signal_data_indeces = df.query(
            "~(s.isin(['endemic', 'non_case']) | d.isin(['endemic', 'non_case']))"
        ).index
        df.loc[signal_data_indeces, "posterior"] = 1 / len(
            set(self.DATA_LABELS) - set(self.MUST_HAVE_LABELS)
        )

        df.loc[(df.loc[:, "d"] == "endemic") & (df.loc[:, "s"] == "endemic"), "posterior"] = 1
        df.loc[(df.loc[:, "d"] == "non_case") & (df.loc[:, "s"] == "non_case"), "posterior"] = 1
        # Only NAs left are entries where dors in ['endemic', 'non_case'] and d!=s
        return df.fillna(0)


class ScoreCalculator(_ScoreBase):
    """Algorithm agnostic evaluation for (disease) outbreak detection.

    The `ScoreCalculator` offers epidemiologically meaningful scores given case count
    data of infectious diseases, information on cases linked through an
    outbreak, and signals for outbreaks generated by outbreak detection algorithms.
    """

    def __init__(
        self,
        cases: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> None:
        r"""Builds scorer given data.

        Args:
            cases: This DataFrame must contain the following columns and no NaNs:

                - ``data_label``. Is the class per outbreak. Must contain ``endemic``
                and must not contain ``non-case``.
                - ``value``. This is the amount of cases in the respective cell.
                This value must be an positive integer.
                - Each other column in the DataFrame is treated as a coordinate
                where each row is one single cell. This coordinate system is
                the evaluation resolution.

            signals: This DataFrame must contain the following columns:

                - ``signal_label``. Is the class per signal. Must contain ``endemic`` and
                ``non-case``.
                - ``value``. This is the signal strength :math:`w` and should be :math:`w \in [0,1]`
                - Each other column in the DataFrame is treated as a coordinate
                where each row is one single cell. Cases coordinates and cells
                must be subset of cases coordinates and cells. Cells outside
                the coordinate system of the cases DataFrame are ignored.
        """
        super().__init__(cases, signals)

    def calc_score(
        self,
        scorer: sk_metrics,
        p_thresh: Optional[float] = None,
        p_hat_thresh: Optional[float] = None,
        weighting: Optional[str] = None,
        time_space_weighting: dict[str, float] = None,
        time_axis: Optional[str] = None,
    ) -> dict[str, Union[float, np.ndarray]]:
        eval_df = self._thresholded_eval_df(p_thresh, p_hat_thresh)
        if weighting is None:
            eval_df["weight"] = 1
        elif weighting == "cases":
            eval_df = self._apply_case_weighting(eval_df)
        elif weighting == "timespace":
            # Underscore is fix due to Optional in calc_score and no Optional in TimeSpaciness
            # https://github.com/python/mypy/issues/7268
            _time_space_weighting = time_space_weighting
            _time_axis = time_axis
            eval_df = self._apply_timespace_weighting(eval_df, _time_space_weighting, _time_axis)
        else:
            raise ValueError("weighting must be None, 'cases', or 'timespace'.")

        return (
            eval_df.groupby("d")
            .apply(lambda x: scorer(x["true"], x["pred"], sample_weight=x["weight"]))
            .to_dict()
        )

    def _apply_case_weighting(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        return eval_df.merge(
            self.cases, left_on=self.COORDS + ["d"], right_on=self.COORDS + ["data_label"]
        ).rename(columns={"value": "weight"})

    def _apply_timespace_weighting(self, eval_df, time_space_weighting, time_axis) -> pd.DataFrame:
        timespaciness = TimeSpaciness(self.cases.query("data_label!='non_case'"), self.signals)
        timespace_weights = timespaciness.timespace_weighting(time_space_weighting, time_axis)
        return eval_df.merge(
            timespace_weights,
            left_on=timespaciness.COORDS + ["d"],
            right_on=timespaciness.COORDS + ["data_label"],
            how="left",
        )

    def _thresholded_eval_df(
        self, p_thresh: Optional[float], p_hat_thresh: Optional[float]
    ) -> pd.DataFrame:
        eval_df = self._eval_df()
        # TODO: change to strict larger than.
        # If p_thresh is one p(d) is one, label positive anyway
        if p_thresh:
            if p_thresh == 1:
                eval_df = eval_df.assign(true=np.where(eval_df["p(d)"] >= p_thresh, 1, 0))
            else:
                eval_df = eval_df.assign(true=np.where(eval_df["p(d)"] > p_thresh, 1, 0))
        else:
            eval_df = eval_df.rename(columns={"p(d)": "true"})

        if p_hat_thresh:
            if p_hat_thresh == 1:
                eval_df = eval_df.assign(pred=np.where(eval_df["p^(d)"] >= p_hat_thresh, 1, 0))
            else:
                eval_df = eval_df.assign(pred=np.where(eval_df["p^(d)"] > p_hat_thresh, 1, 0))
        else:
            eval_df = eval_df.rename(columns={"p^(d)": "pred"})
        return eval_df


class TimeSpaciness(_DataLoader):
    """A class to calculate time space accuracy."""

    def __init__(
        self,
        cases: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> None:
        r"""Builds TimeSpaciness given data.

        Args:
            cases: This DataFrame must contain the following columns and no NaNs:

                - ``data_label``. Is the class per outbreak. Must contain ``endemic``
                and must not contain ``non-case``.
                - ``value``. This is the amount of cases in the respective cell.
                This value must be an positive integer.
                - Each other column in the DataFrame is treated as a coordinate
                where each row is one single cell. This coordinate system is
                the evaluation resolution.

            signals: This DataFrame must contain the following columns:

                - ``signal_label``. Is the class per signal. Must contain ``endemic`` and
                ``non-case``.
                - ``value``. This is the signal strength :math:`w` and should be :math:`w \in [0,1]`
                - Each other column in the DataFrame is treated as a coordinate
                where each row is one single cell. Cases coordinates and cells
                must be subset of cases coordinates and cells. Cells outside
                the coordinate system of the cases DataFrame are ignored.
        """
        super().__init__(cases, signals)

    def timespace_weighting(
        self,
        time_space_weighting: dict[str, float],
        time_axis: Optional[str] = None,
    ) -> pd.DataFrame:
        """Creates spatial gauss weights for scoring.

        Args:
            time_space_weighting: Dict with dimension of the case data that represent space and
                                  the covariance over each spatial dim as the value

        Returns:
            Weights per data_label and spatial dimension.
        """
        gauss_dims = list(time_space_weighting.keys())
        covariance_diag = np.diag(list(time_space_weighting.values()))
        coords_system = self._gauss_coord_system(gauss_dims)
        case_coords_dict = self._coords_where_more_than_one_case_per_label(
            gauss_dims, coords_system
        )
        weights = {}
        for data_label, case_coords in case_coords_dict.items():
            mvns = [multivariate_normal(case_coord, covariance_diag) for case_coord in case_coords]
            values = [mvn.pdf(coords_system) for mvn in mvns]
            score_weight = np.array(values).sum(axis=0)
            weights[data_label] = score_weight
        melted = pd.DataFrame(weights).melt(
            value_name="weight",
            var_name="data_label",
        )
        melted.loc[:, gauss_dims] = np.vstack([coords_system] * len(self.DATA_LABELS))
        if time_axis:
            time_mask = self._time_mask(time_axis)
            melted = melted.merge(time_mask, on=gauss_dims + ["data_label"], how="right")
            melted.loc[:, "masked_weight"] = melted.loc[:, "weight"] * melted.loc[:, "time_mask"]
            return melted
        else:
            return melted

    def _gauss_coord_system(self, gauss_dims: list[str]) -> np.ndarray:
        dim_lengths = [self.cases[col].nunique() for col in gauss_dims]
        dim_ranges = [np.arange(0, dim_length) for dim_length in dim_lengths]
        return np.array(list(product(*dim_ranges)))

    def _coords_where_more_than_one_case_per_label(self, gauss_dims, coords_system):
        case_mask = self.cases.groupby(gauss_dims + ["data_label"]).agg({"value": "sum"})
        case_coords_dict = {}
        for data_label, df in case_mask.groupby("data_label"):
            case_coords = coords_system[np.argwhere(df["value"].values > 0).ravel()]
            case_coords_dict[data_label] = case_coords
        return case_coords_dict

    def _time_mask(self, time_axis: str) -> pd.DataFrame:
        dfs = []
        for _, df in self.cases.groupby("data_label"):
            df = df.merge(self._all_true_after_first_true(df, time_axis), on=self.COORDS)
            dfs.append(df)
        return pd.concat(dfs).drop(columns="value")

    def _all_true_after_first_true(self, df: pd.DataFrame, time_axis: "str"):
        mask = df.groupby(time_axis).agg({"value": "any"}).rename(columns={"value": "time_mask"})
        mask_len = len(mask)
        if mask["time_mask"].sum() == 0:
            mask.loc[:, "time_mask"] = np.zeros(mask_len)
        else:
            first_true_idx = np.argmax(mask["time_mask"] == True)  # NOQA
            mask.loc[:, "time_mask"] = np.hstack(
                (np.zeros(first_true_idx), np.ones(mask_len - first_true_idx))
            )
        return df.merge(mask, on=time_axis, how="left").drop(columns=["value", "data_label"])


class Timeliness(_DataLoader):
    """A class to calculate timeliness of detected outbreak."""

    def __init__(
        self,
        cases: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> None:
        r"""Builds Timeliness given data.

        Args:
            cases: This DataFrame must contain the following columns and no NaNs:

                - ``data_label``. Is the class per outbreak. Must contain ``endemic``
                and must not contain ``non-case``.
                - ``value``. This is the amount of cases in the respective cell.
                This value must be an positive integer.
                - Each other column in the DataFrame is treated as a coordinate
                where each row is one single cell. This coordinate system is
                the evaluation resolution.

            signals: This DataFrame must contain the following columns:

                - ``signal_label``. Is the class per signal. Must contain ``endemic`` and
                ``non-case``.
                - ``value``. This is the signal strength :math:`w` and should be :math:`w \in [0,1]`
                - Each other column in the DataFrame is treated as a coordinate
                where each row is one single cell. Cases coordinates and cells
                must be subset of cases coordinates and cells. Cells outside
                the coordinate system of the cases DataFrame are ignored.
        """
        super().__init__(cases, signals)
        self.outbreak_labels = list(set(self.DATA_LABELS) - set(["endemic", "non_case"]))
        self.outbreak_signals = list(set(self.SIGNALS_LABELS) - set(["endemic", "non_case"]))

    def timeliness(self, time_axis: str, D: int, signal_threshold: float = 0) -> dict[str, float]:
        if not isinstance(time_axis, str):
            raise ValueError("time_axis must be of type str.")

        if not (isinstance(D, int) and (D > 0)):
            raise ValueError("D must be a positive integer.")

        signals_agg = (
            self.signals.query("signal_label.isin(@self.outbreak_signals)")
            .assign(value=lambda x: np.where(x["value"] > signal_threshold, 1, 0))
            .groupby(time_axis)
            .agg({"value": "any"})
        )
        cases_agg = (
            self.cases.query("data_label.isin(@self.outbreak_labels)")
            .groupby([time_axis, "data_label"])
            .agg({"value": "any"})
            .reset_index()
        )

        delays_per_label = (
            cases_agg.merge(
                signals_agg,
                suffixes=("_cases", "_signals"),
                on=time_axis,
            )
            .groupby("data_label")
            .apply(lambda x: self._calc_delay(df=x, D=D))
        )
        return dict((1 - delays_per_label / D))

    @staticmethod
    def _calc_delay(df: pd.DataFrame, D: int) -> int:
        first_case_idx = (df["value_cases"] == 1).argmax()
        first_signal_idx = (df["value_signals"] == 1).argmax()
        delay = first_signal_idx - first_case_idx
        if (df["value_cases"].sum() == 0) or (df["value_signals"].sum() == 0):
            return D
        elif not (0 <= delay <= D):
            return D
        else:
            return delay
