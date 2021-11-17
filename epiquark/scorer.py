from itertools import product
from typing import Callable, Optional
from warnings import warn

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix


class _DataLoader:
    """A class to read, check, and impute data used in this package."""

    def __init__(self, cases: pd.DataFrame, signals: pd.DataFrame) -> None:
        self.cases = cases
        self.signals = signals
        self.MUST_HAVE_LABELS = {"endemic", "non_case"}
        self.COORDS = self._extract_coords(cases)
        self.cases = self._prepare_cases(cases)
        self.signals = self._prepare_signals(signals, self.cases)
        self.SIGNALS_LABELS = self.signals["signal_label"].unique()
        self.DATA_LABELS = self.cases["data_label"].unique()

    def _extract_coords(self, cases: pd.DataFrame) -> list[str]:
        return list(cases.columns[~cases.columns.isin(["data_label", "value"])])

    def _prepare_cases(self, cases: pd.DataFrame) -> pd.DataFrame:
        cases_correct = self._check_cases_correctness(cases)
        return self._impute_non_case(cases_correct)

    def _check_cases_correctness(self, cases_correct: pd.DataFrame) -> pd.DataFrame:
        if cases_correct.isna().any(axis=None):
            raise ValueError("Cases DataFrame must not contain any NaN values.")

        if "non_case" in cases_correct["data_label"].values:
            raise ValueError(
                (
                    "Please remove entries with label 'non_cases' from cases DataFrame. "
                    "This label is included automatically and therefore internally reserved."
                )
            )

        if not (
            pd.api.types.is_integer_dtype(cases_correct["value"])
            and (cases_correct["value"] >= 0).all()
        ):
            raise ValueError("Case counts must be non-negative, whole numbers.")

        return cases_correct

    def _impute_non_case(self, cases: pd.DataFrame) -> pd.DataFrame:
        """Imputes case numbers for non_case column.

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

    def _prepare_signals(self, signals: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
        signals_correct = self._check_signals_correctness(signals, cases)
        return self._impute_signals(signals_correct, cases)

    def _check_signals_correctness(
        self,
        signals_correct: pd.DataFrame,
        cases: pd.DataFrame,
    ) -> pd.DataFrame:
        if not (
            pd.api.types.is_float_dtype(signals_correct["value"])
            and ((1 >= signals_correct["value"]) & (signals_correct["value"] >= 0)).all()
        ):
            raise ValueError("'values' in signal DataFrame must be floats between 0 and 1.")

        if (
            set(signals_correct.loc[:, self.COORDS].apply(tuple, axis=1))
            - set(cases.loc[:, self.COORDS].apply(tuple, axis=1))
            != set()
        ):
            raise ValueError("Coordinates of 'signals' must be a subset of coordinates of 'cases'.")

        if (
            signals_correct.groupby(self.COORDS).size() != signals_correct["signal_label"].nunique()
        ).any():
            raise ValueError("Each coordinate must contain the same amount of signals.")
        return signals_correct

    def _impute_signals(
        self, signals: pd.DataFrame, cases: pd.DataFrame, agg_function: Optional[str] = "min"
    ) -> pd.DataFrame:
        """Calculates signals for endemic and non cases when they are missing."""
        assigns = self._column_imputation(signals)
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
                    on=self.COORDS,
                    how="right",
                )
                .assign(**assigns)
                .groupby(self.COORDS)
                .agg(aggs)
                .reset_index()
                .melt(id_vars=self.COORDS, var_name="signal_label")
            )
            return pd.concat([signals, missing_signals], ignore_index=True)
        else:
            return signals

    def _column_imputation(
        self, signals: pd.DataFrame
    ) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
        assigns = {}
        if not signals["signal_label"].str.contains("w_endemic").any():
            assigns["w_endemic"] = lambda x: (1 - x["value"]) * np.logical_xor(x["non_case"], 1)
            warn("w_endemic is missing and is being imputed.")

        if not signals["signal_label"].str.contains("w_non_case").any():
            assigns["w_non_case"] = lambda x: (1 - x["value"]) * x["non_case"]
            warn("w_non_case is missing and is being imputed.")
        return assigns


class _ScoreBase(_DataLoader):
    """Class that contains main logic to calculate p(d_i) and p^(d_i)."""

    def __init__(self, cases, signals) -> None:
        super().__init__(cases, signals)

    def _eval_df(self) -> pd.DataFrame:
        """Creates DataFrame with p(d_i | x) and p^(d_i | x)"""
        return self._p_di_given_x().merge(
            self._p_hat_di(),
            on=self.COORDS + ["d_i"],
        )

    def _p_hat_di(self) -> pd.DataFrame:
        """Calculates p^(d_i | x) = sum( p^(d_i| s_j, x) p^(s_j, x) )"""
        p_hat_di = self._p_hat_di_given_sj_x().merge(
            self._p_hat_sj_given_x(),
            on=self.COORDS + ["s_j"],
        )
        p_hat_di.loc[:, "p^(d_i)"] = p_hat_di["posterior"] * p_hat_di["prior"]
        p_hat_di = p_hat_di.groupby(self.COORDS + ["d_i"]).agg({"p^(d_i)": sum}).reset_index()
        return p_hat_di

    def _p_di_given_x(self) -> pd.DataFrame:
        return (
            self.cases.assign(
                value=lambda x: x.value / x.groupby(self.COORDS)["value"].transform("sum").values
            )
            .fillna(0)
            .rename(columns={"data_label": "d_i", "value": "p(d_i)"})
        )

    def _p_hat_sj_given_x(self) -> pd.DataFrame:
        """p^ (s_j|x) = w(s, x) / sum_s (w(s,x))"""
        return (
            self.signals.assign(
                prior=lambda x: (x.value / x.groupby(self.COORDS)["value"].transform("sum")).fillna(
                    0
                ),
                s_j=lambda x: x["signal_label"].str.replace("w_", ""),
            )
            .drop(columns=["signal_label", "value"])
            .loc[:, self.COORDS + ["prior", "s_j"]]
        )

    def _p_hat_di_given_sj_x(self) -> pd.DataFrame:
        """Calculates p^(d_i | s_j, x) which depends on the use case."""
        unique_coords = [self.cases[coord].unique() for coord in self.COORDS]
        signal_per_diseases = list(
            product(
                product(*unique_coords),
                product(
                    self.DATA_LABELS,
                    [col.replace("w_", "") for col in self.SIGNALS_LABELS],
                ),
            )
        )
        df = pd.DataFrame(
            [tuple_[0] + tuple_[1] for tuple_ in signal_per_diseases],
            columns=list(self.COORDS) + ["d_i", "s_j"],
        )

        signal_data_indeces = df.query(
            "~(s_j.isin(['endemic', 'non_case']) | d_i.isin(['endemic', 'non_case']))"
        ).index
        df.loc[signal_data_indeces, "posterior"] = 1 / len(
            set(self.DATA_LABELS) - set(self.MUST_HAVE_LABELS)
        )

        non_case_endemic_signal_indeces = df.query("d_i == s_j").index
        df.loc[non_case_endemic_signal_indeces, "posterior"] = 1
        return df.fillna(0)


class Score(_ScoreBase):
    """Algorithm agnostic evaluation for (disease) outbreak detection.

    The `Scorer` offers epidemiologically meaningful scores given case count
    data of infectious diseases, information on cases linked through an
    outbreak, and signals for outbreaks generated by outbreak detection algorithms.
    """

    def __init__(
        self,
        cases: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> None:
        """Builds scorer given data.

        Args:
            cases: Case numbers with coordinate columns, 'data_label' column, and 'value' column.
                'data_label' should contain (outbreak) labels. Must contain 'endemic' and must
                not contain data label 'non_case'. 'value' column contains case numbers
                per cell and data_label. Remaining columns define the coordinate system.
                Coordinates in `cases` is required to be complete.
            signals: Signal with coordinate columns, 'signal_label' column, and 'value' column.
                Coordinates in `signals` must be a subset of the coordinates in `cases`.
                Each signal_label must start with 'w_' and 'w_endemic' and 'w_non_case'
                should be included.
        """
        super().__init__(cases, signals)

    def calc_score(
        self,
        scorer: sk_metrics,
        p_thresh: Optional[float] = None,
        p_hat_thresh: Optional[float] = None,
        weights: Optional[str] = None,
        gauss_dims: Optional[list[str]] = None,
        covariance_diag: Optional[list[float]] = None,
        time_axis: Optional[str] = None,
    ) -> tuple[float, float]:

        eval_df = self._thresholded_eval_df(p_thresh, p_hat_thresh)
        if weights is None:
            eval_df["weight"] = 1
        elif weights == "cases":
            eval_df = self._apply_case_weighting(eval_df)
        elif weights == "timespace":
            # Fix due to Optional in calc_score and no Optional in EpiMetrics
            # https://github.com/python/mypy/issues/7268
            _gauss_dims = gauss_dims
            _covariance_diag = covariance_diag
            _time_axis = time_axis
            eval_df = self._apply_timespace_weighting(
                eval_df, _gauss_dims, _covariance_diag, _time_axis
            )
        else:
            raise ValueError("weights must be None, 'cases', or 'timespace'.")

        return (
            eval_df.groupby("d_i")
            .apply(lambda x: scorer(x["true"], x["pred"], sample_weight=x["weight"]))
            .to_dict()
        )

    def _apply_case_weighting(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        return eval_df.merge(
            self.cases, left_on=self.COORDS + ["d_i"], right_on=self.COORDS + ["data_label"]
        ).rename(columns={"value": "weight"})

    def _apply_timespace_weighting(
        self, eval_df, gauss_dims, covariance_diag, time_axis
    ) -> pd.DataFrame:
        epimetrics = EpiMetrics(self.cases.query("data_label!='non_case'"), self.signals)
        gauss_weights = epimetrics.gauss_weighting(gauss_dims, covariance_diag, time_axis)
        coords = list(set(epimetrics.COORDS).intersection(set(gauss_weights.columns)))
        return eval_df.merge(
            gauss_weights,
            left_on=coords + ["d_i"],
            right_on=coords + ["data_label"],
            how="left",
        )

    def _thresholded_eval_df(
        self, p_thresh: Optional[float], p_hat_thresh: Optional[float]
    ) -> pd.DataFrame:
        eval_df = self._eval_df()
        if p_thresh:
            eval_df = eval_df.assign(true=np.where(eval_df["p(d_i)"] >= p_thresh, 1, 0))
        else:
            eval_df = eval_df.rename(columns={"p(d_i)": "true"})

        if p_hat_thresh:
            eval_df = eval_df.assign(pred=np.where(eval_df["p^(d_i)"] >= p_hat_thresh, 1, 0))
        else:
            eval_df = eval_df.rename(columns={"p^(d_i)": "pred"})
        return eval_df

    def class_based_conf_mat(
        self,
        p_thresh: Optional[float] = None,
        p_hat_thresh: Optional[float] = None,
        weighted: Optional[bool] = False,
    ) -> dict[str, list[list[int]]]:
        """Return confusion matrix for all data labels.

        The rows indicate the true labels and the columns the predicted labels.
        The 0th row and column corresponds to the negative label, the 1st row
        and column to the positive label.

        You can find more on:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        """
        p_thresh = p_thresh or (1 / len(self.MUST_HAVE_LABELS))
        p_hat_thresh = p_hat_thresh or (1 / len(self.DATA_LABELS))

        thresholded_eval = self._thresholded_eval_df(p_thresh, p_hat_thresh)
        thresholded_eval = thresholded_eval.pivot(
            index=self.COORDS, columns="d_i", values=["true", "pred"]
        )
        if weighted:
            cm_list: list[np.ndarray] = []
            for label in thresholded_eval.columns.levels[1]:
                duplicated = self._duplicate_cells_by_cases(thresholded_eval, label)
                cm_list.append(
                    confusion_matrix(duplicated["true"].values, duplicated["pred"].values)
                )
            cm: np.ndarray = np.array(cm_list)

        else:
            cm = multilabel_confusion_matrix(
                thresholded_eval.loc[:, "true"].values,
                thresholded_eval.loc[:, "pred"].values,
            )
        return dict(zip(thresholded_eval.columns.levels[1], cm.tolist()))

    def _duplicate_cells_by_cases(self, thresholded_eval: pd.DataFrame, label: str):
        sliced_eval = thresholded_eval.loc[:, (slice(None), label)]
        cases_per_cell_exceeding_one = (
            self.cases.query("data_label==@label")["value"]
            .where(lambda x: x > 1)
            .dropna()
            .reset_index(drop=True)
        ) - 1
        for idx, factor in cases_per_cell_exceeding_one.iteritems():
            for _ in range(int(factor)):
                # duplicate cells that have more than one case
                sliced_eval = sliced_eval.append(sliced_eval.iloc[idx])
        return sliced_eval


class EpiMetrics(_DataLoader):
    """A class to calculate epidemiologically relevant metrics."""

    def __init__(
        self,
        cases: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> None:
        """Builds EpiMetrics given data.

        Args:
            cases: Case numbers with coordinate columns, 'data_label' column, and 'value' column.
                'data_label' should contain (outbreak) labels. Must contain 'endemic' and must
                not contain data label 'non_case'. 'value' column contains case numbers per cell
                and data_label. Remaining columns define the coordinate system.
                Coordinates in `cases` is required to be complete.
            signals: Signal with coordinate columns, 'signal_label' column, and 'value' column.
                Coordinates in `signals` must be a subset of the coordinates in `cases`.
                Each signal_label must start with 'w_' and 'w_endemic' and 'w_non_case'
                should be included.
        """
        super().__init__(cases, signals)
        self.outbreak_labels = list(set(self.DATA_LABELS) - set(["endemic", "non_case"]))
        self.outbreak_signals = list(set(self.SIGNALS_LABELS) - set(["w_endemic", "w_non_case"]))

    def timeliness(self, time_axis: str, D: int, signal_threshold: float = 0) -> dict[str, float]:
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
            .apply(self._calc_delay)
        )
        return dict((1 - delays_per_label / D).clip(0, 1))

    @staticmethod
    def _calc_delay(df: pd.DataFrame) -> int:
        max_delay = len(df)
        # should ideally work with gauss weighting
        first_case_idx = (df["value_cases"] == 1).argmax()
        first_signal_idx = (df["value_signals"] == 1).argmax()
        # erst delay berechnen, über den delay die bedingung (ist zwischen 0 und D) prüfen
        #  und dann timeliness zurückgeben
        if (df["value_cases"].sum() == 0) or (df["value_signals"].sum() == 0):
            return max_delay
        elif first_signal_idx < first_case_idx:
            return max_delay
        else:
            return max(first_signal_idx - first_case_idx, 0)

    def gauss_weighting(
        self,
        gauss_dims: list[str],
        covariance_diag: Optional[list[float]] = None,
        time_axis: Optional[str] = None,
    ) -> pd.DataFrame:
        """Creates spatial gauss weights for scoring.

        Args:
            gauss_dims: Dimension of the case data that represent space.
            covariance_diag: The covariance over each spatial dim in the same order as gauss_dims.

        Returns:
            Weights per data_label and spatial dimension.
        """
        if covariance_diag is None:
            covariance_diag = np.diag(np.ones(len(gauss_dims)))
        else:
            covariance_diag = np.diag(covariance_diag)

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
            first_true_idx = np.argmax(mask["time_mask"] == True)   # NOQA
            mask.loc[:, "time_mask"] = np.hstack(
                (np.zeros(first_true_idx), np.ones(mask_len - first_true_idx))
            )
        return df.merge(mask, on=time_axis, how="left").drop(columns=["value", "data_label"])
