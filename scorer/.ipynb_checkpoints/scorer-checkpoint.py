from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn as sn
from matplotlib.patches import Rectangle
from scipy.stats import tmean
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    f1_score,
    confusion_matrix,
)


class Score:
    def __init__(self, data, signals, data_labels, missing_signal_agg="mean"):
        self.MUST_HAVE_LABELS = {"endemic", "non_case"}
        self.DATA_LABELS = list(set(data_labels) | self.MUST_HAVE_LABELS)

        self.data = self._pad_non_case(data)
        self.signals = self._pad_signals(
            pd.merge(self.data, signals, on=["x1", "x2"]), missing_signal_agg
        )
        self.SIGNALS_LABELS = list(
            self.signals.columns[self.signals.columns.str.contains("^w")]
        )
        self.COORDS = list(data.columns[data.columns.str.contains("^x")])

    def _pad_non_case(self, data):
        if data["non_case"].isna().any():
            is_non_case = data.loc[:, list(self.DATA_LABELS)].sum(axis=1) == 0
            return data.assign(non_case=np.where(is_non_case, 1, 0))
        else:
            return data

    def _pad_signals(self, signals, aggregation_func):
        """
        Calculates signals for endemic and non cases.

        Input DataFrame must indicate signal columns with 'w_' in their name.
        """
        if not any((signals["endemic"] > 0) & (signals["non_case"] > 0)):
            signal_columns = [c for c in signals.columns if "w_" in c]
            signal_df = signals.copy()

            for signal in signal_columns:
                signal_df.loc[:, f"_{signal}_endemic"] = (
                    1 - signal_df.loc[:, signal]
                ) * (signal_df["non_case"] == 0)
                signal_df.loc[:, f"_{signal}_non_case"] = (
                    1 - signal_df.loc[:, signal]
                ) * (signal_df["non_case"] != 0)

            non_case_signals = signal_df.columns[
                signal_df.columns.str.contains(r"w_._non_case")
            ]
            endemic_signals = signal_df.columns[
                signal_df.columns.str.contains(r"w_._endemic")
            ]

            signal_df.loc[:, "w_non_case"] = signal_df[non_case_signals].agg(
                aggregation_func, axis=1
            )
            signal_df.loc[:, "w_endemic"] = signal_df[endemic_signals].agg(
                aggregation_func, axis=1
            )
            return signal_df
        else:
            return signals

    def plot_map(self, series, title):
        df = self.signals.melt(
            id_vars=self.COORDS,
            value_vars=self.SIGNALS_LABELS,
            var_name="signal",
            value_name="w",
        )
        ax = sns.heatmap(
            series.values.reshape(len(self.DATA_LABELS), -1).T,
            linewidth=2,
            cmap="RdPu",
            cbar=False,
            annot=True,
        )
        cmap = dict(zip(df.signal.unique(), sns.color_palette("tab10")))
        for _, (x1, x2, signal, value) in df.iterrows():
            if signal not in ("w_endemic", "w_non_case"):
                ax.add_patch(
                    Rectangle(
                        (x1, x2),
                        1,
                        1,
                        fill=False,
                        lw=3,
                        alpha=value,
                        label=signal,
                        color=cmap[signal],
                    )
                )
        ax.set_title(title)
        ax.set_ylim(0, len(self.DATA_LABELS))
        ax.set_xlim(0, len(self.DATA_LABELS))
        return ax

    def p_di_given_x(self):
        """Calculate disease probability per cell, p(d_i| x)"""
        disease_proba_df = self.data.copy()
        disease_proba_df.loc[:, self.DATA_LABELS] = disease_proba_df.loc[
            :, self.DATA_LABELS
        ].div(disease_proba_df.loc[:, self.DATA_LABELS].sum(axis=1), axis=0)
        return disease_proba_df.melt(
            id_vars=["x1", "x2"], var_name="d_i", value_name="p(d_i)"
        )

    def p_hat_sj_given_x(self):
        """p^ (s_j|x) = w(s, x) / sum_s (w(s,x))"""
        df_prior_signal = self.signals.copy()

        prior_columns = [col.replace("w", "p") for col in self.SIGNALS_LABELS]
        sj_given_x = (
            df_prior_signal.loc[:, self.SIGNALS_LABELS]
            .div(
                df_prior_signal.loc[:, self.SIGNALS_LABELS].sum(axis=1),
                axis=0,
            )
            .values
        )

        df_prior_signal.loc[:, prior_columns] = sj_given_x

        signals_long = df_prior_signal.melt(
            id_vars=self.COORDS,
            value_vars=prior_columns,
            var_name="s_j_given_x",
            value_name="prior",
        ).assign(s_j=lambda x: x["s_j_given_x"].str.replace("p_", ""))
        return signals_long

    def p_hat_di_given_sj_x(self):
        """Calculates p^(d_i | s_j, x) which is algo based."""
        signal_per_diseases = list(
            product(
                product(range(5), range(5)),
                product(
                    self.DATA_LABELS,
                    [col.replace("w_", "") for col in self.SIGNALS_LABELS],
                ),
            )
        )
        df = pd.DataFrame(
            [tuple_[0] + tuple_[1] for tuple_ in signal_per_diseases],
            columns=["x1", "x2", "d_i", "s_j"],
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

    def p_hat_di(self):
        """Calculates p^(d_i | x) = sum( p^(d_i| s_j, x) p^(s_j, x) )"""
        p_hat_di = self.p_hat_di_given_sj_x().merge(
            self.p_hat_sj_given_x(),
            on=self.COORDS + ["s_j"],
        )
        p_hat_di.loc[:, "p^(d_i)"] = p_hat_di["posterior"] * p_hat_di["prior"]
        p_hat_di = (
            p_hat_di.groupby(["x1", "x2", "d_i"]).agg({"p^(d_i)": sum}).reset_index()
        )
        return p_hat_di

    def eval_df(self):
        return self.p_di_given_x().merge(
            self.p_hat_di(),
            on=self.COORDS + ["d_i"],
        )

    def multi_conf_mat(
        self,
        p_thresh=None,
        p_hat_thresh=None,
    ):
        eval_df = self.eval_df().assign(
            true=lambda x: x["p(d_i)"], pred=lambda x: x["p^(d_i)"]
        )
        if p_thresh is not None:
            eval_df.loc[:, "true"] = np.where(eval_df.loc[:, "p(d_i)"] > p_thresh, 1, 0)
        if p_hat_thresh is not None:
            eval_df.loc[:, "pred"] = np.where(
                eval_df.loc[:, "p^(d_i)"] > p_hat_thresh, 1, 0
            )
        pivot_eval_r = eval_df.pivot(
            index=["x1", "x2"], columns="d_i", values=["true", "pred"]
        )

        rel_cm = (
            pivot_eval_r.loc[:, "true"]
            .reset_index()
            .melt(id_vars=["x1", "x2"], value_name="true")
            .merge(
                pivot_eval_r.loc[:, "pred"].reset_index(),
                on=["x1", "x2"],
                how="left",
            )
            .assign(
                endemic=lambda x: x["true"] * x["endemic"],
                non_case=lambda x: x["true"] * x["non_case"],
                one=lambda x: x["true"] * x["one"],
                two=lambda x: x["true"] * x["two"],
                three=lambda x: x["true"] * x["three"],
            )
            .groupby("d_i")
            .agg(dict(zip(["endemic", "non_case", "one", "three", "two"], ["sum"] * 5)))
        )
        return rel_cm

    def class_based_conf_mat(self, p_thresh=None, p_hat_thresh=None, weighted=False):
        if p_thresh is None:
            p_thresh = 1 / len(self.MUST_HAVE_LABELS)
        if p_hat_thresh is None:
            p_hat_thresh = 1 / len(self.DATA_LABELS)

        pivot_eval = (
            self.eval_df()
            .assign(
                true=lambda x: np.where(x["p(d_i)"] >= p_thresh, 1, 0),
                pred=lambda x: np.where(x["p^(d_i)"] >= p_hat_thresh, 1, 0),
            )
            .pivot(index=["x1", "x2"], columns="d_i", values=["true", "pred"])
        )
        if weighted:
            cm = []
            for label in pivot_eval.columns.levels[1]:
                sliced_eval = pivot_eval.loc[:, (slice(None), label)]
                duplicater = self.data[label].where(lambda x: x > 1).dropna()
                for idx, factor in duplicater.iteritems():
                    for _ in range(int(factor - 1)):
                        sliced_eval = sliced_eval.append(sliced_eval.iloc[idx])
                cm.append(
                    confusion_matrix(
                        sliced_eval["true"].values, sliced_eval["pred"].values
                    )
                )

        else:
            cm = multilabel_confusion_matrix(
                pivot_eval.loc[:, "true"].values,
                pivot_eval.loc[:, "pred"].values,
            )

        f, axes = plt.subplots((len(self.DATA_LABELS) + 1) // 2, 2, figsize=(10, 7))
        axes = axes.ravel()
        for i, label in enumerate(pivot_eval.columns.levels[1]):
            df_cm = pd.DataFrame(cm[i], index=["0", "1"], columns=["0", "1"])
            sn.heatmap(df_cm, annot=True, ax=axes[i]).set(
                title=label, ylabel="True", xlabel="Predicted"
            )
            print(f"**{label.upper()}**")
            print(
                classification_report(
                    pivot_eval.loc[:, ("true", label)].values,
                    pivot_eval.loc[:, ("pred", label)].values,
                )
            )
        if len(axes) - len(self.DATA_LABELS):
            f.delaxes(axes[-1])
        f.tight_layout()

    def mean_score(
        self,
        scorer,
        p_thresh=None,
        p_hat_thresh=None,
    ):
        if p_thresh is None:
            p_thresh = 1 / len(self.MUST_HAVE_LABELS)
        if p_hat_thresh is None:
            p_hat_thresh = 1 / len(self.DATA_LABELS)

        pivot_eval = (
            self.eval_df()
            .assign(
                true=lambda x: np.where(x["p(d_i)"] >= p_thresh, 1, 0),
                pred=lambda x: np.where(x["p^(d_i)"] >= p_hat_thresh, 1, 0),
            )
            .pivot(index=["x1", "x2"], columns="d_i", values=["true", "pred"])
        )
        weights = dict(pivot_eval.loc[:, "true"].sum())

        cm = multilabel_confusion_matrix(
            pivot_eval.loc[:, "true"].values, pivot_eval.loc[:, "pred"].values
        )

        truncated_means = self.data.apply(lambda x: tmean(x, (1, np.inf)))
        max_cases = self.data.max()
        case_weighted = []
        case_weighted_norm = 0
        max_weighted = []
        max_weighted_norm = 0

        scores = []
        for label in pivot_eval.columns.levels[1]:
            score = scorer(
                pivot_eval.loc[:, ("true", label)].values,
                pivot_eval.loc[:, ("pred", label)].values,
            )
            scores.append(score)

            case_weighted.append(truncated_means[label] * score)
            case_weighted_norm += truncated_means[label]
            max_weighted.append(max_cases[label] * score)
            max_weighted_norm += max_cases[label]
        return (
            np.mean(scores),
            np.average(scores, weights=pivot_eval.loc[:, "true"].sum().values),
            (sum(case_weighted) / case_weighted_norm),
            (sum(max_weighted) / max_weighted_norm),
        )

    def timeliness(self, time_axis, D):
        # s = inf wenn nicht erkannt
        outbreak_labels = list(set(self.DATA_LABELS) - set(["endemic", "non_case"]))
        outbreak_signals = list(
            set(self.SIGNALS_LABELS) - set(["w_endemic", "w_non_case"])
        )
        non_time_coords = list(set(self.COORDS) - set([time_axis]))
        delays = self.signals.assign(w=lambda x: x[outbreak_signals].sum(axis=1))
        for label in outbreak_labels:
            is_delayed = ((delays[label] != 0) & (delays["w"] == 0)).astype(int)
            delays.loc[:, f"{label}_delay"] = is_delayed
            delays.loc[:, f"{label}_delay_shift"] = is_delayed.diff(1)
        delayed = {}
        for label in outbreak_labels:
            first_delays = 0
            for _, df in delays.groupby(non_time_coords):
                first_delays += (
                    df[f"{label}_delay"]
                    .iloc[: (df[f"{label}_delay_shift"] == -1).argmax()]
                    .sum()
                )
            delayed[label] = 1 - (first_delays / D)
        return delayed
