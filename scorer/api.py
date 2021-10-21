from dataclasses import dataclass
from typing import Optional

import pandas as pd
import sklearn.metrics as sk_metrics
from sklearn.metrics import confusion_matrix

from .scorer import EpiMetrics, Score


@dataclass
class ThreshRequired:
    p_thresh: bool
    p_hat_thresh: bool

    def check_threshs_correct(
        self, p_thresh: Optional[float], p_hat_thresh: Optional[float]
    ) -> None:
        actual = (p_thresh != None, p_hat_thresh != None)
        threshs_correct = actual == (self.p_thresh, self.p_hat_thresh)
        if not threshs_correct:
            raise ValueError(
                f"This metric {self._thresh_text(self.p_thresh)} p_thresh and {self._thresh_text(self.p_hat_thresh)} p_hat_thresh."
            )

    def _thresh_text(self, thresh: bool):
        if thresh:
            thresh_text = "requires"
        else:
            thresh_text = "must not contain"
        return thresh_text


def check_threshs(
    metric: str, p_thresh: Optional[float] = None, p_hat_thresh: Optional[float] = None
):
    required_treshs = {
        "f1": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "brier": ThreshRequired(p_thresh=True, p_hat_thresh=False),
        "auc": ThreshRequired(p_thresh=True, p_hat_thresh=False),
        "sensitivity": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "recall": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "tpr": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "specificity": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "tnr": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "fpr": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "fnr": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "precision": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "ppv": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "npv": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "matthews": ThreshRequired(p_thresh=True, p_hat_thresh=True),
        "r2": ThreshRequired(p_thresh=False, p_hat_thresh=False),
        "mse": ThreshRequired(p_thresh=False, p_hat_thresh=False),
        "mae": ThreshRequired(p_thresh=False, p_hat_thresh=False),
    }
    try:
        required_tresh = required_treshs[metric]
    except KeyError as e:
        raise KeyError(
            f"This metric is not recognized. Please use one of the following: {', '.join(required_treshs.keys())}"
        )

    required_tresh.check_threshs_correct(p_thresh=p_thresh, p_hat_thresh=p_hat_thresh)


def _sensitivity(true, pred, sample_weight):
    tn, fp, fn, tp = confusion_matrix(true, pred, sample_weight=sample_weight).ravel()
    return tp / (tp + fn)


def _specificity(true, pred, sample_weight):
    tn, fp, fn, tp = confusion_matrix(true, pred, sample_weight=sample_weight).ravel()
    return tn / (tn + fp)


def _fpr(true, pred, sample_weight):
    tn, fp, fn, tp = confusion_matrix(true, pred, sample_weight=sample_weight).ravel()
    return fp / (fp + tn)


def _fnr(true, pred, sample_weight):
    tn, fp, fn, tp = confusion_matrix(true, pred, sample_weight=sample_weight).ravel()
    return fn / (fn + tp)


def _auc(true, pred, sample_weight):
    fpr, tpr, _ = sk_metrics.roc_curve(true, pred, sample_weight=sample_weight)
    return sk_metrics.auc(fpr, tpr)


def _precision(true, pred, sample_weight):
    tn, fp, fn, tp = confusion_matrix(true, pred, sample_weight=sample_weight).ravel()
    return tp / (tp + fp)


def _npv(true, pred, sample_weight):
    tn, fp, fn, tp = confusion_matrix(true, pred, sample_weight=sample_weight).ravel()
    return tn / (tn + fn)


def score(
    cases: pd.DataFrame,
    signals: pd.DataFrame,
    metric: sk_metrics,
    threshsold_true: Optional[float] = None,
    threshsold_pred: Optional[float] = None,
    weights: Optional[str] = None,
    gauss_dims: Optional[list] = None,
    covariance_diag: Optional[list[float]] = None,
    time_axis: Optional[str] = None,
):
    check_threshs(metric, threshsold_true, threshsold_pred)
    metrics = {
        "f1": sk_metrics.f1_score,
        "brier": sk_metrics.brier_score_loss,
        "auc": _auc,
        "sensitivity": _sensitivity,
        "recall": _sensitivity,
        "tpr": _sensitivity,
        "specificity": _specificity,
        "tnr": _specificity,
        "fpr": _fpr,
        "fnr": _fnr,
        "precision": _precision,
        "ppv": _precision,
        "npv": _npv,
        "matthews": sk_metrics.matthews_corrcoef,
        "r2": sk_metrics.r2_score,
        "mse": sk_metrics.mean_squared_error,
        "mae": sk_metrics.mean_absolute_error,
    }
    return Score(cases, signals).calc_score(
        metrics[metric],
        threshsold_true,
        threshsold_pred,
        weights,
        gauss_dims,
        covariance_diag,
        time_axis,
    )


def conf_matrix(
    cases: pd.DataFrame,
    signals: pd.DataFrame,
    threshsold_true: float,
    threshsold_pred: float,
):
    return Score(cases, signals).class_based_conf_mat(threshsold_true, threshsold_pred)


def timeliness(
    cases: pd.DataFrame, signals: pd.DataFrame, time_axis: str, D: int, signal_threshold: float = 0
):
    return EpiMetrics(cases, signals).timeliness(time_axis, D, signal_threshold)
