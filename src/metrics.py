"""Shared metric utilities. Written for Phase 5 (which needs EER to tune on dev)
and reused verbatim by Phase 7.

SCORE CONVENTION, project-wide: **higher score = more bonafide**.

This matches the ASVspoof official baseline `score.txt` files, which hold
log-likelihood ratios oriented so that higher favours bonafide. Keeping the same
orientation everywhere means Phase 7 can join this project's scores against the
four official baselines (CQCC-GMM / LFCC-GMM / LFCC-LCNN / RawNet2) without any
sign flipping -- a small decision now that removes a whole class of silent
sign-error bugs later.

Label encoding that goes with it: **1 = bonafide (positive class), 0 = spoof.**
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_eer(bonafide_scores, spoof_scores) -> tuple[float, float]:
    """Equal Error Rate and the threshold at which it occurs.

    EER is rank-based (it is read off the ROC curve), so it only needs a
    continuous, monotonically-ordered score -- NOT a calibrated probability.
    That is why the SVM here uses `decision_function` rather than
    `probability=True`, which would trigger an internal 5-fold Platt-scaling CV
    at ~5x the fit cost while leaving this number essentially unchanged.
    """
    bonafide_scores = np.asarray(bonafide_scores, dtype=float)
    spoof_scores = np.asarray(spoof_scores, dtype=float)

    scores = np.concatenate([bonafide_scores, spoof_scores])
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])

    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fnr[idx] + fpr[idx]) / 2.0
    return float(eer), float(thresholds[idx])


def eer_from_labels(y_true, scores) -> tuple[float, float]:
    """Convenience wrapper. y_true: 1 = bonafide, 0 = spoof."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    return compute_eer(scores[y_true == 1], scores[y_true == 0])


def metrics_at_threshold(y_true, scores, threshold: float) -> dict:
    """Confusion matrix + the intuitive supplementary metrics at one operating point.

    Per PROJECT_PLAN.md section 7.2/7.3 these are reported at the EER threshold,
    and the thesis should state explicitly that this operating point was chosen
    deliberately -- it is a design decision, not a neutral default.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision_bonafide": float(precision),
        "recall_bonafide": float(recall),
        "f1_bonafide": float(f1),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "tp_bonafide_accepted": int(tp),
        "fn_bonafide_rejected": int(fn),
        "tn_spoof_rejected": int(tn),
        "fp_spoof_accepted": int(fp),
    }


def full_report(y_true, scores) -> dict:
    """EER (primary) plus the supplementary metrics at the EER threshold."""
    eer, threshold = eer_from_labels(y_true, scores)
    report = {"eer": eer}
    report.update(metrics_at_threshold(y_true, scores, threshold))
    return report
