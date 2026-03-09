import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, hamming_loss, accuracy_score
)

CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


def compute_metrics(y_true, y_pred_binary, y_pred_proba):
    """
    Compute full evaluation metrics for multilabel ECG classification.

    Parameters
    ----------
    y_true         : np.ndarray  (N, 5)  ground truth binary labels
    y_pred_binary  : np.ndarray  (N, 5)  thresholded predictions (>=0.5)
    y_pred_proba   : np.ndarray  (N, 5)  raw sigmoid probabilities

    Returns
    -------
    metrics : dict  with macro/per-class scores
    """

    # ── Macro averaged ────────────────────────────────────────────────────────
    macro_f1        = f1_score(y_true, y_pred_binary, average='macro',  zero_division=0)
    macro_precision = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
    macro_recall    = recall_score(y_true, y_pred_binary, average='macro',    zero_division=0)

    # AUC-ROC: only compute for classes that have both positive and negative samples
    try:
        macro_auc = roc_auc_score(y_true, y_pred_proba, average='macro')
    except ValueError:
        macro_auc = None   # happens if a class has no positive samples in the set

    # Hamming loss — fraction of individual label predictions that are wrong
    h_loss = hamming_loss(y_true, y_pred_binary)

    # Subset accuracy — entire label vector must match exactly
    subset_acc = accuracy_score(y_true, y_pred_binary)

    # ── Per-class ─────────────────────────────────────────────────────────────
    per_f1        = f1_score(y_true, y_pred_binary, average=None, zero_division=0)
    per_precision = precision_score(y_true, y_pred_binary, average=None, zero_division=0)
    per_recall    = recall_score(y_true, y_pred_binary, average=None,    zero_division=0)

    per_auc = []
    for i in range(y_true.shape[1]):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred_proba[:, i])
        except ValueError:
            auc = None
        per_auc.append(auc)

    # ── Build output dict ─────────────────────────────────────────────────────
    metrics = {
        # Macro
        "macro_f1":        float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall":    float(macro_recall),
        "macro_auc":       float(macro_auc) if macro_auc is not None else None,
        "hamming_loss":    float(h_loss),
        "subset_accuracy": float(subset_acc),

        # Per-class
        "per_class": {
            CLASS_NAMES[i]: {
                "f1":        float(per_f1[i]),
                "precision": float(per_precision[i]),
                "recall":    float(per_recall[i]),
                "auc":       float(per_auc[i]) if per_auc[i] is not None else None,
            }
            for i in range(len(CLASS_NAMES))
        }
    }

    return metrics


def print_metrics(metrics, label=""):
    """Pretty-print the metrics dict returned by compute_metrics."""
    if label:
        print(f"\n===== RESULTS: {label} =====")
    else:
        print("\n===== RESULTS =====")

    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision : {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall    : {metrics['macro_recall']:.4f}")
    auc_str = f"{metrics['macro_auc']:.4f}" if metrics['macro_auc'] is not None else "N/A"
    print(f"  Macro AUC-ROC   : {auc_str}")
    print(f"  Hamming Loss    : {metrics['hamming_loss']:.4f}")
    print(f"  Subset Accuracy : {metrics['subset_accuracy']:.4f}")

    print(f"\n  {'Class':<6}  {'F1':>6}  {'Prec':>6}  {'Recall':>6}  {'AUC':>6}")
    print(f"  {'-'*42}")
    for c, v in metrics['per_class'].items():
        auc = f"{v['auc']:.4f}" if v['auc'] is not None else "  N/A"
        print(f"  {c:<6}  {v['f1']:>6.4f}  {v['precision']:>6.4f}  {v['recall']:>6.4f}  {auc:>6}")
