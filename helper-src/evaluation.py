"""evaluation.py
Compute the project's test metrics and draw the two required figures.

    Accuracy      share of correct predictions at the threshold
    MacroAUC      area under the ROC curve (for a binary label this is the usual ROC AUC)
    Precision_1   precision for class 1 (chronic pain)
    Recall_1      recall for class 1
    F1_1          F1-score for class 1

Usage:

from evaluation import compute_metrics, plot_confusion_matrix, plot_roc_curve
metrics = compute_metrics(y_test, test_prob)              # test_prob = P(class 1), one per row
plot_confusion_matrix(y_test, test_prob, "my-model-confusion-matrix-iter1.png")
plot_roc_curve(y_test, test_prob, "my-model-roc-curve-iter1.png")

Figures are drawn with matplotlib's Figure class (not pyplot), so they are saved to
file without opening a window and work on a server without a display.
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

METRIC_NAMES = ["Accuracy", "MacroAUC", "Precision_1", "Recall_1", "F1_1"]
CLASS_NAMES = ("No chronic pain (0)", "Chronic pain (1)")

# Colors: one blue for the ROC line, a light-to-dark blue ramp for the confusion matrix
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES_BLUE = "#2a78d6"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def _check_inputs(y_true, y_prob):
    """Return y_true as 0/1 ints and y_prob as floats, or raise a clear error."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(y_true) != len(y_prob):
        raise ValueError(f"y_true has {len(y_true)} values but y_prob has {len(y_prob)}; they must match.")
    if not set(np.unique(y_true)) <= {0, 1}:
        raise ValueError(f"y_true must contain only 0 and 1, found {sorted(np.unique(y_true).tolist())[:10]}")
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true contains only one class, so AUC is undefined. Evaluate on the full test split.")
    if np.isnan(y_prob).any():
        raise ValueError("y_prob contains NaN values. Check your model's output.")
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(
            "y_prob must be probabilities between 0 and 1 (the predicted chance of class 1). "
            "If your network outputs logits, apply a sigmoid first, e.g. torch.sigmoid(logits)."
        )
    return y_true.astype(int), y_prob


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Return {"Accuracy", "MacroAUC", "Precision_1", "Recall_1", "F1_1"} as plain floats.

    y_true: 0/1 labels. y_prob: predicted probability of class 1 for the same rows.
    A row is predicted as class 1 when y_prob >= threshold.
    """
    y_true, y_prob = _check_inputs(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "MacroAUC": float(roc_auc_score(y_true, y_prob)),
        "Precision_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def plot_confusion_matrix(y_true, y_prob, path, threshold: float = 0.5, name: str = None) -> Path:
    """Save a 2x2 confusion matrix (counts and % of each true class) as a PNG. Returns the path.

    name: optional run name shown in the title, e.g. "my-model-iter1".
    """
    y_true, y_prob = _check_inputs(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    row_share = cm / cm.sum(axis=1, keepdims=True)

    fig = Figure(figsize=(5.2, 4.4), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    # 3-point surface-colored edges leave a small gap between the cells
    ax.pcolormesh(row_share, cmap=BLUE_RAMP, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=3)
    ax.invert_yaxis()  # true class 0 on the top row
    for i in range(2):
        for j in range(2):
            ink = "#ffffff" if row_share[i, j] > 0.55 else INK
            ax.text(j + 0.5, i + 0.42, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=ink)
            ax.text(j + 0.5, i + 0.64, f"{row_share[i, j]:.0%} of true {i}", ha="center",
                    va="center", fontsize=8.5, color=ink)

    ax.set_xticks([0.5, 1.5], CLASS_NAMES)
    ax.set_yticks([0.5, 1.5], CLASS_NAMES, rotation=90, va="center")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix{': ' + name if name else ''}\n"
                 f"threshold {threshold:g}, n = {len(y_true):,}", fontsize=10.5, loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _style_axes(ax)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    print(f"Saved confusion matrix to {path.as_posix()}")
    return path


def plot_roc_curve(y_true, y_prob, path, name: str = None) -> Path:
    """Save the ROC curve (with its AUC) as a PNG. Returns the path.

    name: optional run name shown in the title, e.g. "my-model-iter1".
    """
    y_true, y_prob = _check_inputs(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig = Figure(figsize=(4.8, 4.8), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = fig.subplots()
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#9b9a95")
    ax.text(0.72, 0.62, "chance (AUC = 0.5)", rotation=45, rotation_mode="anchor",
            ha="center", va="top", fontsize=8, color=INK_SECONDARY)  # just below the diagonal
    ax.plot(fpr, tpr, linewidth=2, color=SERIES_BLUE, solid_capstyle="round")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(f"ROC curve{': ' + name if name else ''}\n"
                 f"AUC = {auc:.3f}, n = {len(y_true):,}", fontsize=10.5, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    _style_axes(ax)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    print(f"Saved ROC curve to {path.as_posix()}")
    return path
