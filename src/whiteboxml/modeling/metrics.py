"""Machine Learning metrics module.

This module implements utilities to measure the performance of Machine Learning models.
It is built as a wrapper on top of libraries like scikit-learn and Matplotlib, but easier
to use and with extended functionality.
"""

####################################################################################################
# IMPORTS

from typing import Tuple, Iterable

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

####################################################################################################
# FUNCTIONS

def plot_roc_auc_binary(y_pred: Iterable[float],
                        y_true: Iterable[float],
                        figsize: Tuple[int, int] = (8, 8)) -> tuple:
    """Computes the roc curve and auc metrics for a binary classification
    problem.

        Args:
            y_pred: an iterable with the predicted probabilities.
            y_true: an iterable with the ground truth (1s and 0s).

        Returns:
            The roc curve plot with its associated metrics (fpr, tpr, thr, auc_score).
    """

    # parameters
    COLOR_ROC = 'darkorange'
    COLOR_BASELINE = 'navy'
    LINE_STYLE_ROC = None
    LINE_STYLE_BASELINE = '--'
    LINE_WIDTH = 1

    # metrics computation
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    # roc
    ax.plot(fpr, tpr,
            color=COLOR_ROC,
            lw=LINE_WIDTH,
            linestyle=LINE_STYLE_ROC,
            label=f'roc curve (area/auc = {auc_score:.2f})')

    # baseline
    ax.plot([0, 1], [0, 1],
            color=COLOR_BASELINE,
            lw=LINE_WIDTH,
            linestyle=LINE_STYLE_BASELINE)

    # style
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('false positive Rate')
    ax.set_ylabel('true positive Rate')
    ax.set_title('receiver operating characteristic curve')
    ax.legend(loc="lower right")

    return ax, fpr, tpr, thr, auc_score
