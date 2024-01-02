import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def compute_roc_auc(y_true, y_pred_probs, num_classes):
    """
    Compute ROC curve and AUC for each class in a multi-class classification.

    Args:
    y_true: True labels.
    y_pred_probs: Predicted probabilities for each class.
    num_classes: Number of classes.

    Returns:
    Dictionary containing FPR, TPR, and AUC for each class.
    """
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return fpr, tpr, roc_auc


def compute_mean_auc(y_true, y_pred_probs, num_classes):
    """
    Compute mean AUC for multi-class classification.

    Args:
    y_true: True labels.
    y_pred_probs: Predicted probabilities for each class.
    num_classes: Number of classes.

    Returns:
    Mean AUC score.
    """
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))
    return roc_auc_score(y_true_binarized, y_pred_probs, multi_class='ovr')
def plot_roc_curves(fpr, tpr, roc_auc, num_classes, class_names):
    """
    Plots ROC curves for each class.

    Args:
    fpr: Dictionary containing false positive rates for each class.
    tpr: Dictionary containing true positive rates for each class.
    roc_auc: Dictionary containing AUC values for each class.
    num_classes: Number of classes.
    class_names: List of class names for labeling.
    """
    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each class
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for each class')
    plt.legend(loc="lower right")
    plt.show()