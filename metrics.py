import numpy as np

def get_confusion_matrix(true_patch_matrix, pred_patch_matrix):
    """
    Calculate the confusion matrix for the given true and predicted patch matrices.
    """
    # Flatten the matrices
    true_flat = true_patch_matrix.flatten()
    pred_flat = pred_patch_matrix.flatten()

    # Calculate TP, TN, FP, FN
    TP = np.sum((true_flat == 1) & (pred_flat == 1))
    TN = np.sum((true_flat == 0) & (pred_flat == 0))
    FP = np.sum((true_flat == 0) & (pred_flat == 1))
    FN = np.sum((true_flat == 1) & (pred_flat == 0))

    return TP, TN, FP, FN

def calculate_iou(true_patch_matrix, pred_patch_matrix):
    """
    Calculate the Intersection over Union (IoU) for the given true and predicted patch matrices.
    """
    # Flatten the matrices
    true_flat = true_patch_matrix.flatten()
    pred_flat = pred_patch_matrix.flatten()

    # Calculate intersection and union
    intersection = np.sum((true_flat == 1) & (pred_flat == 1))
    union = np.sum((true_flat == 1) | (pred_flat == 1))

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou

def calculate_auc(true_patch_matrices, pred_patch_matrices):
    """
    Calculate the Area Under the Curve (AUC) for the given true and predicted patch matrices.
    """
    # Flatten the matrices
    true_flat = np.concatenate([matrix.flatten() for matrix in true_patch_matrices])
    pred_flat = np.concatenate([matrix.flatten() for matrix in pred_patch_matrices])

    # Sort by predicted scores
    sorted_indices = np.argsort(pred_flat)
    true_flat_sorted = true_flat[sorted_indices]
    pred_flat_sorted = pred_flat[sorted_indices]

    # Calculate AUC
    auc = np.trapz(true_flat_sorted, pred_flat_sorted)
    return auc

