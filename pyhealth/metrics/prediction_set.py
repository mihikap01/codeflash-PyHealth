import numpy as np


def size(y_pred: np.ndarray):
    """Average size of the prediction set."""
    return np.mean(y_pred.sum(1))


def rejection_rate(y_pred: np.ndarray):
    """Rejection rate, defined as the proportion of samples with prediction set size != 1"""
    return np.mean(y_pred.sum(1) != 1)


def _missrate(y_pred: np.ndarray, y_true: np.ndarray, ignore_rejected=False):
    """Computes the class-wise mis-coverage rate (or risk).

    Args:
        y_pred (np.ndarray): prediction scores.
        y_true (np.ndarray): true labels.
        ignore_rejected (bool, optional): If True, we compute the miscoverage rate
            without rejection  (that is, condition on the unrejected samples). Defaults to False.

    Returns:
        np.ndarray: miss-coverage rates for each class.
    """
    # currently handles multilabel and multiclass
    K = y_pred.shape[1]
    # Optimized one-hot conversion (if y_true is 1D)
    if y_true.ndim == 1:
        idx = y_true
        y_true = np.zeros((y_pred.shape[0], K), dtype=bool)
        y_true[np.arange(y_pred.shape[0]), idx] = True
    else:
        y_true = y_true.astype(bool, copy=False)

    # Compute keep_msk in a vectorized way
    if ignore_rejected:
        keep_msk = y_pred.sum(1) == 1
    else:
        keep_msk = None  # None is better for advanced indexing

    # Vectorized filtering: Get mask for "relevant" samples (for each class)
    # mask shape: (n_samples, K), True if sample should be included for each class
    # For each class, a sample is "kept" if
    #  - (not rejected if ignore_rejected) and
    #  - true label for class is True in this sample
    if ignore_rejected:
        included_mask = keep_msk[:, None] & y_true
    else:
        included_mask = y_true

    # For every class: gather predicted value for those samples that are kept and are true for class k
    # This produces an array of shape (n_samples, K); we then process columnwise.
    pred_included = np.where(included_mask, y_pred, 0)

    # Count how many for each class are included (ensure not to divide by zero)
    n_included = included_mask.sum(axis=0)
    # To avoid division by zero, mark as 1 for now, we'll fill with nan later
    n_included_safe = np.where(n_included == 0, 1, n_included)

    # For each class: sum preds for hit ratio, then miss = 1-mean
    mean_pred = pred_included.sum(axis=0) / n_included_safe
    missed = 1 - mean_pred

    # Where there were no samples, set miss to nan (same as np.mean([]) = nan)
    missed = np.where(n_included == 0, np.nan, missed)

    return missed


def miscoverage_ps(y_pred: np.ndarray, y_true: np.ndarray):
    """Miscoverage rates for all samples (similar to recall).

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0],[0, 1, 0]])
        >>> y_true = np.asarray([1,0,1,2])
        >>> error_ps(y_pred, y_true)
        array([0. , 0.5, 1. ])


    Explanation:
    For class 0, the 1-th prediction set ({0}) contains the label, so the miss-coverage is 0/1=0.
    For class 1, the 0-th prediction set ({0}) does not contain the label, the 2-th prediction
    set ({0,1}) contains the label. Thus, the miss-coverage is 1/2=0.5.
    For class 2, the last prediction set is {1} and the label is 2, so the miss-coverage is 1/1=1.
    """
    return _missrate(y_pred, y_true, False)


def error_ps(y_pred: np.ndarray, y_true: np.ndarray):
    """Miscoverage rates for unrejected samples, where rejection is defined to be sets with size !=1).

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0],[0, 1, 0]])
        >>> y_true = np.asarray([1,0,1,2])
        >>> error_ps(y_pred, y_true)
        array([0., 1., 1.])

    Explanation:
    For class 0, the 1-th sample is correct and not rejected, so the error is 0/1=0.
    For class 1, the 0-th sample is incorrerct and not rejected, the 2-th is rejected.
    Thus, the error is 1/1=1.
    For class 2, the last sample is not-rejected but the prediction set is {1}, so the error
    is 1/1=1.
    """
    return _missrate(y_pred, y_true, True)


def miscoverage_overall_ps(y_pred: np.ndarray, y_true: np.ndarray):
    """Miscoverage rate for the true label. Only for multiclass.

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0]])
        >>> y_true = np.asarray([1,0,1])
        >>> miscoverage_overall_ps(y_pred, y_true)
        0.333333

    Explanation:
    The 0-th prediction set is {0} and the label is 1 (not covered).
    The 1-th prediction set is {0} and the label is 0 (covered).
    The 2-th prediction set is {0,1} and the label is 1 (covered).
    Thus the miscoverage rate is 1/3.
    """
    assert len(y_true.shape) == 1
    truth_pred = y_pred[np.arange(len(y_true)), y_true]

    return 1 - np.mean(truth_pred)


def error_overall_ps(y_pred: np.ndarray, y_true: np.ndarray):
    """Overall error rate for the un-rejected samples.

    Example:
        >>> y_pred = np.asarray([[1,0,0],[1,0,0],[1,1,0]])
        >>> y_true = np.asarray([1,0,1])
        >>> error_overall_ps(y_pred, y_true)
        0.5

    Explanation:
    The 0-th prediction set is {0} and the label is 1, so it is an error (no rejection
    as its prediction set has only one class).
    The 1-th sample is not rejected and incurs on error.
    The 2-th sample is rejected, thus excluded from the computation.
    """
    assert len(y_true.shape) == 1
    truth_pred = y_pred[np.arange(len(y_true)), y_true]
    truth_pred = truth_pred[y_pred.sum(1) == 1]
    return 1 - np.mean(truth_pred)
