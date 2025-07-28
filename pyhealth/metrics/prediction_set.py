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
    K = y_pred.shape[1]
    # Fast one-hot encoding only if y_true is 1D (multiclass)
    if y_true.ndim == 1:
        oh = np.zeros((len(y_true), K), dtype=bool)
        oh[np.arange(len(y_true)), y_true] = True
        y_true_bool = oh
    else:
        # Already multilabel: just ensure boolean dtype
        y_true_bool = y_true.astype(bool, copy=False)

    # Mask for unrejected/unmasked samples
    if ignore_rejected:
        keep_msk = y_pred.sum(1) == 1
    else:
        keep_msk = np.ones(y_pred.shape[0], dtype=bool)

    # Vectorized calculation
    # mask: samples to keep for each class
    sel = keep_msk[:, None] & y_true_bool

    # Avoid zero division: num_samples_per_class=min 1, otherwise we get nan (matches np.mean's behavior)
    num_samples_per_class = sel.sum(0)
    # The mask is empty for a class => np.mean([]) returns nan, so we control this behavior

    # sum of predicted-positive in kept true samples per class
    sum_pred = (y_pred * sel).sum(0)
    result = np.ones(K) - np.divide(
        sum_pred,
        num_samples_per_class,
        out=np.full(K, np.nan, dtype=float),  # behave like np.mean([])==nan
        where=(num_samples_per_class != 0),
    )

    return result


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
