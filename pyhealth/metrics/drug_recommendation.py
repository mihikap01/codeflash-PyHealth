from typing import List

import numpy as np


# TODO: this metric is very ad-hoc, need to be improved


def ddi_rate_score(medications: List[np.ndarray], ddi_matrix: np.ndarray) -> float:
    """DDI rate score.

    Args:
        medications: list of medications for each patient, where each medication
            is represented by the corresponding index in the ddi matrix.
        ddi_matrix: array-like of shape (n_classes, n_classes).

    Returns:
        result: DDI rate score.
    """
    all_cnt = 0
    ddi_cnt = 0

    # Precompute element-wise logical OR of ddi_matrix and its transpose for possible asymmetry
    ddi_mtx = np.logical_or(ddi_matrix, ddi_matrix.T)

    for sample in medications:
        n = len(sample)
        if n < 2:
            continue
        # Get all unique pairs (i<j) via np.triu_indices
        med_indices = np.array(sample)
        # Use numpy broadcasting to get upper triangle indices
        a, b = np.triu_indices(n, k=1)
        all_cnt += len(a)
        ddi_cnt += np.count_nonzero(ddi_mtx[med_indices[a], med_indices[b]])

    if all_cnt == 0:
        return 0
    return ddi_cnt / all_cnt
