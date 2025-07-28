from typing import List
import numpy as np

from pyhealth.datasets import BaseEHRDataset


def sensitive_attributes_from_patient_ids(
    dataset: BaseEHRDataset,
    patient_ids: List[str],
    sensitive_attribute: str,
    protected_group: str,
) -> np.ndarray:
    """
    Returns the desired sensitive attribute array from patient_ids.

    Args:
        dataset: Dataset object.
        patient_ids: List of patient IDs.
        sensitive_attribute: Sensitive attribute to extract.
        protected_group: Value of the protected group.

    Returns:
        Sensitive attribute array of shape (n_samples,).
    """

    patients = dataset.patients
    get_attr = sensitive_attribute
    # Build an array using list comprehension for much faster Python code.
    sensitive_attribute_array = np.fromiter(
        (
            1.0 if getattr(patients[pid], get_attr) == protected_group else 0.0
            for pid in patient_ids
        ),
        dtype=float,
        count=len(patient_ids),
    )
    return sensitive_attribute_array
