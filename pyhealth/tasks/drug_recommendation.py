from pyhealth.data import Patient, Visit


def drug_recommendation_mimic3_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key, like this
            {
                "patient_id": xxx,
                "visit_id": xxx,
                "conditions": [list of diag in visit 1, list of diag in visit 2, ..., list of diag in visit N],
                "procedures": [list of prod in visit 1, list of prod in visit 2, ..., list of prod in visit N],
                "drugs_hist": [list of drug in visit 1, list of drug in visit 2, ..., list of drug in visit (N-1)],
                "drugs": list of drug in visit N, # this is the predicted target
            }

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> mimic3_base = MIMIC3Dataset(
        ...    root="/srv/local/data/physionet.org/files/mimiciii/1.4",
        ...    tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        ...    code_mapping={"ICD9CM": "CCSCM"},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_mimic3_fn
        >>> mimic3_sample = mimic3_base.set_task(drug_recommendation_mimic3_fn)
        >>> mimic3_sample.samples[0]
        {
            'visit_id': '174162',
            'patient_id': '107',
            'conditions': [['139', '158', '237', '99', '60', '101', '51', '54', '53', '133', '143', '140', '117', '138', '55']],
            'procedures': [['4443', '4513', '3995']],
            'drugs_hist': [[]],
            'drugs': ['0000', '0033', '5817', '0057', '0090', '0053', '0', '0012', '6332', '1001', '6155', '1001', '6332', '0033', '5539', '6332', '5967', '0033', '0040', '5967', '5846', '0016', '5846', '5107', '5551', '6808', '5107', '0090', '5107', '5416', '0033', '1150', '0005', '6365', '0090', '6155', '0005', '0090', '0000', '6373'],
        }
    """
    # Use patient attribute if it exists for faster access
    visits = getattr(patient, "visits", None)
    if visits is None:
        # Fallback: old style __getitem__ support
        visits = [patient[i] for i in range(len(patient))]

    # Pre-extract relevant codes for each visit, filter valid ones
    conds, procs, drugs_lists, valid_visits = [], [], [], []
    for visit in visits:
        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = [drug[:4] for drug in visit.get_code_list(table="PRESCRIPTIONS")]

        if conditions and procedures and drugs:
            conds.append(conditions)
            procs.append(procedures)
            drugs_lists.append(drugs)
            valid_visits.append(visit)

    n = len(valid_visits)
    # exclude: patients with less than 2 valid visits
    if n < 2:
        return []

    patient_id = patient.patient_id
    out_samples = []
    # We'll build history lists as lists of lists: history[i] is up to visit i
    # The i-th output should have:
    #  - "conditions": [cond0, cond1, ..., condi]
    #  - "procedures": [proc0, proc1, ..., proci]
    #  - "drugs_hist": [drugs0, drugs1, ..., drugs(i-1), []]
    # (with drugs_hist[i] == [] for this target)

    # Instead of inefficient deep copies, we can reuse history containers and add new elements in-place,
    # then shallow-copy for each sample output.

    cond_hist = []
    proc_hist = []
    drug_hist = []

    # We collect samples as we go, appending to the growing lists.
    for i in range(n):
        cond_hist.append(conds[i])
        proc_hist.append(procs[i])
        drug_hist.append(drugs_lists[i])
        # For "drugs_hist": make a shallow copy, then
        # clear the current target index to empty as per requirement:
        drugs_hist_out = drug_hist.copy()
        drugs_hist_out[i] = []
        # Only include up-to-current point in the histories (not beyond)
        sample = {
            "visit_id": valid_visits[i].visit_id,
            "patient_id": patient_id,
            "conditions": cond_hist.copy(),
            "procedures": proc_hist.copy(),
            "drugs_hist": drugs_hist_out,
            "drugs": drugs_lists[i],  # target
        }
        out_samples.append(sample)
    return out_samples


def drug_recommendation_mimic4_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key
            {
                "patient_id": xxx,
                "visit_id": xxx,
                "conditions": [list of diag in visit 1, list of diag in visit 2, ..., list of diag in visit N],
                "procedures": [list of prod in visit 1, list of prod in visit 2, ..., list of prod in visit N],
                "drugs_hist": [list of drug in visit 1, list of drug in visit 2, ..., list of drug in visit (N-1)],
                "drugs": list of drug in visit N, # this is the predicted target
            }

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> mimic4_base = MIMIC4Dataset(
        ...     root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        ...     tables=["diagnoses_icd", "procedures_icd"],
        ...     code_mapping={"ICD10PROC": "CCSPROC"},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_mimic4_fn
        >>> mimic4_sample = mimic4_base.set_task(drug_recommendation_mimic4_fn)
        >>> mimic4_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '19', '122', '98', '663', '58', '51']], 'procedures': [['1']], 'label': [['2', '3', '4']]}]
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # ATC 3 level
        drugs = [drug[:4] for drug in drugs]
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_hist": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_hist"] = [samples[0]["drugs_hist"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_hist"] = samples[i - 1]["drugs_hist"] + [
            samples[i]["drugs_hist"]
        ]

    # remove the target drug from the history
    for i in range(len(samples)):
        samples[i]["drugs_hist"][i] = []

    return samples


def drug_recommendation_eicu_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> eicu_base = eICUDataset(
        ...     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication"],
        ...     code_mapping={},
        ...     dev=True
        ... )
        >>> from pyhealth.tasks import drug_recommendation_eicu_fn
        >>> eicu_sample = eicu_base.set_task(drug_recommendation_eicu_fn)
        >>> eicu_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51']], 'procedures': [['1']], 'label': [['2', '3', '4']]}]
    """
    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="diagnosis")
        procedures = visit.get_code_list(table="physicalExam")
        drugs = visit.get_code_list(table="medication")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples


def drug_recommendation_omop_fn(patient: Patient):
    """Processes a single patient for the drug recommendation task.

    Drug recommendation aims at recommending a set of drugs given the patient health
    history  (e.g., conditions and procedures).

    Args:
        patient: a Patient object

    Returns:
        samples: a list of samples, each sample is a dict with patient_id, visit_id,
            and other task-specific attributes as key

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> omop_base = OMOPDataset(
        ...     root="https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2",
        ...     tables=["condition_occurrence", "procedure_occurrence"],
        ...     code_mapping={},
        ... )
        >>> from pyhealth.tasks import drug_recommendation_omop_fn
        >>> omop_sample = omop_base.set_task(drug_recommendation_eicu_fn)
        >>> omop_sample.samples[0]
        [{'visit_id': '130744', 'patient_id': '103', 'conditions': [['42', '109', '98', '663', '58', '51'], ['98', '663', '58', '51']], 'procedures': [['1'], ['2', '3']], 'label': [['2', '3', '4'], ['0', '1', '4', '5']]}]
    """

    samples = []
    for i in range(len(patient)):
        visit: Visit = patient[i]
        conditions = visit.get_code_list(table="condition_occurrence")
        procedures = visit.get_code_list(table="procedure_occurrence")
        drugs = visit.get_code_list(table="drug_exposure")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        # TODO: should also exclude visit with age < 18
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": conditions,
                "procedures": procedures,
                "drugs": drugs,
                "drugs_all": drugs,
            }
        )
    # exclude: patients with less than 2 visit
    if len(samples) < 2:
        return []
    # add history
    samples[0]["conditions"] = [samples[0]["conditions"]]
    samples[0]["procedures"] = [samples[0]["procedures"]]
    samples[0]["drugs_all"] = [samples[0]["drugs_all"]]

    for i in range(1, len(samples)):
        samples[i]["conditions"] = samples[i - 1]["conditions"] + [
            samples[i]["conditions"]
        ]
        samples[i]["procedures"] = samples[i - 1]["procedures"] + [
            samples[i]["procedures"]
        ]
        samples[i]["drugs_all"] = samples[i - 1]["drugs_all"] + [
            samples[i]["drugs_all"]
        ]

    return samples


if __name__ == "__main__":
    # from pyhealth.datasets import MIMIC3Dataset
    # base_dataset = MIMIC3Dataset(
    #     root="/srv/local/data/physionet.org/files/mimiciii/1.4",
    #     tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    #     dev=True,
    #     code_mapping={"ICD9CM": "CCSCM"},
    #     refresh_cache=False,
    # )
    # sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_mimic3_fn)
    # sample_dataset.stat()
    # print(sample_dataset.available_keys)
    # print(sample_dataset.samples[0])

    from pyhealth.datasets import MIMIC4Dataset

    base_dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
        code_mapping={"NDC": "ATC"},
        refresh_cache=False,
    )
    sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_mimic4_fn)
    sample_dataset.stat()
    print(sample_dataset.available_keys)
    print(sample_dataset.samples[0])

    # from pyhealth.datasets import eICUDataset

    # base_dataset = eICUDataset(
    #     root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
    #     tables=["diagnosis", "medication", "physicalExam"],
    #     dev=True,
    #     refresh_cache=False,
    # )
    # sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_eicu_fn)
    # sample_dataset.stat()
    # print(sample_dataset.available_keys)

    # from pyhealth.datasets import OMOPDataset

    # base_dataset = OMOPDataset(
    #     root="/srv/local/data/zw12/pyhealth/raw_data/synpuf1k_omop_cdm_5.2.2",
    #     tables=["condition_occurrence", "procedure_occurrence", "drug_exposure"],
    #     dev=True,
    #     refresh_cache=False,
    # )
    # sample_dataset = base_dataset.set_task(task_fn=drug_recommendation_omop_fn)
    # sample_dataset.stat()
    # print(sample_dataset.available_keys)
