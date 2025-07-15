"""Functions required for conformal prediction task."""

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def ecfp_generator(smiles_list: List) -> List:
    """
    Generate ECFP (Extended-Connectivity Fingerprint) bit vectors from a list of SMILES strings.

    Parameters
    ----------
    smiles_list : List
        A list of SMILES (Simplified Molecular Input Line Entry System) strings representing molecules.

    Returns
    -------
    List
        A NumPy array of shape (n_samples, 2048), where each row is a 2048-bit ECFP4 fingerprint vector.
    """
    ecfp_fvs = np.zeros((len(smiles_list), 2048), dtype=float)

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES string: {smiles}")
            continue
        ecfp_fvs[i] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return ecfp_fvs


def probability_dataframe(models, folds_dict, name, split_type="val"):
    """
    Generate a combined DataFrame of predicted probabilities and labels,
    and save it to a CSV file.

    Parameters
    ----------
    models : list
        List of trained models.

    folds_dict : dict
        Dictionary containing split data.

    name : str
        Key prefix used to access split data in `folds_dict`.

    split_type : str, optional
        Data split type, either "val" or "test" (default is "val").

    Returns
    -------
    pd.DataFrame
        Combined DataFrame of predicted probabilities and labels.
    """
    pred_df_list = []
    label_list = []

    for i, model in enumerate(models):
        split = folds_dict[f"{name}_split_{i}"]
        pred_df_list.append(
            pd.DataFrame(model.predict_proba(split[f"{split_type}_fv"]))
        )
        label_list.append(split[f"{split_type}_label"])

    label_df = pd.DataFrame(label_list).T
    label_df.columns = [f"label_{i}" for i in range(len(models))]
    pred_df = pd.concat(pred_df_list, axis=1)
    pred_df.columns = [
        f"{cls}_{i}" for i in range(len(models)) for cls in ("active", "inactive")
    ]

    final_df = pd.concat([pred_df, label_df], axis=1)
    final_df.to_csv(
        f"svm_ecfp_{uniprot_to_isoform[name]}_{split_type}_prob.csv", index=False
    )

    return final_df


def run_data(
    df: pd.DataFrame, run_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts probabilities and labels for a specific run.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of predicted probabilities and labels.
    run_idx : int
        Index of a run.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple of predicted probabilities for actives and inactives alongside labels.
    """
    prob_active_col = f"active_{run_idx}"
    prob_inactive_col = f"inactive_{run_idx}"
    label_col = f"label_{run_idx}"

    p_active = df[prob_active_col].values
    p_inactive = df[prob_inactive_col].values
    labels = df[label_col].values.astype(int)
    return p_active, p_inactive, labels


def calculate_conformity_scores(
    p_active_cal: np.ndarray, p_inactive_cal: np.ndarray, labels_cal: np.ndarray
) -> Dict:
    """Calculates class-conditional conformity scores for the calibration set.
    Score = probability of the true class, stored per class.

    Parameters
    ----------
    p_active_cal : np.ndarray
        Predicted probabilities for actives.
    p_inactive_cal : np.ndarray
        Predicted probabilities for inactives.
    labels_cal : np.ndarray
        True labels.

    Returns
    -------
    Dict
        Dictionary of conformity score for both classes.
    """
    conformity_scores_dict = {0: [], 1: []}

    for i in range(len(labels_cal)):
        true_label = labels_cal[i]
        if true_label == 1:
            score = p_active_cal[i]
            conformity_scores_dict[1].append(score)
        elif true_label == 0:
            score = p_inactive_cal[i]
            conformity_scores_dict[0].append(score)

    conformity_scores_dict[0] = np.array(conformity_scores_dict[0])
    conformity_scores_dict[1] = np.array(conformity_scores_dict[1])
    return conformity_scores_dict


def get_conformal_prediction_set(
    test_p_active_single: float,
    test_p_inactive_single: float,
    cal_scores_dict: Dict,
    epsilon: float,
) -> List:
    """Calculates the prediction set for a single test instance using class-conditional conformity scores.

    Parameters
    ----------
    test_p_active_single : float
        Predicted probabilities for actives.
    test_p_inactive_single : float
        Predicted probabilities for inactives.
    cal_scores_dict : Dict
        Dictionary of conformity score for both classes.
    epsilon : float
        Desired confidence.

    Returns
    -------
    List
        Prediction set.
    """

    prediction_set = []

    conformity_scores_class_0 = cal_scores_dict[0]
    n_cal_0 = len(conformity_scores_class_0)
    score_test_as_0 = test_p_inactive_single

    p_value_0 = (np.sum(conformity_scores_class_0 <= score_test_as_0) + 1) / (
        n_cal_0 + 1
    )

    if p_value_0 > epsilon:
        prediction_set.append(0)

    conformity_scores_class_1 = cal_scores_dict[1]
    n_cal_1 = len(conformity_scores_class_1)
    score_test_as_1 = test_p_active_single

    p_value_1 = (np.sum(conformity_scores_class_1 <= score_test_as_1) + 1) / (
        n_cal_1 + 1
    )

    if p_value_1 > epsilon:
        prediction_set.append(1)

    return prediction_set


def evaluate_conformal_predictor(
    test_p_active: np.ndarray,
    test_p_inactive: np.ndarray,
    test_labels: np.ndarray,
    cal_scores_dict: dict,
    epsilon: float,
) -> Tuple[float, float, float]:
    """Evaluates validity and efficiency for a given test set and epsilon,
    using class-conditional conformity scores.

    Parameters
    ----------
    test_p_active : np.ndarray
        Predicted probabilities for actives.
    test_p_inactive : np.ndarray
        Predicted probabilities for inactives.
    test_labels : np.ndarray
        True labels for the test set.
    cal_scores_dict : dict
        Dictionary of conformity score for both classes.
    epsilon : float
        Desired confidence.

    Returns
    -------
    Tuple[float, float, float]
        Validity, Avg Efficiency (Two Label), Avg Empty Rate
    """
    num_test = len(test_labels)

    correct_predictions = 0
    two_label_predictions = 0
    empty_predictions = 0

    prediction_sets = []

    for i in range(num_test):
        pred_set = get_conformal_prediction_set(
            test_p_active[i], test_p_inactive[i], cal_scores_dict, epsilon
        )
        prediction_sets.append(pred_set)

        if test_labels[i] in pred_set:
            correct_predictions += 1
        if len(pred_set) == 2:
            two_label_predictions += 1
        elif len(pred_set) == 0:
            empty_predictions += 1

    validity = correct_predictions / num_test
    efficiency_two_label = two_label_predictions / num_test
    empty_rate = empty_predictions / num_test

    return validity, efficiency_two_label, empty_rate
